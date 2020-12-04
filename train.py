"""
훈련 및 평가 관련 함수들 구현
"""

import torch 
import time
from itertools import chain

import tqdm
from sklearn.metrics import f1_score
import numpy as np

from utils import epoch_time

# multigpu 일 때 설정
context_device = "cuda:0"
gloss_device = "cuda:1"

def train_one_epoch(train_data, gloss_dict, model, optimizer, criterion, model_path, args, save_ratio=0):
    
    logger = args.logger
    gloss_bsz = args.gloss_bsz
    max_grad_norm = args.max_grad_norm
    multigpu = args.multigpu
    
    # 한 에폭을 훈련시키는 함수. 
    model.train()
    total_loss = 0
    
    if save_ratio > 0 and save_ratio <= 1:
        save_each_it = int(len(train_data)*save_ratio)
    else:
        save_each_it = 0
    
    for i, (context_ids, context_attn_mask, context_output_mask, words, sense_ids) in tqdm.tqdm(enumerate(train_data)):
        
        model.zero_grad()
        
        # 컨텍스트 인코더 계산
        if multigpu:
            context_ids = context_ids.to(context_device)
            context_attn_mask = context_attn_mask.to(context_device)
        else:
            context_ids = context_ids.to('cuda')
            context_attn_mask = context_attn_mask.to('cuda')
        

        context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)
        # context_output : (배치 내 다의어 수, hidden_dim)
        # 여기서 '배치 내 다의어 수'는 len(chain(*words))과 같아야함
        # 다의어가 하나도 없는 경우 None을 리턴하므로 패스
        if context_output is None:
            logger.debug(f"No multiwords : {words}")
            continue        
        
        # 의미 인코더 계산
        loss = 0.
        gloss_sz = 0
        
        words_org = list(chain(*words))
        # sense_id 만 저장
        sense_ids_org = chain(*[list(sense_d.values()) for sense_d in sense_ids])
        
        # sense_id 가 -1인 경우 제외
        words = []
        sense_ids = []
        for w, v in zip(words_org, sense_ids_org):
            if v != -1:
                words.append(w)
                sense_ids.append(v)
        
        assert context_output.size(0) == len(words), \
        f"context_output.size(0) = {context_output.size(0)}, len(words) = {len(list(words))}, {i}-th epoch, {context_ids}"
        assert context_output.size(0) == len(list(sense_ids)), "context_output.size(0) != len(sense_ids)"
        
        context_sz = len(list(sense_ids))
        
        for j, (word, sense_id) in enumerate(zip(words, sense_ids)):
            output = context_output.split(1,dim=0)[j]
            # output : (1, hidden_dim) # j번째 다의어 토큰에 대응하는 텐서 
            
            # 의미 임베딩
            # "시·군·구" 같은 단어는 우리말사전에 "시군구"로 등록되어 있으므로
            # 다음처럼 변환해보고, 그럼에도 단어가 사전에 없는 경우는 넘어갈 것
            if word not in gloss_dict.keys() and word.replace("·", "") in gloss_dict.keys():
                word = word.replace("·", "")
            elif word not in gloss_dict.keys():
                logger.debug(f"'{word}'는 사전에 없으므로 학습되지 않습니다.'")
                continue

            gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[word]
                
            if multigpu:
                gloss_ids = gloss_ids.to(gloss_device)
                gloss_attn_mask = gloss_attn_mask.to(gloss_device)
            else:
                gloss_ids = gloss_ids.cuda()
                gloss_attn_mask = gloss_attn_mask.cuda()                    
            
            gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
            # (의미수, hidden_dim)
            gloss_output = gloss_output.transpose(0,1)
            # (hidden_dim, 의미수)
            
            # multigpu인 경우 각각의 GPU에서 연산한 결과를 CPU로 가져옴
            if multigpu:
                output = output.cpu()
                gloss_output = gloss_output.cpu()          
                
            # 다의어 토큰에 대응되는 텐서와 의미 텐서들간 곱셈
            output = torch.mm(output, gloss_output)
            # (1, 의미수)
            
            # loss 계산
            try:
                idx = sense_keys.index(sense_id)
            except:
                logger.debug(f"{word}의 {sense_id}번째 의미가 {sense_keys}에 포함되지 않아 학습되지 않습니다.")
                continue
                
            if multigpu:
                label_tensor = torch.tensor([idx])
            else:
                label_tensor = torch.tensor([idx]).to('cuda')
            
            # 단어별로 loss 계산하여 더함
            loss += criterion[word](output, label_tensor)
            gloss_sz += gloss_output.size(-1) # transpose 했으므로
            
            # 의미 배치 사이즈를 넘어가면 업데이트
            if gloss_bsz != -1 and gloss_sz >= gloss_bsz:
                total_loss += loss.item()
                loss /= gloss_sz
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                # loss, global_sz 리셋
                loss = 0.
                gloss_sz = 0
                
                model.zero_grad()
                
                # 문맥 다시 실행
                context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)
                
                # 다의어가 하나도 없는 경우 None을 리턴하므로 패스
                if context_output is None:
                    continue
        
        # 문맥 배치가 끝나면 업데이트
        if gloss_bsz != -1:
            loss_sz = gloss_sz
        else:
            loss_sz = context_sz
            
        # 문맥 배치사이즈가 -1 아닌 경우,
        # gloss_sz > 0 이라는 건 모델 업데이트가 아직 되지 않은 상태 -> 업데이트
        if loss_sz > 0:
            
            total_loss += loss.item()
            loss /= loss_sz
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()                
        
        
        if save_each_it > 1 and i > 0 and i % save_each_it == 0:
            logger.info(f'Save checkpoint at {i}-th iteration.')
            logger.info(f'Loss at save checkpoint is {total_loss/(i+1):.4f}')
            torch.save(model, model_path+"_"+str(i)) 
            
    return model, optimizer, total_loss

def predict(eval_data, gloss_dict, model, multigpu=False):
    model.eval()
    preds = []
    with torch.no_grad():
        for context_ids, context_attn_mask, context_output_mask, words, sense_ids in eval_data:

            # 컨텍스트 인코더 계산
            if multigpu:
                context_ids = context_ids.to(context_device)
                context_attn_mask = context_attn_mask.to(context_device)
            else:
                context_ids = context_ids.to('cuda')
                context_attn_mask = context_attn_mask.to('cuda')
                
            context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)

            words_org = chain(*words)
            sense_ids_org = chain(*[list(sense_d.values()) for sense_d in sense_ids])
            
            # 의미 인코더 계산
            i = 0
            for word, sense_id in zip(words_org, sense_ids_org):
                # 의미 임베딩
                # "시·군·구" 같은 단어는 우리말사전에 "시군구"로 등록되어 있으므로
                # 다음처럼 변환해보고, 그럼에도 단어가 사전에 없는 경우는 넘어갈 것
                if word not in gloss_dict.keys() and word.replace("·", "") in gloss_dict.keys():
                    word = word.replace("·", "")
                elif word not in gloss_dict.keys():
                    preds.append(-1)
                    continue
                    
                if sense_id == -1:
                    preds.append(-1)
                    continue
                    
                output = context_output[i].unsqueeze(0)
#                 print("Output shape : ", output.shape)
                gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[word]

                if multigpu:
                    gloss_ids = gloss_ids.to(gloss_device)
                    gloss_attn_mask = gloss_attn_mask.to(gloss_device)
                else:
                    gloss_ids = gloss_ids.cuda()
                    gloss_attn_mask = gloss_attn_mask.cuda()      

                gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
                gloss_output = gloss_output.transpose(0,1)

                # Dot product of context output and gloss output
                if multigpu:
                    output = output.cpu()
                    gloss_output = gloss_output.cpu()                    
                output = torch.mm(output, gloss_output)
#                 print("output :", output)
                pred_idx = output.topk(1, dim=-1)[1].squeeze().item()
                pred_label = sense_keys[pred_idx]
                # sense_candidates의 마지막 자리에 후보를 기록
                i += 1
                preds.append(pred_label)
                
    return preds
        

def train(train_data, eval_data, train_gloss_dict, eval_gloss_dict, model, optimizer, criterion, args):
          
    epochs = args.epochs
    gloss_bsz = args.gloss_bsz
    max_grad_norm = args.max_grad_norm 
    logger = args.logger
    if args.multigpu:
        multigpu = args.multigpu
    else:
        multigpu = False
    print(f"The number of iteration for each epoch is {len(train_data)}")

    # 평가 데이터 라벨 기록
    truth = []
    for data in eval_data:
        sense_ids_org = chain(*[list(sense_d.values()) for sense_d in data[4]])
        truth += sense_ids_org

    # 훈련 
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1} initialized.")
        model_path = f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}"

        start_time = time.time()
        
        model, optimizer, total_loss = train_one_epoch(train_data, train_gloss_dict, model, optimizer, criterion, model_path, args)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 평가 데이터셋 예측
        preds = predict(eval_data, eval_gloss_dict, model)
        assert len(preds) == len(truth)
        eval_acc = np.mean(np.array(preds)==np.array(truth))
        eval_f1 = f1_score(truth, preds, average='weighted')        
        
        logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {total_loss:.3f}')
        logger.info(f'\tEval. Acc: {eval_acc*100:.2f}%')
        logger.info(f'\tEval. F1 : {eval_f1*100:.2f}%')
    
        # Saving
        torch.save(model, f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")
        logger.info(f"Checkpoint saved at {args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")
        args.checkpoint_count += 1

