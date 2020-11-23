"""
데이터 훈련 및 평가 로직 구현
"""

import argparse
import time
import random
import os
import json
from itertools import chain

import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AdamW
from tokenization_kobert import KoBertTokenizer

from dataloader import glosses_dataloader, ContextDataset, BatchGenerator, context_dataloader
from model import BiEncoderModel
from utils import epoch_time, gen_checkpoint_id, get_logger, checkpoint_count

# Argparse Setting
parser = argparse.ArgumentParser(description='다의어 분리 모델 파라미터 설정')

#training arguments
parser.add_argument('--rand_seed', type=int, default=42)
parser.add_argument('--max-grad-norm', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.0001)
# parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--multigpu', action='store_true', default=False)
parser.add_argument('--context-max-length', type=int, default=128)
parser.add_argument('--gloss-max-length', type=int, default=128)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--context-bsz', type=int, default=1)
parser.add_argument('--gloss-bsz', type=int, default=32)
parser.add_argument('--encoder-name', type=str, default='distilkobert')
# 	choices=['bert-base', 'bert-large', 'roberta-base', 'roberta-large'])
parser.add_argument('--checkpoint', type=str, default='checkpoint',
	help='filepath at which to save model')
# parser.add_argument('--data-path', type=str, required=True,
# 	help='Location of top-level directory for the Unified WSD Framework')

#evaluation arguments
parser.add_argument('--eval', action='store_true',
	help='Flag to set script to evaluate probe (rather than train)')

# multigpu 일 때 설정
context_device = "cuda:0"
gloss_device = "cuda:1"

def train_one_epoch(train_data, gloss_dict, model, optimizer, criterion, model_path, args):
    
    logger = args.logger
    gloss_bsz = args.gloss_bsz
    max_grad_norm = args.max_grad_norm
    multigpu = args.multigpu
    
    # 한 에폭을 훈련시키는 함수. 
    model.train()
    total_loss = 0
    
    save_each_it = int(len(train_data)/10)
    
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
                logger.warning(f"'{word}'는 사전에 없으므로 학습되지 않습니다.'")
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
                logger.warning(f"{word}의 {sense_id}번째 의미가 {sense_keys}에 포함되지 않아 학습되지 않습니다.")
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
        
        
#         if save_each_it > 1 and i > 0 and i % save_each_it == 0:
#             logger.info(f'Save checkpoint at {i}-th iteration.')
#             logger.info(f'Loss at save checkpoint is {total_loss/(i+1):.4f}')
#             torch.save(model, model_path+"_"+str(i)) 
            
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
            
            words = []
            for w, v in zip(words_org, sense_ids_org):
                if v != -1:
                    words.append(w)
                
            # 배치 내 순번 / 단어 순번 저장
            sense_candidates = []
            for i in range(len(sense_ids)):
                for j in sense_ids[i].keys():
                    sense_candidates.append([i,j,-1])
            
            assert context_output.size(0) == len(list(words)), "context_output.size(0) != len(words)"
#             assert context_output.size(0) == len(sense_candidates)
            
            # 의미 인코더 계산
            for j, (output, word) in enumerate(zip(context_output.split(1, dim=0), words)):
                # 의미 임베딩
                # "시·군·구" 같은 단어는 우리말사전에 "시군구"로 등록되어 있으므로
                # 다음처럼 변환해보고, 그럼에도 단어가 사전에 없는 경우는 넘어갈 것
                if word not in gloss_dict.keys() and word.replace("·", "") in gloss_dict.keys():
                    word = word.replace("·", "")
                elif word not in gloss_dict.keys():
                    logger.warning(f"'{word}'는 사전에 없으므로 평가되지 않습니다.'")
                    continue

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
                pred_idx = output.topk(1, dim=-1)[1].squeeze().item()
                pred_label = sense_keys[pred_idx]
                # sense_candidates의 마지막 자리에 후보를 기록
                sense_candidates[j][-1] = pred_label
                
            preds.append(sense_candidates)
                
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
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1} initialized.")
        model_path = f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}"

        start_time = time.time()
        # train_one_epoch(train_dl, gloss_dict, model, optimizer, criterion, model_path, args):
        model, optimizer, total_loss = train_one_epoch(train_data, train_gloss_dict, model, optimizer, criterion, model_path, args)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
#         # 예측 결과
#         preds = predict(eval_data, eval_gloss_dict, model, multigpu)
#         # 실제 결과
#         total_labels = 0
#         correct_labels = 0
#         for i, data in enumerate(eval_data):
#             # 단어들 갯수 저장
#             words = chain(*data[3])
#             sense_ids = chain(*data[4])
            
#             total_labels += len(list(words))
#             for j, label_d in enumerate(data[4]):
#                 # 배치마다 실제 WSD 라벨  
#                 preds_d = {d[1]:d[2] for d in preds[i][j]}
#                 for k, v in label_d.items():
#                     if k in preds_d.keys() and v == preds_d[k]:
#                         correct_labels += 1
                
            
#         pred_acc = float(correct_labels/total_labels)
        
        logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {total_loss:.3f}')
#         logger.info(f'\tEval. Acc: {pred_acc*100:.2f}%')
    
#         # Saving
#         torch.save(model, f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")
#         logger.info(f"Checkpoint saved at {args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")
#         args.checkpoint_count += 1

def evaluate(eval_data, eval_gloss_dict, epochs, model):
    pass
    

if __name__ == "__main__":
    args = parser.parse_args()
    args.checkpoint = os.path.join(args.checkpoint,gen_checkpoint_id(args))
    
    
    #set random seeds
    torch.manual_seed(args.rand_seed)
    os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)   
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True

    # Pretrained 모델 및 토크나이저 불러오기
    bert_model = AutoModel.from_pretrained("monologg/distilkobert")
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

    with open('Dict/processed_dictionary.json','r') as f:
        urimal_dict = json.load(f)

    batch_generator = BatchGenerator(tokenizer, args.context_max_length)
    
    # 평가 데이터 불러오기
    eval_df = pd.read_csv('Data/processed_eval.csv')
    eval_df = eval_df.iloc[:30]
    eval_ds = ContextDataset(eval_df)
    eval_dl = context_dataloader(eval_ds, batch_generator, args.context_bsz)

    eval_gloss_dict, eval_gloss_weight = glosses_dataloader(eval_df, tokenizer, urimal_dict, args.gloss_max_length)
    
    # 모델 로딩
    model = BiEncoderModel(bert_model)
    if args.multigpu: 
        model.gloss_encoder = model.gloss_encoder.to(gloss_device)
        model.context_encoder = model.context_encoder.to(context_device)
    else:
        model = model.to('cuda')
    
    # If checkpoint path exists, load the last model
    if os.path.isdir(args.checkpoint):
        # EXAMPLE: "{engine_name}_{task_name}_{timestamp}/saved_checkpoint_1"     
        args.checkpoint_count = checkpoint_count(args.checkpoint)
        logger = get_logger(args)
        logger.info(f"Checkpoint path directory exists")
        logger.info(f"Loading model from saved_checkpoint_{args.checkpoint_count}")
        model = torch.load(f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}") 
        
        args.checkpoint_count += 1 #
    # If there is none, create a checkpoint folder and train from scratch
    else:
        try:
            os.makedirs(args.checkpoint)
        except:
            print("Ignoring Existing File Path ...")

        args.checkpoint_count = 0
        logger = get_logger(args)

        logger.info(f"Creating a new directory for {args.checkpoint}")

    args.logger = logger
    
        
    if args.eval:
        pass
    else:
        # 훈련 데이터 로드
        train_df = pd.read_csv('Data/processed_train.csv')
        train_df = train_df.iloc[:10]
        train_ds = ContextDataset(train_df)
        train_dl = context_dataloader(train_ds, batch_generator, args.context_bsz)

        train_gloss_dict, train_gloss_weight = glosses_dataloader(train_df, tokenizer, urimal_dict, args.gloss_max_length)        
        
        # 훈련 스텝
        optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
        criterion = {}
        for key in train_gloss_dict:
            # If reduction is 'none', then the same size as the target
            criterion[key] = torch.nn.CrossEntropyLoss(reduction='none')

        train(train_dl, eval_dl, train_gloss_dict, eval_gloss_dict, model, optimizer, criterion, args)
        
