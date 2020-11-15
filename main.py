"""
데이터 훈련 및 평가 로직 구현
"""

import pickle
import argparse
import time
import random
import os

import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AdamW
from tokenization_kobert import KoBertTokenizer

from dataloader import dataloader_glosses, dataloader_context
from model import BiEncoderModel
from utils import epoch_time, gen_checkpoint_id, get_logger, checkpoint_count

# Argparse Setting
parser = argparse.ArgumentParser(description='다의어 분리 모델 파라미터 설정')

#training arguments
parser.add_argument('--rand_seed', type=int, default=42)
parser.add_argument('--grad-norm', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.0001)
# parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--context-max-length', type=int, default=192)
parser.add_argument('--gloss-max-length', type=int, default=128)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--context-bsz', type=int, default=4)
parser.add_argument('--gloss-bsz', type=int, default=64)
parser.add_argument('--encoder-name', type=str, default='distilkobert')
# 	choices=['bert-base', 'bert-large', 'roberta-base', 'roberta-large'])
parser.add_argument('--checkpoint', type=str, default='checkpoint',
	help='filepath at which to save best probing model (on dev set)')
# parser.add_argument('--data-path', type=str, required=True,
# 	help='Location of top-level directory for the Unified WSD Framework')

#evaluation arguments
parser.add_argument('--eval', action='store_true',
	help='Flag to set script to evaluate probe (rather than train)')


def train_one_epoch(train_data, gloss_dict, model, optimizer, criterion, gloss_bsz, max_grad_norm):
    # 한 에폭을 훈련시키는 함수. 
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for i, (context_ids, context_attn_mask, context_output_mask, example_keys, labels) in tqdm.tqdm(enumerate(train_data)):
        
        model.zero_grad()
        
        # 컨텍스트 인코더 계산
        context_ids = context_ids.to('cuda')
        context_attn_mask = context_attn_mask.to('cuda')
        context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)
    
        # 의미 인코더 계산
        loss = 0.
        gloss_sz = 0
        context_sz = len(labels)
        
        for j, (key, label) in enumerate(zip(example_keys, labels)):
            output = context_output.split(1,dim=0)[j]
            
            # 의미 임베딩
            # "시·군·구" 같은 단어는 우리말사전에 "시군구"로 등록되어 있으므로
            # 다음처럼 변환해보고, 그럼에도 단어가 사전에 없는 경우는 넘어갈 것
            if key not in gloss_dict.keys() and key.replace("·", "") in gloss_dict.keys():
                key = key.replace("·", "")
            elif key not in gloss_dict.keys():
                continue
            
            gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]
            
            gloss_ids = gloss_ids.cuda()
            gloss_attn_mask = gloss_attn_mask.cuda()
            
            gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
            gloss_output = gloss_output.transpose(0,1)
            
            # Dot product of context output and gloss output
            output = torch.mm(output, gloss_output)
            
            # loss 계산
            idx = sense_keys.index(label)
            label_tensor = torch.tensor([idx]).to('cuda')
            
            loss += criterion[key](output, label_tensor)
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
            
        if loss_sz > 0:
            total_loss += loss.item()
            loss /= loss_sz
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()                

    return model, optimizer, total_loss

def predict(eval_data, gloss_dict, model):
    model.eval()
    preds = []
    with torch.no_grad():
        for context_ids, context_attn_mask, context_output_mask, example_keys, labels in eval_data:

            # 컨텍스트 인코더 계산
            context_ids = context_ids.to('cuda')
            context_attn_mask = context_attn_mask.to('cuda')
            context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)

            # 의미 인코더 계산
            for output, key in zip(context_output.split(1, dim=0), example_keys):
                # 의미 임베딩
                gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]
                gloss_ids = gloss_ids.cuda()
                gloss_attn_mask = gloss_attn_mask.cuda()

                gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
                gloss_output = gloss_output.transpose(0,1)

                # Dot product of context output and gloss output
                output = torch.mm(output, gloss_output)
                pred_idx = output.topk(1, dim=-1)[1].squeeze().item()
                pred_label = sense_keys[pred_idx]
                preds.append(pred_label)
                
    return np.array(preds)
            
def train(train_data, eval_data, train_gloss_dict, eval_gloss_dict, epochs, model, optimizer, criterion, gloss_bsz, max_grad_norm, logger):
    print(f"The number of iteration for each epoch is {len(train_data)}")
    
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1} initialized.")
        start_time = time.time()
        model, optimizer, total_loss = train_one_epoch(train_data, train_gloss_dict, model, optimizer, criterion, gloss_bsz, max_grad_norm)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # 예측 결과
        preds = predict(eval_data, eval_gloss_dict, model)
        # 실제 결과
        labels = []
        for data in eval_data:
            labels += data[4]
        pred_acc = np.mean(preds == np.array(labels))
        
        logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {total_loss:.3f}')
        logger.info(f'\tEval. Acc: {pred_acc*100:.2f}%')
        
        # Saving
        torch.save(model, f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")
        logger.info(f"Checkpoint saved at {args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")
        args.checkpoint_count += 1


def evaluate(eval_data, gloss_dict, model, optimizer, criterion, gloss_bsz, max_grad_norm):
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

    with open('Data/processed_eval.pickle', 'rb') as f:
        eval_data = pickle.load(f)
    with open('Dict/processed_dictionary.pickle', 'rb') as f:
        urimal_dict = pickle.load(f)

    # 평가 데이터와 사전 토크나이즈
    eval_gloss_dict, eval_gloss_weight = dataloader_glosses(eval_data, tokenizer, urimal_dict, args.gloss_max_length)
    eval_data = dataloader_context(eval_data[:12], tokenizer, bsz=args.context_bsz, max_len=args.context_max_length)
    
    # 모델 로딩
    model = BiEncoderModel(bert_model)
    model.to('cuda')
    
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
        with open('Data/processed_train.pickle', 'rb') as f:
            train_data = pickle.load(f)
        train_gloss_dict, train_gloss_weight = dataloader_glosses(train_data, tokenizer, urimal_dict, args.gloss_max_length)        
        
        # 훈련 스텝
        optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
        criterion = {}
        for key in train_gloss_dict:
            criterion[key] = torch.nn.CrossEntropyLoss(reduction='none')
                    
        train_data = dataloader_context(train_data[:12], tokenizer, bsz=args.context_bsz, max_len=args.context_max_length)

        train(train_data, eval_data, train_gloss_dict, eval_gloss_dict, args.epochs, model, optimizer, criterion, args.gloss_bsz, args.grad_norm, logger)
        