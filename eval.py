"""
임의로 주어진 문장에 포함된 다의어의 의미 분석
"""

import pickle
import json
import time
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel
from tokenization_kobert import KoBertTokenizer
from konlpy.tag import Mecab

from dataloader import glosses_dataloader, ContextDataset, BatchGenerator, context_dataloader
from model import BiEncoderModel
from train import predict


# Argparse Setting
parser = argparse.ArgumentParser(description='다의어 분리 모델 실험')

#training arguments
parser.add_argument('--model_date', type=str, default='distilkobert_202011201741')
parser.add_argument('--text', type=str, required=True)
parser.add_argument('--multigpu', action='store_true', default=False)

# multigpu 일 때 설정
context_device = "cuda:0"
gloss_device = "cuda:1"
# device = torch.device('cuda')

mecab = Mecab()

def text_process(text, urimal_dict):
    text_processed = mecab.pos(text)
    idx = 0
    prev_idx = 0
    word_id = 1
    text2 = text
    wsd = []
    for word, pos in text_processed:
        if pos in ('NNP', 'NNG', 'NNB', 'NP') and word in urimal_dict.keys():
            prev_idx = idx
            text2 = text2[prev_idx:]
            idx = text2.find(word)
            wsd_d = {'word':word,
                    'sense_id':1,
                     'pos':pos,
                     'begin':idx+prev_idx,
                     'end':idx+prev_idx+len(word),
                     'word_id':word_id
                    }
            word_id += 1

            wsd.append(wsd_d)
    return pd.DataFrame([{'form':text, 'WSD':str(wsd)}])

if __name__ == "__main__":
    args = parser.parse_args()
    text = args.text
    multigpu = args.multigpu
    
    with open('Dict/processed_dictionary.json', 'rb') as f:
        urimal_dict = json.load(f)


    bert_model = AutoModel.from_pretrained("monologg/distilkobert")
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    model = BiEncoderModel(bert_model)
    model.to('cuda')    
    
    model_list = os.listdir(f"checkpoint/{args.model_date}") 
    model_fname = 'saved_checkpoint_0'
   
    model = torch.load(f"checkpoint/{args.model_date}/{model_fname}") 
    model.eval()

    batch_generator = BatchGenerator(tokenizer, 192)
    eval_df = text_process(text, urimal_dict)   
    eval_ds = ContextDataset(eval_df)
    eval_dl = context_dataloader(eval_ds, batch_generator, 1)

    eval_gloss_dict, eval_gloss_weight = glosses_dataloader(eval_df, tokenizer, urimal_dict, 192)
#     print(eval_dat)
#     print(eval_data)
    
    preds = []
    with torch.no_grad():
        for context_ids, context_attn_mask, context_output_mask, example_keys, labels in eval_dl:
            # 컨텍스트 인코더 계산
            if multigpu:
                context_ids = context_ids.to(context_device)
                context_attn_mask = context_attn_mask.to(context_device)
            else:
                context_ids = context_ids.to('cuda')
                context_attn_mask = context_attn_mask.to('cuda')
            context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)

            # 의미 인코더 계산
            for output, key in zip(context_output.split(1, dim=0), example_keys):
                # 의미 임베딩
                gloss_ids, gloss_attn_mask, sense_keys = eval_gloss_dict[key]
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
                preds.append(pred_label)


#     print(preds)
    print("-"*100)
    print(f"문장 : {text}")
#     print(f"토크나이즈 결과 : {mecab.morphs(text)}")
    print("-"*100)
    indices = eval_data[0][-1]
    for i, (word, pos, sense_no) in enumerate(eval_dat[0]):
        if sense_no != -1 and i in indices:
            idx = indices.index(i)
            label = preds[idx]
            idx2 = urimal_dict[word]['sense_no'].index(label)
            print(f"'{word}'의 의미 : {urimal_dict[word]['definition'][idx2]}")
