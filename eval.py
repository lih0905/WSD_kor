"""
임의로 주어진 문장에 포함된 다의어의 의미 분석
"""

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
    """
    문장과 우리말샘 사전을 입력으로 받아서
    말뭉치 데이터 형식으로 변환
    
    Args:
        text : string
        urimal_dict : dictionary
        
    Return:
        pandas.DataFrame
    """
    text_processed = mecab.pos(text)
    end = 0
    word_id = 1
    text2 = text
    wsd = []
    for word, pos in text_processed:
        if pos in ('NNP', 'NNG', 'NNB', 'NP') and word in urimal_dict.keys():
            idx = text2.find(word)
            start = end + idx
            end = start + len(word)
            
            wsd_d = {'word':word,
                    'sense_id':1,
                     'pos':pos,
                     'begin':start,
                     'end':end,
                     'word_id':word_id
                    }
            wsd.append(wsd_d)
            
            text2 = text[end:]
            word_id += 1

#     return wsd
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
    model_fname = 'saved_checkpoint_fin'
   
#     model = torch.load(f"checkpoint/{args.model_date}/{model_fname}") 
    model = torch.load(f"checkpoint/WSD_v2/{model_fname}", map_location='cuda') 
    model.eval()

    batch_generator = BatchGenerator(tokenizer, 128)
    eval_df = text_process(text, urimal_dict)   
    eval_ds = ContextDataset(eval_df)
    eval_dl = context_dataloader(eval_ds, batch_generator, 1)

    eval_gloss_dict, eval_gloss_weight = glosses_dataloader(eval_df, tokenizer, urimal_dict, 128)
#     print(eval_dat)
#     print(eval_data)
    
    preds = predict(eval_dl, eval_gloss_dict, model)
    wsd = eval(eval_df.iloc[0,1])
#     print(preds)
    print("-"*100)
    print(f"문장 : {text}")
#     print(f"토크나이즈 결과 : {mecab.morphs(text)}")
    print("-"*100)
    for wsd_d, pred in zip(eval(eval_df.iloc[0,1]), preds):
        if pred != -1:
            word = wsd_d['word']
            idx = urimal_dict[word]['sense_no'].index(pred)
            meaning = urimal_dict[word]['definition'][idx]

            print(f"'{word}'의 의미 : {meaning}")