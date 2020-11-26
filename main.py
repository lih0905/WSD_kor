"""
데이터 훈련 및 평가 로직 구현
"""

import argparse
import random
import os
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AdamW
from tokenization_kobert import KoBertTokenizer

from dataloader import glosses_dataloader, ContextDataset, BatchGenerator, context_dataloader
from model import BiEncoderModel
from train import train
from utils import gen_checkpoint_id, get_logger, checkpoint_count

# Argparse Setting
parser = argparse.ArgumentParser(description='다의어 분리 모델 파라미터 설정')

#training arguments
parser.add_argument('--rand_seed', type=int, default=42)
parser.add_argument('--max-grad-norm', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.0001)
# parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--multigpu', action='store_true', default=False)
parser.add_argument('--context-max-length', type=int, default=64)
parser.add_argument('--gloss-max-length', type=int, default=64)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--context-bsz', type=int, default=4)
parser.add_argument('--gloss-bsz', type=int, default=16)
parser.add_argument('--encoder-name', type=str, default='distilkobert')
# 	choices=['bert-base', 'bert-large', 'roberta-base', 'roberta-large'])
parser.add_argument('--checkpoint', type=str, default='checkpoint',
	help='filepath at which to save model')
# parser.add_argument('--data-path', type=str, required=True,
# 	help='Location of top-level directory for the Unified WSD Framework')

#evaluation arguments
parser.add_argument('--eval', action='store_true',
	help='Flag to set script to evaluate probe (rather than train)')


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
#     eval_df = eval_df.iloc[:30]
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
#         train_df = train_df.iloc[251270:251300]
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
        
