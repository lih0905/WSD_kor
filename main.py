"""
데이터 훈련 및 평가 로직 구현
"""

import pickle
import argparse
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel

# from tokenization_kobert import KoBertTokenizer
# tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
# model = AutoModel.from_pretrained("monologg/kobert")

from dataloader import dataloader_glosses, dataloader_context
from model import BiEncoderModel

# with open('Data/processed_eval.pickle', 'rb') as f:
#     eval_data = pickle.load(f)
# with open('Dict/processed_dictionary.pickle', 'rb') as f:
#     urimal_dict = pickle.load(f)
# gloss_dict, gloss_weight = dataloader_glosses(eval_data, tokenizer, urimal_dict, 128)

