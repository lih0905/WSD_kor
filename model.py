"""
모델 구현
 : Gloss와 Context 별로 모델을 구현한 후 결합하는 모델 별도 생성
"""


import torch
import torch.nn as nn
from torch.nn import functional as F

def process_encoder_outputs(output, mask, as_tensor=False):
    combined_outputs = []
    position = -1
    avg_arr = []
    for idx, rep in zip(mask, torch.split(output, 1, dim=0)):
        # 배치마다 각각의 텐서로 분리
        # 즉 (batch_size, max_len, hidden_dim) 텐서를 길이가 batch_size인 (1, max_len, hidden_dim)인
        # 텐서들을 담은 tuple로 변환        

        #ignore unlabeled words
        if idx == -1: continue
        #average representations for units in same example
        elif position < idx: 
            position=idx
            if len(avg_arr) > 0: 
                # 현재까지 저장한 avg_arr이 있으며 다음 단어로 넘어간 경우, 평균을 취한다.
                combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
            # 이번 단어의 표현으로 대체
            avg_arr = [rep]
        else:a
            assert position == idx 
            avg_arr.append(rep)
    #get last example from avg_arr
    if len(avg_arr) > 0: combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
    if as_tensor: return torch.cat(combined_outputs, dim=0)
    else: return combined_outputs

class GlossEncoder(nn.Module):
    def __init__(self, gloss_encoder)#, hidden_dim):
        super(self).__init__()
        self.gloss_encoder = gloss_encoder
#         self.gloss_hdim = hidden_dim
        
    def forward(self, input_ids, attn_mask):
        gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        # (의미수, 의미 최대 토큰수, 토큰당 임베딩)
        return gloss_output[:,0,:] # 각 의미별 [CLS] 토큰의 임베딩 출력
    
class ContextEncoder(nn.Module):
    def __init__(self, context_encoder)#, hidden_dim):
        super(self).__init__()
        self.context_encoder = context_encoder
#         self.context_hdim = hidden_dim
        
    def forward(self, input_ids, attn_mask, output_mask):
        context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]

        #average representations over target word(s)
        example_arr = []        
        for i in range(context_output.size(0)): # 배치마다 수행
            example_arr.append(process_encoder_outputs(context_output[i], output_mask[i], as_tensor=True))
        context_output = torch.cat(example_arr, dim=0)

        return context_output
    
class BiEncoderModel(nn.Module):
    def __init__(self, encoder):
        super(self).__init__()
        
        self.context_encoder = ContextEncoder(encoder)
        self.gloss_encoder = GlossEncoder(encoder)
        
    def context_forward(self, context_input, context_input_mask, context_example_mask):
        return self.context_encoder.forward(context_input, context_input_mask, context_example_mask)
    
    def gloss_forward(self, gloss_input, gloss_mask)
        return self.gloss_encoder.forward(gloss_input, gloss_mask)