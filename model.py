"""
모델 구현
 : Gloss와 Context 별로 모델을 구현한 후 결합하는 모델 별도 생성
"""

import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

### Gloss Encoder

class GlossEncoder(nn.Module):
    def __init__(self, gloss_encoder):#, hidden_dim):
        super(GlossEncoder, self).__init__()
        self.gloss_encoder = copy.deepcopy(gloss_encoder)
#         self.gloss_hdim = hidden_dim
        
    def forward(self, input_ids, attn_mask):
        gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        # (의미수, 의미 최대 토큰수, hidden_dim)
        return gloss_output[:,0,:] # 각 의미별 [CLS] 토큰의 임베딩 출력

### Context Encoder    
    
def process_encoder_outputs(output, mask, as_tensor=False):
    """
    주어진 문장 텐서에서 다의어 토큰 위치에 대응되는 텐서만 추출한다.
    하나의 단어가 여러개의 토큰으로 토크나이즈 된 경우, 각 텐서의 평균을 return
    
    - 입력
        output : bert 모델에 context_ids를 통과시켜 얻은 output tensor batch중 하나
                (max_len, hidden_dim)
        mask : 주어진 문장의 토큰들의 다의어 마스킹 (동일한 단어로부터 분리된 토큰은 같은 값) 
                (max_len)
        as_tensor : 리턴값을 텐서로 할지 여부
        
    - 출력
        combined_outputs : 각 다의어 별로 output을 평균한 텐서
                            (다의어 수, hidden_dim)
    """
    combined_outputs = []
    position = -1
    avg_arr = []
    
    assert len(mask) == output.size(0), "len(mask) != output.size(0)"
    
    for idx, rep in zip(mask, torch.split(output, 1, dim=0)):
        # 각 토큰마다 각각의 텐서로 분리
        # 즉 (max_len, hidden_dim) 텐서를 길이가 max_len인 (1, hidden_dim)
        # 텐서들을 담은 tuple로 변환        

        # idx는 다의어 번호를 나타내는데, -1인 경우는 다의어가 아니므로 다음 토큰으로 넘어감
        if idx == -1: 
            continue
        # 다의어의 토큰을 처음으로 만나는 경우
        elif position < idx: 
            # 토큰 위치를 업데이트
            position = idx
            if len(avg_arr) > 0:
                # 이전 다의어의 텐서들이 저장되어있는 경우, 해당 텐서들의 평균을 취하여
                # combined_outputs에 저장
                combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
                # (1 , hidden_dim)
            # 이번 단어의 표현으로 대체
            avg_arr = [rep]
        # 다의어의 쪼개진 토큰인 경우
        else:
            assert position == idx 
            avg_arr.append(rep)
            
    # 마지막 다의어의 평균을 취함
    if len(avg_arr) > 0: 
        combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
    # combined_output = 다의어 수 * (1, hidden_dim)
    
    # 다의어가 토크나이즈되지 않아서 하나도 걸리지 않은 경우
    # torch.cat 적용하면 예외 발생
    if len(combined_outputs) == 0: 
        return None
#         combined_outputs = [torch.zeros((1, output.size(-1))) for _ in range(len(word))]
    elif as_tensor: 
        return torch.cat(combined_outputs, dim=0)
    else: 
        return combined_outputs    

class ContextEncoder(nn.Module):
    def __init__(self, context_encoder):
        super(ContextEncoder, self).__init__()
        self.context_encoder = copy.deepcopy(context_encoder)
        
    def forward(self, input_ids, attn_mask, output_mask):
        context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]
        # (batch_size, max_len, hiddin_dim)

        # 다의어의 토큰들은 평균을 취한 값으로 사용
        example_arr = []        
        for i in range(context_output.size(0)): # 배치마다 수행
            res = process_encoder_outputs(context_output[i], output_mask[i], as_tensor=True)
            if res is not None:
                example_arr.append(res)
            # batch_size * (다의어 수, hidden_dim)

        try:
            context_output = torch.cat(example_arr, dim=0)
            # (배치 내 다의어 수, hidden_dim)
        except RuntimeError:
            # 배치 내 다의어가 하나도 없을 경우 context_output이 empty
            return None       

        return context_output
    
class BiEncoderModel(nn.Module):
    def __init__(self, encoder):
        super(BiEncoderModel, self).__init__()
        
        self.context_encoder = ContextEncoder(encoder)
        self.gloss_encoder = GlossEncoder(encoder)
        
    def context_forward(self, context_input, context_input_mask, context_example_mask):
        return self.context_encoder.forward(context_input, context_input_mask, context_example_mask)
    
    def gloss_forward(self, gloss_input, gloss_mask):
        return self.gloss_encoder.forward(gloss_input, gloss_mask)