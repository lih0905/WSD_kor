"""
전처리된 데이터셋을 모델에 넣을 형식으로 변환하여 Dataloader 구현

토크나이저는 Huggingface의 KoBert 모델을 사용한다.
"""

import torch
from torch.utils.data import Dataset

from tokenization_kobert import KoBertTokenizer


# 최대 길이로 맞춰주는 함수 구현
def normalize_length(ids, attn_mask, o_mask, max_len, pad_id):
    if max_len == -1:
        return ids, attn_mask, o_mask
    else:
        if len(ids) < max_len:
            while len(ids) < max_len:
                ids.append(torch.tensor([[pad_id]]))
                attn_mask.append(0)
                o_mask.append(-1)
        else:
            ids = ids[:max_len-1]+[ids[-1]]
            attn_mask = attn_mask[:max_len]
            o_mask = o_mask[:max_len]

        assert len(ids) == max_len
        assert len(attn_mask) == max_len
        assert len(o_mask) == max_len
        
        return ids, attn_mask, o_mask


# Gloss loader
def tokenize_glosses(text_arr, tokenizer, max_len):
    glosses = []
    masks = []
    for text in text_arr:
        encoded_tensors = tokenizer.encode_plus(text)
        g_ids = [torch.tensor([[x]]) for x in encoded_tensors['input_ids']]
        g_attn_mask = encoded_tensors['attention_mask']
        g_fake_mask = [-1] * len(g_ids)
        g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len, pad_id=tokenizer.pad_token_id)
        g_ids = torch.cat(g_ids, dim=-1)
        g_attn_mask = torch.tensor(g_attn_mask)
        glosses.append(g_ids)
        masks.append(g_attn_mask)
    return glosses, masks

# creates a sense label/ gloss dictionary for training/using the gloss encoder
def dataloader_glosses(data, tokenizer, gloss_dict, max_len=-1):
    sense_glosses = {}
    sense_weights = {}

    gloss_lengths = []

    for sent in data:
        for word, pos, sense_no in sent:
            if sense_no == -1 or sense_no in (777, 888, 999):
                continue # 다의어가 아닌 경우나 고유명사인 경우는 패스
            else:
                word = word.replace('·','')
                key = word + "_" + pos
                if key not in sense_glosses.keys():
                    # 단어의 모든 의미 불러옴
                    try: 
                        gloss_arr = gloss_dict[word]['definition']
                        if len(gloss_arr) >= 1:
                                sensekey_arr = gloss_dict[word]['sense_no']

                                #preprocess glosses into tensors
                                gloss_ids, gloss_masks = tokenize_glosses(gloss_arr, tokenizer, max_len)
                                gloss_ids = torch.cat(gloss_ids, dim=0)
                                gloss_masks = torch.stack(gloss_masks, dim=0)
                                sense_glosses[key] = (gloss_ids, gloss_masks, sensekey_arr)
                                # sense_glosses[key] = ( len(sensekey_arr) * max_len, len(sensekey_arr) * max_len, len(sensekey_arr) )

                                #intialize weights for balancing senses
                                sense_weights[key] = [0]*len(gloss_arr)
                                w_idx = sensekey_arr.index(sense_no)
                                sense_weights[key][w_idx] += 1
                        else: # 사전에 단어 없는 경우 넘어감
                            pass
                    except ValueError:
                        pass # 의미 번호가 제대로 매핑되어 있지 않은 경우
                    except KeyError:
                        pass
                            
                else:
                    #update sense weight counts
                    try:
                        w_idx = sense_glosses[key][2].index(sense_no)
                        sense_weights[key][w_idx] += 1
                    except ValueError:
                        pass # 의미 번호가 제대로 매핑되어 있지 않은 경우
                    except KeyError:
                        pass # 의미 번호가 제대로 매핑되어 있지 않은 경우
                
    #normalize weights
    for key in sense_weights:
        total_w = sum(sense_weights[key])
        sense_weights[key] = torch.FloatTensor([total_w/x if x !=0 else 0 for x in sense_weights[key]])

    return sense_glosses, sense_weights

# Context data loader 구현
def dataloader_context(text_data, tokenizer, bsz=1, max_len=-1):
    if max_len == -1: assert bsz==1 #otherwise need max_length for padding

    context_ids = []
    context_attn_masks = []

    example_keys = []

    context_output_masks = []
    
    labels = []
    indices = []

    #tensorize data
    for sent in text_data:
        # 시작 토큰 지정
        c_ids = [torch.tensor([[tokenizer.cls_token_id]])] #cls token aka sos token, returns a list with index
        o_masks = [-1] # 다의어 마스킹
        sent_keys = []
        sent_labels = []
        sent_indices = []

        # 각 단어에 대해서
        for idx, (word, pos, sense_no) in enumerate(sent):
            # 각 단어를 토크나이즈
            word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower(), add_special_tokens=False)]
            c_ids.extend(word_ids)

            # 다의어이면서 고유명사가 아닌 경우
            if sense_no != -1 and sense_no not in (777, 888, 999):
                # 다의어 위치 마킹
                o_masks.extend([idx]*len(word_ids))
                #track example instance keys to get glosses
                ex_key = word + '_' + pos
                sent_keys.append(ex_key)
                sent_labels.append(sense_no)
                sent_indices.append(idx)
            else:
                #mask out output of context encoder for WSD task (not labeled)
                o_masks.extend([-1]*len(word_ids))

            #break if we reach max len
            if max_len != -1 and len(c_ids) >= (max_len-1):
                break

        # EOS 토큰 추가
        c_ids.append(torch.tensor([[tokenizer.sep_token_id]])) 
        c_attn_mask = [1]*len(c_ids)
        o_masks.append(-1)
        c_ids, c_attn_masks, o_masks = normalize_length(c_ids, c_attn_mask, o_masks, max_len, pad_id=tokenizer.pad_token_id)

#         y = torch.tensor([1]*len(sent_insts), dtype=torch.float)
        #not including examples sentences with no annotated sense data
        if len(sent_keys) > 0:
            context_ids.append(torch.cat(c_ids, dim=-1))
            context_attn_masks.append(torch.tensor(c_attn_masks).unsqueeze(dim=0))
            context_output_masks.append(torch.tensor(o_masks).unsqueeze(dim=0))
            example_keys.append(sent_keys)
            labels.append(sent_labels)
            indices.append(sent_indices)

    #package data
    data = list(zip(context_ids, context_attn_masks, context_output_masks, example_keys, labels, indices))

    #batch data if bsz > 1
    if bsz > 1:
#         print('Batching data with bsz={}...'.format(bsz))
        batched_data = []
        for idx in range(0, len(data), bsz):
            if idx+bsz <=len(data): b = data[idx:idx+bsz]
            else: b = data[idx:]
            context_ids = torch.cat([x for x,_,_,_,_,_ in b], dim=0)
            context_attn_mask = torch.cat([x for _,x,_,_,_,_ in b], dim=0)
            context_output_mask = torch.cat([x for _,_,x,_,_,_ in b], dim=0)
            example_keys = []
            for _,_,_,x,_,_ in b: example_keys.extend(x)
            labels = []
            for _,_,_,_,x,_ in b: labels.extend(x)
            indices = []
            for _,_,_,_,_,x in b: indices.extend(x)
            batched_data.append((context_ids, context_attn_mask, context_output_mask, example_keys, labels, indices))
        return batched_data
    else:  
        return data