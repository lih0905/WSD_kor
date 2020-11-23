"""
전처리된 데이터셋을 모델에 넣을 형식으로 변환하여 Dataloader 구현

토크나이저는 Huggingface의 KoBert 모델을 사용한다.
"""

import torch
from torch.utils.data import Dataset, DataLoader


### Gloss Data Loader 

# 단어별 가중치 생성
def glosses_dataloader(data_df, tokenizer, gloss_dict, max_len=-1):
    sense_glosses = {}
    sense_weights = {}

    for wsd_list in data_df['WSD']:
        wsd_list = eval(wsd_list)
        for wsd_d in wsd_list:
            word = wsd_d['word']
            pos = wsd_d['pos']
            sense_no = wsd_d['sense_id']

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
                            res = tokenizer(gloss_arr, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
                            gloss_ids, gloss_masks = res['input_ids'], res['attention_mask']
                            sense_glosses[key] = (gloss_ids, gloss_masks, sensekey_arr)
                            # sense_glosses[key] = ( len(sensekey_arr) * max_len, len(sensekey_arr) * max_len, len(sensekey_arr) )

                            #intialize weights for balancing senses
                            sense_weights[key] = [0]*len(gloss_arr)
                            w_idx = sensekey_arr.index(sense_no)
                            sense_weights[key][w_idx] += 1
                        else: # 사전에 단어 없는 경우 넘어감
                            pass
                    except ValueError:
                        continue # 의미 번호가 제대로 매핑되어 있지 않은 경우
                    except KeyError:
                        continue

                else: # 이미 단어장에 등록된 단어면 가중치만 업데이트
                    try:
                        w_idx = sense_glosses[key][2].index(sense_no)
                        sense_weights[key][w_idx] += 1
                    except ValueError:
                        continue # 의미 번호가 제대로 매핑되어 있지 않은 경우
                    except KeyError:
                        continue # 의미 번호가 제대로 매핑되어 있지 않은 경우

    # 가중치 정규화
    for key in sense_weights:
        total_w = sum(sense_weights[key])
        sense_weights[key] = torch.FloatTensor([total_w/x if x !=0 else 0 for x in sense_weights[key]])

    return sense_glosses, sense_weights

### Context Data Loader 

def tokenizer_wsd(tokenizer, sent, wsd, max_len):
    """문장을 토크나이즈 한 후 WSD에 등장하는 단어들의 의미를 마스킹"""
    
    # 단어별로 ID 지정
    for i in range(len(wsd)):
        wsd[i]['word_id'] = i+1
        
    tokens = []
    word_pos = []
    sense_ids = {}
    word_ids = []
    sent_e = sent
    start_id = 0
    tkd = tokenizer.tokenize(sent)

    for i, tk in enumerate(tkd):
        if i == max_len:
            break
            
        diff = len(sent) - len(sent_e)
        if tk.startswith('▁'):
            word = tk[1:]
        else:
            word = tk
            
        
        try:
            start_id = sent_e.index(word) + diff
        except ValueError:
            # 토크나이즈 과정에서 쪼개지는 단어는 WSD에 등장하지 않음
            tokens.append(tk)
            word_ids.append(-1)
            continue
            
            
        end_id = start_id + len(word)
        sent_e = sent[end_id:]
        tokens.append(tk)
    
        for w_d in wsd:
            tokenized = False
            if start_id == w_d['begin']: 
                # 토크나이즈 된 첫번째 토큰일 경우
                # 단어+pos, sense_id, word_id 모두 기록
                if w_d['sense_id'] not in (777, 888, 999):
                    sense_ids[w_d['word_id']] = w_d['sense_id']
                    word_ids.append(w_d['word_id'])
                    tokenized = True
                    break
            elif start_id > w_d['begin'] and end_id <= w_d['end'] and tokenized:
                # 두번째 이후 토큰일 경우 word_id만 계속해서 기록
                if w_d['sense_id'] not in (777, 888, 999):
                    word_ids.append(w_d['word_id'])
                    break
        else:
            word_ids.append(-1)
    
    if len(tokens) < max_len:
        fill_len = max_len - len(tokens)
        tokens += [""] * fill_len
        word_ids += [-1] * fill_len
        
    assert len(tokens) == max_len
    
    word_pos = [wsd_d['word']+"_"+wsd_d['pos'] for wsd_d in wsd]
    # 등장하지 않은 word_id에 대해서는 sense_id 를 -1 로 채워줌
    for j in range(1, len(word_pos)+1):
        if j not in sense_ids.keys():
            sense_ids[j] = -1
            
    sense_ids = dict(sorted(sense_ids.items()))
    #[sense_ids[j] for j in range(1, len(word_pos)+1)]
#     dict(sorted(sense_ids[key].items(), key=lambda x: x[1], reverse=True))
    
    res = {'tokens':tokens, 'words':word_pos, 'sense_ids':sense_ids, 'word_ids':word_ids}

    return res


class ContextDataset(Dataset):
    """데이터프레임 형태의 데이터를 읽는 데이터셋"""

    def __init__(self, data_df):
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data['form'][index], self.data['WSD'][index]


class BatchGenerator:
    """데이터로더가 후처리과정에서 토크나이즈 및 필요 정보 기록하는 함수"""
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __call__(self, batch):

        if isinstance(batch[0], str):
            sentences = [batch[0]]
            wsds = [batch[1]]
        elif isinstance(batch, list):
            sentences = [item[0] for item in batch]
            wsds = [item[1] for item in batch]
        else:
            sentences = list(item for item in batch[0])
            wsds = list(item for item in batch[1])

        res = self.tokenizer(sentences, 
                                    padding='max_length', 
                                    max_length=self.max_len,
                                    truncation=True,
                                    return_tensors='pt')
        context_ids = res['input_ids']
        context_attn_masks = res['attention_mask']
        
        tokd_wsd = [tokenizer_wsd(self.tokenizer, sentence, eval(wsd), self.max_len) \
                    for sentence, wsd in zip(sentences, wsds)]
        
        context_output_masks = torch.tensor([tokd['word_ids'] for tokd in tokd_wsd])

        sense_ids = [tokd['sense_ids'] for tokd in tokd_wsd]
        word_pos = [tokd['words'] for tokd in tokd_wsd]
        
        return context_ids, context_attn_masks, context_output_masks, word_pos, sense_ids
    
def context_dataloader(dataset, batch_generator, batch_size=4):
    """데이터 로더"""
    data_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              collate_fn=batch_generator,
                              num_workers=4)
    return data_loader
    