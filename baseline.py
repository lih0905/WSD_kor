"""
우리말샘 사전을 기준으로 단어의 의미가 2개 이상인 단어에 대하여,
훈련 데이터셋 기준 최고 빈도 의미를 매핑하는 Baseline Model.
(빈도가 동일한 의미가 2개 이상일 경우 랜덤 샘플링)

Seed=42 일 때 정확도 86.232% 달성
"""

import random
import pickle
from collections import defaultdict

import numpy as np

def count_meaning(data):
    """문장들이 주어지면 단어별로 의미의 등장횟수를 카운트하여 딕셔너리 반환
       이때 등장빈도 순으로 정렬한다.
    ex:
        d[word] = {2:5, 1:3, 3:1}
    """
    count_d = defaultdict(dict)
    for sent in data:
        for word, pos, sense_no in sent:
            if pos in ('NNP', 'NNG', 'NNB', 'NP') and word in urimal_dict.keys():
                try:
                    count_d[word][sense_no] += 1
                except KeyError:
                    count_d[word][sense_no] = 1
                    
    for key in count_d.keys():
        count_d[key] = dict(sorted(count_d[key].items(), key=lambda x: x[1], reverse=True))

    return count_d


if __name__ == '__main__':

    # SEED 고정
    np.random.seed(42)
    random.seed(42)
    
    # 전처리된 데이터셋 로드
    with open('Data/processed_train.pickle', 'rb') as f:
        train_data = pickle.load(f)
    with open('Data/processed_eval.pickle', 'rb') as f:
        eval_data = pickle.load(f)
    with open('Dict/processed_dictionary.pickle', 'rb') as f:
        urimal_dict = pickle.load(f)
        
    # 단어별로 훈련셋에서 의미의 등장횟수 카운트 
    train_count_d = count_meaning(train_data)
    
    # 최다 빈도 의미 단어 추출
    train_count_d_max = {}
    for word, sample in train_count_d.items():
        train_count_d_max[word] = [sense_no for sense_no, value in sample.items() \
                                   if value==max(sample.values())]
        
    # 우리말샘 기준 다의어 추출
    multiwords = set(word for word, d in urimal_dict.items() if len(d['definition'])>1)
    
    # 평가 데이터셋의 의미번호 추출 (명사 한정)
    result = []
    for sent in eval_data:
        result += [sense_no for word, pos, sense_no in sent \
                   if pos in ('NNP', 'NNG', 'NNB', 'NP') and word in multiwords]
        
    # 최다 빈도 기준 의미번호 산출
    # 이때 빈도가 같은 의미가 있는 단어들은 랜덤 샘플링
    preds = []
    for sent in eval_data:
        for word, pos, _ in sent:
            if pos in ('NNP', 'NNG', 'NNB', 'NP') and word in multiwords:
                try:
                    sense_no = random.choice(train_count_d_max[word])
#                     sense_no = train_count_d_max[word][0]
                except KeyError: # 훈련셋에 없었던 단어는 -1 입력 
                    sense_no = -1
                preds.append(sense_no)    
    
    # 최종 정확도 출력
    eval_acc = np.mean(np.array(result) == np.array(preds))
    print(f"의미의 빈도 기반으로 산출한 정확도는 {eval_acc*100:.3f}%입니다.")