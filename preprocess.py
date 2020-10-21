"""
국립국어원 말뭉치 데이터를 불러온 후 다음 형태로 변환한다.

    [(단어1, 품사1, 의미순번1), (단어2, 품사2, 의미순번2), ... ]
    * 다의어가 아닌 경우는 의미순번 = -1

또한 다운로드한 [우리말샘](https://opendict.korean.go.kr/) 사전을 CSV 파일 형태로 변환한다.
"""

import os
import json
import pickle
import xml.etree.ElementTree as elemTree

import pandas as pd

data_path = 'Data'
fname1 = 'NXLS1902008050.json'
fname2 = 'SXLS1902008030.json'
train_fname = 'processed_train.pickle'
eval_fname = 'processed_eval.pickle'

def process(sent_input):
    """말뭉치 데이터의 문장이 주어졌을 때 다음 형식으로 변환
    [(단어1, 품사1, 의미순번1), (단어2, 품사2, 의미순번2), ... ]
    """
    result = []
    for word_d in sent_input['morpheme']:
        # {'id': 1, 'form': '2012', 'label': 'SN', 'word_id': 1, 'position': 1},
        for wsd_d in sent_input['WSD']:
            # {'word': '년', 'sense_id': 2, 'pos': 'NNB', 'begin': 4, 'end': 5, 'word_id': 1}
            if word_d['form'] == wsd_d['word'] and word_d['word_id'] == wsd_d['word_id'] and word_d['label'] == wsd_d['pos']:
                # 이 경우 (단어, 품사, 의미순번) 
                result.append((word_d['form'], word_d['label'], wsd_d['sense_id']))
                break
        else: # 다의어가 아닌 경우는 의미순번을 제외
            result.append((word_d['form'], word_d['label'], -1))
    return result

def dict_to_list(dirname):
    """우리말샘 사전 데이터를 딕셔너리로 변환"""
    dict_list = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        if filename.endswith('xml'):
            full_filename = os.path.join(dirname, filename)
            root = elemTree.parse(full_filename).getroot()
            for it in root.iter('item'):
                word = it.find('wordInfo').find('word').text
                sense_no = it.find('senseInfo').find('sense_no').text
                try:
                    pos = it.find('senseInfo').find('pos').text
                except:
                    pos = ''
                definition = it.find('senseInfo').find('definition').text
                dict_list.append([word, sense_no, pos, definition])
    
    return dict_list


if __name__ == "__main__":

    print("말뭉치 데이터 로딩...")
    with open(os.path.join(data_path, fname1), "r") as json_file:
        f1 = json.load(json_file)

    with open(os.path.join(data_path, fname2), "r") as json_file:
        f2 = json.load(json_file)

    # 데이터의 key는 'id', 'metadata', 'document' 이며, 이중 'document'가 필요한 데이터 부분
    # 'document'는 기사들의 list로 이루어져 있으며, item은 딕셔너리이다.
    # 이 딕셔너리 또한 'sentence'라는 key를 조회하면 실제 데이터를 얻을 수 있다.
    data = f1['document'] + f2['document']
    print(f"문맥 데이터는 {len(data)}개입니다.")

    print("말뭉치 데이터 변환 시작...")
    train_results = []
    eval_results = []
    # 문맥 데이터 중 10번째마다 eval_data로 저장,
    # 그외는 train_data로 저장
    for i, paragraph in enumerate(data):
        para_result = []
        for sentence in paragraph['sentence']:
            # 일부 오류나는 라인이 있어 강제로 패스
            try:
                para_result.append(process(sentence))
            except:
                break
        # 오류가 나지 않은 문맥만 추가
        else:
            if i % 10 != 0:
                train_results.append(para_result)
            elif i % 10 == 0:
                eval_results.append(para_result)

    print("변환된 말뭉치 데이터 파일로 저장...")
    # 완료된 데이터를 pickle 데이터로 저장
    with open(os.path.join(data_path, train_fname), "wb") as train_f:
        pickle.dump(train_results, train_f)
    with open(os.path.join(data_path, eval_fname), "wb") as eval_f:
        pickle.dump(eval_results, eval_f)            
        
        
    print("우리말샘 사전 데이터 로딩...")
    # 우리말샘 사전 데이터 불러온 후 CSV 파일로 저장
    dict_list = dict_to_list('Dict')
    
    df = pd.DataFrame(dict_list, columns=['word', 'sense_no', 'pos', 'definition'])
    # '제주-도' 등을 '제주도'로 변환
    df['word'] = df['word'].apply(lambda x: x.replace('-',''))
    df['sense_id'] = df['sense_no'].astype(int)
    print(f'사전 데이터는 총 {len(df)}개 입니다.')
    
    print("우리말샘 사전 데이터 저장...")
    # 변환 파일 저장
    df.to_csv('Dict/dict_processed.csv', index=False)
    
    print("전처리가 끝났습니다.")