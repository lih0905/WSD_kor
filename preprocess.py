"""
다운로드한 [우리말샘](https://opendict.korean.go.kr/) 사전을 json 파일 형태로 변환한다.

또한 국립국어원 말뭉치 데이터를 불러온 후 다음 형태로 변환하여 csv 파일로 저장한다.

    [(문장1, 다의어1), (문장2, 다의어2), ...]
    
이때 다의어 중 우리말샘 기준으로 의미가 1개인 단어는 삭제한다.
"""

import os
import json
import argparse
import pickle
import xml.etree.ElementTree as elemTree

import pandas as pd

# Argparse Setting
parser = argparse.ArgumentParser(description='전처리 파라미터 설정')

# training arguments
parser.add_argument('--dict-path', type=str, default='./Dict')
parser.add_argument('--data-path', type=str, default='./Data')

fname1 = 'NXLS1902008050.json'
fname2 = 'SXLS1902008030.json'
train_fname = 'processed_train.csv'
eval_fname = 'processed_eval.csv'
dict_fname = 'processed_dictionary.json'


def dict_to_data(dirname):
    """
    우리말샘 사전 데이터를 딕셔너리로 변환

    Args:
        dirname : string (우리말샘 사전 파일이 저장된 폴더)

    Return:
        dictionary ('key' : 단어+품사, 'definition' : 의미, 'sense_no' : 단어 의미 순번)
    """
    dict_data = {}
    filenames = os.listdir(dirname)
    for filename in filenames:
        if filename.endswith('xml'):
            full_filename = os.path.join(dirname, filename)
            root = elemTree.parse(full_filename).getroot()
            for it in root.iter('item'):
                word = it.find('wordInfo').find('word').text
                word = word.replace('-', '')
                word = word.replace('^', '')
                sense_no = it.find('senseInfo').find('sense_no').text
                try:
                    pos = it.find('senseInfo').find('pos').text
                except:
                    pos = ''
                definition = it.find('senseInfo').find('definition').text
                if word not in dict_data.keys():
                    w_dict = {}
                    w_dict['key'] = [word+"_"+pos]
                    w_dict['definition'] = [definition]
                    w_dict['sense_no'] = [int(sense_no)]
                    dict_data[word] = w_dict
                else:
                    dict_data[word]['key'].append(word+"_"+pos)
                    dict_data[word]['definition'].append(definition)
                    dict_data[word]['sense_no'].append(int(sense_no))
    
    return dict_data

def process(sent_input, multiwords_set):
    """
    말뭉치 데이터의 문장이 주어졌을 때 문장과 WSD만 남기며, 또한 다의어 중
    우리말 사전 기준으로 의미 갯수가 2개 이상인 것만 남긴다.
    
    Args:
        sent_input : dictionary(key=['id', 'form', 'word', 'morpheme', 'WSD'])
        multiwords_set : set(우리말샘 사전 기준 의미가 2개 이상인 명사의 집합)
    Return:
        dictionary(key=['form', 'WSD'])
    """
    
    sent_input2 = sent_input.copy()
    # 사용하지 않는 key 제거
    del sent_input2['id']
    del sent_input2['word']
    del sent_input2['morpheme']
    
    WSD_multiwords = []
    for wsd_d in sent_input2['WSD']:
        # 우리말샘 기준 의미가 2개 이상인 단어이면서
        # WSD 데이터의 의미번호가 777, 888, 999 이 아닌 다의어로 한정
        # 777 : 우리말샘에 해당 형태가 없는 경우
        # 888 : 우리말샘에 형태는 있되 해당 의미가 없는 경우
        # 999 : 말뭉치 원어절에 오타, 탈자가 있는 경우
        if wsd_d['word'] in multiwords_set and wsd_d['sense_id'] not in (777,888,999):
            WSD_multiwords.append(wsd_d)
    sent_input2['WSD'] = WSD_multiwords
    
    return sent_input2


if __name__ == "__main__":

    args = parser.parse_args()
    dict_path = args.dict_path
    data_path = args.data_path

    print("우리말샘 사전 데이터 로딩...")
    # 우리말샘 사전 데이터 불러온 후 json 파일로 저장
    dict_data = dict_to_data(dict_path)
    
    print(f'사전 데이터는 총 {len(dict_data)}개 입니다.')
    
    print("우리말샘 사전 데이터 저장...")
    # 변환 파일 저장
    with open(os.path.join(dict_path, dict_fname), 'w') as f:
        json.dump(dict_data, f)
    
    # 다의어 추출
    multiwords_set = set(word for word, d in dict_data.items() if len(d['definition'])>1)

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
    i = 0
    for paragraph in data:
        for sentence in paragraph['sentence']:
            sent = process(sentence, multiwords_set)
            if len(sent['WSD']) > 0:
                i += 1
                if i % 10 != 0:
                    train_results.append(sent)
                elif i % 10 == 0:
                    eval_results.append(sent)
            else:
                continue

    train_df = pd.DataFrame(train_results)
    eval_df = pd.DataFrame(eval_results)

    print("변환된 말뭉치 데이터 파일로 저장...")
    # 완료된 데이터를 csv 파일로 저장
    train_df.to_csv(os.path.join(data_path, train_fname), index=False)
    eval_df.to_csv(os.path.join(data_path, eval_fname), index=False)
       
    print("전처리가 끝났습니다.")
