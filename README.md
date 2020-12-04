# 어휘 의미 분석 모델 

## 개요

```
* '형'은 자상하고 애정이 깊었으며 언제나 너그러웠다.
* 그는 법정에서 12년 '형'을 선고 받았다.
```

위 두 문장에서 공통적으로 사용된 단어 '형'은 여러가지 의미를 가지는 <b>다의어</b>입니다. 첫 번째 문장에서 '형'은 '같은 부모에게서 태어난 남자 중 손윗사람'라는 의미로, 두 번째 '형'은 '범죄에 대한 법적 제재'이라는 의미로 사용되었습니다. '형'이라는 단어는 이외에도 '형태'라는 의미로 쓰이기도 합니다. 이런 다의어가 주어진 문장에서 어떤 의미로 사용되었는지를 분석(Word Sense Disambiguation, 이하 WSD)하는 딥러닝 기반 모델을 구현하는 것이 목표입니다.

해당 모델의 구현을 위한 데이터셋은 <b>국립국어원</b>에서 제공하는 [<b>어휘 의미 분석 말뭉치</b>](https://corpus.korean.go.kr/)를 이용하였습니다. 이 말뭉치에서는 [<b>우리말샘</b>](https://opendict.korean.go.kr/main) 사전의 의미들을 기준으로, 주어진 문장에서 다의어 <b>명사</b>들이 가지는 의미를 정리하였습니다. 따라서 이번 분석에서 다의어는 <b>우리말샘 사전 기준 의미가 2개 이상인 명사</b>로 한정하였습니다.

이 데이터를 통해 문장의 다의어를 분석하기 위해 사용한 모델은 <b>Facebook</b>에서 [Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders (ACL, 2020)](https://blvns.github.io/papers/acl2020.pdf) 이라는 논문을 통해 공개한 <b>Gloss Informed Bi-encoders</b> 입니다. 영문 데이터에서 이미 우수한 성능이 입증된 모델을 한글에 적용하였을 때 어떤 결과를 내는지 파악해보기 위해, 사전학습된 [DistilKobert](https://github.com/monologg/DistilKoBERT)를 기반으로 [Pytorch](https://pytorch.org/)와 [Huggingface Transformers](https://github.com/huggingface/transformers)를 사용하여 해당 모델을 구현하였습니다.

(*) 모델은 [구름IDE](https://www.goorm.io/)의 GPU 체험 이벤트를 통해 학습하였습니다.

## 어휘 의미 분석 말뭉치 

어휘 의미 분석 말뭉치는 문어와 구어로 이루어져 있으며, '문맥/문장/단어/다의어 의미 정보'의 딕셔너리로 구축되어 있습니다. 어휘 의미 분석 말뭉치의 통계 정보는 다음과 같습니다. (각 문어, 구어, 합계 순)

* 문맥 : 7265 + 423 = 7688개
* 문장 : 150,082 + 223,962 = 374,044개
* 어절 : 2,000,213 + 1,006,447 = 3,006,660개
* 다의어 : 1,672,688 + 467,480 = 2,140,168개

이 데이터셋은 문맥/문장 단위로 다음과 같은 JSON 형태로 주어집니다.

```
{
 'id': 'SARW1800000001.1',
 'form': '요즘처럼 추운 날씨에는',
 'word': [{'begin': 0, 'end': 4, 'form': '요즘처럼', 'id': 1},
          {'begin': 5, 'end': 7, 'form': '추운', 'id': 2},
          {'begin': 8, 'end': 12, 'form': '날씨에는', 'id': 3}],
 'morpheme': [{'form': '요즘',
               'id': 1,
               'label': 'NNG',
               'position': 1,
               'word_id': 1},
              {'form': '처럼',
               'id': 2,
               'label': 'JKB',
               'position': 2,
               'word_id': 1},
              {'form': '춥',
               'id': 3,
               'label': 'VA',
               'position': 1,
               'word_id': 2},
              {'form': 'ㄴ',
               'id': 4,
               'label': 'ETM',
               'position': 2,
               'word_id': 2},
              {'form': '날씨',
               'id': 5,
               'label': 'NNG',
               'position': 1,
               'word_id': 3},
              {'form': '에',
               'id': 6,
               'label': 'JKB',
               'position': 2,
               'word_id': 3},
              {'form': '는',
               'id': 7,
               'label': 'JX',
               'position': 3,
               'word_id': 3}],
 'WSD': [{'begin': 0,
          'end': 2,
          'pos': 'NNG',
          'sense_id': 1,
          'word': '요즘',
          'word_id': 1},
         {'begin': 8,
          'end': 10,
          'pos': 'NNG',
          'sense_id': 1,
          'word': '날씨',
          'word_id': 3}],          
}
```
각 key에 대응되는 정보는 다음과 같습니다.

* id : 문장별 고유번호
* form : 문장
* word : 문장을 구성하는 어절
* morpheme : 문장을 구성하는 형태소
* WSD : 문장의 다의어 

다의어의 의미에 대한 정보는 `WSD`라는 키에 대응되어 있으며, 이를 통해 문장에서 해당 다의어의 인덱스 정보(`begin`, `end`), 형태소 정보(`pos`), 우리말샘 기준 의미 정보(`sense_id`) 등의 정보를 제공합니다. 

어휘 뭉치 데이터의 구축은 다음 프로세스를 통해 이루어졌습니다.

![diagram](https://github.com/lih0905/WSD_kor/blob/master/images/diagram.png?raw=true)

어휘 의미 말뭉치는 국립국어원 말뭉치 홈페이지에서 신청서를 작성, 허가 후 다운로드 받을 수 있습니다. 더 자세한 사항은 해당 데이터를 구축한 고려대학교 연구진이 국립국어원에 제출한 [분석보고서](https://korean.go.kr/common/download.do;front=705CF43F5B77029E1B5BE09E8910830F?file_path=reportData&c_file_name=f7222492-4580-40c6-864f-b66caeeeab3c_0.pdf&o_file_name=%EC%B5%9C%EC%A2%85%20%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%96%B4%ED%9C%98%EC%9D%98%EB%AF%B8%20%EB%B6%84%EC%84%9D%20%EB%A7%90%EB%AD%89%EC%B9%98%20%EA%B5%AC%EC%B6%95.pdf)에 수록되어 있습니다.

이 모델에서는 어휘 말뭉치 중 매 10번째 문맥 데이터를 평가 데이터, 그 외를 훈련 데이터로 사용하였습니다.


## 우리말샘 사전

다음은 우리말샘 사전 [홈페이지](https://opendict.korean.go.kr/main)에서 발췌한 우리말샘 소개글입니다.

> 우리말의 쓰임이 궁금할 때 국어사전을 찾게 됩니다. 그런데 막상 사전을 찾아도 정보가 없거나 설명이 어려워 아쉬움을 느낄 때가 있습니다. 그동안 간행된 사전들은 여러 가지 제약이 있어 정보를 압축하여 제한적으로 수록하였기 때문입니다.<br> 사용자가 참여하는 ‘우리말샘’은 이런 문제점을 극복하고자 기획되었습니다. 한국어를 사용하는 우리 모두가 주체가 되어 예전에 사용되었거나 현재 사용되고 있는 어휘를 더욱 다양하고 알기 쉽게 수록하고자 합니다. 또한 전통적인 사전 정보 이외에 다양한 언어 지식도 실어 한국어에 관한 많은 궁금증을 푸는 통로가 되고자 합니다.

우리말샘은 다음과 같이 전문가가 감수한 전통적인 의미에 더해 참여자가 제안한 정보 또한 수록되어, 갈수록 다양해지고 있는 어휘의 의미들을 품기에 적합한 사전입니다. 

![우리말](https://github.com/lih0905/WSD_kor/blob/master/images/urimal.png?raw=true)

우리말샘은 누구나 자유롭게 이용할 수 있는 `크리에이티브 커먼즈 저작자표시-동일조건변경허락 2.0 대한민국 라이선스`에 따라 배포되며, 회원 가입 후 사전 전체를 XML 파일로 다운로드 할 수 있습니다. 


## 다의어 정보 인덱스 전처리

문장과 다의어 정보로 구성된 데이터셋은 다음과 같은 형태입니다.

```
'form' : '실제 애플이 겨냥하는 상대는 구글이지 삼성전자가 아니라는 평가도 나온다.'
 'WSD' :  [{'word': '겨냥',
            'sense_id': 1,
            'pos': 'NNG',
            'begin': 7,
            'end': 9,
            'word_id': 3},
            ...]
```

여기서 `WSD` 데이터의 인덱스는 주어진 문장에서 다의어의 위치를 음절 단위로 평가한 것입니다. 그러나 해당 인덱스는 토크나이즈된 이후에는 더이상 의미가 없으므로, 토크나이즈 후 토큰별 인덱스를 기준으로 한 다의어 마스킹을 필요로 합니다.

이를 다음과 같이 구현하였습니다.

```python
"""
    Args:
        tokenizer: transformers.Tokenizer
        sent : string
        wsd : dictionary(key=['word', 'sense_id', 'pos', begin', 'end', 'word_id'])
"""

tkd = tokenizer.tokenize(sent)
sent_e = sent

# 문장에 있는 다의어의 의미 번호 기록
sense_ids = {}
# 다의어에 해당하는 토큰은 해당 다의어의 발생 순번을 기록
word_ids = []
# 현재 토큰이 다의어의 일부에 해당하는지 여부를 기록
tokenized = False

    for tk in tkd:
        diff = len(sent) - len(sent_e)
        # 토큰의 원문 기준 인덱스 파악
        start_id = sent_e.index(word) + diff
        end_id = start_id + len(word)
        # 원문 중 현재 토큰 이후 부분만 남김
        sent_e = sent[end_id:]
    
        # 다의어 정보 중 현재 토큰을 오버랩하는 다의어 탐색
        for w_d in wsd:
            if start_id == w_d['begin']: 
                # 토크나이즈 된 첫번째 토큰일 경우
                # sense_id, word_id 모두 기록
                if w_d['sense_id'] not in (777, 888, 999):
                    # WSD 데이터의 의미번호가 777, 888, 999 이 아닌 다의어로 한정
                    # 777 : 우리말샘에 해당 형태가 없는 경우
                    # 888 : 우리말샘에 형태는 있되 해당 의미가 없는 경우
                    # 999 : 말뭉치 원어절에 오타, 탈자가 있는 경우
                    sense_ids[w_d['word_id']] = w_d['sense_id']
                    word_ids.append(w_d['word_id'])
                    tokenized = True
                    # 하나의 토큰이 여러 다의어에 걸쳐 있는 경우는 없으므로
                    # 토큰을 오버랩하는 다의어를 찾으면 바로 다음 토큰으로 넘어감
                    break
            elif start_id > w_d['begin'] and end_id <= w_d['end'] and tokenized:
                # 현재 토큰이 토크나이즈된 다의어의 두번째 이후 토큰일 경우 
                # word_id만 계속해서 기록
                if w_d['sense_id'] not in (777, 888, 999):
                    word_ids.append(w_d['word_id'])
                    break
        else:
            # 현재 토큰을 오버랩하는 다의어가 없는 경우는 단어 순번을 -1 로 기록
            word_ids.append(-1)
```

이 함수는 각 토큰이 다의어에 해당하는지 여부를 마스킹하여 리턴합니다. 이 정보를 토대로 모델은 문맥 인코더를 통해 토크나이즈된 문장과 다의어 정보를 같이 학습하게 됩니다.

## 모델

* Baseline : 훈련 데이터셋 기준 각 단어별 최고 빈출 단어로 추론
* [Gloss Informed Bi-encoders](https://github.com/facebookresearch/wsd-biencoders) : 

![Model](https://github.com/facebookresearch/wsd-biencoders/raw/master/docs/wsd_biencoder_architecture.jpg)

영문 WSD 분야에서 State-of-the-art 성능을 달성한 바 있는 이 모델은 사전 학습된 BERT 계열 모델을 기반으로 하는 두 개의 인코더로 이루어져 있습니다. 문맥(context) 인코더를 통해 다의어와 다의어가 포함된 문장의 표현, 어구(gloss) 인코더를 통해 해당 다의어가 가지는 여러 의미들의 표현을 얻습니다. 이 두 인코더를 통해 출력된 벡터간의 내적을 통해 해당 단어의 의미를 분석합니다. 학습된 가중치 데이터는 [여기](https://drive.google.com/file/d/1YYerGnZ76KOKp8Ik2tUP1a6HsD9-bMNU/view?usp=sharing)에서 다운받을 수 있습니다.

한국어 BERT 모델은 사전학습된 [DistilKoBERT](https://github.com/monologg/DistilKoBERT)를 사용하였습니다. 

## 모델 학습 결과 

| Model      	| Accuracy 	| F1(weighted) 	|
|------------	|----------	|--------------	|
| Baseline   	| 0.85949  	| 0.85864      	|
| Bi-encoder 	| 0.88327  	| 0.88286       |

## 사용법

* 필요 라이브러리
    * Python >= 3.7
    * Pytorch >= 1.6.0
    * Huggingface Transformers >= 3.3.0
    * Pandas
    * Numpy
    * tqdm

* 전처리

```bash
python preprocess.py
```

* Baseline 모델 훈련

```bash
python baseline.py
```


* Bi-encoder 모델 훈련

```bash
python main.py
```

* Bi-encoder 모델 추론

```bash
python eval.py --text "형은 자상하고 애정이 깊었으며 언제나 너그러웠다." 
```

```
----------------------------------------------------------------------------------------------------
문장 : 형은 자상하고 애정이 깊었으며 언제나 너그러웠다.
----------------------------------------------------------------------------------------------------
'형'의 의미 : 같은 부모에게서 태어난 사이이거나 일가친척 가운데 항렬이 같은 남자들 사이에서 손윗사람을 이르거나 부르는 말. 주로 남자 형제 사이에 많이 쓴다.
'애정'의 의미 : 사랑하는 마음.
```

```bash
python eval.py --text "그는 법정에서 12년 형을 선고 받았다."
```

```
----------------------------------------------------------------------------------------------------
문장 : 그는 법정에서 12년 형을 선고 받았다.
----------------------------------------------------------------------------------------------------
'그'의 의미 : 말하는 이와 듣는 이가 아닌 사람을 가리키는 삼인칭 대명사. 앞에서 이미 이야기하였거나 듣는 이가 생각하고 있는 사람을 가리킨다. 주로 남자를 가리킬 때 쓴다.
'법정'의 의미 : 법원이 소송 절차에 따라 송사를 심리하고 판결하는 곳.
'형'의 의미 : 범죄에 대한 법률의 효과로서 국가 따위가 범죄자에게 제재를 가함. 또는 그 제재. 이에는 사형, 징역, 금고, 자격 상실, 자격 정지, 벌금, 구류, 과료, 몰수가 있다.
'선고'의 의미 : 형사 사건을 심사하는 법정에서 재판장이 판결을 알리는 일
```


## References

* [국립국어원(2020). 국립국어원 어휘 의미 분석 말뭉치(버전 1.0)](https://corpus.korean.go.kr/)

* [Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders](https://blvns.github.io/papers/acl2020.pdf) [[Code]](https://github.com/facebookresearch/wsd-biencoders)

* [DistilKoBert : Distillation of KoBERT](https://github.com/monologg/DistilKoBERT)