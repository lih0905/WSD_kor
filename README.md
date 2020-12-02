# 어휘 의미 분석 모델 

## 개요

```
* 그의 '꿈'은 커서 대통령이 되는 것이다.
* 어젯밤 꾼 '꿈' 때문에 하루종일 멍했다.
```

위 두 문장에서 공통적으로 사용된 단어 '꿈'은 여러가지 의미를 가지는 <b>다의어</b>입니다. 첫 번째 문장에서 '꿈'은 '희망이나 목표'라는 의미로, 두 번째 '꿈'은 '자면서 느끼는 정신 현상'이라는 의미로 사용되었습니다. 이런 다의어가 주어진 문장에서 어떤 의미로 사용되었는지를 분석(Word Sense Disambiguation, 이하 WSD)하는 딥러닝 기반 모델을 구현하는 것이 목표입니다.

해당 모델의 구현을 위한 데이터셋은 <b>국립국어원</b>에서 제공하는 [<b>어휘 의미 분석 말뭉치</b>](https://corpus.korean.go.kr/)를 이용하였습니다. 이 말뭉치에서는 [<b>우리말샘</b>](https://opendict.korean.go.kr/main) 사전의 의미들을 기준으로, 주어진 문장에서 다의어 <b>명사</b>들이 가지는 의미를 정리하였습니다. 따라서 이번 분석에서 다의어는 <b>우리말샘 사전 기준 의미가 2개 이상인 명사</b>로 한정하였습니다.

이 데이터를 통해 문장의 다의어를 분석하기 위해 사용한 모델은 <b>Facebook</b>에서 [Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders (ACL, 2020)](https://blvns.github.io/papers/acl2020.pdf) 이라는 논문을 통해 공개한 <b>Gloss Informed Bi-encoders</b> 입니다. `Pytorch`를 통해 해당 모델을 구현하여, 영문 데이터에서 이미 우수한 성능이 입증된 모델을 한글에 적용하였을 때 어떤 결과를 내는지 파악해보고자 합니다.



## 어휘 의미 분석 말뭉치 

* [분석보고서](https://korean.go.kr/common/download.do;front=705CF43F5B77029E1B5BE09E8910830F?file_path=reportData&c_file_name=f7222492-4580-40c6-864f-b66caeeeab3c_0.pdf&o_file_name=%EC%B5%9C%EC%A2%85%20%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%96%B4%ED%9C%98%EC%9D%98%EB%AF%B8%20%EB%B6%84%EC%84%9D%20%EB%A7%90%EB%AD%89%EC%B9%98%20%EA%B5%AC%EC%B6%95.pdf) : 해당 데이터를 구축한 고려대학교 연구진이 국립국어원에 제출한 보고서

## 우리말샘 사전

* 함께 만들고 모두 누리는 우리말 사전

## 전처리 과정

* TBA

## 모델

* Baseline : 훈련 데이터셋 기준 각 단어별 최고 빈출 단어로 추론
* [Gloss Informed Bi-encoders](https://github.com/facebookresearch/wsd-biencoders) : 

![Model](https://github.com/facebookresearch/wsd-biencoders/raw/master/docs/wsd_biencoder_architecture.jpg)

이 모델은 사전 학습된 BERT 모델을 기반으로 한 두 개의 인코더로 이루어져 있습니다. 문맥(context) 인코더를 통해 다의어와 다의어가 포함된 문장의 표현, 어구(gloss) 인코더를 통해 해당 다의어가 가지는 여러 의미들의 표현을 얻습니다. 이 두 인코더를 통해 출력된 벡터간의 내적을 통해 해당 단어의 의미를 분석합니다. 이 모델은 영문 WSD 분야에서 State-of-the-art 성능을 달성한 바 있습니다.

## 결과 

| Model      	| Accuracy 	| F1(weighted) 	|
|------------	|----------	|--------------	|
| Baseline   	| 0.85949  	| 0.85864      	|
| Bi-encoder 	| 0.88327  	| 0.88286       |

## Requirements

* Python >= 3.7
* Pytorch >= 1.6.0
* Huggingface Transformers >= 3.3.0
* Pandas
* Numpy
* tqdm

## Process

* 전처리

```python
python preprocess.py
```

* Baseline 결과

```python
python baseline.py
```

```
의미의 빈도 기반으로 산출한 평가 데이터의 정확도는 0.85949, F1 점수는 0.85864 입니다.
```

* Bi-encoder 모델 훈련

```python
python main.py
```

* Bi-encoder 모델 추론

```python
python eval.py --text "그의 꿈은 커서 대통령이 되는 것이다."   
```

```
----------------------------------------------------------------------------------------------------
문장 : 그의 꿈은 커서 대통령이 되는 것이다.
----------------------------------------------------------------------------------------------------
'그'의 의미 : 말하는 이와 듣는 이가 아닌 사람을 가리키는 삼인칭 대명사. 앞에서 이미 이야기하였거나 듣는 이가 생각하고 있는 사람을 가리킨다. 주로 남자를 가리킬 때 쓴다.
'꿈'의 의미 : 실현하고 싶은 희망이나 이상.
'대통령'의 의미 : 외국에 대하여 국가를 대표하는 국가의 원수. 행정부의 실질적인 권한을 갖는 경우와 형식적인 권한만을 가지는 경우가 있는데 우리나라는 전자에 속한다.
'것'의 의미 : 사물, 일, 현상 따위를 추상적으로 이르는 말.
```

```python
python eval.py --text "사실 그의 꿈이 이루어질 가능성은 거의 없습니다."   
```

```
----------------------------------------------------------------------------------------------------
문장 : 꿈깨라 멍청한 녀석
----------------------------------------------------------------------------------------------------
'꿈'의 의미 : 실현될 가능성이 아주 적거나 전혀 없는 헛된 기대나 생각.
'녀석'의 의미 : 사내아이를 귀엽게 이르는 말.
```


## References

* [국립국어원(2020). 국립국어원 어휘 의미 분석 말뭉치(버전 1.0)](https://corpus.korean.go.kr/)

* [Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders](https://blvns.github.io/papers/acl2020.pdf)
