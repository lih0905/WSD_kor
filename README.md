# 어휘 의미 분석 모델 

다의어가 주어진 문장에서 구체적으로 어떤 의미를 가지는 지 분석하는 모델 구현. 데이터셋은 국립국어원 어휘 의미 분석 말뭉치를 이용한다.

* 데이터셋 : 모두의 말뭉치(어휘 의미 분석 말뭉치)([https://corpus.korean.go.kr/](https://corpus.korean.go.kr/))
* 사전 : 우리말샘([https://opendict.korean.go.kr/main](https://opendict.korean.go.kr/main))
* 다의어 : 우리말샘 기준 의미가 2개 이상인 단어 (이 프로젝트에서는 명사로 한정)

## 어휘 의미 분석 말뭉치 

* TBA

## 전처리 과정

* TBA

## 모델

* Baseline : 훈련 데이터셋 기준 각 단어별 최고 빈출 단어로 추론
* [Gloss Informed Bi-encoders for WSD](https://github.com/facebookresearch/wsd-biencoders) : 

![Model](https://github.com/facebookresearch/wsd-biencoders/raw/master/docs/wsd_biencoder_architecture.jpg)

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
python eval.py --text "꿈깨라 멍청한 녀석"   
```

```
----------------------------------------------------------------------------------------------------
문장 : 꿈깨라 멍청한 녀석
----------------------------------------------------------------------------------------------------
'꿈'의 의미 : 실현될 가능성이 아주 적거나 전혀 없는 헛된 기대나 생각.
'녀석'의 의미 : 사내아이를 귀엽게 이르는 말.
```
