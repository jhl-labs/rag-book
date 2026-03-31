# 2. 임베딩 모델 — 텍스트를 숫자로 번역하기

> **이 챕터를 읽고 나면...**
>
> - 임베딩이 무엇인지, 왜 필요한지 설명할 수 있다
> - 벡터, 차원, 코사인 유사도 같은 핵심 개념을 이해한다
> - Python으로 직접 임베딩을 생성하고 비교할 수 있다
> - 자신의 프로젝트에 맞는 임베딩 모델을 선택할 수 있다

---

## 2.1 임베딩이란 무엇인가?

컴퓨터는 본질적으로 수를 다루는 기계입니다. 텍스트, 이미지, 음성 — 모든 것은 결국 숫자로 변환되어야 컴퓨터가 처리할 수 있습니다.

그런데 단순히 글자를 숫자로 바꾸는 것(예: 'A'=65, 'B'=66)은 **의미**를 담지 못합니다. "고양이"와 "냥이"는 같은 동물을 가리키지만, 이런 방식으로는 전혀 관련 없는 숫자가 됩니다.

**임베딩(Embedding)**은 이 문제를 해결합니다. 텍스트의 **의미(semantics)**를 보존하면서 숫자 벡터로 변환하는 기술입니다.

```
"고양이가 쥐를 쫓는다"   → [0.12, -0.34, 0.56, 0.89, ...]  (1536개의 숫자)
"캣이 마우스를 추격한다" → [0.13, -0.33, 0.55, 0.90, ...]  ← 거의 같은 숫자들!
"오늘 주식이 폭락했다"   → [0.78,  0.92, -0.21, 0.04, ...]  ← 완전히 다른 숫자들
```

의미가 비슷한 문장은 **비슷한 숫자들의 묶음**으로 변환됩니다. 이것이 임베딩의 핵심입니다.

---

### 벡터(Vector)란?

!!! note "용어 설명: 벡터(Vector)"
    **벡터**는 여러 개의 숫자를 순서대로 나열한 것입니다.

    - 1차원: `[3.5]` — 숫자 하나
    - 2차원: `[3.5, -1.2]` — 숫자 두 개 (2D 좌표처럼 생각)
    - 3차원: `[3.5, -1.2, 0.8]` — 숫자 세 개 (3D 좌표처럼 생각)
    - **1536차원**: `[0.12, -0.34, 0.56, ...]` — 숫자 1536개

    임베딩 벡터는 보통 수백~수천 차원이며, 각 숫자는 텍스트의 어떤 의미적 특성을 나타냅니다.

---

### 차원(Dimension)이란?

!!! note "용어 설명: 차원(Dimension)"
    벡터에 담긴 숫자의 **개수**를 차원이라고 합니다.

    - `[0.1, 0.2]` → 2차원 벡터
    - `[0.1, 0.2, 0.3, 0.4]` → 4차원 벡터
    - `[0.1, 0.2, ..., 1536개]` → 1536차원 벡터

    차원이 높을수록 더 많은 정보를 담을 수 있지만, 그만큼 저장 공간과 계산 비용도 늘어납니다.

---

!!! note "실생활 비유: GPS 좌표"
    임베딩은 마치 **도시의 GPS 좌표**와 같습니다.

    - 서울: `[37.5665, 126.9780]`
    - 인천: `[37.4563, 126.7052]`
    - 부산: `[35.1796, 129.0756]`
    - 도쿄: `[35.6762, 139.6503]`

    서울과 인천은 좌표가 비슷합니다 (가까운 도시). 서울과 부산은 좌표 차이가 큽니다 (먼 도시).

    임베딩도 마찬가지입니다. "고양이"와 "냥이"는 의미적으로 가까우므로 벡터도 비슷하고, "고양이"와 "주식"은 의미가 달라 벡터도 멀리 떨어져 있습니다.

    단, 실제 임베딩은 2차원이 아닌 1536차원 이상의 "공간"에 점을 찍는 것이라고 생각하면 됩니다.

---

### 왜 이게 중요한가? (임베딩의 필요성)

!!! warning "왜 이게 중요한가?"
    RAG 시스템에서 임베딩은 **심장**과 같습니다.

    사용자가 "파이썬으로 웹 크롤링하는 방법"을 물어봤을 때, 문서 데이터베이스에는 다음과 같은 문서들이 있다고 가정합시다:

    1. "Python requests 라이브러리로 HTTP 요청 보내기"
    2. "BeautifulSoup을 이용한 HTML 파싱"
    3. "자바스크립트 fetch API 사용법"
    4. "오늘 점심 메뉴 추천"

    키워드 검색만 하면 "파이썬"이나 "크롤링"이라는 단어가 정확히 없는 문서 1, 2는 못 찾을 수도 있습니다.

    하지만 임베딩을 사용하면 **의미**가 비슷한 문서 1, 2를 정확하게 찾아냅니다. 이것이 시맨틱 검색(Semantic Search)입니다.

---

## 2.2 임베딩이 작동하는 원리

### 토크나이제이션(Tokenization)

!!! note "용어 설명: 토크나이제이션(Tokenization)"
    임베딩 모델이 텍스트를 처리하기 전, 텍스트를 작은 조각으로 나누는 과정입니다.
    이 조각들을 **토큰(Token)**이라고 합니다.

    ```
    "임베딩은 재미있다" → ["임베딩", "은", " 재미", "있다"]
    "I love Python"     → ["I", " love", " Python"]
    ```

    토큰은 단어일 수도, 단어의 일부일 수도, 구두점일 수도 있습니다.
    **최대 토큰** 수는 모델이 한 번에 처리할 수 있는 텍스트의 길이를 의미합니다.
    예: 최대 8,191 토큰 ≈ 약 6,000~8,000 단어 ≈ 약 20~30페이지 분량

---

### 임베딩이 만들어지는 과정 (개념적 설명)

```
┌─────────────────────────────────────────────────────────────────┐
│                    임베딩 생성 과정                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  입력 텍스트: "고양이가 야옹 울었다"                               │
│       │                                                           │
│       ▼ (1단계: 토크나이제이션)                                    │
│  ["고양이", "가", " 야옹", " 울었다"]                             │
│       │                                                           │
│       ▼ (2단계: 토큰을 숫자 ID로 변환)                            │
│  [4521, 12, 8834, 9921]                                          │
│       │                                                           │
│       ▼ (3단계: 신경망을 통한 의미 추출)                           │
│  [수백 개의 트랜스포머 레이어를 거치며 문맥 이해]                 │
│       │                                                           │
│       ▼ (4단계: 하나의 벡터로 집약)                               │
│  [0.12, -0.34, 0.56, 0.89, ..., 0.23]  (1536차원)               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

### 벡터 공간에서의 시각화

2차원으로 단순화하면 이렇게 볼 수 있습니다:

```
     ↑ 동물 관련
     │
     │   고양이
     │       냥이
     │           길고양이
     │
─────┼──────────────────────── → 다른 축
     │
     │           주식
     │               코스피
     │                   투자
     │
     ↓ 금융 관련
```

실제로는 1536개의 축이 있어서 훨씬 정밀하게 의미를 구분할 수 있습니다.

---

## 2.3 유사도 측정: 코사인 유사도

두 텍스트가 얼마나 비슷한지 측정하려면 **두 벡터 사이의 거리**를 계산합니다.

### 코사인 유사도(Cosine Similarity)란?

!!! note "용어 설명: 코사인 유사도(Cosine Similarity)"
    두 벡터 사이의 **각도**를 이용해 유사도를 측정하는 방법입니다.

    - 값의 범위: -1 ~ 1
    - 1에 가까울수록: 매우 유사 (같은 방향을 가리킴)
    - 0에 가까울수록: 관련 없음 (직각 방향)
    - -1에 가까울수록: 반대 의미 (반대 방향)

    ```
    벡터 A ──────────────►
    벡터 B ──────────────►
            각도 ≈ 0°  →  cos(0°) = 1  →  완전히 같은 의미

    벡터 A ──────────────►
    벡터 B         ▲
                   │
            각도 = 90°  →  cos(90°) = 0  →  관련 없음
    ```

    왜 코사인을 쓸까요? 벡터의 **방향**이 의미를 나타내기 때문입니다.
    긴 문서와 짧은 문서라도 같은 주제면 같은 방향을 가리킵니다.

---

### 직접 계산해보기

```python
import numpy as np

# 두 문장의 임베딩 벡터 (예시 - 실제로는 수천 차원)
vector_a = np.array([0.12, -0.34, 0.56, 0.89])  # "고양이가 야옹 울었다"
vector_b = np.array([0.13, -0.33, 0.55, 0.90])  # "고양이가 소리를 냈다"
vector_c = np.array([0.78,  0.92, -0.21, 0.04]) # "주식이 폭락했다"

def cosine_similarity(v1, v2):
    """
    코사인 유사도 계산 함수

    공식: cos(θ) = (A · B) / (|A| × |B|)
    - A · B : 내적 (dot product) - 각 위치의 숫자를 곱해서 모두 더함
    - |A|   : 벡터의 크기 (norm) - 제곱합의 제곱근
    """
    # 내적 계산: [a1*b1 + a2*b2 + ...]
    dot_product = np.dot(v1, v2)

    # 각 벡터의 크기(norm) 계산
    norm_v1 = np.linalg.norm(v1)  # root(0.12^2 + 0.34^2 + ...)
    norm_v2 = np.linalg.norm(v2)

    # 코사인 유사도 = 내적 / (크기1 × 크기2)
    return dot_product / (norm_v1 * norm_v2)

# 유사도 계산
sim_ab = cosine_similarity(vector_a, vector_b)
sim_ac = cosine_similarity(vector_a, vector_c)

print(f"'고양이 야옹' vs '고양이 소리': {sim_ab:.4f}")  # 높은 유사도
print(f"'고양이 야옹' vs '주식 폭락':   {sim_ac:.4f}")  # 낮은 유사도
```

**예상 출력:**
```
'고양이 야옹' vs '고양이 소리': 0.9998
'고양이 야옹' vs '주식 폭락':   0.2341
```

---

## 2.4 첫 번째 임베딩 만들기

### 사전 준비

```bash
# 필요한 패키지 설치
pip install openai sentence-transformers numpy
```

### OpenAI 임베딩 (가장 쉬운 시작)

OpenAI API를 사용하면 코드 몇 줄로 고품질 임베딩을 만들 수 있습니다.

```python
# 1단계: 라이브러리 가져오기
from openai import OpenAI  # OpenAI의 공식 Python 라이브러리

# 2단계: 클라이언트 초기화
# OPENAI_API_KEY 환경변수를 자동으로 읽습니다
client = OpenAI()

# 3단계: 임베딩 생성
response = client.embeddings.create(
    model="text-embedding-3-small",  # 사용할 모델 이름
    input="안녕하세요, 임베딩 테스트입니다"  # 변환할 텍스트
)

# 4단계: 벡터 추출
# response.data[0].embedding: 첫 번째 텍스트의 임베딩 벡터
vector = response.data[0].embedding

# 5단계: 결과 확인
print(f"벡터 타입: {type(vector)}")         # <class 'list'>
print(f"벡터 차원: {len(vector)}")           # 1536
print(f"처음 5개 값: {vector[:5]}")          # [-0.023, 0.012, ...]
print(f"최솟값: {min(vector):.4f}")
print(f"최댓값: {max(vector):.4f}")
```

**예상 출력:**
```
벡터 타입: <class 'list'>
벡터 차원: 1536
처음 5개 값: [-0.0231, 0.0128, -0.0089, 0.0312, -0.0156]
최솟값: -0.1234
최댓값: 0.1567
```

---

### LangChain을 이용한 OpenAI 임베딩

LangChain은 AI 애플리케이션 개발을 위한 프레임워크입니다. RAG 파이프라인을 만들 때 자주 사용됩니다.

```python
# 라이브러리 가져오기
from langchain_openai import OpenAIEmbeddings
# langchain_openai: LangChain의 OpenAI 연동 패키지

# 기본 사용 (3072차원)
embeddings_large = OpenAIEmbeddings(
    model="text-embedding-3-large"  # 고성능 모델, 3072차원
)

# 차원 축소 버전 (비용 절감)
# 3072차원을 1024차원으로 줄여도 성능이 크게 떨어지지 않습니다
# 차원이 줄어들면 → 저장 공간 절약, 검색 속도 향상, API 비용 감소
embeddings_reduced = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024  # 3072 → 1024로 축소 (약 67% 절약)
)

# 단일 텍스트 임베딩
# embed_query: 검색 쿼리 하나를 임베딩할 때 사용
query_vector = embeddings_large.embed_query("RAG는 무엇인가요?")
print(f"쿼리 벡터 차원: {len(query_vector)}")  # 3072

# 여러 텍스트 임베딩
# embed_documents: 여러 문서를 한 번에 임베딩할 때 사용
documents = [
    "RAG는 검색 증강 생성(Retrieval Augmented Generation)입니다.",
    "벡터 데이터베이스는 임베딩을 저장하고 검색합니다.",
    "LangChain은 AI 애플리케이션 개발 프레임워크입니다."
]

# embed_documents는 리스트를 반환 (각 문서마다 벡터 하나)
doc_vectors = embeddings_large.embed_documents(documents)

print(f"문서 수: {len(doc_vectors)}")           # 3
print(f"각 벡터 차원: {len(doc_vectors[0])}")   # 3072
```

**예상 출력:**
```
쿼리 벡터 차원: 3072
문서 수: 3
각 벡터 차원: 3072
```

---

### 실제 유사도 비교 실험

```python
import numpy as np
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 테스트할 문장들
query = "파이썬으로 데이터를 분석하는 방법"

documents = [
    "Python pandas를 사용하면 데이터 분석을 쉽게 할 수 있습니다",     # 매우 관련
    "Python은 데이터 사이언스에서 가장 많이 사용되는 언어입니다",      # 관련
    "자바로 안드로이드 앱을 개발하는 방법",                             # 약간 관련 (프로그래밍)
    "오늘 점심은 김치찌개를 먹었습니다",                                 # 무관
]

# 임베딩 생성
# 쿼리 임베딩 (검색할 질문)
query_vector = np.array(embeddings.embed_query(query))

# 문서 임베딩 (검색 대상 문서들)
doc_vectors = embeddings.embed_documents(documents)

# 유사도 계산
print(f"검색 쿼리: '{query}'\n")
print("=" * 60)

for i, (doc, doc_vec) in enumerate(zip(documents, doc_vectors)):
    doc_vec = np.array(doc_vec)

    # 코사인 유사도 계산
    similarity = np.dot(query_vector, doc_vec) / (
        np.linalg.norm(query_vector) * np.linalg.norm(doc_vec)
    )

    # 시각적 막대 그래프 (유사도를 직관적으로 표시)
    bar_length = int(similarity * 30)  # 최대 30칸
    bar = "█" * bar_length + "░" * (30 - bar_length)

    print(f"[{bar}] {similarity:.4f}")
    print(f"  → {doc[:40]}...")
    print()
```

**예상 출력:**
```
검색 쿼리: '파이썬으로 데이터를 분석하는 방법'

============================================================
[████████████████████████░░░░░░] 0.8234
  → Python pandas를 사용하면 데이터 분석을 쉽게 할 수...

[███████████████████░░░░░░░░░░░] 0.7156
  → Python은 데이터 사이언스에서 가장 많이 사용되는...

[████████████░░░░░░░░░░░░░░░░░░] 0.4521
  → 자바로 안드로이드 앱을 개발하는 방법...

[████░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.1234
  → 오늘 점심은 김치찌개를 먹었습니다...
```

---

!!! tip "직접 해보기 1: 다양한 문장 실험"
    다음 코드를 실행해보고, 어떤 문장 쌍이 높은 유사도를 보이는지 예측해보세요.

    ```python
    from langchain_openai import OpenAIEmbeddings
    import numpy as np

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 테스트할 문장 쌍들 - 각 쌍의 유사도를 예측해보세요!
    pairs = [
        ("강아지가 공원을 뛰어다닌다", "개가 공원에서 달린다"),
        ("자동차 엔진 수리 방법", "차 고치는 법"),
        ("파이썬 프로그래밍", "뱀의 생태"),
        ("주식 투자 전략", "요리 레시피"),
        ("머신러닝이란 무엇인가", "What is machine learning?"),
    ]

    print("문장 쌍 유사도 측정 실험")
    print("=" * 70)

    for sent1, sent2 in pairs:
        v1 = np.array(embeddings.embed_query(sent1))
        v2 = np.array(embeddings.embed_query(sent2))

        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        if sim > 0.8:
            label = "매우 유사"
        elif sim > 0.6:
            label = "관련있음"
        elif sim > 0.4:
            label = "약간 관련"
        else:
            label = "무관"

        print(f"문장1: {sent1}")
        print(f"문장2: {sent2}")
        print(f"유사도: {sim:.4f} ({label})")
        print()
    ```

    **마지막 쌍(한국어 vs 영어)이 높은 유사도를 보인다면?**
    그것은 임베딩 모델이 다국어를 이해한다는 증거입니다!

---

## 2.5 오픈소스 임베딩 모델

API 비용 없이 자신의 컴퓨터에서 실행할 수 있는 오픈소스 모델들이 있습니다.

### sentence-transformers 라이브러리

```python
# 설치: pip install sentence-transformers

# 라이브러리 가져오기
from sentence_transformers import SentenceTransformer
# SentenceTransformer: 문장 임베딩에 특화된 라이브러리
# Hugging Face에 공개된 수백 개의 모델을 사용 가능

# 모델 로드
# BAAI/bge-m3: 한국어 포함 100개+ 언어를 지원하는 다국어 모델
# 처음 실행 시 모델 파일(약 2GB)을 자동으로 다운로드합니다
model = SentenceTransformer("BAAI/bge-m3")

# 테스트 문장들
sentences = [
    "RAG는 검색 기반 생성 기법이다",                          # 문장 0
    "Retrieval Augmented Generation은 문서 검색을 활용한다",  # 문장 1 (유사)
    "오늘 날씨가 매우 맑고 화창하다"                           # 문장 2 (무관)
]

# 임베딩 생성
# model.encode(): 리스트를 받아 numpy 배열 형태로 임베딩 반환
# 반환 형태: (문장 수, 차원 수) 크기의 2D 배열
embeddings = model.encode(sentences)

print(f"임베딩 배열 크기: {embeddings.shape}")
# → (3, 1024): 3개 문장, 각 1024차원

# 코사인 유사도 계산
from sentence_transformers.util import cos_sim
# cos_sim: sentence-transformers 내장 코사인 유사도 함수
# 결과는 -1 ~ 1 사이의 값

# 문장 0 vs 문장 1 (RAG 설명 두 가지)
sim_01 = cos_sim(embeddings[0], embeddings[1])

# 문장 0 vs 문장 2 (RAG vs 날씨)
sim_02 = cos_sim(embeddings[0], embeddings[2])

print(f"\n유사도 비교:")
print(f"'RAG' vs 'Retrieval Augmented Generation': {sim_01.item():.4f}")
print(f"'RAG' vs '오늘 날씨':                       {sim_02.item():.4f}")
```

**예상 출력:**
```
임베딩 배열 크기: (3, 1024)

유사도 비교:
'RAG' vs 'Retrieval Augmented Generation': 0.8923
'RAG' vs '오늘 날씨':                       0.1245
```

---

### 여러 문장을 한번에 검색하기

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("BAAI/bge-m3")

# 문서 데이터베이스 역할을 하는 텍스트들
corpus = [
    "Python은 간결한 문법으로 유명한 프로그래밍 언어입니다",
    "머신러닝은 데이터에서 패턴을 학습하는 AI 기술입니다",
    "서울은 대한민국의 수도이며 인구 약 1000만 명의 도시입니다",
    "벡터 데이터베이스는 임베딩 벡터를 효율적으로 저장·검색합니다",
    "RAG는 LLM에 외부 지식을 주입하는 기술입니다",
    "김치찌개는 한국의 전통 음식으로 발효된 김치로 만듭니다",
]

# 검색 쿼리
query = "AI 모델에게 외부 정보를 제공하는 방법"

# 임베딩 생성
# 문서들을 미리 임베딩해두기 (실제 시스템에선 한 번만 하고 저장)
corpus_embeddings = model.encode(corpus)

# 쿼리 임베딩
query_embedding = model.encode(query)

# 유사도 계산 및 정렬
similarities = cos_sim(query_embedding, corpus_embeddings)[0]
# cos_sim은 2D tensor를 반환하므로 [0]으로 1D로 만듦

# 인덱스를 유사도 내림차순으로 정렬
sorted_indices = similarities.argsort(descending=True)

# 결과 출력
print(f"검색 쿼리: '{query}'")
print("\n검색 결과 (유사도 순):")
print("-" * 60)

for rank, idx in enumerate(sorted_indices):
    score = similarities[idx].item()
    doc = corpus[idx]
    print(f"{rank+1}위 (유사도: {score:.4f}): {doc}")
```

**예상 출력:**
```
검색 쿼리: 'AI 모델에게 외부 정보를 제공하는 방법'

검색 결과 (유사도 순):
------------------------------------------------------------
1위 (유사도: 0.8234): RAG는 LLM에 외부 지식을 주입하는 기술입니다
2위 (유사도: 0.6521): 벡터 데이터베이스는 임베딩 벡터를 효율적으로 저장·검색합니다
3위 (유사도: 0.5234): 머신러닝은 데이터에서 패턴을 학습하는 AI 기술입니다
4위 (유사도: 0.3421): Python은 간결한 문법으로 유명한 프로그래밍 언어입니다
5위 (유사도: 0.2103): 서울은 대한민국의 수도이며 인구 약 1000만 명의 도시입니다
6위 (유사도: 0.1234): 김치찌개는 한국의 전통 음식으로 발효된 김치로 만듭니다
```

---

!!! tip "직접 해보기 2: 나만의 미니 검색 엔진"
    아래 코드를 완성해서 자신만의 미니 검색 엔진을 만들어보세요.

    ```python
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    model = SentenceTransformer("BAAI/bge-m3")

    # TODO: 자신이 관심 있는 주제의 문서 10개 이상 추가하기
    my_corpus = [
        # 여기에 문서 추가...
    ]

    # 문서 임베딩 (미리 계산)
    corpus_embeddings = model.encode(my_corpus, show_progress_bar=True)

    # 대화형 검색
    while True:
        query = input("\n검색어 입력 (종료: q): ").strip()
        if query.lower() == 'q':
            break

        query_emb = model.encode(query)
        scores = cos_sim(query_emb, corpus_embeddings)[0]

        # 상위 3개 결과 출력
        top_indices = scores.argsort(descending=True)[:3]
        print("\n--- 검색 결과 ---")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. [{scores[idx]:.3f}] {my_corpus[idx]}")
    ```

---

## 2.6 Cohere 임베딩: input_type의 중요성

Cohere 임베딩 모델의 독특한 특징은 `input_type`을 구분한다는 점입니다.

### 왜 input_type을 구분해야 하는가?

!!! warning "왜 이게 중요한가?"
    대부분의 임베딩 모델은 문서와 쿼리를 같은 방식으로 처리합니다. 하지만 생각해보면 이 둘은 성격이 다릅니다:

    - **문서(Document)**: "Python의 리스트는 순서가 있는 가변 컬렉션으로, append(), extend() 등의 메서드를 제공합니다."
    - **쿼리(Query)**: "파이썬 리스트에 요소 추가하는 방법"

    문서는 **완전한 정보를 담은 긴 텍스트**이고, 쿼리는 **원하는 정보를 묻는 짧은 질문**입니다.

    Cohere는 이 차이를 인식하고 각각에 최적화된 임베딩을 생성합니다.
    **input_type을 잘못 설정하면 검색 품질이 크게 떨어집니다!**

---

```python
# 설치: pip install cohere

import cohere
# cohere: Cohere API를 사용하기 위한 공식 Python 라이브러리

# 클라이언트 초기화
# COHERE_API_KEY 환경변수를 자동으로 읽습니다
co = cohere.ClientV2()

# 문서 임베딩 (인덱싱할 때 사용)
# input_type="search_document": "이것은 검색될 문서입니다"라고 모델에게 알림
doc_response = co.embed(
    texts=[
        "RAG 파이프라인의 구조와 동작 원리에 대한 설명",
        "벡터 데이터베이스 종류 비교: Pinecone, Weaviate, Chroma"
    ],
    model="embed-v4.0",           # 사용할 모델
    input_type="search_document", # 핵심! 문서 임베딩임을 명시
    embedding_types=["float"]     # 부동소수점 형태로 반환
)

# 쿼리 임베딩 (검색할 때 사용)
# input_type="search_query": "이것은 검색 쿼리입니다"라고 모델에게 알림
query_response = co.embed(
    texts=["RAG란 무엇인가요?"],
    model="embed-v4.0",
    input_type="search_query",    # 핵심! 쿼리 임베딩임을 명시
    embedding_types=["float"]
)

# 임베딩 추출
# .embeddings.float: float 타입 임베딩 리스트
doc_vectors = doc_response.embeddings.float
query_vector = query_response.embeddings.float[0]

print(f"문서 임베딩 수: {len(doc_vectors)}")          # 2
print(f"쿼리 임베딩 차원: {len(query_vector)}")        # 1024
print(f"문서1 임베딩 차원: {len(doc_vectors[0])}")     # 1024

# 다른 input_type 옵션들:
# "classification"  : 텍스트 분류 작업
# "clustering"      : 군집화 작업
# "image"           : 이미지 임베딩 (멀티모달)
```

**예상 출력:**
```
문서 임베딩 수: 2
쿼리 임베딩 차원: 1024
문서1 임베딩 차원: 1024
```

---

!!! note "실생활 비유: 질문과 답의 다른 언어"
    도서관에서 책을 찾는 상황을 생각해보세요.

    - **사서(도서관의 문서)**: "이 책은 파이썬 프로그래밍 언어의 기초부터 고급까지 다루며, 특히 데이터 분석 라이브러리인 pandas와 numpy의 활용법을 중점적으로 설명합니다."

    - **방문객(검색 쿼리)**: "파이썬 배우는 책 있어요?"

    방문객의 짧은 질문과 사서의 상세한 설명은 다른 방식으로 표현됩니다. Cohere는 이 차이를 학습해서 더 정확한 매칭을 합니다.

---

## 2.7 최신 임베딩 모델 비교 (2025)

### 상용 모델

| 모델 | 제공사 | 차원 | 최대 토큰 | 특징 | 추천 용도 |
|------|--------|------|-----------|------|-----------|
| **text-embedding-3-large** | OpenAI | 3072 | 8,191 | 차원 축소 가능, 범용적 | 일반 RAG, 범용 검색 |
| **text-embedding-3-small** | OpenAI | 1536 | 8,191 | 저비용, 빠른 속도 | 예산 제한, 대용량 처리 |
| **embed-v4.0** | Cohere | 1024 | 128,000 | 초장문 지원, input_type | 긴 문서 처리, 기업용 |
| **voyage-3-large** | Voyage AI | 1024 | 32,000 | 코드 임베딩 강점 | 코드 검색, 기술 문서 |
| **Gemini Embedding** | Google | 3072 | 8,192 | Task type 지정 가능 | Google 생태계 통합 |

### 오픈소스 모델

| 모델 | 차원 | 최대 토큰 | 특징 | 추천 용도 |
|------|------|-----------|------|-----------|
| **GTE-Qwen2-7B-instruct** | 3584 | 131,072 | 초장문 지원, 최상위 성능 | 긴 문서 RAG, 로컬 실행 |
| **NV-Embed-v2** | 4096 | 32,768 | NVIDIA 최적화 | GPU 보유 시, 고성능 |
| **multilingual-e5-large-instruct** | 1024 | 514 | 다국어 강점 | 다국어 서비스 |
| **bge-m3** | 1024 | 8,192 | 한국어 포함 다국어 | 한국어 RAG |
| **nomic-embed-text-v2-moe** | 768 | 8,192 | 경량, 효율적 | 리소스 제한 환경 |

---

### 모델 선택 결정 트리

```
나의 상황에 맞는 임베딩 모델 선택하기
============================================================

비용을 지불할 수 있나요?
│
├─── 아니오 (오픈소스 선택) ────────────────────────────────
│
│    한국어/다국어가 필요한가?
│    ├─ 예 → bge-m3 (균형잡힌 성능과 언어 지원)
│    │        multilingual-e5-large-instruct
│    │
│    └─ 아니오
│         ├─ 긴 문서(10만 토큰+) → GTE-Qwen2-7B-instruct
│         ├─ GPU 보유 + 최고 성능 → NV-Embed-v2
│         └─ 가볍고 빠르게 → nomic-embed-text-v2-moe
│
└─── 예 (상용 API 선택) ─────────────────────────────────────
     │
     ├─ 일반 텍스트 RAG → OpenAI text-embedding-3-large
     │
     ├─ 예산 제한 있음 → OpenAI text-embedding-3-small
     │                   (3-large의 70% 성능, 20% 비용)
     │
     ├─ 매우 긴 문서(128K 토큰) → Cohere embed-v4.0
     │
     ├─ 코드/기술 문서 검색 → Voyage voyage-3-large
     │
     └─ Google Cloud 사용 중 → Gemini Embedding
```

---

## 2.8 MTEB 벤치마크 완전 이해

### MTEB란 무엇인가?

MTEB(Massive Text Embedding Benchmark)는 임베딩 모델의 성능을 표준화된 방법으로 측정하는 벤치마크입니다. 2022년 Hugging Face 연구팀이 발표했으며, 현재 임베딩 모델 평가의 사실상 표준이 되었습니다.

!!! note "실생활 비유: 대학 입시 시험"
    MTEB는 임베딩 모델의 "수능 시험"과 같습니다.

    - 수능에 국어, 수학, 영어 등 다양한 과목이 있듯이
    - MTEB에는 Retrieval, STS, Classification 등 다양한 태스크가 있습니다.

    수능 총점이 높아도 특정 과목이 약할 수 있듯이,
    MTEB 평균이 높아도 특정 태스크에서는 약한 모델이 있을 수 있습니다.

    **RAG를 만든다면 "Retrieval 점수"가 가장 중요합니다!**

---

### MTEB 태스크 상세 설명

| 태스크 | 영문명 | 의미 | RAG 관련성 | 측정 방법 |
|--------|--------|------|-----------|-----------|
| **검색** | Retrieval | 쿼리에 맞는 문서 찾기 | 매우 높음 | nDCG@10 |
| **문장 유사도** | STS | 두 문장이 얼마나 비슷한가 | 높음 | Spearman 상관계수 |
| **분류** | Classification | 텍스트를 카테고리로 분류 | 보통 | 정확도 |
| **군집화** | Clustering | 비슷한 문서끼리 그룹화 | 보통 | V-measure |
| **재순위** | Reranking | 검색 결과를 재정렬 | 매우 높음 | MAP |
| **이중언어 검색** | Bitext Mining | 번역 문장 쌍 찾기 | 낮음 | F1 점수 |
| **요약 평가** | Summarization | 요약의 품질 측정 | 보통 | Spearman 상관계수 |

---

### 주요 평가 지표 설명

#### nDCG@10 (Retrieval 지표)

!!! note "용어 설명: nDCG@10"
    **nDCG** = Normalized Discounted Cumulative Gain

    쉽게 말하면: "상위 10개 검색 결과가 얼마나 좋은가?"

    - 상위에 좋은 결과가 있을수록 높은 점수
    - 7위에 좋은 결과 있는 것보다 1위에 있는 게 더 높은 점수
    - 0 ~ 1 사이 값, 1에 가까울수록 좋음

    **예시:**
    ```
    검색 결과:    관련성:    점수:
    1위           O 관련    높음
    2위           O 관련    높음 (1위보다 약간 낮음)
    3위           X 무관    0
    4위           O 관련    (3위보다 낮음 - 늦게 나왔으므로)
    ...
    10위          X 무관    0
    ```

#### MAP (Reranking 지표)

!!! note "용어 설명: MAP (Mean Average Precision)"
    관련 문서들이 전체 결과 목록에서 얼마나 **앞쪽에** 있는지 측정합니다.

    - 모든 관련 문서가 1위~N위에 있으면 MAP = 1.0
    - 관련 문서가 뒤쪽에 섞여 있을수록 MAP 낮아짐

#### Spearman 상관계수 (STS 지표)

!!! note "용어 설명: Spearman 상관계수"
    모델이 매긴 유사도 순서와 사람이 매긴 유사도 순서가 얼마나 일치하는가.

    - 1.0: 완벽하게 사람과 같은 판단
    - 0.0: 전혀 관련 없는 판단
    - -1.0: 반대로 판단 (매우 나쁨)

---

### MTEB 리더보드 읽는 방법

```
MTEB 리더보드 예시 (허구의 데이터):

모델명                | Avg  | Retr | STS  | Class | Clust
─────────────────────────────────────────────────────────
SuperEmbed-XL        | 72.3 | 78.5 | 85.2 | 76.1  | 50.3
GoodEmbed-Large      | 71.1 | 75.2 | 87.8 | 74.3  | 48.9
FastEmbed-Small      | 65.4 | 68.1 | 79.4 | 70.2  | 44.1

     ↑전체 평균  ↑검색  ↑문장유사도  ↑분류  ↑군집화
```

**RAG를 만들고 싶다면?**

- `Avg`(평균)보다 `Retr`(Retrieval) 점수가 높은 모델 선택!
- FastEmbed-Small의 `Retr`이 68.1이고 SuperEmbed-XL이 78.5면, FastEmbed가 저렴해도 RAG엔 SuperEmbed를 써야 합니다.

!!! tip "모델 선택 팁"
    1. [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)에서 **Retrieval** 탭 클릭
    2. 한국어가 필요하면 **Multilingual** 필터 적용
    3. 모델 크기(MB) 확인 — 너무 큰 모델은 로컬 실행 어려움
    4. 라이선스 확인 — 상업적 사용 가능 여부

---

## 2.9 임베딩 성능을 직접 측정하기

벤치마크 점수를 믿는 것도 중요하지만, **내 데이터로 직접 테스트**하는 것이 더 중요합니다.

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np

def check_retrieval(model_name: str, test_cases: list) -> dict:
    """
    임베딩 모델의 검색 성능을 직접 측정하는 함수

    Parameters:
    - model_name: 테스트할 모델 이름
    - test_cases: [(쿼리, 정답_문서_인덱스, 문서_리스트), ...] 형식의 테스트 데이터

    Returns:
    - {'accuracy': float, 'avg_rank': float} 형식의 결과
    """
    print(f"\n모델 로드 중: {model_name}")
    model = SentenceTransformer(model_name)

    correct = 0      # 1위에서 정답을 찾은 횟수
    total_rank = 0   # 정답의 평균 순위

    for query, correct_idx, corpus in test_cases:
        # 모든 문서와 쿼리 임베딩
        corpus_embs = model.encode(corpus)
        query_emb = model.encode(query)

        # 유사도 계산 및 순위 매기기
        scores = cos_sim(query_emb, corpus_embs)[0]

        # 정답 문서의 순위 (1이 가장 좋음)
        ranking = scores.argsort(descending=True).tolist()
        rank = ranking.index(correct_idx) + 1  # 0-indexed → 1-indexed

        total_rank += rank
        if rank == 1:
            correct += 1

    n = len(test_cases)
    accuracy = correct / n        # 1위 정답률
    avg_rank = total_rank / n     # 평균 순위

    print(f"  정답 1위 비율: {accuracy:.1%}")
    print(f"  평균 순위: {avg_rank:.1f}")

    return {'accuracy': accuracy, 'avg_rank': avg_rank}


# 테스트 데이터 준비
corpus = [
    "Python의 리스트 자료구조 사용법",                      # 인덱스 0
    "JavaScript 배열 메서드 정리",                          # 인덱스 1
    "RAG 시스템 구축 방법",                                 # 인덱스 2
    "벡터 데이터베이스 Chroma 사용 가이드",                 # 인덱스 3
    "대한민국 역사와 문화",                                 # 인덱스 4
]

test_cases = [
    ("파이썬 리스트 어떻게 쓰나요?", 0, corpus),             # 정답: 0번 문서
    ("RAG 파이프라인 만들기", 2, corpus),                    # 정답: 2번 문서
    ("Chroma DB 설치하고 사용하기", 3, corpus),              # 정답: 3번 문서
]

# 모델 비교
models_to_compare = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large-instruct",
]

results = {}
for model_name in models_to_compare:
    results[model_name] = check_retrieval(model_name, test_cases)

# 결과 비교 출력
print("\n" + "=" * 50)
print("모델 성능 비교 결과")
print("=" * 50)
for model, result in results.items():
    short_name = model.split("/")[-1]
    print(f"{short_name:<40} 정답률: {result['accuracy']:.1%}, 평균순위: {result['avg_rank']:.1f}")
```

---

!!! tip "직접 해보기 3: 내 도메인 데이터로 측정하기"
    위 코드를 수정해서 **자신의 프로젝트 데이터**로 모델을 비교해보세요.

    1. `corpus`에 자신의 실제 문서 10~20개 추가
    2. `test_cases`에 사용자가 실제로 할 것 같은 질문 5~10개 작성
    3. 각 질문의 정답 문서 인덱스 지정
    4. 여러 모델 비교 실행

    이렇게 하면 **MTEB 점수와 상관없이** 자신의 데이터에서 실제로 잘 동작하는 모델을 찾을 수 있습니다!

---

## 2.10 비용과 성능의 트레이드오프

실제 프로덕션 시스템을 만들 때는 성능만이 아닌 비용도 고려해야 합니다.

### 비용 계산 예시

```python
# 임베딩 비용 계산기 (2025년 기준 OpenAI 가격)

def calculate_embedding_cost(
    num_documents: int,
    avg_tokens_per_doc: int,
    model: str = "text-embedding-3-small"
) -> dict:
    """
    임베딩 비용 계산 함수

    가격 (2025년 기준):
    - text-embedding-3-small: $0.020 per 1M tokens
    - text-embedding-3-large: $0.130 per 1M tokens
    """
    pricing = {
        "text-embedding-3-small": 0.020,   # $ per 1M tokens
        "text-embedding-3-large": 0.130,   # $ per 1M tokens
    }

    # 총 토큰 수 계산
    total_tokens = num_documents * avg_tokens_per_doc

    # 비용 계산 ($ per million tokens)
    price_per_million = pricing.get(model, 0.020)
    cost_usd = (total_tokens / 1_000_000) * price_per_million
    cost_krw = cost_usd * 1350  # 환율 가정

    return {
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "cost_krw": cost_krw,
    }


# 시나리오별 비용 계산
scenarios = [
    {
        "label": "소규모 스타트업 RAG (문서 1만개)",
        "num_documents": 10_000,
        "avg_tokens": 500
    },
    {
        "label": "중규모 기업 지식베이스 (문서 100만개)",
        "num_documents": 1_000_000,
        "avg_tokens": 500
    },
    {
        "label": "대규모 뉴스 아카이브 (문서 1000만개)",
        "num_documents": 10_000_000,
        "avg_tokens": 800
    },
]

print("임베딩 비용 시나리오 분석")
print("=" * 70)

for scenario in scenarios:
    print(f"\n[{scenario['label']}]")

    for model in ["text-embedding-3-small", "text-embedding-3-large"]:
        result = calculate_embedding_cost(
            scenario["num_documents"],
            scenario["avg_tokens"],
            model
        )

        short_model = model.replace("text-embedding-", "")
        print(
            f"  {short_model:<15} "
            f"토큰: {result['total_tokens']:>15,}  "
            f"비용: ${result['cost_usd']:>8.2f}  "
            f"(약 {result['cost_krw']:>10,.0f}원)"
        )
```

**예상 출력:**
```
임베딩 비용 시나리오 분석
======================================================================

[소규모 스타트업 RAG (문서 1만개)]
  3-small         토큰:       5,000,000  비용: $    0.10  (약      135원)
  3-large         토큰:       5,000,000  비용: $    0.65  (약      878원)

[중규모 기업 지식베이스 (문서 100만개)]
  3-small         토큰:     500,000,000  비용: $   10.00  (약   13,500원)
  3-large         토큰:     500,000,000  비용: $   65.00  (약   87,750원)

[대규모 뉴스 아카이브 (문서 1000만개)]
  3-small         토큰:   8,000,000,000  비용: $  160.00  (약  216,000원)
  3-large         토큰:   8,000,000,000  비용: $ 1,040.00  (약 1,404,000원)
```

!!! info "결론"
    - 소규모 프로젝트는 비용 걱정 없이 3-large 사용 가능
    - 대규모 프로덕션에서는 비용이 수백만원 단위가 될 수 있음
    - 이 경우 **오픈소스 + 자체 서버** 또는 3-small이 현명한 선택

---

## 2.11 실전: 완전한 임베딩 파이프라인

지금까지 배운 내용을 종합해서 실전에서 사용할 수 있는 임베딩 파이프라인을 만들어봅니다.

```python
"""
실전 임베딩 파이프라인
- 문서를 로드하고 임베딩하여 저장 및 검색하는 전체 흐름
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from typing import List, Dict

class SimpleVectorStore:
    """
    간단한 벡터 저장소 구현
    실제 프로덕션에서는 Chroma, Pinecone, Weaviate 같은
    전문 벡터 DB를 사용하지만, 학습용으로 직접 구현해봅니다.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        초기화 메서드
        model_name: 사용할 임베딩 모델 이름
        """
        print(f"모델 로딩: {model_name}")
        self.model = SentenceTransformer(model_name)

        # 문서와 벡터를 저장하는 리스트
        self.documents: List[str] = []           # 원본 텍스트
        self.metadata: List[Dict] = []           # 추가 정보 (출처, 날짜 등)
        self.embeddings: np.ndarray = None       # 임베딩 벡터 배열

        print("준비 완료!")

    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """
        문서들을 벡터 저장소에 추가하는 메서드

        documents: 추가할 텍스트 리스트
        metadata: 각 문서의 부가 정보 (선택적)
        """
        if metadata is None:
            metadata = [{} for _ in documents]

        print(f"{len(documents)}개 문서 임베딩 생성 중...")

        # 문서들을 임베딩으로 변환
        # show_progress_bar=True: 진행 상황 막대 표시
        new_embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            batch_size=32  # 한 번에 처리할 문서 수
        )

        # 기존 데이터에 추가
        self.documents.extend(documents)
        self.metadata.extend(metadata)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            # 기존 임베딩에 새 임베딩 추가 (행 방향으로 합치기)
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        print(f"총 {len(self.documents)}개 문서 저장됨")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        쿼리와 가장 유사한 문서를 찾는 메서드

        query: 검색할 텍스트
        top_k: 반환할 최대 문서 수
        """
        if len(self.documents) == 0:
            return []

        # 쿼리 임베딩 생성
        query_embedding = self.model.encode(query)

        # 모든 문서와의 코사인 유사도 계산
        similarities = cos_sim(query_embedding, self.embeddings)[0]

        # 유사도 내림차순으로 인덱스 정렬
        top_indices = similarities.argsort(descending=True)[:top_k]

        # 결과 구성
        results = []
        for idx in top_indices:
            idx = idx.item()  # 텐서 → 파이썬 정수 변환
            results.append({
                "text": self.documents[idx],
                "score": similarities[idx].item(),
                "metadata": self.metadata[idx],
                "index": idx
            })

        return results

    def save(self, path: str):
        """벡터 저장소를 파일로 저장"""
        data = {
            "documents": self.documents,
            "metadata": self.metadata,
            "embeddings": self.embeddings.tolist()  # numpy → 리스트
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"저장 완료: {path}")

    def load(self, path: str):
        """파일에서 벡터 저장소 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.documents = data["documents"]
        self.metadata = data["metadata"]
        self.embeddings = np.array(data["embeddings"])  # 리스트 → numpy
        print(f"로드 완료: {len(self.documents)}개 문서")


# 사용 예시
if __name__ == "__main__":
    # 1. 벡터 저장소 생성
    store = SimpleVectorStore("BAAI/bge-m3")

    # 2. 문서 추가
    documents = [
        "Python은 1991년 Guido van Rossum이 만든 고급 프로그래밍 언어입니다.",
        "머신러닝은 데이터에서 패턴을 학습하고 예측하는 AI 기술입니다.",
        "RAG(검색 증강 생성)는 LLM에 외부 지식을 실시간으로 제공하는 기술입니다.",
        "벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 검색합니다.",
        "트랜스포머(Transformer)는 2017년 구글이 발표한 딥러닝 아키텍처입니다.",
        "임베딩은 텍스트를 의미를 보존하는 고차원 수치 벡터로 변환하는 기술입니다.",
    ]

    metadata = [
        {"출처": "Python 공식문서", "날짜": "2024-01"},
        {"출처": "ML 교과서", "날짜": "2024-02"},
        {"출처": "RAG 논문", "날짜": "2023-12"},
        {"출처": "DB 비교 블로그", "날짜": "2024-03"},
        {"출처": "Attention Is All You Need", "날짜": "2017-06"},
        {"출처": "임베딩 튜토리얼", "날짜": "2024-04"},
    ]

    store.add_documents(documents, metadata)

    # 3. 검색 테스트
    print("\n" + "=" * 60)
    print("검색 테스트")
    print("=" * 60)

    queries = [
        "AI에게 외부 정보를 주는 방법",
        "딥러닝 모델 구조",
        "텍스트를 숫자로 변환하기",
    ]

    for query in queries:
        print(f"\n쿼리: '{query}'")
        results = store.search(query, top_k=2)

        for i, result in enumerate(results, 1):
            print(f"  {i}위 (점수: {result['score']:.4f}): {result['text'][:50]}...")
            if result['metadata']:
                print(f"      출처: {result['metadata'].get('출처', '없음')}")

    # 4. 저장 및 재로드
    store.save("/tmp/my_vector_store.json")

    new_store = SimpleVectorStore("BAAI/bge-m3")
    new_store.load("/tmp/my_vector_store.json")

    print(f"\n저장소 재로드 후 문서 수: {len(new_store.documents)}")
```

---

## 2.12 자주 묻는 질문 (FAQ)

??? question "Q1. 임베딩 벡터의 각 숫자가 무엇을 의미하나요?"
    각 숫자 자체는 사람이 해석할 수 있는 명확한 의미가 없습니다.

    예를 들어 "감정" 축, "장소" 축, "시간" 축처럼 이름을 붙일 수 없습니다.
    임베딩 모델은 학습 과정에서 자동으로 의미 있는 방향을 찾아내며,
    그 방향들이 1536개(또는 그 이상)의 숫자로 표현됩니다.

    중요한 것은 개별 숫자가 아니라 **전체 벡터의 방향(패턴)**입니다.

??? question "Q2. 차원이 높을수록 무조건 좋은가요?"
    아닙니다. 높은 차원은 더 많은 정보를 담을 수 있지만, 단점도 있습니다:

    - **저장 비용**: 1536차원 대신 3072차원이면 파일 크기가 2배
    - **검색 속도**: 차원이 높으면 유사도 계산이 느려짐
    - **차원의 저주**: 극도로 높은 차원에서는 오히려 거리가 무의미해질 수 있음

    OpenAI text-embedding-3-large는 3072차원이지만, 1024차원으로 줄여도 성능이 크게 떨어지지 않습니다.
    **용도에 맞는 적절한 차원**을 선택하는 것이 중요합니다.

??? question "Q3. 임베딩 모델을 직접 훈련할 수 있나요?"
    가능하지만 매우 어렵고 비쌉니다.

    일반적인 선택지:

    1. **그대로 사용**: 기존 모델을 그대로 활용 (99% 사례)
    2. **파인튜닝(Fine-tuning)**: 기존 모델을 내 데이터로 추가 학습
       - `sentence-transformers` 라이브러리로 가능
       - 특수 도메인(의학, 법률, 특정 언어)에 유용
    3. **처음부터 훈련**: 수십억 개의 텍스트 쌍과 수백 개의 GPU 필요
       - 일반적으로 불필요하고 비현실적

??? question "Q4. 한국어는 어떤 모델이 가장 좋나요?"
    2025년 기준 추천:

    - **오픈소스**: `BAAI/bge-m3` — 한국어 포함 100개 언어, 균형잡힌 성능
    - **오픈소스 대안**: `intfloat/multilingual-e5-large-instruct`
    - **상용**: OpenAI `text-embedding-3-large` — 다국어 지원이 뛰어남
    - **한국어 특화**: HuggingFace에서 `ko-sroberta` 등 한국어 전용 모델 검색

    한국어만 처리한다면 한국어 전용 모델이 다국어 모델보다 좋을 수 있습니다.
    하지만 영어 문서도 함께 처리해야 한다면 다국어 모델을 선택하세요.

??? question "Q5. embed_query와 embed_documents의 차이는?"
    LangChain에서:

    - `embed_query(text)`: **단일 텍스트** 입력, **리스트** 반환 (벡터 하나)
    - `embed_documents(texts)`: **리스트** 입력, **리스트의 리스트** 반환 (벡터 여러 개)

    일부 모델(Cohere 등)은 쿼리와 문서에 다른 최적화를 적용합니다.
    그래서 쿼리를 임베딩할 때는 `embed_query`, 문서를 임베딩할 때는 `embed_documents`를 사용해야 합니다.

??? question "Q6. 임베딩이 실시간으로 바뀌나요?"
    아닙니다. 같은 텍스트를 같은 모델로 임베딩하면 **항상 같은 벡터**가 나옵니다.

    이 덕분에 문서를 한 번만 임베딩해서 데이터베이스에 저장해두면, 이후 검색할 때마다 다시 임베딩하지 않아도 됩니다. 이것이 임베딩 기반 검색이 효율적인 이유 중 하나입니다.

    단, 모델이 업데이트되면 같은 텍스트라도 다른 벡터가 나올 수 있으므로, 모델을 바꿀 때는 전체 문서를 다시 임베딩해야 합니다.

??? question "Q7. 최대 토큰을 초과하는 긴 문서는 어떻게 처리하나요?"
    두 가지 방법이 있습니다:

    **방법 1: 청킹(Chunking)** — 문서를 작은 조각으로 나누기

    ```python
    def chunk_text(text, chunk_size=500, overlap=50):
        """텍스트를 chunk_size 단어씩, overlap 단어 겹침으로 분할"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    ```

    **방법 2: 긴 컨텍스트 모델 사용** — Cohere embed-v4.0(128K 토큰), GTE-Qwen2-7B(131K 토큰)

    대부분의 RAG 시스템은 방법 1(청킹)을 사용합니다. 다음 챕터에서 자세히 다룹니다!

??? question "Q8. 임베딩 모델을 로컬에서 실행하려면 어느 정도 사양이 필요한가요?"
    모델 크기별 대략적 요구사항:

    | 모델 크기 | RAM | GPU VRAM | 예시 모델 |
    |---------|-----|----------|---------|
    | 소형 (~80MB) | 2GB | 불필요 | all-MiniLM-L6-v2 |
    | 중형 (~300MB) | 4GB | 불필요 | bge-m3 |
    | 대형 (~1.5GB) | 8GB | 권장 4GB+ | multilingual-e5-large |
    | 초대형 (~14GB) | 32GB | 24GB+ | GTE-Qwen2-7B |

    CPU만으로도 실행 가능하지만 속도가 느립니다. 대용량 처리에는 GPU를 권장합니다.

---

## 2.13 핵심 요약

!!! abstract "이것만 기억하자"
    **임베딩의 본질**

    1. 임베딩 = 텍스트의 **의미**를 보존하는 숫자 벡터로 변환
    2. 의미가 비슷한 텍스트 = 비슷한 벡터 = 벡터 공간에서 가까운 위치
    3. 코사인 유사도로 두 벡터(텍스트)가 얼마나 비슷한지 측정

    **모델 선택 기준**

    4. **상용**: OpenAI(범용/가성비), Cohere(장문 128K), Voyage(코드)
    5. **오픈소스**: bge-m3(한국어 다국어), GTE-Qwen2(장문), nomic(경량)
    6. RAG에선 **MTEB Retrieval 점수**가 핵심 지표

    **실전 주의사항**

    7. Cohere처럼 `input_type`을 구분하는 모델은 반드시 올바르게 설정
    8. 문서는 `embed_documents`, 쿼리는 `embed_query` 사용
    9. 모델 바꿀 때는 전체 문서를 다시 임베딩해야 함
    10. MTEB보다 **자신의 데이터로 직접 테스트**하는 것이 더 신뢰할 수 있음

---

!!! tip "다음 챕터 예고"
    임베딩을 만들었으면 이제 어디에 **저장**하고 어떻게 **빠르게 검색**할까요?

    다음 챕터에서는 벡터 데이터베이스(Vector Database)를 다룹니다.
    수백만 개의 임베딩 중에서 가장 유사한 것을 밀리초 안에 찾는 방법을 알아봅니다!
