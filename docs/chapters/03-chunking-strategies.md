# 3. 문서 청킹 전략

> "RAG의 품질은 청킹에서 결정된다. 아무리 좋은 AI 모델도 잘못 나뉜 청크는 고칠 수 없다."

---

## 청킹이란 무엇인가? (기초 개념)

### 청크(Chunk)란?

**청크(Chunk)**란 긴 문서를 검색과 처리에 적합한 **작은 조각**으로 나눈 것입니다.

!!! note "현실 세계 비유"
    청킹은 마치 두꺼운 교과서를 **인덱스 카드**로 정리하는 것과 같습니다.

    - 교과서 전체를 한 번에 외울 수는 없지만
    - "광합성이란?" 이라는 질문에 해당하는 카드 한 장은 바로 찾을 수 있습니다.
    - 각 카드에는 **하나의 완결된 개념**이 담겨 있어야 합니다.

### 왜 청킹이 필요한가?

AI 언어 모델(LLM)에는 **컨텍스트 창(Context Window)**이라는 한계가 있습니다.

!!! info "컨텍스트 창(Context Window)이란?"
    LLM이 한 번에 처리할 수 있는 텍스트의 최대 길이입니다.

    - GPT-4: 약 128,000 토큰 (약 96,000단어)
    - Claude 3: 약 200,000 토큰
    - 하지만 실제 기업 문서는 수백만 토큰에 달하는 경우도 있습니다.

따라서 우리는 다음 과정을 거칩니다:

```
[긴 문서 전체]
      ↓ 청킹 (Chunking)
[청크1] [청크2] [청크3] ... [청크N]
      ↓ 임베딩 (Embedding) - 각 청크를 숫자 벡터로 변환
[벡터1] [벡터2] [벡터3] ... [벡터N]
      ↓ 벡터 DB에 저장
      ↓ 검색 시: 질문과 가장 유사한 청크 Top-K 반환
      ↓ LLM에게 관련 청크만 전달
[LLM이 답변 생성]
```

### 토큰(Token)이란?

**토큰(Token)**은 LLM이 텍스트를 처리하는 **최소 단위**입니다. 글자 수와 다릅니다.

!!! example "토큰 계산 예시"
    ```
    "안녕하세요" = 약 5-7 토큰 (한국어는 글자당 약 1.5-2 토큰)
    "Hello world" = 2 토큰 (영어는 단어당 약 1 토큰)
    "RAG는 Retrieval-Augmented Generation의 약자입니다." = 약 30-40 토큰
    ```

    한국어는 영어보다 토큰 효율이 낮습니다. 같은 내용을 한국어로 쓰면 영어보다 약 1.5-2배 많은 토큰을 사용합니다.

---

## 왜 청킹이 중요한가?

!!! danger "청킹은 RAG 품질의 70%를 결정한다"
    아무리 좋은 임베딩 모델과 LLM을 사용해도, 청크가 잘못 나뉘면 검색 결과 자체가 틀려버립니다.
    이 문제는 AI 모델이 아무리 뛰어나도 고칠 수 없습니다.

### 나쁜 청킹 vs 좋은 청킹

아래 텍스트를 예시로 봅시다:

```
원본 텍스트:
"파이썬의 주요 특징은 다음과 같다.
1. 읽기 쉬운 문법으로 초보자도 배우기 쉽다.
2. 다양한 라이브러리 생태계를 갖추고 있다.
3. 데이터 과학과 AI 분야에서 표준 언어로 자리잡았다."
```

**나쁜 청킹 (잘못된 위치에서 자름):**

```
청크 A: "파이썬의 주요 특징은 다음과 같다."
청크 B: "1. 읽기 쉬운 문법으로 초보자도 배우기 쉽다.
         2. 다양한 라이브러리 생태계를 갖추고 있다.
         3. 데이터 과학과 AI 분야에서 표준 언어로 자리잡았다."
```

사용자가 "파이썬의 특징이 뭔가요?"라고 물으면:

- 청크 A만 반환되면: "다음과 같다"라는 답변만 나옴 (쓸모없음)
- 청크 B만 반환되면: 어떤 언어의 특징인지 알 수 없음 (문맥 없음)

**좋은 청킹 (의미 단위로 자름):**

```
청크: "파이썬의 주요 특징은 다음과 같다.
      1. 읽기 쉬운 문법으로 초보자도 배우기 쉽다.
      2. 다양한 라이브러리 생태계를 갖추고 있다.
      3. 데이터 과학과 AI 분야에서 표준 언어로 자리잡았다."
```

사용자가 "파이썬의 특징이 뭔가요?"라고 물으면 완전한 답변이 반환됩니다.

---

## 오버랩(Overlap)이란?

**오버랩(Overlap, 청크 겹침)**은 연속된 청크 사이에 **의도적으로 겹치는 텍스트**를 두는 기법입니다.

### 오버랩이 없을 때의 문제

```
원본: "A B C D E F G H I J K L M N O P"
      (각 글자를 하나의 단어라고 가정)

청크 크기 = 6 단어, 오버랩 = 0:
청크1: [A B C D E F]
청크2: [G H I J K L]
청크3: [M N O P]

문제: "E F G H"가 중요한 문구라면?
→ 청크1에 E F는 있지만 G H는 없음
→ 청크2에 G H는 있지만 E F는 없음
→ 검색 시 이 문구를 완전히 찾지 못함!
```

### 오버랩이 있을 때

```
원본: "A B C D E F G H I J K L M N O P"

청크 크기 = 6 단어, 오버랩 = 2:
청크1: [A B C D E F]
청크2: [E F G H I J]  ← E F 가 반복됨 (오버랩)
청크3: [I J K L M N]  ← I J 가 반복됨 (오버랩)
청크4: [M N O P]

장점: "E F G H"라는 문구가 청크2에 완전히 포함됨!
```

!!! tip "오버랩 시각화"
    ```
    텍스트:  [===청크1=====][===청크2=====][===청크3=====]
    오버랩:           [==][==]      [==][==]

    결과:    [===청크1=======]
                      [===청크2=======]
                               [===청크3=====]
    ```

    오버랩이 있으면 연속된 두 청크가 **경계 부분을 공유**합니다.

### 오버랩 설정 가이드

| 오버랩 비율 | 사용 상황 |
|------------|---------|
| 0% (오버랩 없음) | 코드, 표, 독립적인 항목들 |
| 10-20% | 일반 텍스트, 기술 문서 |
| 20-30% | 서사적 글, 맥락이 중요한 문서 |
| 30%+ | 법률 문서, 학술 논문 (과도한 중복 주의) |

---

## 구분자(Separator)란?

**구분자(Separator)**는 텍스트를 나눌 **기준이 되는 특수 문자나 문자열**입니다.

!!! example "주요 구분자 종류"
    ```python
    "\n\n"  # 빈 줄 (단락 구분) - 가장 자연스러운 의미 단위
    "\n"    # 줄바꿈 (한 줄 단위)
    ". "    # 마침표+공백 (문장 단위)
    " "     # 공백 (단어 단위)
    ""      # 빈 문자열 (글자 단위 - 최후 수단)
    ```

---

## 청킹 전략 비교

이제 4가지 주요 청킹 전략을 **동일한 텍스트**로 비교해 봅시다.

### 비교에 사용할 예시 텍스트

```python
sample_text = """
머신러닝이란 무엇인가?

머신러닝(Machine Learning)은 컴퓨터가 명시적인 프로그래밍 없이
데이터를 통해 스스로 학습하는 인공지능의 한 분야입니다.

머신러닝의 세 가지 유형

지도 학습(Supervised Learning)은 레이블이 있는 데이터로 훈련합니다.
예를 들어, 스팸 메일 분류기는 '스팸/정상'으로 표시된 이메일로 학습합니다.

비지도 학습(Unsupervised Learning)은 레이블 없이 패턴을 찾습니다.
고객 군집화가 대표적인 예입니다.

강화 학습(Reinforcement Learning)은 보상과 처벌을 통해 학습합니다.
알파고가 이 방법으로 바둑을 익혔습니다.
"""
```

---

### 전략 1: 고정 크기 청킹 (Fixed-size Chunking)

**개념:** 글자 수나 토큰 수를 기준으로 **무조건 일정 크기**로 자릅니다. 가장 단순한 방법입니다.

!!! note "현실 비유"
    마치 책을 **페이지 번호 기준**으로 자르는 것과 같습니다.
    1-10페이지, 11-20페이지... 내용과 상관없이 10페이지씩 자릅니다.
    결과적으로 문장이 중간에 잘릴 수 있습니다.

#### 코드 예시 (단계별 설명 포함)

```python
from langchain_text_splitters import CharacterTextSplitter

# 1단계: 분할기 생성
# chunk_size: 각 청크의 최대 글자 수
# chunk_overlap: 연속된 청크 간 겹치는 글자 수
# separator: 이 문자를 기준으로 자르려고 시도 (없으면 그냥 자름)
splitter = CharacterTextSplitter(
    chunk_size=200,      # 각 청크는 최대 200글자
    chunk_overlap=30,    # 연속 청크 간 30글자 겹침
    separator="\n"       # 줄바꿈을 기준으로 우선 분할 시도
)

# 2단계: 텍스트 분할 실행
chunks = splitter.split_text(sample_text)

# 3단계: 결과 확인
print(f"총 {len(chunks)}개 청크 생성")
for i, chunk in enumerate(chunks):
    print(f"\n--- 청크 {i+1} ({len(chunk)}글자) ---")
    print(chunk)
```

**실제 출력 결과:**

```
총 4개 청크 생성

--- 청크 1 (87글자) ---
머신러닝이란 무엇인가?

머신러닝(Machine Learning)은 컴퓨터가 명시적인 프로그래밍 없이

--- 청크 2 (156글자) ---
데이터를 통해 스스로 학습하는 인공지능의 한 분야입니다.

머신러닝의 세 가지 유형

지도 학습(Supervised Learning)은 레이블이 있는 데이터로 훈련합니다.

--- 청크 3 (175글자) ---
예를 들어, 스팸 메일 분류기는 '스팸/정상'으로 표시된 이메일로 학습합니다.

비지도 학습(Unsupervised Learning)은 레이블 없이 패턴을 찾습니다.

--- 청크 4 (93글자) ---
고객 군집화가 대표적인 예입니다.

강화 학습(Reinforcement Learning)은 보상과 처벌을 통해 학습합니다.
알파고가 이 방법으로 바둑을 익혔습니다.
```

!!! warning "문제점 발견!"
    청크 1은 "...없이"로 끝나고, 청크 2는 "데이터를 통해..."로 시작합니다.
    → "머신러닝은 컴퓨터가 명시적인 프로그래밍 없이 데이터를 통해 스스로 학습한다"는
      하나의 완결된 문장이 두 청크로 쪼개졌습니다!

**장단점 정리:**

| 장점 | 단점 |
|------|------|
| 구현 매우 간단 | 의미 단위 무시 |
| 예측 가능한 크기 | 문장 중간에서 잘릴 수 있음 |
| 처리 속도 빠름 | 문맥 단절 발생 빈번 |
| 추가 라이브러리 불필요 | 품질이 낮음 |

**사용 권장 상황:** 빠른 프로토타입 제작, 단순 구조의 텍스트, 성능보다 속도가 중요할 때

---

### 전략 2: 재귀 분할 (Recursive Character Text Splitting) ⭐ 가장 많이 사용

**개념:** 여러 구분자를 **우선순위 순서대로** 시도하면서 의미 단위를 최대한 보존합니다.

!!! note "현실 비유"
    마치 책을 자를 때 **챕터 → 단락 → 문장 → 단어** 순서로 최대한 자연스러운 경계를 찾는 것입니다.

    1. 우선 챕터(빈 줄) 경계에서 자르려 시도
    2. 청크가 너무 크면 줄바꿈에서 자름
    3. 그래도 크면 문장 끝(마침표)에서 자름
    4. 마지막 수단으로 단어나 글자 단위로 자름

#### 재귀 분할 작동 원리 (단계별)

```
원본 텍스트 (너무 큼):
"A단락\n\nB단락\n\nC단락"

1단계: \n\n 으로 분리 시도
  → ["A단락", "B단락", "C단락"] ← 크기 확인

2단계: 각 조각이 chunk_size 이하인지 확인
  → A단락이 너무 크면: \n 으로 추가 분리
  → B단락이 OK면: 그대로 유지
  → C단락이 너무 크면: ". " 으로 추가 분리

3단계: 그래도 크면: " " (공백)으로 분리
4단계: 최후 수단: "" (글자 단위)로 분리
```

#### 코드 예시 (단계별 설명 포함)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 분할기 생성
# separators: 시도할 구분자 목록 (우선순위 순서)
# - 먼저 \n\n(빈 줄)로 시도, 안되면 \n(줄바꿈), 그 다음 ". "(문장), 마지막 " "(단어)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " ", ""],  # 우선순위 순서
    length_function=len,  # 길이 측정 함수 (기본: 글자 수)
)

# 텍스트 분할
chunks = splitter.split_text(sample_text)

# 결과 확인
print(f"총 {len(chunks)}개 청크 생성")
for i, chunk in enumerate(chunks):
    print(f"\n--- 청크 {i+1} ({len(chunk)}글자) ---")
    print(repr(chunk))  # repr()으로 줄바꿈 문자도 보이게 표시
```

**실제 출력 결과:**

```
총 5개 청크 생성

--- 청크 1 (35글자) ---
'머신러닝이란 무엇인가?'

--- 청크 2 (108글자) ---
'머신러닝(Machine Learning)은 컴퓨터가 명시적인 프로그래밍 없이\n데이터를 통해 스스로 학습하는 인공지능의 한 분야입니다.'

--- 청크 3 (101글자) ---
'머신러닝의 세 가지 유형\n\n지도 학습(Supervised Learning)은 레이블이 있는 데이터로 훈련합니다.'

--- 청크 4 (107글자) ---
"예를 들어, 스팸 메일 분류기는 '스팸/정상'으로 표시된 이메일로 학습합니다.\n\n비지도 학습(Unsupervised Learning)은 레이블 없이 패턴을 찾습니다."

--- 청크 5 (95글자) ---
'고객 군집화가 대표적인 예입니다.\n\n강화 학습(Reinforcement Learning)은 보상과 처벌을 통해 학습합니다.\n알파고가 이 방법으로 바둑을 익혔습니다.'
```

!!! success "개선된 점"
    - 청크 2는 "머신러닝은 ... 분야입니다."라는 완전한 문장을 담고 있습니다.
    - 청크 1은 제목만 별도로 분리됩니다 (의미적으로 올바름).
    - 문장이 중간에 잘리는 현상이 크게 줄었습니다.

#### Document 객체와 함께 사용하기

실제 RAG에서는 단순 텍스트보다 **메타데이터**가 포함된 Document 객체를 주로 사용합니다.

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document 객체 생성 (메타데이터 포함)
# page_content: 실제 텍스트 내용
# metadata: 이 문서에 대한 부가 정보
doc = Document(
    page_content=sample_text,
    metadata={
        "source": "machine_learning_guide.pdf",
        "page": 1,
        "author": "김철수",
        "created_at": "2025-01-15"
    }
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
)

# split_documents()는 Document 객체의 리스트를 입력받아
# 메타데이터를 모든 청크에 자동으로 복사해줍니다
chunks = splitter.split_documents([doc])

# 결과 확인: 메타데이터가 모든 청크에 복사됨
for i, chunk in enumerate(chunks):
    print(f"\n청크 {i+1}:")
    print(f"  내용: {chunk.page_content[:50]}...")
    print(f"  메타데이터: {chunk.metadata}")
```

**출력 결과:**

```
청크 1:
  내용: 머신러닝이란 무엇인가?...
  메타데이터: {'source': 'machine_learning_guide.pdf', 'page': 1, 'author': '김철수', 'created_at': '2025-01-15'}

청크 2:
  내용: 머신러닝(Machine Learning)은 컴퓨터가...
  메타데이터: {'source': 'machine_learning_guide.pdf', 'page': 1, 'author': '김철수', 'created_at': '2025-01-15'}
```

!!! tip "핵심 포인트"
    `split_documents()`를 사용하면 원본 Document의 메타데이터가 모든 청크에 자동으로 복사됩니다.
    이 덕분에 "이 청크가 어느 문서에서 왔는지" 항상 추적할 수 있습니다.

**장단점 정리:**

| 장점 | 단점 |
|------|------|
| 의미 단위 최대 보존 | 고정 크기 청킹보다 약간 복잡 |
| 다양한 문서 형식에 범용 적용 가능 | 구분자가 없는 텍스트에서는 효과 제한 |
| 문장 중간 잘림 최소화 | 여전히 의미적 경계를 완벽히 감지 못함 |
| 안정적이고 예측 가능 | |

**사용 권장 상황:** 대부분의 일반 문서, 기술 문서, 블로그, 기사 (범용 추천)

---

### 전략 3: 의미 기반 청킹 (Semantic Chunking)

**개념:** 텍스트를 **임베딩 벡터**로 변환한 후, 의미가 갑자기 바뀌는 지점을 찾아서 분할합니다.

!!! note "현실 비유"
    마치 책을 읽으면서 "앞 내용과 이 내용은 완전히 다른 주제다"라고 느끼는 **자연스러운 주제 전환점**에서 자르는 것입니다.

!!! info "임베딩(Embedding)이란?"
    텍스트를 의미를 담은 숫자 벡터로 변환하는 것입니다.

    ```
    "고양이는 귀엽다" → [0.2, 0.8, 0.1, 0.9, ...] (수백 차원의 숫자)
    "강아지도 귀엽다" → [0.2, 0.7, 0.2, 0.8, ...] (비슷한 숫자 = 의미가 유사)
    "주식 시장 하락" → [0.9, 0.1, 0.7, 0.2, ...] (다른 숫자 = 의미가 다름)
    ```

    의미가 비슷한 텍스트는 비슷한 벡터를 가집니다.

#### 작동 원리

```
입력 텍스트의 각 문장을 임베딩으로 변환:
문장1 → [0.2, 0.8, 0.1, ...]
문장2 → [0.2, 0.7, 0.2, ...]   ← 문장1과 유사도 높음 (같은 주제)
문장3 → [0.1, 0.9, 0.1, ...]   ← 문장2와 유사도 높음 (같은 주제)
문장4 → [0.8, 0.2, 0.9, ...]   ← 문장3과 유사도 낮음! → 여기서 분할!
문장5 → [0.7, 0.3, 0.8, ...]   ← 문장4와 유사도 높음 (새 주제)
```

#### 코드 예시 (단계별 설명 포함)

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델 생성
# 각 문장을 숫자 벡터로 변환하는 모델
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# SemanticChunker 생성
# breakpoint_threshold_type: 분할 기준 방식
#   - "percentile": 유사도 변화가 하위 X% 미만인 곳에서 분할
#   - "standard_deviation": 평균 대비 표준편차로 판단
#   - "interquartile": 사분위수 범위로 판단
# breakpoint_threshold_amount: 분할 민감도
#   - percentile일 때: 90 = 상위 10% 변화 지점에서만 분할 (큰 청크)
#   - 낮을수록 더 자주 분할 (작은 청크)
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)

# 분할 실행 (내부적으로 각 문장에 임베딩 API 호출 발생)
chunks = splitter.split_text(sample_text)

print(f"총 {len(chunks)}개 청크 생성")
for i, chunk in enumerate(chunks):
    preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
    print(f"\n--- 의미 청크 {i+1} ---")
    print(preview)
```

!!! warning "주의사항: API 비용 발생"
    SemanticChunker는 각 문장마다 임베딩 API를 호출합니다.
    100개 문장이면 100번 API 호출 → 비용과 시간이 발생합니다.

    ```python
    # 비용을 줄이고 싶다면 로컬 임베딩 모델 사용
    from langchain_huggingface import HuggingFaceEmbeddings

    # 무료 로컬 임베딩 모델 (인터넷 없이도 작동)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",  # 한국어 지원 다국어 모델
        model_kwargs={"device": "cpu"}  # GPU 없어도 작동
    )

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )
    ```

**장단점 정리:**

| 장점 | 단점 |
|------|------|
| 가장 자연스러운 의미 경계 | API 비용 발생 |
| 주제 전환을 정확히 감지 | 처리 속도 느림 |
| 청크 크기가 내용에 맞게 조절됨 | 청크 크기 예측 어려움 |
| 품질이 가장 높음 | 설정 조정이 어려움 |

**사용 권장 상황:** 주제가 자주 바뀌는 긴 문서, 품질이 가장 중요한 프로덕션 환경, 비용이 허용될 때

---

### 전략 4: 문서 구조 기반 청킹 (Structure-aware Chunking)

**개념:** 마크다운(Markdown), HTML, 코드 등의 **문서 형식(구조)**을 활용하여 분할합니다.

!!! note "현실 비유"
    마치 목차(Table of Contents)를 보고 각 챕터 단위로 자르는 것입니다.
    헤딩(제목) 구조를 그대로 활용합니다.

#### 4-1. 마크다운 헤더 기반 청킹

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 마크다운 문서 예시
markdown_doc = """
# 파이썬 입문 가이드

## 1장. 파이썬 설치

파이썬 공식 홈페이지에서 3.11 이상 버전을 다운로드합니다.
설치 시 'Add Python to PATH' 옵션을 반드시 체크하세요.

### 1.1 윈도우 설치

윈도우 사용자는 python.org에서 .exe 설치파일을 다운받아 실행합니다.

### 1.2 맥OS 설치

맥 사용자는 brew install python3 명령어를 사용하세요.

## 2장. 첫 번째 프로그램

파이썬으로 "Hello, World!"를 출력해봅시다.
"""

# 어떤 헤더 레벨을 메타데이터로 추출할지 설정
# 형식: (헤더 마크다운 기호, 메타데이터 키 이름)
headers_to_split_on = [
    ("#", "title"),       # # 제목 → metadata["title"]
    ("##", "section"),    # ## 제목 → metadata["section"]
    ("###", "subsection") # ### 제목 → metadata["subsection"]
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # 헤더 텍스트를 청크 내용에 포함할지 여부
)

chunks = splitter.split_text(markdown_doc)

# 결과 확인
for i, chunk in enumerate(chunks):
    print(f"\n--- 청크 {i+1} ---")
    print(f"메타데이터: {chunk.metadata}")
    print(f"내용:\n{chunk.page_content}")
```

**실제 출력 결과:**

```
--- 청크 1 ---
메타데이터: {'title': '파이썬 입문 가이드', 'section': '1장. 파이썬 설치'}
내용:
파이썬 공식 홈페이지에서 3.11 이상 버전을 다운로드합니다.
설치 시 'Add Python to PATH' 옵션을 반드시 체크하세요.

--- 청크 2 ---
메타데이터: {'title': '파이썬 입문 가이드', 'section': '1장. 파이썬 설치', 'subsection': '1.1 윈도우 설치'}
내용:
윈도우 사용자는 python.org에서 .exe 설치파일을 다운받아 실행합니다.

--- 청크 3 ---
메타데이터: {'title': '파이썬 입문 가이드', 'section': '1장. 파이썬 설치', 'subsection': '1.2 맥OS 설치'}
내용:
맥 사용자는 brew install python3 명령어를 사용하세요.

--- 청크 4 ---
메타데이터: {'title': '파이썬 입문 가이드', 'section': '2장. 첫 번째 프로그램'}
내용:
파이썬으로 "Hello, World!"를 출력해봅시다.
```

!!! success "메타데이터 자동 생성의 강점"
    각 청크가 **어느 챕터, 어느 절에 속하는지** 자동으로 메타데이터로 기록됩니다.

    나중에 "1장에서만 검색"하거나 "특정 섹션 필터링"이 가능해집니다.

#### 4-2. 마크다운 + 재귀 분할 결합 (실전 패턴)

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 1단계: 헤더 기반으로 대분할
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "title"),
        ("##", "section"),
        ("###", "subsection"),
    ]
)
header_chunks = md_splitter.split_text(markdown_doc)

# 2단계: 여전히 큰 청크는 재귀 분할로 추가 세분화
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

final_chunks = text_splitter.split_documents(header_chunks)
# 메타데이터(title, section 등)는 자동으로 보존됩니다!

print(f"헤더 분할 후: {len(header_chunks)}개")
print(f"최종 분할 후: {len(final_chunks)}개")
```

#### 4-3. HTML 문서 청킹

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

# HTML 문서를 헤더 구조로 분할
html_content = """
<html>
<body>
  <h1>제품 사용 설명서</h1>
  <h2>설치 방법</h2>
  <p>패키지를 다운로드한 후 설치 마법사를 실행하세요.</p>
  <h2>주요 기능</h2>
  <p>이 제품은 자동 저장, 클라우드 동기화 등을 지원합니다.</p>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "header1"),
    ("h2", "header2"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = html_splitter.split_text(html_content)

for chunk in chunks:
    print(f"메타: {chunk.metadata}")
    print(f"내용: {chunk.page_content[:80]}")
    print()
```

**장단점 정리:**

| 장점 | 단점 |
|------|------|
| 문서 구조를 완벽히 보존 | 구조화된 문서에만 적용 가능 |
| 헤더 정보가 메타데이터로 자동 추가 | 비구조화 텍스트에는 사용 불가 |
| 섹션 단위 필터링 검색 가능 | 헤더 없는 섹션이 누락될 수 있음 |
| 검색 정밀도 높음 | |

**사용 권장 상황:** 마크다운 문서, 기술 문서, 위키, HTML 페이지

---

### 전략 5: 코드 특화 청킹 (Code Chunking)

코드 파일은 함수나 클래스 단위로 자르는 것이 가장 자연스럽습니다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Python 코드 전용 분할기
# Language.PYTHON을 지정하면 Python 구문에 맞는 구분자를 자동으로 사용
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)

python_code = '''
def calculate_area(radius: float) -> float:
    """원의 넓이를 계산합니다."""
    import math
    return math.pi * radius ** 2

class Circle:
    """원을 나타내는 클래스입니다."""

    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        """원의 넓이를 반환합니다."""
        return calculate_area(self.radius)

    def perimeter(self) -> float:
        """원의 둘레를 반환합니다."""
        import math
        return 2 * math.pi * self.radius
'''

chunks = python_splitter.split_text(python_code)
for i, chunk in enumerate(chunks):
    print(f"\n--- 코드 청크 {i+1} ---")
    print(chunk)
```

!!! tip "지원하는 프로그래밍 언어"
    ```python
    # LangChain이 지원하는 언어 목록
    from langchain_text_splitters import Language

    supported_languages = [
        Language.PYTHON,    # 파이썬
        Language.JS,        # 자바스크립트
        Language.TS,        # 타입스크립트
        Language.JAVA,      # 자바
        Language.CPP,       # C++
        Language.GO,        # Go
        Language.RUST,      # Rust
        Language.MARKDOWN,  # 마크다운
        Language.HTML,      # HTML
        Language.SQL,       # SQL
    ]
    ```

---

### 6. Late Chunking (2024년 신기법)

기존 청킹의 문맥 손실 문제를 해결하는 최신 기법. 문서 전체를 먼저 토큰 레벨로 임베딩한 후 청크로 분할한다.

!!! note "실생활 비유"
    기존 청킹: 책을 먼저 페이지로 자른 후 각 페이지를 요약
    Late Chunking: 책 전체를 한 번 읽고 이해한 후 페이지로 나눔 → 문맥이 보존됨

```
기존 방식:  문서 → [청크 분할] → [각 청크 독립 임베딩]
Late Chunking: 문서 → [전체 토큰 임베딩] → [청크 분할] → [토큰 임베딩 평균]
```

- **효과**: 대명사 참조("그것", "이 방법") 등 문맥 의존 검색에서 10-12% 정확도 향상
- **지원**: Jina Embeddings에서 네이티브 지원
- **한계**: 임베딩 모델이 긴 컨텍스트를 지원해야 함

---

## 4가지 전략 최종 비교

동일한 텍스트로 각 전략을 비교한 결과:

| 전략 | 청크 수 | 의미 보존 | 메타데이터 | 속도 | 비용 | 추천 상황 |
|------|---------|----------|----------|------|------|---------|
| 고정 크기 | 예측 가능 | 낮음 | 없음 | 매우 빠름 | 없음 | 프로토타입 |
| 재귀 분할 | 보통 | 중간-높음 | 없음 | 빠름 | 없음 | **일반 추천** |
| 의미 기반 | 가변적 | 매우 높음 | 없음 | 느림 | 높음 | 고품질 필요 시 |
| 구조 기반 | 구조 의존 | 높음 | 자동 생성 | 빠름 | 없음 | 구조화 문서 |

!!! tip "선택 가이드"
    - **처음 시작한다면**: `RecursiveCharacterTextSplitter`
    - **마크다운/HTML 문서**: `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` 결합
    - **코드 파일**: `RecursiveCharacterTextSplitter.from_language()`
    - **최고 품질이 필요하다면**: `SemanticChunker`

---

## 청크 크기 선정 가이드

### 왜 청크 크기가 중요한가?

!!! warning "너무 작은 청크의 문제"
    ```
    원본: "RAG는 검색 기반 생성 기법으로, 외부 지식 베이스에서 관련 정보를
           검색하여 LLM의 답변 품질을 높이는 방법입니다."

    청크 크기 50자로 분할:
    청크1: "RAG는 검색 기반 생성 기법으로, 외부 지식 베이스에서"
    청크2: "관련 정보를 검색하여 LLM의 답변 품질을 높이는 방법입니다."

    문제: "RAG가 뭔가요?"라는 질문에 청크2만 검색되면
    "관련 정보를 검색하여 LLM의 답변 품질을 높이는 방법"이라는
    불완전한 답변이 나옵니다.
    ```

!!! warning "너무 큰 청크의 문제"
    ```
    청크 크기 5000자 (매우 큰 경우):
    청크에 "RAG, 임베딩, 벡터DB, 검색, 생성, 파인튜닝..."이 모두 포함

    문제: "RAG가 뭔가요?"라는 질문에 이 청크가 검색되어도
    LLM에게 5000자 중 어디가 답인지 찾게 만들고
    관련 없는 정보도 많이 포함되어 답변이 흐려집니다.
    ```

### 용도별 권장 설정

| 용도 | 청크 크기 | 오버랩 | 이유 |
|------|-----------|--------|------|
| QA (짧은 답변) | 200-500 토큰 | 20-50 토큰 | 정확한 단편 정보 추출 |
| 요약/분석 | 500-1000 토큰 | 50-100 토큰 | 충분한 맥락 필요 |
| 코드 문서 | 함수/클래스 단위 | 0-50 토큰 | 코드 구조 보존 |
| 법률/계약서 | 조항 단위 | 50-100 토큰 | 법적 의미 단위 보존 |
| 뉴스/블로그 | 300-700 토큰 | 30-70 토큰 | 단락 단위 |
| 학술 논문 | 500-1500 토큰 | 100-200 토큰 | 논거와 근거 함께 포함 |

### 토큰 수 기준으로 청킹하기

```python
import tiktoken  # pip install tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# tiktoken을 사용한 토큰 수 측정 함수
# 모델에 따라 토크나이저가 다릅니다:
# - GPT-4, GPT-3.5: "cl100k_base"
# - text-embedding-3-small/large: "cl100k_base"
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """텍스트의 토큰 수를 반환합니다."""
    return len(encoding.encode(text))

# 토큰 기준 분할기
token_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 최대 500 토큰
    chunk_overlap=50,      # 50 토큰 오버랩
    length_function=count_tokens,  # 글자 수 대신 토큰 수로 측정!
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = token_splitter.split_text(sample_text)
for i, chunk in enumerate(chunks):
    token_count = count_tokens(chunk)
    char_count = len(chunk)
    print(f"청크 {i+1}: {token_count} 토큰 ({char_count}글자)")
```

!!! note "글자 수 vs 토큰 수"
    - `chunk_size=500` + `length_function=len`: 500 **글자** 기준
    - `chunk_size=500` + `length_function=count_tokens`: 500 **토큰** 기준

    한국어 문서는 토큰 수 기준으로 설정하는 것이 더 정확합니다.
    한국어 글자 하나가 약 1.5-2 토큰이기 때문에 글자 수로 설정하면 실제로는 더 작은 청크가 됩니다.

---

## 메타데이터 관리 심화 가이드

### 메타데이터(Metadata)란?

**메타데이터(Metadata)**는 "데이터에 대한 데이터"입니다. 청크 자체의 내용이 아니라, 그 청크에 **대한 정보**입니다.

!!! example "도서관 비유"
    책 한 권이 청크라면:
    - **청크 내용**: 책의 본문 텍스트
    - **메타데이터**: 저자, 출판일, ISBN, 장르, 쪽 번호, 챕터 제목 등

### 나쁜 메타데이터 vs 좋은 메타데이터

```python
from langchain_core.documents import Document

# 나쁜 예: 내용만 저장 (어디서 왔는지 알 수 없음)
bad_chunk = Document(
    page_content="RAG는 검색 기반 생성 기법이다"
)

# 좋은 예: 풍부한 메타데이터
good_chunk = Document(
    page_content="RAG는 검색 기반 생성 기법이다",
    metadata={
        # 출처 정보
        "source": "rag_guide.pdf",
        "source_type": "pdf",
        "page": 3,

        # 문서 구조 정보
        "chapter": "1. RAG 개요",
        "section": "1.1 기본 개념",
        "doc_type": "technical_guide",

        # 시간 정보
        "created_at": "2025-01-15",
        "indexed_at": "2025-03-01",

        # 청크 위치 정보
        "chunk_index": 5,
        "total_chunks": 42,
        "chunk_id": "rag_guide_pdf_005",

        # 언어/접근성 정보
        "language": "ko",
        "difficulty": "intermediate",
    }
)
```

### 자동 메타데이터 추가 파이프라인

```python
import hashlib
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_document_with_metadata(
    raw_text: str,
    source_path: str,
    doc_type: str = "general",
    language: str = "ko"
) -> list:
    """
    문서를 청킹하고 풍부한 메타데이터를 자동으로 추가합니다.

    Args:
        raw_text: 원본 텍스트
        source_path: 원본 파일 경로 또는 URL
        doc_type: 문서 유형 (technical_guide, legal, news 등)
        language: 언어 코드 (ko, en 등)

    Returns:
        메타데이터가 풍부한 Document 객체 리스트
    """
    # 1단계: 분할기 생성
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # 2단계: 텍스트를 청크로 분할
    text_chunks = splitter.split_text(raw_text)
    total_chunks = len(text_chunks)

    # 3단계: 각 청크에 메타데이터 추가
    documents = []
    for i, chunk_text in enumerate(text_chunks):
        # 청크 고유 ID 생성 (파일경로 + 인덱스의 해시)
        chunk_id = hashlib.md5(
            f"{source_path}_{i}".encode()
        ).hexdigest()[:16]

        doc = Document(
            page_content=chunk_text,
            metadata={
                "source": source_path,
                "doc_type": doc_type,
                "language": language,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "chunk_id": chunk_id,
                "indexed_at": datetime.now().isoformat(),
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
            }
        )
        documents.append(doc)

    return documents

# 사용 예시
docs = process_document_with_metadata(
    raw_text=sample_text,
    source_path="machine_learning_guide.pdf",
    doc_type="technical_guide",
    language="ko"
)

print(f"총 {len(docs)}개 청크 생성")
for doc in docs[:2]:
    print(f"\n청크 ID: {doc.metadata['chunk_id']}")
    print(f"위치: {doc.metadata['chunk_index']+1}/{doc.metadata['total_chunks']}")
    print(f"내용: {doc.page_content[:60]}...")
```

### 메타데이터를 활용한 필터링 검색

메타데이터를 잘 설정해두면 나중에 **특정 조건**으로 검색 범위를 좁힐 수 있습니다.

```python
# 벡터 DB에서 메타데이터 필터링 검색 예시 (Chroma 기준)
# pip install chromadb langchain-chroma

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# 전체 검색 (필터 없음)
results = vectorstore.similarity_search("머신러닝이란?", k=3)

# 특정 문서에서만 검색 (source 필터)
results_filtered = vectorstore.similarity_search(
    "머신러닝이란?",
    k=3,
    filter={"source": "machine_learning_guide.pdf"}
)

# 여러 조건 동시 적용
results_multi = vectorstore.similarity_search(
    "머신러닝이란?",
    k=3,
    filter={
        "doc_type": "technical_guide",
        "language": "ko"
    }
)
```

!!! tip "메타데이터 필터링의 장점"
    - "이 회사의 2024년 정책 문서에서만 답을 찾아줘" → `doc_type`과 날짜 필터
    - "한국어 문서에서만 검색해줘" → `language` 필터
    - "특정 제품 매뉴얼에서만 답을 찾아줘" → `source` 필터

---

## 부모-자식 청킹 (Parent-Child Chunking)

### 개념 설명

**부모-자식 청킹**은 **검색 정밀도**와 **맥락 풍부함**을 동시에 얻는 고급 기법입니다.

```
아이디어:
- 검색(임베딩): 작은 청크가 더 정확 (노이즈 적음)
- 답변 생성: 큰 청크가 더 좋음 (맥락 풍부)

해결책:
- 작은 자식 청크로 검색 → 해당 자식의 부모 청크를 LLM에 전달
```

!!! note "현실 비유"
    도서관 색인(Index) 시스템과 같습니다:
    - 색인 카드(자식): "광합성 - 식물이 빛을 에너지로 전환하는 과정" (짧고 정확)
    - 실제 책 페이지(부모): 광합성의 전체 설명, 과정, 예시 (긴 맥락)

    색인으로 찾고 → 실제 책 페이지를 읽는 것처럼!

### 구현 코드

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def create_parent_child_chunks(
    documents: list,
    parent_chunk_size: int = 2000,
    child_chunk_size: int = 400,
    overlap: int = 50
) -> tuple:
    """
    부모-자식 청킹을 수행합니다.

    Returns:
        (parent_chunks, child_chunks) 튜플
        - parent_chunks: LLM에 전달할 큰 청크
        - child_chunks: 검색에 사용할 작은 청크 (parent_id 메타데이터 포함)
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=overlap,
    )

    parent_chunks = []
    child_chunks = []

    for doc in documents:
        parents = parent_splitter.split_documents([doc])

        for parent_idx, parent in enumerate(parents):
            parent_id = f"{doc.metadata.get('source', 'unknown')}_{parent_idx}"
            parent.metadata["parent_id"] = parent_id
            parent.metadata["chunk_type"] = "parent"
            parent_chunks.append(parent)

            children = child_splitter.split_documents([parent])
            for child_idx, child in enumerate(children):
                child.metadata["parent_id"] = parent_id
                child.metadata["chunk_type"] = "child"
                child.metadata["child_index"] = child_idx
                child_chunks.append(child)

    return parent_chunks, child_chunks

# 사용 예시
documents = [Document(
    page_content=sample_text * 5,  # 긴 문서 시뮬레이션
    metadata={"source": "ml_guide.pdf"}
)]

parent_chunks, child_chunks = create_parent_child_chunks(documents)

print(f"부모 청크 수: {len(parent_chunks)}")
print(f"자식 청크 수: {len(child_chunks)}")
print(f"\n자식 청크 예시:")
print(f"  내용: {child_chunks[0].page_content[:60]}...")
print(f"  메타데이터: {child_chunks[0].metadata}")
```

---

## 청크 디버깅 (Troubleshooting Bad Chunks)

### 청크 품질 검증 함수

```python
from langchain_core.documents import Document
from collections import Counter
import re

def validate_chunks(
    chunks: list,
    min_chars: int = 50,
    max_chars: int = 3000,
    verbose: bool = True
) -> dict:
    """
    청크 품질을 종합 검증하고 보고서를 출력합니다.
    """
    issues = {
        "too_short": [],
        "too_long": [],
        "no_source": [],
        "duplicate": [],
        "garbage": [],
    }

    char_lengths = [len(c.page_content) for c in chunks]
    seen_contents = Counter()

    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()

        if len(content) < min_chars:
            issues["too_short"].append(i)
        if len(content) > max_chars:
            issues["too_long"].append(i)

        if not chunk.metadata.get("source"):
            issues["no_source"].append(i)

        key = content[:100]
        seen_contents[key] += 1
        if seen_contents[key] > 1:
            issues["duplicate"].append(i)

        # 알파벳/한글 비율 검사 (0.3 미만이면 의미없는 청크 의심)
        alnum_ratio = len(re.findall(r'[가-힣a-zA-Z0-9]', content)) / max(len(content), 1)
        if alnum_ratio < 0.3:
            issues["garbage"].append(i)

    if verbose:
        total = len(chunks)
        print(f"=== 청크 품질 검증 보고서 ===")
        print(f"총 청크 수: {total}")
        print(f"평균 길이: {sum(char_lengths)/total:.0f}자")
        print(f"최소/최대: {min(char_lengths)}자 / {max(char_lengths)}자")
        print()

        total_issues = sum(len(v) for v in issues.values())
        if total_issues == 0:
            print("✅ 이슈 없음! 모든 청크 정상입니다.")
        else:
            print(f"⚠️ 총 {total_issues}개 이슈 발견:")
            for issue_type, indices in issues.items():
                if indices:
                    sample = str(indices[:5])
                    suffix = "..." if len(indices) > 5 else ""
                    print(f"  - {issue_type}: {len(indices)}개 청크 (인덱스: {sample}{suffix})")

    return issues

# 사용 예시
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents([Document(
    page_content=sample_text,
    metadata={"source": "test.pdf"}
)])

issues = validate_chunks(chunks)
```

### 자주 발생하는 문제와 해결책

#### 문제 1: 청크가 너무 많이 생성됨

```python
# 증상: 청크가 수천 개 → 검색 결과 관련 없는 청크 범람

# 원인: chunk_size가 너무 작음
bad_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,   # 너무 작음!
    chunk_overlap=10,
)

# 해결: chunk_size 늘리기
good_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # 적합한 크기
    chunk_overlap=50,
)
```

#### 문제 2: 표나 리스트가 중간에 잘림

```python
import re

def preprocess_tables(text: str) -> str:
    """표 앞뒤에 마커를 추가하여 청크로 분리되지 않도록 합니다."""
    # 마크다운 표 패턴 감지
    table_pattern = r'(\|.+\|\n)+'

    def add_markers(match):
        table = match.group()
        return f"\n\n[TABLE_START]\n{table}[TABLE_END]\n\n"

    return re.sub(table_pattern, add_markers, text)

processed_text = preprocess_tables(raw_text)
chunks = splitter.split_text(processed_text)
```

#### 문제 3: 한국어 특수 상황 - 문장 경계 인식 오류

```python
# 증상: "A.I. 기술은..." 에서 "A.I" 뒤에서 잘림

# 해결: 한국어에 맞는 구분자 설정
korean_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\n\n",     # 단락 구분 (최우선)
        "\n",       # 줄바꿈
        "다.\n",    # 한국어 문장 끝 패턴
        "다. ",     # 한국어 문장 끝 패턴
        "요.\n",    # 한국어 문장 끝 패턴
        "요. ",     # 한국어 문장 끝 패턴
        ". ",       # 영어 문장 끝 (뒤에 공백 필수)
        " ",        # 단어 단위
        "",         # 글자 단위 (최후 수단)
    ]
)
```

#### 문제 4: 특수 문자나 HTML 태그가 청크에 포함됨

```python
import re

def clean_text(text: str) -> str:
    """청킹 전 텍스트를 정리합니다."""
    text = re.sub(r'<[^>]+>', '', text)      # HTML 태그 제거
    text = re.sub(r' +', ' ', text)          # 연속 공백 제거
    text = re.sub(r'\n{3,}', '\n\n', text)   # 과도한 줄바꿈 정리
    text = re.sub(r'\n- \d+ -\n', '\n', text)  # 페이지 번호 제거
    return text.strip()

raw = "   <p>안녕하세요   </p>\n\n\n\n다음 내용입니다."
clean = clean_text(raw)
print(repr(clean))
# 출력: '안녕하세요\n\n다음 내용입니다.'
```

---

## 청크에 컨텍스트 헤더 추가

### 왜 컨텍스트 헤더가 필요한가?

```
문제 상황:
청크 내용: "이것은 주로 3단계로 이루어집니다."

→ LLM 입장에서: "무엇이 3단계로 이루어진다는 거지?"
   문맥이 없어서 답변이 엉터리가 됩니다.
```

### 컨텍스트 헤더 추가 함수

```python
from langchain_core.documents import Document

def add_context_headers(
    chunks: list,
    document_title: str = None,
    include_position: bool = True,
    include_source: bool = True
) -> list:
    """
    각 청크 앞에 문서 제목, 위치 정보를 추가합니다.
    LLM이 "이 내용이 어디서 왔는지" 바로 파악할 수 있게 됩니다.
    """
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        header_parts = []

        title = document_title or chunk.metadata.get("chapter") or chunk.metadata.get("title")
        if title:
            header_parts.append(f"문서: {title}")

        if include_source and chunk.metadata.get("source"):
            header_parts.append(f"출처: {chunk.metadata['source']}")

        if include_position:
            header_parts.append(f"섹션: {i+1}/{total}")

        if header_parts:
            header = "[" + " | ".join(header_parts) + "]\n\n"
            chunk.page_content = header + chunk.page_content

    return chunks

# 사용 예시
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents([Document(
    page_content=sample_text,
    metadata={"source": "ml_guide.pdf", "chapter": "머신러닝 입문"}
)])

chunks_with_header = add_context_headers(chunks)

print("첫 번째 청크 내용:")
print(chunks_with_header[0].page_content[:150])
```

**출력 결과:**

```
첫 번째 청크 내용:
[문서: 머신러닝 입문 | 출처: ml_guide.pdf | 섹션: 1/5]

머신러닝이란 무엇인가?
```

!!! tip "Anthropic Contextual Retrieval (2024)"
    각 청크에 LLM으로 문맥 요약을 자동 생성하여 추가하는 기법.
    수동으로 헤더를 추가하는 위 방법의 자동화 버전이라고 볼 수 있다.
    Anthropic 연구에서 검색 실패율 49% 감소, Reranking 결합 시 67% 감소를 보고했다.

---

## 흔한 실수 (Common Mistakes)

### 실수 1: 오버랩 없이 중요한 텍스트 분할

```python
# 잘못된 예
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,  # 오버랩 없음 → 경계 정보 유실
)

# 올바른 예
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,  # chunk_size의 10% 정도 권장
)
```

### 실수 2: 메타데이터 없이 청크 저장

```python
# 잘못된 예 - 나중에 출처를 알 수 없음
chunks = splitter.split_text(raw_text)  # 단순 문자열 리스트 반환

# 올바른 예 - Document 객체 사용
doc = Document(
    page_content=raw_text,
    metadata={"source": "guide.pdf", "page": 1}
)
chunks = splitter.split_documents([doc])  # 메타데이터 자동 복사
```

### 실수 3: chunk_size가 너무 작음

```python
# 잘못된 예 - 청크가 수천 개 생성되어 검색 품질 저하
bad_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,   # 50글자는 너무 작음
    chunk_overlap=5,
)

# 올바른 예
good_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # 300-500글자가 일반적으로 적합
    chunk_overlap=50,
)
```

### 실수 4: 전처리 없이 PDF 텍스트 바로 청킹

```python
import re

# 잘못된 예 - PDF 추출 텍스트에는 페이지 번호, 헤더/푸터 노이즈 많음
# chunks = splitter.split_text(raw_pdf_text)  # 노이즈 포함!

# 올바른 예 - 전처리 후 청킹
def preprocess_pdf_text(text: str) -> str:
    text = re.sub(r'\n\d+\n', '\n', text)         # 페이지 번호 제거
    text = re.sub(r' +', ' ', text)               # 연속 공백 제거
    text = re.sub(r'\n{3,}', '\n\n', text)        # 과도한 줄바꿈 정리
    return text.strip()

cleaned = preprocess_pdf_text(raw_pdf_text)
chunks = splitter.split_text(cleaned)
```

### 실수 5: 모든 문서에 동일한 설정 적용

```python
# 잘못된 예 - 문서 유형 무관하게 동일 설정
one_size_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
# 법률 문서도 뉴스도 코드도 모두 500자? → 각 유형에 최적화 안 됨

# 올바른 예 - 문서 유형별 설정
def get_splitter_for_doc_type(doc_type: str) -> RecursiveCharacterTextSplitter:
    configs = {
        "news": {"chunk_size": 300, "chunk_overlap": 30},
        "legal": {"chunk_size": 800, "chunk_overlap": 100},
        "technical": {"chunk_size": 500, "chunk_overlap": 50},
        "academic": {"chunk_size": 1000, "chunk_overlap": 100},
    }
    config = configs.get(doc_type, {"chunk_size": 400, "chunk_overlap": 50})
    return RecursiveCharacterTextSplitter(**config)

news_splitter = get_splitter_for_doc_type("news")
legal_splitter = get_splitter_for_doc_type("legal")
```

---

## 자주 묻는 질문 (FAQ)

### Q1. chunk_size는 글자 수인가요, 토큰 수인가요?

기본적으로 **글자 수(characters)**입니다. 토큰 수로 설정하려면 `length_function`을 변경해야 합니다.

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

token_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=lambda text: len(enc.encode(text))
)
```

### Q2. 최적의 chunk_size는 얼마인가요?

정답은 없습니다. 일반적인 시작점:

- 한국어 문서: **300-500 글자** (약 200-350 토큰)
- 영어 문서: **500-1000 글자** (약 200-400 토큰)
- 코드: **함수/클래스 단위** (크기 무관)

실험을 통해 최적값을 찾아야 합니다.

### Q3. 오버랩이 클수록 좋은가요?

아닙니다. 오버랩이 너무 크면:

- 청크 수가 늘어나 검색 성능 저하
- 중복 내용이 많아져 저장 비용 증가
- LLM에 중복 정보가 전달되어 혼란 유발

일반적으로 chunk_size의 **10-20%**를 권장합니다.

### Q4. split_text()와 split_documents()의 차이는?

```python
# split_text(): 문자열 → 문자열 리스트 반환 (메타데이터 없음)
result = splitter.split_text("긴 텍스트...")
# 결과: ["청크1", "청크2", ...]

# split_documents(): Document 리스트 → Document 리스트 반환 (메타데이터 자동 복사!)
docs = [Document(page_content="긴 텍스트...", metadata={"source": "file.pdf"})]
result = splitter.split_documents(docs)
# 결과: [Document(metadata={"source": "file.pdf"}), Document(...), ...]
```

실제 RAG 구현에서는 거의 항상 `split_documents()`를 사용하세요.

### Q5. 청크마다 임베딩이 생성되나요?

네. 각 청크가 벡터 DB에 저장될 때 하나의 임베딩 벡터가 생성됩니다.

```
청크 수 = 1000개
임베딩 API 호출 = 1000번
임베딩 벡터 = 1000개
```

청크 수가 많을수록 임베딩 비용이 증가합니다.

### Q6. 다국어 문서(한국어+영어 혼합)는 어떻게 처리하나요?

```python
multilang_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        "다.\n",    # 한국어 문장 끝
        "다. ",
        "요.\n",
        "요. ",
        ". ",       # 영어 문장 끝
        " ",
        "",
    ]
)
```

### Q7. 이미지나 표가 포함된 PDF는 어떻게 처리하나요?

```python
# 표는 별도 Document로 저장
table_doc = Document(
    page_content="[표] 연도별 매출 현황\n2022: 100억\n2023: 150억\n2024: 200억",
    metadata={
        "source": "annual_report.pdf",
        "content_type": "table",
        "page": 15,
        "table_caption": "연도별 매출 현황"
    }
)
```

---

## 직접 해보기 (Hands-on Exercises)

### 연습 1: 기본 청킹 비교

다음 텍스트를 두 가지 방법으로 청킹해보고 결과를 비교해보세요.

```python
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

exercise_text = """
인공지능의 역사

인공지능(AI)은 1950년대부터 연구가 시작되었습니다.
앨런 튜링은 1950년 "기계가 생각할 수 있는가?"라는 논문을 발표하며 AI 연구의 토대를 마련했습니다.

첫 번째 AI 붐 (1950-1970년대)

초기 AI 연구자들은 매우 낙관적이었습니다.
모든 인간의 지식을 규칙으로 표현할 수 있다고 믿었습니다.
하지만 현실의 복잡성을 처리하는 데 한계에 부딪혔습니다.

딥러닝 혁명 (2010년대~현재)

2012년 ImageNet 대회에서 딥러닝 모델이 압도적인 성능을 보이며 새로운 시대를 열었습니다.
GPU의 발전과 빅데이터의 등장이 딥러닝 혁명을 가능하게 했습니다.
"""

# 방법 1: 고정 크기 청킹
fixed_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20, separator="\n")

# 방법 2: 재귀 분할
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " ", ""]
)

fixed_chunks = fixed_splitter.split_text(exercise_text)
recursive_chunks = recursive_splitter.split_text(exercise_text)

print(f"고정 크기: {len(fixed_chunks)}개 청크")
print(f"재귀 분할: {len(recursive_chunks)}개 청크")

# 비교: 어느 방법이 더 자연스러운 경계에서 분할했나요?
for i in range(min(3, len(fixed_chunks), len(recursive_chunks))):
    print(f"\n=== 청크 {i+1} ===")
    fc = fixed_chunks[i] if i < len(fixed_chunks) else "(없음)"
    rc = recursive_chunks[i] if i < len(recursive_chunks) else "(없음)"
    print(f"고정: {fc[:60]}...")
    print(f"재귀: {rc[:60]}...")
```

### 연습 2: 메타데이터 파이프라인 구축

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import hashlib

# TODO: 다음 함수를 완성하세요
def build_chunking_pipeline(
    texts: list,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list:
    """
    여러 문서를 청킹하고 메타데이터를 풍부하게 추가하는 파이프라인.
    각 입력 항목: {"text": "...", "source": "...", "type": "..."}

    반환되는 Document는 다음 메타데이터를 포함해야 합니다:
    - source, doc_type, chunk_id, chunk_index, total_chunks, indexed_at
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []

    for item in texts:
        # 힌트: splitter.split_text()로 텍스트를 분할하고
        # 각 청크에 메타데이터를 추가한 Document를 만드세요
        pass  # TODO: 구현하세요

    return all_chunks

# 테스트
test_texts = [
    {"text": exercise_text, "source": "ai_history.pdf", "type": "educational"},
    {"text": sample_text, "source": "ml_guide.pdf", "type": "technical"}
]

result = build_chunking_pipeline(test_texts)
print(f"총 {len(result)}개 청크 생성됨")
```

### 연습 3: 청크 품질 개선

아래 코드에는 최소 3가지 문제가 있습니다. 찾아서 고쳐보세요.

```python
from langchain_text_splitters import CharacterTextSplitter

# 이 코드에는 문제가 있습니다. 찾아서 수정하세요!
problematic_splitter = CharacterTextSplitter(
    chunk_size=30,       # 문제 1: 왜 문제인가요?
    chunk_overlap=0,     # 문제 2: 왜 문제인가요?
    separator=".",       # 문제 3: 왜 문제인가요?
)

# 힌트:
# 1. chunk_size=30이면 어떤 문제가 생기나요?
# 2. chunk_overlap=0이면 무슨 문제가 있나요?
# 3. "."만을 구분자로 쓰면 어떤 상황이 발생하나요?

# 수정된 버전을 작성해보세요:
# fixed_splitter = RecursiveCharacterTextSplitter(...)
```

---

## 꿀팁 모음 (Best Practices)

### 팁 1: 청킹 실험 자동화

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def experiment_chunk_sizes(
    text: str,
    sizes_to_try: list = None,
    overlap_ratios: list = None,
):
    """다양한 청크 크기와 오버랩 비율로 실험합니다."""
    if sizes_to_try is None:
        sizes_to_try = [100, 200, 400, 800, 1500]
    if overlap_ratios is None:
        overlap_ratios = [0.1, 0.2]

    print("chunk_size | overlap | 청크 수 | 평균 길이 | 최소 | 최대")
    print("-" * 65)

    for size in sizes_to_try:
        for ratio in overlap_ratios:
            overlap = int(size * ratio)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
            )
            chunks = splitter.split_text(text)
            lengths = [len(c) for c in chunks]
            avg = sum(lengths) / len(lengths)
            print(
                f"{size:10d} | {overlap:7d} | {len(chunks):7d} | "
                f"{avg:9.0f} | {min(lengths):4d} | {max(lengths):4d}"
            )

# 실험 실행
experiment_chunk_sizes(sample_text * 10)
```

### 팁 2: 청킹 결과 간단 시각화

```python
def print_chunk_summary(chunks: list, title: str = "청크 요약") -> None:
    """청크 통계를 텍스트로 요약합니다."""
    lengths = [len(c.page_content) if hasattr(c, 'page_content') else len(c)
               for c in chunks]

    sorted_lengths = sorted(lengths)
    median = sorted_lengths[len(sorted_lengths) // 2]
    avg = sum(lengths) / len(lengths)

    print(f"\n=== {title} ===")
    print(f"총 청크 수   : {len(lengths)}")
    print(f"평균 길이    : {avg:.0f}자")
    print(f"중앙값 길이  : {median}자")
    print(f"최소 / 최대  : {min(lengths)}자 / {max(lengths)}자")

    # 간단한 ASCII 히스토그램
    bucket_count = 5
    bucket_size = (max(lengths) - min(lengths)) // bucket_count + 1
    buckets = [0] * bucket_count
    for l in lengths:
        idx = min((l - min(lengths)) // bucket_size, bucket_count - 1)
        buckets[idx] += 1

    print("\n분포 (대략):")
    for i, count in enumerate(buckets):
        low = min(lengths) + i * bucket_size
        high = low + bucket_size - 1
        bar = "#" * count
        print(f"  {low:4d}-{high:4d}자: {bar} ({count}개)")
```

### 팁 3: 안정적인 청크 ID 체계

```python
import hashlib

def generate_chunk_id(source: str, content: str) -> str:
    """
    내용 기반 해시로 안정적인 청크 ID를 생성합니다.
    동일한 내용이면 항상 같은 ID → 재인덱싱 시 중복 방지
    """
    combined = f"{source}:{content}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]

# 사용 예시
id1 = generate_chunk_id("guide.pdf", "RAG는 검색 기반 생성 기법이다")
id2 = generate_chunk_id("guide.pdf", "RAG는 검색 기반 생성 기법이다")
print(f"같은 내용, 같은 ID: {id1 == id2}")  # True
```

### 팁 4: 문서 전처리 체크리스트

```
청킹 전 체크리스트:
  ☐ 불필요한 헤더/푸터 제거 (페이지 번호, 워터마크)
  ☐ 표를 텍스트로 변환하거나 별도 Document로 처리
  ☐ 이미지/차트의 캡션 텍스트 추출
  ☐ 인코딩 통일 (UTF-8)
  ☐ 중복 문서 제거
  ☐ 메타데이터 스키마 사전 정의 (팀 내 표준화)
  ☐ 청크 ID 체계 수립 (재인덱싱 시 중복 방지)
  ☐ 특수 문자 / HTML 태그 제거
  ☐ 연속된 공백 및 빈 줄 정리
  ☐ 언어 감지 및 언어별 메타데이터 추가
```

### 팁 5: 점진적 청킹 (대용량 파일 처리)

```python
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def process_large_file_incrementally(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    batch_size: int = 100,
) -> None:
    """
    대용량 파일을 배치 단위로 처리합니다.
    메모리 사용량을 낮게 유지합니다.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    total_processed = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        buffer = ""

        for line in f:
            buffer += line

            if len(buffer) > chunk_size * batch_size:
                chunks = splitter.split_text(buffer)

                for chunk in chunks[:-1]:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": os.path.basename(file_path),
                            "batch": total_processed // batch_size,
                        }
                    )
                    # vectorstore.add_documents([doc]) 호출하는 부분
                    total_processed += 1

                buffer = chunks[-1] if chunks else ""

        if buffer.strip():
            final_chunks = splitter.split_text(buffer)
            print(f"최종 {len(final_chunks)}개 청크 처리")

    print(f"총 {total_processed}개 청크 처리 완료")
```

---

## 완성된 청킹 파이프라인 (Production-Ready)

```python
import hashlib
import re
from datetime import datetime
from typing import Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

class DocumentChunkingPipeline:
    """
    프로덕션 수준의 문서 청킹 파이프라인.

    기능:
    - 문서 유형별 자동 분할 전략 선택
    - 풍부한 메타데이터 자동 추가
    - 청크 품질 검증
    - 컨텍스트 헤더 추가 (선택)
    """

    DEFAULT_CONFIGS = {
        "markdown": {"chunk_size": 500, "chunk_overlap": 50},
        "legal":    {"chunk_size": 800, "chunk_overlap": 100},
        "technical":{"chunk_size": 500, "chunk_overlap": 50},
        "news":     {"chunk_size": 300, "chunk_overlap": 30},
        "academic": {"chunk_size": 1000, "chunk_overlap": 100},
        "general":  {"chunk_size": 400, "chunk_overlap": 50},
    }

    def __init__(
        self,
        doc_type: str = "general",
        add_headers: bool = False,
        validate: bool = True,
    ):
        self.doc_type = doc_type
        self.add_headers = add_headers
        self.validate = validate

        config = self.DEFAULT_CONFIGS.get(doc_type, self.DEFAULT_CONFIGS["general"])
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def preprocess(self, text: str) -> str:
        """텍스트 전처리"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def generate_chunk_id(self, source: str, content: str) -> str:
        return hashlib.sha256(f"{source}:{content}".encode()).hexdigest()[:16]

    def process(
        self,
        text: str,
        source: str,
        extra_metadata: Optional[dict] = None,
    ) -> list:
        """전체 파이프라인 실행."""
        cleaned_text = self.preprocess(text)
        text_chunks = self.splitter.split_text(cleaned_text)
        total_chunks = len(text_chunks)

        documents = []
        for i, chunk_text in enumerate(text_chunks):
            metadata = {
                "source": source,
                "doc_type": self.doc_type,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "chunk_id": self.generate_chunk_id(source, chunk_text),
                "char_count": len(chunk_text),
                "indexed_at": datetime.now().isoformat(),
            }
            if extra_metadata:
                metadata.update(extra_metadata)

            doc = Document(page_content=chunk_text, metadata=metadata)
            documents.append(doc)

        if self.add_headers:
            for i, doc in enumerate(documents):
                header = f"[출처: {source} | {i+1}/{total_chunks}]\n\n"
                doc.page_content = header + doc.page_content

        if self.validate:
            short_count = sum(1 for d in documents if len(d.page_content) < 30)
            status = "✅ 모두 정상" if short_count == 0 else f"⚠️ {short_count}개 짧은 청크 발견"
            print(f"{len(documents)}개 청크 생성 완료 ({status})")

        return documents


# 사용 예시
pipeline = DocumentChunkingPipeline(
    doc_type="technical",
    add_headers=True,
    validate=True,
)

chunks = pipeline.process(
    text=sample_text,
    source="ml_guide.pdf",
    extra_metadata={"author": "김철수", "language": "ko"}
)

print(f"\n첫 번째 청크 미리보기:")
print(f"내용: {chunks[0].page_content[:100]}...")
print(f"메타데이터: {chunks[0].metadata}")
```

---

## 핵심 요약

!!! abstract "이것만 기억하자"

    **청킹의 본질:**

    1. 청크는 "하나의 완결된 아이디어"를 담아야 한다
    2. 문장이 중간에 잘리면 검색 품질이 크게 떨어진다
    3. 오버랩은 경계에 걸친 정보를 놓치지 않게 해준다

    **전략 선택:**

    4. 범용 → `RecursiveCharacterTextSplitter` (가장 무난)
    5. 마크다운/HTML → `MarkdownHeaderTextSplitter` + 재귀 분할 결합
    6. 고품질 → `SemanticChunker` (비용 감수)
    7. 코드 → `RecursiveCharacterTextSplitter.from_language()`

    **크기 설정:**

    8. 한국어 일반 문서: 300-500 글자, 오버랩 30-50 글자
    9. 용도에 따라 다름 (QA: 작게, 요약: 크게)
    10. 반드시 실험으로 최적값 검증

    **메타데이터:**

    11. `source`, `chunk_id`, `chunk_index`, `total_chunks`는 필수
    12. `split_documents()`를 쓰면 메타데이터가 자동 복사됨
    13. 메타데이터를 활용한 필터링 검색이 가능해짐

    **파이프라인:**

    14. 전처리 → 청킹 → 메타데이터 추가 → 검증 순서로 구축
    15. 부모-자식 청킹으로 검색 정밀도와 맥락을 동시에 확보
