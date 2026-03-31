# 2. 임베딩 모델

## 임베딩이란?

텍스트를 **고차원 수치 벡터**로 변환하는 것. 의미적으로 유사한 텍스트는 벡터 공간에서 가까이 위치한다.

```
"고양이가 쥐를 쫓는다" → [0.12, -0.34, 0.56, ...]  (1536차원)
"캣이 마우스를 추격한다" → [0.13, -0.33, 0.55, ...]  ← 가까운 벡터!
```

---

## 최신 임베딩 모델 비교 (2025)

### 상용 모델

| 모델 | 제공사 | 차원 | 최대 토큰 | MTEB 순위 | 특징 |
|------|--------|------|-----------|-----------|------|
| **text-embedding-3-large** | OpenAI | 3072 | 8,191 | 상위권 | 차원 축소 가능, 가성비 |
| **text-embedding-3-small** | OpenAI | 1536 | 8,191 | 중위권 | 저비용, 빠른 속도 |
| **embed-v4.0** | Cohere | 1024 | 128,000 | 최상위권 | 긴 문서 지원, 다국어 |
| **voyage-3-large** | Voyage AI | 1024 | 32,000 | 최상위권 | 코드 임베딩 강점 |
| **Gemini Embedding** | Google | 3072 | 8,192 | 최상위권 | Task type 지정 가능 |

### 오픈소스 모델

| 모델 | 차원 | 최대 토큰 | MTEB 순위 | 특징 |
|------|------|-----------|-----------|------|
| **GTE-Qwen2-7B-instruct** | 3584 | 131,072 | 최상위 | 초장문 지원 |
| **NV-Embed-v2** | 4096 | 32,768 | 최상위 | NVIDIA, 고성능 |
| **multilingual-e5-large-instruct** | 1024 | 514 | 상위권 | 다국어 강점 |
| **bge-m3** | 1024 | 8,192 | 상위권 | 다국어, 다기능 |
| **nomic-embed-text-v2-moe** | 768 | 8,192 | 상위권 | 경량, 효율적 |

---

## 모델 선택 가이드

```
선택 기준 결정 트리:

비용 제한 있음?
├─ Yes → 오픈소스 (sentence-transformers로 로컬 실행)
│        ├─ 다국어 필요 → bge-m3 또는 multilingual-e5
│        ├─ 장문 처리 → GTE-Qwen2-7B
│        └─ 경량/빠른 속도 → nomic-embed-text-v2-moe
│
└─ No → 상용 API
         ├─ 범용 → OpenAI text-embedding-3-large
         ├─ 장문 문서 → Cohere embed-v4.0
         └─ 코드 검색 → Voyage voyage-3-large
```

---

## Python 사용 예시

### OpenAI 임베딩

```python
from langchain_openai import OpenAIEmbeddings

# 기본 사용
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 차원 축소 (비용 절감)
embeddings_small = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024  # 3072 → 1024로 축소
)

vector = embeddings.embed_query("RAG는 검색 증강 생성입니다")
print(f"벡터 차원: {len(vector)}")  # 3072
```

### 오픈소스 임베딩 (sentence-transformers)

```python
from sentence_transformers import SentenceTransformer

# 로컬 실행 — API 비용 없음
model = SentenceTransformer("BAAI/bge-m3")

sentences = [
    "RAG는 검색 기반 생성 기법이다",
    "Retrieval Augmented Generation은 문서 검색을 활용한다",
    "오늘 날씨가 좋다"
]

embeddings = model.encode(sentences)

# 유사도 계산
from sentence_transformers.util import cos_sim
similarities = cos_sim(embeddings[0], embeddings[1:])
print(f"RAG 문장 간 유사도: {similarities[0][0]:.4f}")  # 높음
print(f"무관 문장 유사도: {similarities[0][1]:.4f}")     # 낮음
```

### Cohere 임베딩

```python
import cohere

co = cohere.ClientV2()

# 검색용 임베딩 (input_type 구분이 핵심!)
doc_embeddings = co.embed(
    texts=["RAG 파이프라인의 구조", "벡터 DB 비교"],
    model="embed-v4.0",
    input_type="search_document",  # 문서 인덱싱 시
    embedding_types=["float"]
)

query_embedding = co.embed(
    texts=["RAG란 무엇인가?"],
    model="embed-v4.0",
    input_type="search_query",  # 검색 쿼리 시
    embedding_types=["float"]
)
```

---

## 임베딩 성능 측정

### MTEB (Massive Text Embedding Benchmark)

임베딩 모델의 표준 벤치마크. 다양한 태스크에서 성능을 종합 평가한다.

| 태스크 | 설명 | 예시 |
|--------|------|------|
| Retrieval | 쿼리-문서 매칭 | RAG에 직접 해당 |
| STS | 문장 유사도 | 의미 유사도 판단 |
| Classification | 텍스트 분류 | 감성분석 등 |
| Clustering | 문서 군집화 | 주제 분류 |

!!! tip "모델 선택 팁"
    - RAG 용도라면 **Retrieval** 태스크 점수를 우선 확인하라
    - 한국어 RAG라면 다국어 지원 모델(bge-m3, multilingual-e5) 선택
    - [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)에서 최신 순위 확인

---

## 핵심 요약

!!! summary "이것만 기억하자"
    1. 임베딩 = 텍스트의 의미를 숫자 벡터로 변환
    2. **상용**: OpenAI(범용), Cohere(장문), Voyage(코드) — **오픈소스**: bge-m3(다국어), GTE-Qwen2(장문)
    3. RAG에선 **Retrieval** 성능이 핵심 지표
    4. Cohere처럼 `input_type`(document vs query)을 구분하는 모델은 반드시 올바르게 설정
