# 4. 벡터 데이터베이스

!!! abstract "이 챕터에서 배우는 것"
    - 벡터 데이터베이스가 무엇인지, 왜 일반 DB와 다른지 이해한다
    - 유사도 검색(Similarity Search), k-NN, ANN, HNSW 등 핵심 개념을 완전히 이해한다
    - OpenSearch, pgvector, Meilisearch, Chroma 각각의 설치부터 실전 사용까지 익힌다
    - 하이브리드 검색(벡터 + 키워드)의 원리와 구현 방법을 배운다
    - 본인 프로젝트에 맞는 벡터 DB를 선택하는 기준을 갖는다

---

## 1. 벡터 데이터베이스란 무엇인가?

### 1.1 일반 데이터베이스 vs 벡터 데이터베이스

일반적인 데이터베이스(MySQL, PostgreSQL 등)는 **정확한 값으로 검색**한다.

```sql
-- 일반 DB: 정확히 일치하거나 LIKE 패턴으로만 검색
SELECT * FROM products WHERE name = '아이폰';
SELECT * FROM products WHERE name LIKE '%폰%';
```

이런 방식으로는 "스마트폰과 비슷한 제품"이나 "RAG가 뭔지 설명하는 문서"를 찾을 수 없다.
"비슷함"을 수치로 표현하고 검색하는 것이 불가능하기 때문이다.

**벡터 데이터베이스**는 다르다. 텍스트, 이미지, 음성 등을 **숫자 배열(벡터)**로 변환해서 저장하고, **의미적으로 비슷한 것**을 찾아준다.

!!! info "벡터(Vector)란?"
    벡터는 숫자들의 배열이다. 예를 들어 "고양이"라는 단어를 임베딩 모델에 넣으면
    `[0.12, -0.34, 0.87, 0.05, ...]` 같은 수백~수천 개의 숫자 배열이 나온다.

    이 숫자들은 의미(semantics)를 담고 있어서, "고양이"와 "강아지"의 벡터는 가깝고,
    "고양이"와 "자동차"의 벡터는 멀다.

### 1.2 도서관 비유로 이해하기

!!! example "현실 비유: 두 가지 도서관"

    **일반 데이터베이스 = 알파벳/가나다순 도서관**

    - 책이 제목 순서로 정렬되어 있다
    - "파이썬"을 찾으면 정확히 제목에 "파이썬"이 있는 책만 나온다
    - "프로그래밍 언어를 배우고 싶어요"라고 하면 찾을 수 없다

    **벡터 데이터베이스 = 주제별 의미 분류 도서관**

    - 책들이 "주제의 유사함"으로 배치되어 있다
    - 파이썬 책 근처에 자바, C++, 프로그래밍 입문서가 모여 있다
    - "프로그래밍 언어 배우고 싶어요"라고 하면 → 프로그래밍 관련 책들이 쫙 나온다
    - 심지어 "코딩 시작하기"처럼 다른 표현도 찾아준다

### 1.3 왜 RAG에서 벡터 DB가 필수인가?

RAG(Retrieval-Augmented Generation)의 핵심은 "질문과 관련된 문서를 찾아오는 것"이다.

```
사용자 질문: "회사 환불 정책이 어떻게 되나요?"

벡터 DB 없이: 정확히 "환불 정책"이라는 단어가 있는 문서만 검색됨
             → "반품 규정", "구매 취소 절차" 같은 관련 문서를 못 찾음

벡터 DB 있으면: "환불"과 의미적으로 유사한 모든 문서 검색
              → "반품 규정", "구매 취소", "결제 취소 방법" 모두 찾아줌
```

!!! tip "왜 이게 중요한가?"
    실제 사용자는 문서에 있는 정확한 단어로 질문하지 않는다.
    "배가 아파요" → 의료 문서의 "복통", "위장 장애" 항목을 찾아야 한다.
    벡터 DB가 이런 의미적 연결을 가능하게 한다.

---

## 2. 핵심 개념 완전 정복

### 2.1 임베딩(Embedding)

임베딩은 텍스트(또는 이미지 등)를 숫자 벡터로 변환하는 과정이다.

```python
# 임베딩의 실제 동작 예시
from openai import OpenAI

client = OpenAI()

# 텍스트 → 벡터 변환
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="RAG는 검색 증강 생성 기법이다"
)

vector = response.data[0].embedding
print(f"벡터 차원 수: {len(vector)}")   # 1536
print(f"첫 5개 숫자: {vector[:5]}")     # [-0.006929, 0.005001, ...]
```

!!! info "차원(Dimension)이란?"
    벡터의 숫자 개수를 "차원"이라고 한다.

    - `text-embedding-3-small`: 1536차원
    - `text-embedding-ada-002`: 1536차원
    - `all-MiniLM-L6-v2` (무료): 384차원

    차원이 클수록 더 많은 의미를 담을 수 있지만, 저장 공간과 검색 시간이 늘어난다.

### 2.2 유사도 검색(Similarity Search)

두 벡터가 얼마나 비슷한지 수치로 표현하는 방법이 여러 가지 있다.

#### 코사인 유사도 (Cosine Similarity) — 가장 많이 사용

```
코사인 유사도 = 두 벡터 사이의 각도의 코사인 값

값 범위: -1 ~ 1
- 1에 가까울수록: 매우 비슷 (같은 방향)
- 0에 가까울수록: 관계 없음 (직각)
- -1에 가까울수록: 반대 의미 (반대 방향)
```

!!! example "코사인 유사도 비유"
    두 사람이 나침반을 들고 있다고 상상해보자.

    - 두 사람이 같은 방향(북쪽)을 가리키면 → 코사인 유사도 = 1 (완전히 같음)
    - 한 사람은 북쪽, 한 사람은 동쪽 → 코사인 유사도 = 0 (무관함)
    - 한 사람은 북쪽, 한 사람은 남쪽 → 코사인 유사도 = -1 (반대)

    텍스트 임베딩에서는 의미가 비슷할수록 같은 방향을 가리킨다.

```python
# Python으로 코사인 유사도 계산해보기
import numpy as np

def cosine_similarity(vec_a, vec_b):
    """두 벡터의 코사인 유사도를 계산한다"""
    # 내적(dot product) / (각 벡터의 크기 곱)
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    return dot_product / (magnitude_a * magnitude_b)

# 예시: 간단한 2차원 벡터로 테스트
cat_vector   = np.array([0.9, 0.1])   # "고양이"를 나타내는 가상의 벡터
dog_vector   = np.array([0.8, 0.2])   # "강아지"
car_vector   = np.array([0.1, 0.9])   # "자동차"

print(f"고양이 vs 강아지: {cosine_similarity(cat_vector, dog_vector):.3f}")  # 높음
print(f"고양이 vs 자동차: {cosine_similarity(cat_vector, car_vector):.3f}")  # 낮음
```

#### 유클리드 거리 (Euclidean Distance / L2)

두 벡터 사이의 직선 거리. 값이 작을수록 비슷하다.

#### 내적 (Inner Product / Dot Product)

정규화된 벡터에서는 코사인 유사도와 동일. 속도가 빠르다.

### 2.3 k-NN vs ANN

!!! info "k-NN (k-Nearest Neighbors): k개의 가장 가까운 이웃"
    주어진 쿼리 벡터에서 가장 가까운 k개의 벡터를 찾는 알고리즘.

    **정확한 k-NN (Exact k-NN)**:
    - 모든 벡터와 거리를 하나하나 계산
    - 100% 정확하지만, 데이터가 많아지면 극도로 느려짐
    - 100만 개 벡터 × 1536차원 = 수십억 번의 곱셈 연산

    **ANN (Approximate Nearest Neighbors): 근사 최근접 이웃**:
    - 정확하지 않아도 되니까 빠르게 찾자는 접근
    - 99% 정도의 정확도로 1000배 이상 빠른 검색 가능
    - 실제 프로덕션에서는 거의 항상 ANN을 사용

!!! example "ANN 비유"
    서울에서 가장 가까운 편의점을 찾는다고 해보자.

    **정확한 k-NN**: 대한민국 모든 편의점의 GPS 좌표를 하나씩 확인
    → 정확하지만 오래 걸림

    **ANN**: "강남구 → 서초동 → 반포동" 순서로 좁혀가며 찾기
    → 99.9% 정확하고 훨씬 빠름

### 2.4 HNSW 알고리즘 (Hierarchical Navigable Small World)

현재 가장 널리 사용되는 ANN 알고리즘. pgvector, OpenSearch 모두 지원한다.

```
HNSW 구조 (계층적 그래프):

레이어 3 (희박):  [A] -------- [B]
                    \
레이어 2 (중간):  [A] -- [C] -- [B] -- [D]
                    \    /         \
레이어 1 (촘촘):  [A]-[E]-[C]-[F]-[B]-[G]-[D]-[H]
```

!!! info "HNSW 동작 원리"
    1. 데이터를 여러 레이어의 그래프로 구성
    2. 검색 시 가장 위(희박한) 레이어에서 시작
    3. 점점 아래(촘촘한) 레이어로 내려가며 정확도를 높임
    4. 마치 지도에서 국가 → 도시 → 동네 순으로 좁혀가는 것과 같음

    **주요 파라미터**:
    - `m`: 각 노드가 연결할 이웃 수 (기본값: 16, 높을수록 정확하지만 메모리 증가)
    - `ef_construction`: 인덱스 구축 시 탐색 범위 (기본값: 64, 높을수록 정확하지만 구축 느림)
    - `ef_search`: 검색 시 탐색 범위 (높을수록 정확하지만 검색 느림)

### 2.5 IVFFlat 알고리즘

HNSW의 대안. 대용량 데이터에서 메모리 효율이 좋다.

```
IVFFlat 구조 (클러스터링):

      중심점들
     /   |   \
  [C1] [C2] [C3]     ← 데이터를 클러스터로 나눔
  /|\   |   /|\
 벡터들...          ← 쿼리가 들어오면 가까운 클러스터만 검색
```

!!! info "IVFFlat 동작 원리"
    1. 학습 단계: 데이터를 N개의 클러스터로 나눔 (K-means 클러스터링)
    2. 검색 단계: 쿼리 벡터와 가장 가까운 클러스터 몇 개만 탐색
    3. 클러스터 수(`lists`): 많을수록 정확하지만 느림

    **주의**: HNSW보다 메모리는 적게 쓰지만 검색 속도는 느린 경우가 많다.
    소규모에서는 HNSW를 우선 사용하자.

### 2.6 BM25 — 키워드 검색의 왕

!!! info "BM25 (Best Match 25)"
    전통적인 텍스트 검색 알고리즘. Google이 나오기 전부터 사용된 방법으로,
    단어 빈도(TF)와 역문서 빈도(IDF)를 조합해 관련성을 계산한다.

    - **TF (Term Frequency)**: 문서 내에서 단어가 얼마나 자주 나오는가
    - **IDF (Inverse Document Frequency)**: 해당 단어가 전체 문서에서 얼마나 희귀한가

    "그리고", "이다" 같은 흔한 단어는 점수가 낮고,
    "HNSW", "pgvector" 같은 특수 용어는 점수가 높다.

    **BM25의 장점**: 정확한 키워드 매칭, 빠름, 특수 용어/코드/이름에 강함
    **BM25의 단점**: "반품"이라고 검색하면 "환불"이 나오지 않음 (의미를 모름)

### 2.7 하이브리드 검색 (Hybrid Search)

벡터 검색과 키워드 검색(BM25)의 장점을 합친 방법.

```
하이브리드 검색 = 벡터 검색 점수 × α + BM25 점수 × (1-α)

α = 0.0: 순수 키워드 검색 (정확한 단어 매칭)
α = 0.5: 50:50 균형
α = 1.0: 순수 벡터 검색 (의미 기반)
```

!!! example "하이브리드 검색이 필요한 상황"

    **벡터 검색만 쓸 때의 문제**:
    - "에러 코드 E-4012" 검색 → 벡터가 에러 코드의 정확한 숫자를 잘 구분 못 함
    - 고유명사(제품명, 사람 이름) 검색에 약함

    **키워드 검색만 쓸 때의 문제**:
    - "로그인이 안 돼요" 검색 → "인증 실패", "세션 만료" 같은 관련 문서를 못 찾음
    - 다른 표현, 동의어에 약함

    **하이브리드가 최고인 이유**:
    - "에러 코드 E-4012가 뭔가요?" → 키워드로 정확한 에러 코드 문서 찾고 + 벡터로 관련 해결책도 찾음

---

## 3. 주요 벡터 DB 비교

| 특성 | **OpenSearch** | **pgvector** | **Meilisearch** | **Chroma** |
|------|---------------|-------------|----------------|------------|
| 유형 | 검색 엔진 | PostgreSQL 확장 | 검색 엔진 | 전용 벡터 DB |
| 벡터 검색 | HNSW, IVFFlat | HNSW, IVFFlat | 벡터 검색 | HNSW |
| 전문 검색(BM25) | 매우 강력 | 제한적 | 매우 강력 | 없음 |
| 하이브리드 검색 | 네이티브 지원 | 수동 구현 필요 | 네이티브 지원 | 없음 |
| 설치 난이도 | 중간 | 낮음 | 매우 낮음 | 극히 낮음 |
| 확장성 | 매우 높음 | 중간 | 중간 | 낮음 |
| 프로덕션 적합성 | 매우 높음 | 높음 | 중간 | 낮음(프로토타입용) |
| 무료 여부 | 오픈소스 | 오픈소스 | 오픈소스 | 오픈소스 |
| 추천 사용 규모 | 대규모 | 소~중규모 | 소~중규모 | 프로토타입 |

---

## 4. OpenSearch

### 4.1 OpenSearch란?

!!! info "OpenSearch 소개"
    원래 Elasticsearch를 AWS가 포크(fork)해서 만든 오픈소스 검색 엔진이다.

    - **AWS OpenSearch Service**: 관리형 클라우드 서비스로도 제공
    - **Elasticsearch 호환 API**: 기존 Elasticsearch 코드 대부분 그대로 사용 가능
    - **벡터 검색 + 전문 검색**: 두 가지를 함께 지원하는 강력한 하이브리드 검색 엔진

    **언제 OpenSearch를 써야 하나?**
    - 수십만 ~ 수천만 개 이상의 대규모 문서
    - 벡터 검색 + 키워드 검색을 동시에 써야 할 때
    - 기업 환경에서 안정적인 운영이 필요할 때
    - AWS 인프라를 주로 사용할 때

### 4.2 왜 이게 중요한가?

!!! tip "왜 이게 중요한가?"
    OpenSearch는 RAG에서 "검색 엔진의 끝판왕"이다.

    - 하이브리드 검색을 한 번의 쿼리로 처리 (다른 DB는 두 번 검색 후 코드로 합침)
    - 수평 확장 가능 → 노드를 추가하면 성능이 선형으로 증가
    - 실시간 인덱싱 → 문서 추가 즉시 검색 가능
    - AWS 완전 관리형 서비스 → 운영 부담 없음

### 4.3 완전한 설치 가이드

```bash
# ① Docker로 OpenSearch 단일 노드 실행
docker run -d \
  --name opensearch \
  -p 9200:9200 \
  -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  -e "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m" \
  opensearchproject/opensearch:2.11.0

# ② 컨테이너가 정상 기동될 때까지 대기 (약 30~60초)
docker logs -f opensearch
# "Node] started" 메시지가 보이면 완료

# ③ 정상 동작 확인
curl -X GET "http://localhost:9200"
# 응답 예시:
# {
#   "name" : "opensearch",
#   "cluster_name" : "docker-cluster",
#   "version" : { "number" : "2.11.0", ... },
#   "tagline" : "The OpenSearch Project: https://opensearch.org/"
# }

# ④ 클러스터 상태 확인
curl -X GET "http://localhost:9200/_cluster/health?pretty"
# "status" : "green" 또는 "yellow" 이면 정상

# ⑤ Python 라이브러리 설치
pip install opensearch-py langchain-community openai
```

!!! warning "메모리 설정 주의"
    OpenSearch는 기본적으로 JVM 힙 메모리를 많이 사용한다.
    `-e "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m"` 옵션으로
    최소/최대 힙을 512MB로 제한했다.
    프로덕션에서는 서버 RAM의 50% 정도를 할당하는 것이 권장된다.

### 4.4 인덱스 생성 — 완전한 CRUD 예제

아래는 OpenSearch에서 문서를 저장하고 검색하는 완전한 예제다.

```python
# opensearch_crud.py
from opensearchpy import OpenSearch
import json

# ── 연결 설정 ───────────────────────────────────────────────
# OpenSearch 서버에 연결한다
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False,
    verify_certs=False
)

# 연결 확인
info = client.info()
print(f"OpenSearch 버전: {info['version']['number']}")
# 출력: OpenSearch 버전: 2.11.0

# ── CREATE: 인덱스(테이블) 생성 ──────────────────────────────
INDEX_NAME = "rag-documents"

# 먼저 기존 인덱스가 있으면 삭제 (개발 시 편의를 위해)
if client.indices.exists(index=INDEX_NAME):
    client.indices.delete(index=INDEX_NAME)
    print(f"기존 인덱스 '{INDEX_NAME}' 삭제 완료")

# 인덱스 설정 정의
# - settings: 인덱스 동작 방식 설정
# - mappings: 각 필드의 타입 정의 (RDB의 스키마와 유사)
index_body = {
    "settings": {
        "index": {
            "knn": True,            # k-NN(벡터 검색) 활성화
            "knn.algo_param.ef_search": 100  # 검색 시 탐색 범위
        }
    },
    "mappings": {
        "properties": {
            # 실제 텍스트 내용 — BM25 키워드 검색에 사용됨
            "content": {
                "type": "text",
                "analyzer": "standard"    # 텍스트를 단어 단위로 분석
            },
            # 임베딩 벡터 — 의미 검색에 사용됨
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,        # OpenAI text-embedding-3-small 차원
                "method": {
                    "name": "hnsw",       # 알고리즘: HNSW 사용
                    "space_type": "cosinesimil",  # 유사도 측정 방법: 코사인
                    "engine": "faiss",    # faiss 권장 (nmslib는 OpenSearch 2.19+에서 deprecated)
                    "parameters": {
                        "m": 16,          # 각 노드의 연결 수 (많을수록 정확, 메모리 증가)
                        "ef_construction": 64  # 인덱스 구축 시 탐색 범위
                    }
                }
            },
            # 소스 파일명
            "source": {"type": "keyword"},    # keyword: 정확한 값 매칭용
            # 페이지 번호
            "page": {"type": "integer"},
            # 문서 타입 (예: "manual", "faq", "policy")
            "doc_type": {"type": "keyword"},
            # 생성 시간
            "created_at": {"type": "date"}
        }
    }
}

client.indices.create(index=INDEX_NAME, body=index_body)
print(f"인덱스 '{INDEX_NAME}' 생성 완료!")
```

### 4.5 문서 저장(INSERT)

```python
# opensearch_insert.py
import numpy as np
from datetime import datetime
from opensearchpy import OpenSearch
from openai import OpenAI

openai_client = OpenAI()
os_client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False
)

def get_embedding(text: str) -> list[float]:
    """텍스트를 임베딩 벡터로 변환한다"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 예시 문서들
sample_documents = [
    {
        "content": "RAG(Retrieval-Augmented Generation)는 검색 증강 생성 기법으로, LLM이 외부 지식 베이스에서 관련 정보를 검색해 답변의 정확성을 높이는 방법이다.",
        "source": "rag_guide.pdf",
        "page": 1,
        "doc_type": "manual"
    },
    {
        "content": "벡터 데이터베이스는 임베딩 벡터를 저장하고 유사도 기반으로 빠르게 검색하는 특수 목적 데이터베이스다. HNSW 알고리즘을 사용해 수백만 개의 벡터도 밀리초 단위로 검색한다.",
        "source": "vector_db_overview.pdf",
        "page": 5,
        "doc_type": "technical"
    },
    {
        "content": "환불 정책: 구매 후 30일 이내에 미사용 상품은 전액 환불 가능합니다. 사용한 상품은 반품 불가하며, 불량품의 경우 구매일로부터 1년 이내 교환 가능합니다.",
        "source": "policy.pdf",
        "page": 12,
        "doc_type": "policy"
    },
]

# 각 문서에 임베딩을 생성하고 OpenSearch에 저장
for i, doc in enumerate(sample_documents):
    # 임베딩 생성 (텍스트 → 벡터)
    embedding = get_embedding(doc["content"])

    # OpenSearch에 저장할 문서 구성
    document = {
        "content": doc["content"],
        "embedding": embedding,          # 1536차원 벡터
        "source": doc["source"],
        "page": doc["page"],
        "doc_type": doc["doc_type"],
        "created_at": datetime.now().isoformat()
    }

    # 문서 저장 (_id를 지정하면 해당 ID로 저장됨)
    response = os_client.index(
        index="rag-documents",
        id=str(i + 1),      # 문서 ID (없으면 자동 생성)
        body=document
    )
    print(f"문서 {i+1} 저장: {response['result']}")
    # 출력: 문서 1 저장: created

# 인덱스 새로고침 (방금 저장한 문서를 즉시 검색 가능하게)
os_client.indices.refresh(index="rag-documents")
print("인덱스 새로고침 완료 — 이제 검색 가능!")
```

### 4.6 벡터 검색(READ)

```python
# opensearch_search.py
def vector_search(query_text: str, k: int = 3) -> list[dict]:
    """
    텍스트 쿼리를 받아 의미적으로 유사한 문서를 k개 반환한다.

    Args:
        query_text: 검색할 텍스트
        k: 반환할 문서 수

    Returns:
        유사도 순으로 정렬된 문서 목록
    """
    # ① 쿼리 텍스트를 벡터로 변환
    query_vector = get_embedding(query_text)

    # ② OpenSearch에 k-NN 쿼리 전송
    query_body = {
        "size": k,          # 반환할 결과 수
        "_source": ["content", "source", "page", "doc_type"],  # 반환할 필드
        "query": {
            "knn": {        # k-NN 벡터 검색
                "embedding": {
                    "vector": query_vector,   # 쿼리 벡터
                    "k": k                    # 가장 가까운 k개 반환
                }
            }
        }
    }

    results = os_client.search(index="rag-documents", body=query_body)

    # ③ 결과 파싱
    docs = []
    for hit in results["hits"]["hits"]:
        docs.append({
            "score": hit["_score"],          # 유사도 점수 (높을수록 유사)
            "content": hit["_source"]["content"],
            "source": hit["_source"]["source"],
            "doc_type": hit["_source"]["doc_type"]
        })

    return docs

# 실제 검색 실행
results = vector_search("반품하고 싶어요", k=3)

for i, doc in enumerate(results, 1):
    print(f"\n[{i}위] 유사도 점수: {doc['score']:.4f}")
    print(f"소스: {doc['source']}")
    print(f"내용: {doc['content'][:100]}...")

# 출력 예시:
# [1위] 유사도 점수: 0.9234
# 소스: policy.pdf
# 내용: 환불 정책: 구매 후 30일 이내에 미사용 상품은 전액 환불 가능합니다...
```

### 4.7 문서 업데이트(UPDATE)와 삭제(DELETE)

```python
# opensearch_update_delete.py

# ── UPDATE: 특정 문서 내용 업데이트 ─────────────────────────
def update_document(doc_id: str, new_content: str):
    """문서 내용과 임베딩을 함께 업데이트한다"""
    # 새 임베딩 생성 (내용이 바뀌었으므로 벡터도 새로 만들어야 함!)
    new_embedding = get_embedding(new_content)

    response = os_client.update(
        index="rag-documents",
        id=doc_id,
        body={
            "doc": {
                "content": new_content,
                "embedding": new_embedding,
                "updated_at": datetime.now().isoformat()
            }
        }
    )
    print(f"문서 {doc_id} 업데이트: {response['result']}")
    # 출력: 문서 1 업데이트: updated

update_document("1", "RAG는 검색 증강 생성(Retrieval-Augmented Generation)으로, LLM의 환각(Hallucination)을 줄이고 최신 정보를 활용할 수 있게 한다.")

# ── DELETE: 특정 문서 삭제 ───────────────────────────────────
def delete_document(doc_id: str):
    """특정 문서를 삭제한다"""
    response = os_client.delete(
        index="rag-documents",
        id=doc_id
    )
    print(f"문서 {doc_id} 삭제: {response['result']}")
    # 출력: 문서 3 삭제: deleted

delete_document("3")

# ── DELETE: 조건에 맞는 여러 문서 일괄 삭제 ─────────────────
def delete_by_source(source_name: str):
    """특정 소스 파일의 모든 문서를 삭제한다"""
    response = os_client.delete_by_query(
        index="rag-documents",
        body={
            "query": {
                "term": {"source": source_name}
            }
        }
    )
    print(f"'{source_name}' 문서 {response['deleted']}개 삭제 완료")

delete_by_source("policy.pdf")
```

### 4.8 하이브리드 검색 (벡터 + BM25)

OpenSearch의 진정한 강점. 한 번의 쿼리로 두 가지 검색을 동시에 실행한다.

```python
# opensearch_hybrid.py
def hybrid_search(
    query_text: str,
    k: int = 5,
    semantic_weight: float = 0.5  # 벡터:키워드 비율 (0~1)
) -> list[dict]:
    """
    벡터 검색과 BM25 키워드 검색을 결합한 하이브리드 검색.

    Args:
        query_text: 검색 쿼리
        k: 반환할 결과 수
        semantic_weight: 벡터 검색 가중치 (0=키워드만, 1=벡터만)

    Returns:
        하이브리드 점수로 정렬된 문서 목록
    """
    query_vector = get_embedding(query_text)
    keyword_weight = 1.0 - semantic_weight

    # OpenSearch 하이브리드 쿼리 구조
    query_body = {
        "size": k,
        "_source": ["content", "source", "doc_type"],
        "query": {
            "hybrid": {          # 하이브리드 쿼리 (OpenSearch 2.10+ 지원)
                "queries": [
                    # ① BM25 키워드 검색
                    {
                        "match": {
                            "content": {
                                "query": query_text,
                                "boost": keyword_weight   # 키워드 가중치
                            }
                        }
                    },
                    # ② 벡터 유사도 검색
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": k,
                                "boost": semantic_weight  # 벡터 가중치
                            }
                        }
                    }
                ]
            }
        }
    }

    results = os_client.search(index="rag-documents", body=query_body)
    return [
        {
            "score": hit["_score"],
            "content": hit["_source"]["content"],
            "source": hit["_source"]["source"]
        }
        for hit in results["hits"]["hits"]
    ]

# 상황별 검색 전략
print("=== 에러 코드 검색 (키워드 중심) ===")
results = hybrid_search("에러 E-4012 처리 방법", semantic_weight=0.2)
for r in results:
    print(f"  [{r['score']:.3f}] {r['content'][:80]}")

print("\n=== 의미 검색 (벡터 중심) ===")
results = hybrid_search("시스템이 갑자기 멈춰요", semantic_weight=0.8)
for r in results:
    print(f"  [{r['score']:.3f}] {r['content'][:80]}")
```

!!! warning "하이브리드 검색 가중치 설정"
    OpenSearch의 hybrid 쿼리에서 `boost`로는 정밀한 가중치 조절이 어렵습니다.
    프로덕션에서는 **Search Pipeline**의 `normalization-processor`를 사용하여
    가중치를 설정하는 것이 권장됩니다.

### 4.9 LangChain 연동

```python
# opensearch_langchain.py
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# OpenSearch 벡터스토어 연결
vectorstore = OpenSearchVectorSearch(
    opensearch_url="http://localhost:9200",
    index_name="rag-documents-lc",          # LangChain용 별도 인덱스
    embedding_function=embeddings,
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
)

# 문서 객체 생성 (LangChain Document 형식)
documents = [
    Document(
        page_content="RAG 파이프라인의 핵심은 검색의 정확도다.",
        metadata={"source": "guide.pdf", "page": 1, "type": "manual"}
    ),
    Document(
        page_content="벡터 데이터베이스는 의미 검색의 핵심 인프라다.",
        metadata={"source": "guide.pdf", "page": 2, "type": "manual"}
    ),
]

# 문서 일괄 추가 (내부적으로 임베딩 생성 후 저장)
vectorstore.add_documents(documents)
print("문서 추가 완료!")

# 유사도 검색 (가장 간단한 방법)
results = vectorstore.similarity_search("RAG가 뭔가요?", k=3)
for doc in results:
    print(f"\n내용: {doc.page_content}")
    print(f"메타데이터: {doc.metadata}")

# 유사도 점수와 함께 검색
results_with_scores = vectorstore.similarity_search_with_score("벡터 검색", k=3)
for doc, score in results_with_scores:
    print(f"\n[점수: {score:.4f}] {doc.page_content}")
```

### 4.10 성능 튜닝 팁

!!! tip "OpenSearch 성능 최적화"

    **인덱스 설정 최적화**:
    ```python
    # m과 ef_construction 튜닝
    # m이 클수록: 정확도↑, 메모리↑, 인덱싱 속도↓
    # ef_construction이 클수록: 정확도↑, 인덱싱 속도↓

    # 소규모 (~100만 문서): m=16, ef_construction=64 (기본값)
    # 대규모 (100만+):      m=32, ef_construction=128
    ```

    **샤드 설정**:
    ```python
    "settings": {
        "number_of_shards": 1,     # 소규모: 1개 샤드
        "number_of_replicas": 1,   # 복제본 1개 (고가용성)
        "refresh_interval": "30s"  # 실시간성보다 성능 우선 시
    }
    ```

    **검색 시 ef_search 조절**:
    ```python
    # ef_search: 검색 시 탐색 범위 (높을수록 정확하지만 느림)
    # 기본값: 512, 빠른 검색: 64, 정확한 검색: 1024
    query_body = {
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": 10,
                    "method_parameters": {
                        "ef_search": 256   # 검색별 튜닝 가능
                    }
                }
            }
        }
    }
    ```

### 4.11 장단점 정리

| 장점 | 단점 |
|------|------|
| 하이브리드 검색 네이티브 지원 | 설정이 다소 복잡함 |
| 수평 확장(클러스터링) 가능 | 메모리 사용량이 큼 (JVM) |
| AWS 완전 관리형 서비스 제공 | 단일 노드에서는 오버스펙일 수 있음 |
| Elasticsearch 생태계 호환 | 학습 곡선이 있음 |
| 강력한 집계/분석 기능 | 트랜잭션 미지원 |
| 실시간 인덱싱 | 소규모 프로젝트에 무거움 |

---

## 5. pgvector

### 5.1 pgvector란?

!!! info "pgvector 소개"
    PostgreSQL의 확장 모듈로, 일반 PostgreSQL에 벡터 검색 기능을 추가한다.

    - **기존 PostgreSQL 그대로 활용**: 새로운 DB를 추가하지 않아도 됨
    - **SQL로 벡터 검색**: 익숙한 SQL 문법으로 벡터 검색 가능
    - **트랜잭션, JOIN, 필터링**: RDB의 모든 기능과 벡터 검색을 함께 사용

    **언제 pgvector를 써야 하나?**
    - 이미 PostgreSQL을 사용 중인 프로젝트
    - 벡터 검색 + 복잡한 조건 필터링이 필요할 때
    - 트랜잭션 안전성이 필요할 때 (예: 결제 관련 문서)
    - 소~중규모 (수십만 문서까지)

### 5.2 왜 이게 중요한가?

!!! tip "왜 이게 중요한가?"
    많은 웹 서비스가 이미 PostgreSQL을 사용 중이다.
    pgvector를 쓰면 **기존 인프라에 벡터 검색을 추가**할 수 있다.

    별도의 벡터 DB 서버를 운영하지 않아도 되므로:
    - 운영 비용 절감
    - 데이터 일관성 유지 (같은 DB 트랜잭션 내에서 처리)
    - 익숙한 SQL로 복잡한 조건 검색 가능

### 5.3 완전한 설치 가이드

```bash
# ① Docker로 PostgreSQL + pgvector 실행
docker run -d \
  --name pgvector-db \
  -p 5432:5432 \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=ragpassword \
  -e POSTGRES_DB=ragdb \
  -v pgvector_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16

# ② 컨테이너 상태 확인
docker ps | grep pgvector-db
# CONTAINER ID   IMAGE                      STATUS    PORTS
# abc123456789   pgvector/pgvector:pg16     Up        0.0.0.0:5432->5432/tcp

# ③ PostgreSQL 접속 테스트
docker exec -it pgvector-db psql -U raguser -d ragdb -c "SELECT version();"
# 출력: PostgreSQL 16.x ...

# ④ pgvector 확장 확인
docker exec -it pgvector-db psql -U raguser -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
# 출력: CREATE EXTENSION

docker exec -it pgvector-db psql -U raguser -d ragdb -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
# extname | ... → vector 행이 보이면 성공

# ⑤ Python 라이브러리 설치
pip install psycopg2-binary pgvector langchain-community sqlalchemy openai
```

### 5.4 테이블 생성 및 CRUD 예제

```python
# pgvector_crud.py
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
from openai import OpenAI

openai_client = OpenAI()

# ── 데이터베이스 연결 ────────────────────────────────────────
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="ragdb",
    user="raguser",
    password="ragpassword"
)
# RealDictCursor: 결과를 딕셔너리 형태로 반환 (편의성)
cur = conn.cursor(cursor_factory=RealDictCursor)

# ── pgvector 확장 활성화 ─────────────────────────────────────
# 최초 1회만 실행하면 됨
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()
print("pgvector 확장 활성화 완료")

# ── CREATE: 테이블 생성 ──────────────────────────────────────
# vector(1536): 1536차원 벡터를 저장하는 컬럼 타입
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id          SERIAL PRIMARY KEY,
        content     TEXT NOT NULL,
        embedding   vector(1536),            -- 핵심: 벡터 타입
        source      VARCHAR(500),
        page        INTEGER,
        doc_type    VARCHAR(50),
        metadata    JSONB DEFAULT '{}',      -- 유연한 추가 데이터
        created_at  TIMESTAMP DEFAULT NOW(),
        updated_at  TIMESTAMP DEFAULT NOW()
    );
""")
conn.commit()
print("documents 테이블 생성 완료")

# ── HNSW 인덱스 생성 ────────────────────────────────────────
# 벡터 검색 속도를 크게 향상시키는 핵심 단계
# vector_cosine_ops: 코사인 유사도로 검색
cur.execute("""
    CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (
        m = 16,              -- 각 노드당 연결 수
        ef_construction = 64 -- 인덱스 구축 시 탐색 범위
    );
""")
conn.commit()
print("HNSW 인덱스 생성 완료 — 빠른 벡터 검색 준비됨")

def get_embedding(text: str) -> list[float]:
    """텍스트를 임베딩 벡터로 변환"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
```

### 5.5 데이터 삽입과 조회

```python
# pgvector_insert_query.py (위 코드 이어서)

# ── INSERT: 문서 저장 ────────────────────────────────────────
def insert_document(content: str, source: str, page: int, doc_type: str, metadata: dict = None) -> int:
    """
    문서를 임베딩과 함께 PostgreSQL에 저장한다.

    Returns:
        생성된 문서의 ID
    """
    # 임베딩 생성
    embedding = get_embedding(content)

    # pgvector는 Python 리스트를 벡터로 자동 변환
    # metadata는 JSON 문자열로 변환해서 저장
    cur.execute("""
        INSERT INTO documents (content, embedding, source, page, doc_type, metadata)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
    """, (
        content,
        embedding,                           # 리스트 → vector 자동 변환
        source,
        page,
        doc_type,
        json.dumps(metadata or {})
    ))

    result = cur.fetchone()
    conn.commit()
    doc_id = result["id"]
    print(f"문서 저장 완료 (ID: {doc_id})")
    return doc_id

# 샘플 데이터 삽입
id1 = insert_document(
    content="RAG는 검색 증강 생성 기법으로 LLM의 성능을 향상시킨다.",
    source="rag_intro.pdf",
    page=1,
    doc_type="manual",
    metadata={"author": "홍길동", "version": "1.0"}
)

id2 = insert_document(
    content="pgvector는 PostgreSQL에서 벡터 검색을 가능하게 하는 확장 모듈이다.",
    source="pgvector_guide.pdf",
    page=3,
    doc_type="technical"
)

id3 = insert_document(
    content="환불은 구매 후 30일 이내에만 가능하며, 사용한 제품은 환불 불가합니다.",
    source="refund_policy.pdf",
    page=1,
    doc_type="policy"
)

# ── SELECT: 기본 조회 ────────────────────────────────────────
# 전체 문서 개수 확인
cur.execute("SELECT COUNT(*) as total FROM documents;")
result = cur.fetchone()
print(f"\n저장된 문서 수: {result['total']}개")

# 특정 ID 문서 조회
cur.execute("SELECT id, content, source, doc_type FROM documents WHERE id = %s;", (id1,))
doc = cur.fetchone()
print(f"\n문서 {id1}: {doc['content'][:50]}...")
```

### 5.6 벡터 유사도 검색

```python
# pgvector_search.py
def vector_search(query_text: str, limit: int = 5) -> list[dict]:
    """
    코사인 유사도 기반 벡터 검색.

    pgvector 거리 연산자:
    - <=>  : 코사인 거리 (1 - 코사인 유사도) → 작을수록 유사
    - <->  : 유클리드 거리 (L2) → 작을수록 유사
    - <#>  : 내적 (negative inner product) → 작을수록 유사
    """
    query_vector = get_embedding(query_text)

    cur.execute("""
        SELECT
            id,
            content,
            source,
            doc_type,
            -- 코사인 거리를 유사도 점수로 변환 (1 - 거리 = 유사도)
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM documents
        ORDER BY embedding <=> %s::vector   -- 거리가 작은 순 (유사한 순)
        LIMIT %s;
    """, (query_vector, query_vector, limit))
    # 참고: query_vector를 두 번 전달하는 이유는 SELECT와 ORDER BY에 각각 사용하기 때문

    results = cur.fetchall()
    return [dict(row) for row in results]

# 검색 테스트
print("=== '반품하고 싶어요' 검색 결과 ===")
results = vector_search("반품하고 싶어요", limit=3)
for r in results:
    print(f"  [유사도: {r['similarity_score']:.4f}] {r['content'][:60]}")
    print(f"  소스: {r['source']} | 타입: {r['doc_type']}")
    print()

# 출력 예시:
# [유사도: 0.8923] 환불은 구매 후 30일 이내에만 가능하며...
# 소스: refund_policy.pdf | 타입: policy
```

### 5.7 SQL 필터링 + 벡터 검색 (pgvector의 최강 장점)

```python
# pgvector_filtered_search.py
def filtered_vector_search(
    query_text: str,
    doc_type: str = None,
    source: str = None,
    min_similarity: float = 0.7,
    limit: int = 5
) -> list[dict]:
    """
    SQL 조건 필터링과 벡터 검색을 동시에 수행.

    이것이 pgvector의 가장 큰 장점:
    벡터 DB로는 불가능한 복잡한 SQL 조건을 자유롭게 추가할 수 있다.
    """
    query_vector = get_embedding(query_text)

    # 동적 WHERE 조건 구성
    conditions = ["1=1"]  # 기본 조건 (항상 참)
    params = []

    if doc_type:
        conditions.append("doc_type = %s")
        params.append(doc_type)

    if source:
        conditions.append("source = %s")
        params.append(source)

    where_clause = " AND ".join(conditions)

    # 유사도 점수 필터링: HAVING 대신 서브쿼리 사용
    query = f"""
        SELECT *
        FROM (
            SELECT
                id,
                content,
                source,
                doc_type,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity_score
            FROM documents
            WHERE {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ) AS subq
        WHERE similarity_score >= %s   -- 최소 유사도 필터
        ORDER BY similarity_score DESC;
    """

    cur.execute(query, [query_vector] + params + [query_vector, limit, min_similarity])
    return [dict(row) for row in cur.fetchall()]

# 활용 예시 1: 정책 문서에서만 검색
print("=== 정책 문서에서만 검색 ===")
results = filtered_vector_search(
    "환불 가능한 기간",
    doc_type="policy",
    min_similarity=0.6
)
for r in results:
    print(f"  [유사도: {r['similarity_score']:.4f}] {r['content'][:60]}")

# 활용 예시 2: 날짜 범위 + 벡터 검색 (SQL의 자유도)
cur.execute("""
    SELECT
        id,
        content,
        1 - (embedding <=> %s::vector) AS similarity_score
    FROM documents
    WHERE doc_type = 'manual'
      AND created_at > '2024-01-01'
      AND metadata->>'version' = '1.0'    -- JSONB 조건도 가능!
    ORDER BY embedding <=> %s::vector
    LIMIT 5;
""", (query_vector, query_vector))
results = cur.fetchall()
print(f"\n조건 필터링 결과: {len(results)}개")
```

### 5.8 UPDATE와 DELETE

```python
# pgvector_update_delete.py

# ── UPDATE: 문서 업데이트 ────────────────────────────────────
def update_document(doc_id: int, new_content: str):
    """
    문서 내용을 업데이트하고, 임베딩도 새로 생성한다.
    내용이 바뀌면 반드시 임베딩도 새로 만들어야 검색이 정확해진다.
    """
    new_embedding = get_embedding(new_content)

    cur.execute("""
        UPDATE documents
        SET
            content = %s,
            embedding = %s,
            updated_at = NOW()
        WHERE id = %s;
    """, (new_content, new_embedding, doc_id))
    conn.commit()
    print(f"문서 {doc_id} 업데이트 완료")

update_document(
    id1,
    "RAG(Retrieval-Augmented Generation)는 LLM이 외부 지식 베이스를 참조해 더 정확한 답변을 생성하는 기법이다."
)

# ── DELETE: 단건 삭제 ────────────────────────────────────────
def delete_document(doc_id: int):
    cur.execute("DELETE FROM documents WHERE id = %s;", (doc_id,))
    conn.commit()
    print(f"문서 {doc_id} 삭제 완료")

# ── DELETE: 조건부 삭제 ──────────────────────────────────────
def delete_by_type(doc_type: str):
    cur.execute(
        "DELETE FROM documents WHERE doc_type = %s RETURNING id;",
        (doc_type,)
    )
    deleted_ids = [row[0] for row in cur.fetchall()]
    conn.commit()
    print(f"'{doc_type}' 타입 문서 {len(deleted_ids)}개 삭제: ID {deleted_ids}")

# 모두 실행
delete_document(id3)
```

### 5.9 LangChain 연동

!!! warning "LangChain PGVector 패키지 변경"
    `langchain_community.vectorstores.PGVector`는 deprecated 되었습니다.
    신규 프로젝트에서는 `langchain-postgres` 패키지 사용이 권장됩니다:
    ```bash
    pip install langchain-postgres
    ```
    ```python
    from langchain_postgres.vectorstores import PGVector  # 신규 권장 방식
    ```

```python
# pgvector_langchain.py
from langchain_community.vectorstores import PGVector  # deprecated: langchain-postgres 패키지 권장
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 연결 문자열 형식: postgresql://유저:비밀번호@호스트:포트/DB명
CONNECTION_STRING = "postgresql://raguser:ragpassword@localhost:5432/ragdb"

# 임베딩 함수
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# PGVector 벡터스토어 초기화
# collection_name: PostgreSQL 내에서 사용할 컬렉션(테이블) 이름
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name="langchain_docs",    # 자동으로 테이블 생성됨
    pre_delete_collection=False,         # True이면 기존 데이터 초기화
)

# 문서 추가
docs = [
    Document(
        page_content="LangChain은 LLM 애플리케이션 개발 프레임워크다.",
        metadata={"source": "langchain_intro.pdf", "page": 1}
    ),
    Document(
        page_content="pgvector를 LangChain과 연동하면 RAG 파이프라인을 쉽게 구성할 수 있다.",
        metadata={"source": "pgvector_guide.pdf", "page": 5}
    ),
]

ids = vectorstore.add_documents(docs)
print(f"추가된 문서 ID: {ids}")

# 검색
results = vectorstore.similarity_search("LLM 개발 도구", k=2)
for doc in results:
    print(f"\n내용: {doc.page_content}")
    print(f"메타데이터: {doc.metadata}")

# 검색 + 필터 (메타데이터로 필터링)
results = vectorstore.similarity_search(
    "벡터 검색",
    k=2,
    filter={"source": "pgvector_guide.pdf"}  # 특정 소스만 검색
)
```

### 5.10 성능 튜닝 팁

!!! tip "pgvector 성능 최적화"

    **HNSW vs IVFFlat 선택**:
    ```sql
    -- HNSW (권장): 검색 빠름, 메모리 많이 사용
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

    -- IVFFlat: 메모리 적음, 검색 느림 (대용량 데이터 시 고려)
    -- lists: 클러스터 수 (데이터 수의 제곱근이 권장값)
    CREATE INDEX ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);   -- 10만 문서면 lists=316 추천

    -- ⚠️ IVFFlat 주의: 인덱스 생성 후 반드시 VACUUM을 실행해야
    -- 클러스터 학습 데이터가 확정되어 검색이 정상 동작합니다.
    -- VACUUM documents;
    ```

    **검색 시 ef_search 조절**:
    ```sql
    -- 세션별 ef_search 설정 (높을수록 정확하지만 느림)
    SET hnsw.ef_search = 100;  -- 기본값: 40
    ```

    **인덱스 없이 검색하는 경우**:
    ```sql
    -- 인덱스 구축 전 데이터 벌크 삽입 (속도 향상)
    -- 1. 인덱스 없이 INSERT
    -- 2. 모든 데이터 삽입 완료 후 인덱스 생성
    -- (역순으로 하면 매 INSERT마다 인덱스 갱신으로 느려짐)
    ```

### 5.11 장단점 정리

| 장점 | 단점 |
|------|------|
| 기존 PostgreSQL 인프라 재사용 | 대규모 데이터에서 성능 한계 |
| SQL로 복잡한 조건 필터링 | 하이브리드 검색은 수동 구현 필요 |
| 트랜잭션 안전성 보장 | 분산 처리 어려움 |
| JOIN으로 다른 테이블과 결합 | 전용 벡터 DB보다 기능 제한 |
| 낮은 운영 복잡도 | ef_search 등 수동 튜닝 필요 |
| JSONB 메타데이터 강력 지원 | 클러스터 구성이 복잡 |

---

## 6. Meilisearch

### 6.1 Meilisearch란?

!!! info "Meilisearch 소개"
    Rust로 작성된 초고속 오픈소스 검색 엔진. 설정이 거의 없이 바로 사용할 수 있다.

    - **오타 허용(Typo-tolerance)**: "파이썬" 대신 "파이쏜"이라고 써도 찾아줌
    - **한국어 지원**: 형태소 분석기 내장
    - **벡터 + 키워드 하이브리드**: 간단한 파라미터 하나로 비율 조절
    - **REST API**: 어느 언어에서든 HTTP로 사용 가능

    **언제 Meilisearch를 써야 하나?**
    - 빠른 프로토타입이나 MVP 개발
    - 사용자 대면 검색 기능 (빠른 응답, 오타 허용)
    - 중소규모 데이터 (수십만 문서 수준)
    - 설정에 시간 쓰기 싫을 때

### 6.2 왜 이게 중요한가?

!!! tip "왜 이게 중요한가?"
    스타트업이나 개인 프로젝트에서 OpenSearch 같은 복잡한 시스템은 부담스럽다.
    Meilisearch는 Docker 명령 하나로 실행하고, 5분 안에 검색 기능을 붙일 수 있다.

    오타 허용은 실제 사용자 경험에서 매우 중요하다.
    사용자들은 오타를 자주 내고, 이걸 잡아주지 않으면 "아무것도 안 나와요"라는 불만이 생긴다.

### 6.3 완전한 설치 가이드

```bash
# ① Docker로 Meilisearch 실행
docker run -d \
  --name meilisearch \
  -p 7700:7700 \
  -e MEILI_MASTER_KEY="my-super-secret-key-change-this" \
  -e MEILI_ENV="development" \
  -v meili_data:/meili_data \
  getmeili/meilisearch:v1.6

# ② 실행 확인 (약 5~10초 후)
curl http://localhost:7700/health
# 응답: {"status":"available"}

# ③ 버전 확인 (인증 키 필요)
curl -H "Authorization: Bearer my-super-secret-key-change-this" \
     http://localhost:7700/version
# 응답: {"pkgVersion":"1.6.0", ...}

# ④ Python 라이브러리 설치
pip install meilisearch openai

# ⑤ Meilisearch Dashboard 접속 (브라우저)
# http://localhost:7700 → Master Key 입력 → GUI로 데이터 확인 가능!
```

!!! warning "벡터 검색 활성화 필수"
    Meilisearch의 벡터 검색은 실험적 기능이므로, Docker 실행 후 반드시 활성화해야 합니다:
    ```bash
    curl -X PATCH 'http://localhost:7700/experimental-features/' \
      -H 'Content-Type: application/json' \
      -H 'Authorization: Bearer your-master-key' \
      --data-binary '{"vectorStore": true}'
    ```

!!! tip "Meilisearch 대시보드"
    `http://localhost:7700`에 접속하면 웹 GUI가 열린다.
    Master Key를 입력하면 저장된 문서를 직접 보고 검색을 테스트할 수 있다.
    개발 중 매우 유용하다!

### 6.4 인덱스 설정 및 문서 추가

```python
# meilisearch_crud.py
import meilisearch
import time
from openai import OpenAI

openai_client = OpenAI()
ms_client = meilisearch.Client(
    "http://localhost:7700",
    "my-super-secret-key-change-this"
)

def get_embedding(text: str) -> list[float]:
    """텍스트를 임베딩 벡터로 변환"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ── CREATE: 인덱스 생성 ──────────────────────────────────────
# Meilisearch에서 인덱스는 RDB의 테이블과 비슷한 개념
index = ms_client.create_index(
    "documents",
    {"primaryKey": "id"}   # 문서를 고유하게 식별하는 필드
)
print("인덱스 생성 완료")

# ── 벡터 검색 설정 ───────────────────────────────────────────
# "embedders" 설정으로 벡터 검색 기능 활성화
task = ms_client.index("documents").update_settings({
    # 검색 가능한 텍스트 필드
    "searchableAttributes": ["content", "source", "doc_type"],

    # 결과에서 필터로 사용 가능한 필드
    "filterableAttributes": ["doc_type", "source", "page"],

    # 정렬 가능한 필드
    "sortableAttributes": ["page", "created_at"],

    # 임베더(벡터 생성기) 설정
    "embedders": {
        "default": {
            "source": "userProvided",  # 우리가 직접 벡터를 제공
            "dimensions": 1536          # 벡터 차원 수
        }
    }
})

# Meilisearch는 비동기로 설정을 처리하므로 완료까지 대기
task_result = ms_client.wait_for_task(task.task_uid)
print(f"설정 완료: {task_result.status}")  # succeeded

# ── INSERT: 문서 추가 ────────────────────────────────────────
sample_docs = [
    {"id": 1, "content": "RAG는 검색 증강 생성 기법으로 LLM의 정확도를 높인다.", "source": "rag_guide.pdf", "page": 1, "doc_type": "manual"},
    {"id": 2, "content": "Meilisearch는 초고속 오픈소스 검색 엔진으로 설정이 간단하다.", "source": "search_guide.pdf", "page": 5, "doc_type": "technical"},
    {"id": 3, "content": "환불 정책: 30일 이내 미사용 제품에 한해 전액 환불 가능합니다.", "source": "policy.pdf", "page": 1, "doc_type": "policy"},
    {"id": 4, "content": "벡터 데이터베이스는 임베딩 벡터를 저장하고 유사도 검색을 수행한다.", "source": "vector_db.pdf", "page": 2, "doc_type": "technical"},
]

# 각 문서에 임베딩 추가 후 일괄 업로드
documents_with_vectors = []
for doc in sample_docs:
    embedding = get_embedding(doc["content"])
    doc_with_vector = {
        **doc,
        "_vectors": {
            "default": embedding    # "_vectors" 특수 필드에 벡터 저장
        }
    }
    documents_with_vectors.append(doc_with_vector)

# 일괄 추가 (batch 처리가 효율적)
task = ms_client.index("documents").add_documents(documents_with_vectors)
task_result = ms_client.wait_for_task(task.task_uid)
print(f"문서 추가: {task_result.status}")
print(f"추가된 문서 수: {ms_client.index('documents').get_stats().number_of_documents}")
```

### 6.5 검색 예제

```python
# meilisearch_search.py

# ── 기본 키워드 검색 ─────────────────────────────────────────
def keyword_search(query: str, limit: int = 5) -> dict:
    """BM25 기반 키워드 검색"""
    results = ms_client.index("documents").search(
        query,
        {
            "limit": limit,
            "attributesToHighlight": ["content"],  # 검색어 하이라이트
        }
    )
    return results

# ── 벡터 검색 ────────────────────────────────────────────────
def semantic_search(query: str, limit: int = 5) -> dict:
    """순수 벡터(의미) 검색"""
    query_vector = get_embedding(query)
    results = ms_client.index("documents").search(
        "",      # 키워드는 비움 (순수 벡터 검색)
        {
            "vector": query_vector,
            "hybrid": {
                "semanticRatio": 1.0,  # 100% 벡터 검색
                "embedder": "default"
            },
            "limit": limit
        }
    )
    return results

# ── 하이브리드 검색 (핵심 기능) ─────────────────────────────
def hybrid_search(query: str, semantic_ratio: float = 0.5, limit: int = 5) -> dict:
    """
    키워드 검색 + 벡터 검색 하이브리드.

    Args:
        semantic_ratio: 벡터 검색 비율
            0.0 = 키워드만
            0.5 = 50:50 (균형)
            1.0 = 벡터만
    """
    query_vector = get_embedding(query)

    results = ms_client.index("documents").search(
        query,
        {
            "vector": query_vector,
            "hybrid": {
                "semanticRatio": semantic_ratio,
                "embedder": "default"
            },
            "limit": limit
        }
    )
    return results

# ── 검색 결과 출력 함수 ──────────────────────────────────────
def print_results(results: dict, title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"총 {results.get('estimatedTotalHits', 0)}개 결과")
    for i, hit in enumerate(results["hits"], 1):
        score = hit.get("_rankingScore", "N/A")
        print(f"\n[{i}위] 점수: {score}")
        print(f"  내용: {hit['content'][:70]}")
        print(f"  소스: {hit['source']} | 타입: {hit['doc_type']}")

# 실제 검색 테스트
print_results(keyword_search("RAG"), "키워드 검색: 'RAG'")
print_results(semantic_search("반품하고 싶어요"), "의미 검색: '반품하고 싶어요'")
print_results(hybrid_search("검색 시스템 구축", 0.5), "하이브리드 검색: '검색 시스템 구축'")
```

### 6.6 필터링과 정렬

```python
# meilisearch_filter.py

# ── 필터 + 하이브리드 검색 ──────────────────────────────────
def filtered_hybrid_search(
    query: str,
    doc_type: str = None,
    source: str = None,
    semantic_ratio: float = 0.5,
    limit: int = 5
) -> dict:
    """조건 필터링과 하이브리드 검색 결합"""
    query_vector = get_embedding(query)

    # 필터 조건 구성 (Meilisearch 필터 문법)
    filters = []
    if doc_type:
        filters.append(f'doc_type = "{doc_type}"')
    if source:
        filters.append(f'source = "{source}"')

    filter_str = " AND ".join(filters) if filters else None

    search_params = {
        "vector": query_vector,
        "hybrid": {
            "semanticRatio": semantic_ratio,
            "embedder": "default"
        },
        "limit": limit
    }

    if filter_str:
        search_params["filter"] = filter_str

    return ms_client.index("documents").search(query, search_params)

# 활용 예시
results = filtered_hybrid_search(
    "규정 알려주세요",
    doc_type="policy",      # 정책 문서만 검색
    semantic_ratio=0.7      # 의미 검색 70%
)
print_results(results, "정책 문서에서 '규정 알려주세요' 검색")
```

### 6.7 UPDATE와 DELETE

```python
# meilisearch_update_delete.py

# ── UPDATE: 문서 업데이트 ────────────────────────────────────
def update_document(doc_id: int, new_content: str):
    """문서 내용과 임베딩을 업데이트한다"""
    new_embedding = get_embedding(new_content)

    # Meilisearch에서 업데이트는 전체 필드를 다시 전송
    task = ms_client.index("documents").update_documents([
        {
            "id": doc_id,
            "content": new_content,
            "_vectors": {"default": new_embedding}
        }
    ])
    ms_client.wait_for_task(task.task_uid)
    print(f"문서 {doc_id} 업데이트 완료")

update_document(1, "RAG(Retrieval-Augmented Generation)는 LLM의 환각을 줄이고 최신 정보를 반영하는 핵심 기술이다.")

# ── DELETE: 단건 삭제 ────────────────────────────────────────
task = ms_client.index("documents").delete_document(3)
ms_client.wait_for_task(task.task_uid)
print("문서 3 삭제 완료")

# ── DELETE: 조건부 삭제 ──────────────────────────────────────
task = ms_client.index("documents").delete_documents_by_filter(
    'doc_type = "policy"'   # 정책 문서 전체 삭제
)
ms_client.wait_for_task(task.task_uid)
print("policy 타입 문서 전체 삭제 완료")
```

### 6.8 오타 허용 테스트

```python
# Meilisearch의 자랑: 오타 허용(Typo Tolerance)

# 정상 검색
results = ms_client.index("documents").search("meilisearch")
print(f"정상 검색 결과: {len(results['hits'])}개")

# 오타가 있어도 찾아줌!
results = ms_client.index("documents").search("meiisearch")   # 오타
print(f"오타 검색 결과: {len(results['hits'])}개")  # 여전히 찾아줌

results = ms_client.index("documents").search("벡터데이타베이스")  # "데이터" 오타
print(f"오타 검색 결과: {len(results['hits'])}개")  # 찾아줌
```

### 6.9 장단점 정리

| 장점 | 단점 |
|------|------|
| 설치와 설정이 매우 간단 | 대규모 데이터에서 한계 |
| 오타 허용 기본 내장 | 트랜잭션 미지원 |
| 한국어 지원 | SQL 같은 복잡한 쿼리 불가 |
| 하이브리드 검색 파라미터 하나로 조절 | OpenSearch보다 확장성 낮음 |
| 웹 GUI 대시보드 제공 | 클러스터링 미지원 |
| 매우 빠른 키워드 검색 | 고급 집계/분석 기능 없음 |

---

## 7. Chroma (보너스: 프로토타입의 왕)

### 7.1 Chroma란?

!!! info "Chroma 소개"
    Python으로 만든 전용 벡터 데이터베이스. 설치가 `pip install chromadb` 하나로 끝난다.

    - **인메모리 모드**: DB 서버 없이 메모리에서 바로 실행 (테스트용)
    - **영구 저장 모드**: 로컬 파일에 저장 (소규모 프로젝트)
    - **서버 모드**: Docker로 실행해서 API로 접근
    - **LangChain 완벽 지원**: 가장 많이 쓰이는 LangChain 벡터스토어

    **언제 Chroma를 써야 하나?**
    - 빠른 프로토타이핑, 튜토리얼, 학습 목적
    - 혼자 쓰는 소규모 프로젝트
    - 프로덕션 배포 전 로컬 테스트

!!! warning "Chroma의 한계"
    Chroma는 프로덕션 환경에서는 권장하지 않는다.
    대규모 데이터, 고가용성, 분산 처리가 필요하면 OpenSearch나 pgvector를 사용하자.

### 7.2 설치 및 기본 사용법

```bash
# 설치 (Docker 없이!)
pip install chromadb langchain-community openai
```

```python
# chroma_basic.py
import chromadb
from openai import OpenAI

openai_client = OpenAI()

# ── 모드 1: 인메모리 (테스트용, 프로그램 종료 시 데이터 사라짐) ──
client = chromadb.Client()  # 구 API; 신규 API는 chromadb.EphemeralClient() 권장

# ── 모드 2: 영구 저장 (로컬 파일에 저장) ──
# client = chromadb.PersistentClient(path="./chroma_db")

# ── 컬렉션 생성 (인덱스와 유사) ──────────────────────────────
# cosine: 코사인 유사도 사용
collection = client.get_or_create_collection(
    name="rag_documents",
    metadata={"hnsw:space": "cosine"}   # 유사도 측정 방법
)

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ── 문서 추가 ────────────────────────────────────────────────
documents = [
    "RAG는 검색 증강 생성으로 LLM의 정확도를 높인다.",
    "Chroma는 Python 전용 벡터 데이터베이스다.",
    "환불은 구매 후 30일 이내에만 가능하다.",
]

embeddings = [get_embedding(doc) for doc in documents]

collection.add(
    ids=["doc1", "doc2", "doc3"],           # 고유 ID (필수)
    documents=documents,                     # 원본 텍스트
    embeddings=embeddings,                   # 벡터
    metadatas=[                             # 메타데이터 (선택)
        {"source": "rag_guide.pdf", "type": "manual"},
        {"source": "chroma_guide.pdf", "type": "technical"},
        {"source": "policy.pdf", "type": "policy"},
    ]
)
print(f"총 {collection.count()}개 문서 저장 완료")

# ── 벡터 검색 ────────────────────────────────────────────────
query = "반품하고 싶어요"
query_embedding = get_embedding(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,                              # 반환할 결과 수
    include=["documents", "metadatas", "distances"]  # 포함할 정보
)

print(f"\n쿼리: '{query}'")
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
), 1):
    # Chroma는 거리(distance)를 반환 (작을수록 유사)
    similarity = 1 - dist   # 코사인 유사도로 변환
    print(f"\n[{i}위] 유사도: {similarity:.4f}")
    print(f"  내용: {doc}")
    print(f"  소스: {meta['source']}")

# ── 메타데이터 필터 검색 ─────────────────────────────────────
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    where={"type": "policy"}    # 정책 문서만 검색
)
print(f"\n정책 문서 필터 검색 결과: {len(results['documents'][0])}개")

# ── UPDATE ───────────────────────────────────────────────────
collection.update(
    ids=["doc1"],
    documents=["RAG(Retrieval-Augmented Generation)는 LLM 성능을 높이는 핵심 기술이다."],
    embeddings=[get_embedding("RAG(Retrieval-Augmented Generation)는 LLM 성능을 높이는 핵심 기술이다.")]
)
print("doc1 업데이트 완료")

# ── DELETE ───────────────────────────────────────────────────
collection.delete(ids=["doc3"])
print(f"doc3 삭제 완료, 남은 문서: {collection.count()}개")
```

### 7.3 LangChain + Chroma (가장 간단한 RAG 세팅)

```python
# chroma_langchain.py
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 임베딩 함수
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma 벡터스토어 (로컬 저장)
vectorstore = Chroma(
    collection_name="langchain_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"  # 영구 저장 경로
)

# 문서 추가
docs = [
    Document(page_content="RAG 파이프라인의 핵심은 검색 단계이다.", metadata={"source": "guide.pdf"}),
    Document(page_content="Chroma는 LangChain과 가장 잘 통합된 벡터 DB다.", metadata={"source": "chroma.pdf"}),
]
vectorstore.add_documents(docs)

# 검색
results = vectorstore.similarity_search("검색 파이프라인", k=2)
for doc in results:
    print(f"내용: {doc.page_content}")

# 리트리버로 변환 (RAG 파이프라인 연결용)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
retrieved = retriever.invoke("RAG란?")
print(f"\n리트리버 결과: {len(retrieved)}개 문서")
```

---

## 8. 직접 해보기 (Hands-on Exercise)

### 실습 1: 나만의 RAG 검색 시스템 만들기

!!! example "직접 해보기 — 실습 1"
    아래 코드를 복사해서 실행해보자. 어떤 벡터 DB도 없이 Chroma + OpenAI로
    5분 안에 작동하는 RAG 검색 시스템을 만들 수 있다.

```python
# hands_on_exercise_1.py
"""
실습: 나만의 FAQ 검색 시스템
목표: 사용자 질문 → 관련 FAQ 찾기 → 답변 생성

설치: pip install chromadb openai langchain langchain-openai langchain-community
"""
import chromadb
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────────────────────
# 1단계: FAQ 데이터 준비 (실제 프로젝트에서는 파일에서 읽어옴)
# ─────────────────────────────────────────────────────────────
faq_data = [
    {
        "question": "배송은 얼마나 걸리나요?",
        "answer": "일반 배송은 2~3 영업일, 빠른 배송은 당일~익일 도착합니다."
    },
    {
        "question": "환불 정책이 어떻게 되나요?",
        "answer": "구매 후 30일 이내 미사용 제품은 전액 환불 가능합니다. 사용한 제품은 환불 불가합니다."
    },
    {
        "question": "회원 탈퇴는 어떻게 하나요?",
        "answer": "설정 → 계정 관리 → 탈퇴 신청에서 진행할 수 있습니다. 탈퇴 후 30일간 재가입이 불가합니다."
    },
    {
        "question": "비밀번호를 잊어버렸어요",
        "answer": "로그인 화면에서 '비밀번호 찾기'를 클릭하고 이메일로 재설정 링크를 받으세요."
    },
    {
        "question": "결제 수단은 어떤 것이 지원되나요?",
        "answer": "신용카드, 체크카드, 카카오페이, 네이버페이, 무통장입금을 지원합니다."
    },
]

# ─────────────────────────────────────────────────────────────
# 2단계: 문서 임베딩 및 벡터스토어 저장
# ─────────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# FAQ를 Document 객체로 변환
documents = [
    Document(
        # 질문과 답변을 합쳐서 임베딩 (더 좋은 검색 결과)
        page_content=f"Q: {item['question']}\nA: {item['answer']}",
        metadata={"question": item["question"], "answer": item["answer"]}
    )
    for item in faq_data
]

# Chroma에 저장 (처음 실행 시 임베딩 생성, 이후 재사용)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./faq_chroma_db"
)
print(f"FAQ {len(documents)}개 저장 완료!")

# ─────────────────────────────────────────────────────────────
# 3단계: RAG 파이프라인 구성
# ─────────────────────────────────────────────────────────────
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_template("""
당신은 친절한 고객 서비스 담당자입니다.
아래 FAQ 내용을 참고해서 사용자의 질문에 답변해주세요.
FAQ에 없는 내용은 "해당 내용은 FAQ에서 찾을 수 없습니다"라고 답변하세요.

FAQ 내용:
{context}

사용자 질문: {question}

답변:
""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# RAG 체인 구성
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ─────────────────────────────────────────────────────────────
# 4단계: 실제 질문 테스트
# ─────────────────────────────────────────────────────────────
test_questions = [
    "물건을 반품하고 싶어요",           # "환불"과 다른 표현 → 벡터 검색이 연결
    "배송이 언제 오나요?",              # 직접 매칭
    "패스워드 재설정 방법 알려주세요",   # "비밀번호"와 "패스워드" 연결
    "암호화폐로 결제 가능한가요?",       # FAQ에 없는 질문
]

for q in test_questions:
    print(f"\n{'='*50}")
    print(f"질문: {q}")
    print(f"{'='*50}")
    answer = rag_chain.invoke(q)
    print(f"답변: {answer}")
```

### 실습 2: 하이브리드 검색 비교 실험

!!! example "직접 해보기 — 실습 2"
    벡터 검색과 키워드 검색, 하이브리드 검색의 차이를 직접 비교해보자.

```python
# hands_on_exercise_2.py
"""
실습: 검색 방식별 결과 비교
목표: 같은 쿼리로 세 가지 검색 방식의 결과가 어떻게 다른지 확인
"""
# Meilisearch 설치 후 실행 (위 설치 가이드 참고)
import meilisearch
from openai import OpenAI

openai_client = OpenAI()
ms_client = meilisearch.Client("http://localhost:7700", "my-super-secret-key-change-this")

# 기술 문서 샘플
tech_docs = [
    {"id": 1, "content": "Python은 데이터 과학에서 가장 많이 사용되는 프로그래밍 언어다.", "category": "python"},
    {"id": 2, "content": "NumPy는 Python의 수치 계산 라이브러리로 행렬 연산을 지원한다.", "category": "python"},
    {"id": 3, "content": "에러 코드 E-1001은 데이터베이스 연결 실패를 의미한다.", "category": "error"},
    {"id": 4, "content": "에러 코드 E-2048은 메모리 부족 오류다.", "category": "error"},
    {"id": 5, "content": "JavaScript는 웹 프론트엔드 개발의 표준 언어다.", "category": "javascript"},
    {"id": 6, "content": "코딩을 시작하려면 Python이나 JavaScript를 추천한다.", "category": "beginner"},
]

# 인덱스 설정 및 문서 추가 (이전 코드 참고)
# ... (설정 코드 생략 - 위의 Meilisearch 설치 가이드 참고)

# 실험용 쿼리들
test_queries = [
    {
        "query": "E-2048",                     # 정확한 키워드 → 키워드 검색이 유리
        "description": "에러 코드 검색 (정확한 용어)"
    },
    {
        "query": "프로그래밍 입문",              # 의미 검색 → 벡터 검색이 유리
        "description": "의미 기반 검색"
    },
    {
        "query": "파이썬 수학 계산",             # 중간 케이스 → 하이브리드가 유리
        "description": "혼합 검색"
    },
]

def compare_search_methods(query: str, description: str):
    query_vector = [get_embedding(query)]  # get_embedding 함수는 위에서 정의됨

    print(f"\n{'='*60}")
    print(f"쿼리: '{query}' ({description})")
    print(f"{'='*60}")

    for method, ratio in [("키워드만", 0.0), ("하이브리드", 0.5), ("벡터만", 1.0)]:
        results = ms_client.index("tech_docs").search(
            query,
            {
                "vector": query_vector[0],
                "hybrid": {"semanticRatio": ratio, "embedder": "default"},
                "limit": 2
            }
        )
        print(f"\n  [{method} (ratio={ratio})]")
        for hit in results["hits"]:
            print(f"    - {hit['content'][:50]}...")

for test in test_queries:
    compare_search_methods(test["query"], test["description"])
```

---

## 9. 성능 비교 (Benchmarks)

!!! info "성능 비교 기준"
    아래 수치는 일반적인 참고용이며, 실제 환경(서버 사양, 데이터 특성, 쿼리 패턴)에 따라 크게 달라진다.

### 9.1 검색 속도 비교

| 벡터 DB | 10만 문서 (ms) | 100만 문서 (ms) | 비고 |
|---------|--------------|----------------|------|
| Chroma (메모리) | 1~5 | N/A (비권장) | 단일 서버, 인메모리 |
| pgvector (HNSW) | 2~10 | 10~50 | 인덱스 설정 필수 |
| Meilisearch | 1~5 | 10~30 | 키워드 검색 특히 빠름 |
| OpenSearch | 5~20 | 10~30 | 분산 처리로 확장 가능 |

### 9.2 인덱싱 속도 비교

| 벡터 DB | 초당 문서 수 | 인덱스 크기 (10만 문서, 1536차원) |
|---------|------------|--------------------------------|
| Chroma | 500~1000 | ~2.5 GB |
| pgvector (HNSW) | 200~500 | ~3.5 GB |
| Meilisearch | 1000~3000 | ~3 GB |
| OpenSearch | 500~2000 | ~3 GB |

### 9.3 검색 품질 (Recall@10)

!!! info "Recall@10이란?"
    정확한 상위 10개 결과 중에서 ANN이 얼마나 많이 찾아내는지의 비율.
    100%이면 완벽, 낮을수록 부정확한 결과가 섞임.

| 알고리즘 | Recall@10 | 검색 속도 |
|---------|----------|---------|
| 정확한 k-NN (Exact) | 100% | 매우 느림 |
| HNSW (m=16, ef=64) | ~95% | 빠름 |
| HNSW (m=32, ef=128) | ~99% | 중간 |
| IVFFlat (lists=100) | ~90% | 중간 |

### 9.4 하이브리드 검색 효과

실제 RAG 시스템에서 측정한 검색 품질 (정답 포함 여부 기준):

```
순수 BM25 키워드 검색:  62% 정답 포함
순수 벡터 검색:         71% 정답 포함
하이브리드 (0.5:0.5):   79% 정답 포함  ← 최고
```

!!! success "결론"
    **하이브리드 검색이 대부분의 경우 최고 성능을 낸다.**
    단, 특수 용어/코드 검색은 키워드 비중을 높이고,
    자연어 질문은 벡터 비중을 높이는 것이 좋다.

---

## 10. 선택 가이드

### 10.1 결정 트리

```
어떤 벡터 DB를 선택해야 할까?
│
├─ 지금 당장 빠르게 테스트/프로토타입만 필요한가?
│   └─ YES → Chroma (pip install chromadb 하나로 끝)
│
├─ 이미 PostgreSQL을 사용 중인가?
│   └─ YES → pgvector (추가 인프라 없이 확장)
│       ├─ 복잡한 SQL 필터 필요? → pgvector 확정
│       └─ 트랜잭션 안전성 필요? → pgvector 확정
│
├─ 데이터 규모가 100만 문서 이상인가?
│   └─ YES → OpenSearch (분산 처리, 수평 확장)
│
├─ 빠른 설정으로 하이브리드 검색이 필요한가?
│   └─ YES → Meilisearch (semanticRatio 하나로 조절)
│
├─ 엔터프라이즈 환경 / AWS 기반인가?
│   └─ YES → OpenSearch (Amazon OpenSearch Service)
│
└─ 하이브리드 검색 + 대규모 + 세밀한 제어?
    └─ YES → OpenSearch (가장 강력)
```

### 10.2 프로젝트 단계별 권장

| 단계 | 권장 DB | 이유 |
|------|---------|------|
| 학습/실험 | Chroma | 설치 없이 즉시 시작 |
| MVP/프로토타입 | Meilisearch | 빠른 설정, 오타 허용 |
| 소규모 서비스 | pgvector | 기존 PostgreSQL 활용 |
| 중규모 서비스 | pgvector 또는 Meilisearch | 안정성 vs 기능 |
| 대규모 서비스 | OpenSearch | 확장성, 하이브리드 검색 |
| AWS 기반 서비스 | Amazon OpenSearch | 완전 관리형 |

---

## 11. 자주 묻는 질문 (FAQ)

!!! question "Q1: 벡터 DB와 일반 DB를 함께 써야 하나요?"
    대부분의 경우 **함께 쓴다**.

    - **일반 DB (PostgreSQL 등)**: 사용자 정보, 주문 정보, 설정 등 구조화된 데이터
    - **벡터 DB**: 문서 임베딩, 의미 검색용 데이터

    단, pgvector를 쓰면 하나의 PostgreSQL에 두 역할을 동시에 할 수 있다.

!!! question "Q2: 임베딩 모델이 바뀌면 어떻게 되나요?"
    **모든 데이터를 새 모델로 재임베딩해야 한다.**

    임베딩 모델이 다르면 벡터 공간이 완전히 달라지므로,
    `text-embedding-ada-002`로 저장한 데이터를 `text-embedding-3-small`로 검색하면 엉터리 결과가 나온다.

    따라서 임베딩 모델을 바꿀 때는 전체 재인덱싱이 필요하다.

!!! question "Q3: HNSW 인덱스 없이 검색하면 어떻게 되나요?"
    인덱스 없이 검색하면 **순차 스캔(브루트 포스)**이 발생한다.

    - 10개 문서: 인덱스 있으나 없으나 거의 차이 없음
    - 10,000개 문서: 인덱스 없으면 수초 걸림
    - 1,000,000개 문서: 인덱스 없으면 수십 초 ~ 수분 걸림

    **실제 서비스에서는 반드시 인덱스를 생성해야 한다.**

!!! question "Q4: 벡터 검색 결과의 점수(score)가 얼마 이상이어야 좋은 결과인가요?"
    DB마다 다르고, 모델마다 다르다. 절대적인 기준은 없지만 일반적으로:

    - 코사인 유사도 기준: 0.8 이상 = 매우 유사, 0.6~0.8 = 관련 있음, 0.6 미만 = 관련 낮음

    실제로는 직접 여러 쿼리를 테스트해보고 threshold를 정해야 한다.
    보통 `min_score = 0.7` 정도를 시작점으로 잡는다.

!!! question "Q5: 한국어 텍스트도 잘 검색되나요?"
    **임베딩 모델에 따라 다르다.**

    - `text-embedding-3-small/large` (OpenAI): 한국어 매우 잘 지원
    - `multilingual-e5-large`: 한국어 포함 다국어 지원
    - `all-MiniLM-L6-v2`: 영어 위주, 한국어 품질 낮음

    RAG에서 한국어를 쓴다면 OpenAI 임베딩이나 다국어 모델을 사용하자.

!!! question "Q6: 벡터 DB를 클라우드 서비스로 쓸 수 있나요?"
    | 클라우드 서비스 | 제공 업체 |
    |--------------|---------|
    | Amazon OpenSearch Service | AWS |
    | Azure Cognitive Search | Microsoft |
    | Pinecone | Pinecone (전용 벡터 DB SaaS) |
    | Weaviate Cloud | Weaviate |
    | Qdrant Cloud | Qdrant |
    | Supabase (pgvector) | Supabase |

    Pinecone, Weaviate, Qdrant는 벡터 검색 전용 클라우드 서비스로,
    설정 없이 API로 바로 사용할 수 있다.

!!! question "Q7: 메모리가 부족하면 어떻게 하나요?"
    - **HNSW → IVFFlat으로 전환**: HNSW는 메모리를 많이 사용함
    - **차원 축소**: `text-embedding-3-small` (1536차원) → 384~512차원으로 축소 가능
    - **PQ (Product Quantization)**: 벡터를 압축 저장 (OpenSearch에서 지원)
    - **디스크 기반 인덱스**: 메모리 대신 SSD 활용 (속도 저하 감수)

---

## 12. 핵심 요약

!!! abstract "핵심 요약"

    **개념 정리**:

    1. **벡터 DB** = 의미(Semantics)로 검색하는 데이터베이스. "반품"으로 "환불" 문서를 찾는다.
    2. **임베딩** = 텍스트를 숫자 배열(벡터)로 변환. 의미가 비슷하면 벡터도 가깝다.
    3. **코사인 유사도** = 두 벡터의 방향이 얼마나 같은지 (1=완전 같음, 0=무관, -1=반대)
    4. **HNSW** = 빠른 ANN 검색 알고리즘. 그래프 계층 구조로 검색 범위를 좁혀간다.
    5. **하이브리드 검색** = 벡터 검색 + BM25 키워드 검색 결합. 대부분의 경우 가장 좋은 결과.

    **DB 선택 기준**:

    6. **Chroma**: 학습/테스트용. 설치 0초.
    7. **pgvector**: 이미 PostgreSQL 사용 중. SQL 필터링 필요.
    8. **Meilisearch**: 빠른 프로토타입. 오타 허용. 간단한 하이브리드.
    9. **OpenSearch**: 대규모. 엔터프라이즈. 강력한 하이브리드 검색.

    **실전 원칙**:

    10. 항상 **하이브리드 검색**을 먼저 시도하라 — 순수 벡터보다 대부분 더 좋다.
    11. **HNSW 인덱스**는 필수 — 없으면 데이터가 늘수록 기하급수적으로 느려진다.
    12. **임베딩 모델을 바꾸면 전체 재인덱싱** — 처음부터 모델을 잘 선택하자.
    13. **유사도 threshold** 설정 — 점수가 낮은 결과는 LLM에 전달하지 말 것.
