# 4. 벡터 데이터베이스

## 벡터 DB의 역할

임베딩 벡터를 저장하고 **유사도 검색**을 수행하는 데이터베이스. RAG에서 "검색" 단계의 핵심 인프라이다.

---

## 주요 벡터 DB 비교

| 특성 | **OpenSearch** | **pgvector** | **Meilisearch** |
|------|---------------|-------------|----------------|
| 유형 | 검색 엔진 | PostgreSQL 확장 | 검색 엔진 |
| 벡터 검색 | ✅ k-NN | ✅ ANN (IVFFlat, HNSW) | ✅ 벡터 검색 |
| 전문 검색 | ✅ 강력 | ❌ (별도 설정) | ✅ 강력 |
| 하이브리드 검색 | ✅ 네이티브 | ⚠️ 수동 구현 | ✅ 네이티브 |
| 설치 복잡도 | 중간 | 낮음 (기존 PG) | 매우 낮음 |
| 확장성 | 매우 높음 | 중간 | 중간 |
| 적합한 경우 | 대규모, 하이브리드 | 기존 PG 활용 | 빠른 프로토타입 |

---

## OpenSearch

### 특징
- AWS에서 관리형 서비스 제공 (Amazon OpenSearch Service)
- **하이브리드 검색** (벡터 + 키워드) 네이티브 지원
- Elasticsearch 호환 API
- 대규모 데이터에 강점

### 설치 & 설정

```bash
# Docker로 실행
docker run -d --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  opensearchproject/opensearch:latest
```

### Python 사용법

```python
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False
)

# 벡터 인덱스 생성
index_body = {
    "settings": {
        "index": {"knn": True}
    },
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib"
                }
            },
            "metadata": {"type": "object"}
        }
    }
}
client.indices.create("rag-docs", body=index_body)

# 문서 인덱싱
doc = {
    "content": "RAG는 검색 증강 생성 기법이다",
    "embedding": embedding_vector,  # 1536차원 벡터
    "metadata": {"source": "guide.pdf", "page": 1}
}
client.index(index="rag-docs", body=doc)

# 벡터 검색
query = {
    "size": 3,
    "query": {
        "knn": {
            "embedding": {
                "vector": query_vector,
                "k": 3
            }
        }
    }
}
results = client.search(index="rag-docs", body=query)
```

### 하이브리드 검색 (벡터 + 키워드)

```python
# OpenSearch의 강력한 하이브리드 검색
hybrid_query = {
    "size": 5,
    "query": {
        "hybrid": {
            "queries": [
                {
                    "match": {
                        "content": {
                            "query": "RAG 파이프라인"
                        }
                    }
                },
                {
                    "knn": {
                        "embedding": {
                            "vector": query_vector,
                            "k": 5
                        }
                    }
                }
            ]
        }
    }
}
```

### LangChain 연동

```python
from langchain_community.vectorstores import OpenSearchVectorSearch

vectorstore = OpenSearchVectorSearch(
    opensearch_url="http://localhost:9200",
    index_name="rag-docs",
    embedding_function=OpenAIEmbeddings()
)

# 문서 추가
vectorstore.add_documents(chunks)

# 검색
docs = vectorstore.similarity_search("RAG란?", k=3)
```

---

## pgvector

### 특징
- PostgreSQL 확장 → **기존 PG 인프라 활용 가능**
- SQL로 벡터 검색 가능
- 트랜잭션, JOIN, 필터링 등 RDB 기능 전부 사용
- 소~중규모에 적합

### 설치 & 설정

```bash
# Docker로 PostgreSQL + pgvector
docker run -d --name pgvector \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  pgvector/pgvector:pg16
```

### Python 사용법

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="password"
)
cur = conn.cursor()

# pgvector 확장 활성화
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

# 테이블 생성
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector(1536),
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    )
""")

# HNSW 인덱스 (권장 — 빠른 검색)
cur.execute("""
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
""")

# 문서 삽입
cur.execute(
    "INSERT INTO documents (content, embedding, metadata) VALUES (%s, %s, %s)",
    ("RAG 개요 문서", str(embedding_vector), '{"source": "guide.pdf"}')
)

# 유사도 검색 (코사인 거리)
cur.execute("""
    SELECT content, metadata, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 3
""", (str(query_vector), str(query_vector)))

results = cur.fetchall()
conn.commit()
```

### pgvector의 강점: SQL 필터링 + 벡터 검색

```python
# 메타데이터 필터 + 벡터 검색을 SQL로 한 번에
cur.execute("""
    SELECT content, metadata,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    WHERE metadata->>'doc_type' = 'technical'
      AND created_at > '2025-01-01'
    ORDER BY embedding <=> %s::vector
    LIMIT 5
""", (str(query_vector), str(query_vector)))
```

### LangChain 연동

```python
from langchain_community.vectorstores import PGVector

CONNECTION_STRING = "postgresql://postgres:password@localhost:5432/postgres"

vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=OpenAIEmbeddings(),
    collection_name="rag_docs"
)

vectorstore.add_documents(chunks)
docs = vectorstore.similarity_search("검색 쿼리", k=3)
```

---

## Meilisearch

### 특징
- **설정 거의 불필요** — 즉시 사용 가능
- 오타 허용 (typo-tolerance) 내장
- 매우 빠른 검색 속도
- 벡터 검색 + 전문 검색 하이브리드 지원
- REST API 기반

### 설치 & 설정

```bash
# Docker로 실행 (벡터 검색 활성화)
docker run -d --name meilisearch \
  -p 7700:7700 \
  -e MEILI_MASTER_KEY="your-master-key" \
  getmeili/meilisearch:latest
```

### Python 사용법

```python
import meilisearch

client = meilisearch.Client("http://localhost:7700", "your-master-key")

# 인덱스 생성 & 벡터 검색 활성화
index = client.index("documents")
client.index("documents").update_settings({
    "embedders": {
        "default": {
            "source": "userProvided",
            "dimensions": 1536
        }
    }
})

# 문서 추가 (벡터 포함)
documents = [
    {
        "id": 1,
        "content": "RAG는 검색 증강 생성이다",
        "source": "guide.pdf",
        "_vectors": {
            "default": embedding_vector
        }
    }
]
index.add_documents(documents)

# 하이브리드 검색 (키워드 + 벡터)
results = index.search("RAG 파이프라인", {
    "hybrid": {
        "semanticRatio": 0.5,  # 0=키워드만, 1=벡터만
        "embedder": "default"
    },
    "vector": query_vector
})
```

### Meilisearch의 강점: 간편한 하이브리드 검색

```python
# semanticRatio로 키워드/벡터 비율 조절
# 정확한 용어 검색이 중요할 때
results = index.search("에러 코드 E-4012", {
    "hybrid": {"semanticRatio": 0.2}  # 키워드 80% + 벡터 20%
})

# 의미 검색이 중요할 때
results = index.search("시스템이 자꾸 멈춰요", {
    "hybrid": {"semanticRatio": 0.8}  # 키워드 20% + 벡터 80%
})
```

---

## 선택 가이드

```
어떤 벡터 DB를 쓸까?

이미 PostgreSQL 쓰고 있음?
├─ Yes → pgvector (추가 인프라 불필요)
└─ No
    ├─ 대규모 (100만+ 문서)?
    │   └─ OpenSearch (분산 처리, 확장성)
    ├─ 빠른 프로토타입?
    │   └─ Meilisearch (설정 최소, 즉시 사용)
    └─ 하이브리드 검색 필수?
        ├─ 세밀한 제어 → OpenSearch
        └─ 간편하게 → Meilisearch
```

---

## 핵심 요약

!!! summary "이것만 기억하자"
    1. **pgvector**: 기존 PostgreSQL 활용, SQL 필터링 강점, 소~중규모
    2. **OpenSearch**: 대규모, 하이브리드 검색, 엔터프라이즈급
    3. **Meilisearch**: 빠른 프로토타입, 간편 설정, 오타 허용
    4. 하이브리드 검색(벡터+키워드)이 순수 벡터 검색보다 대부분 우수
    5. 프로덕션에선 인덱스 설정(HNSW 파라미터 등)이 성능에 큰 영향
