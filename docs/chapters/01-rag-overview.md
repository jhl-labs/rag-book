# 1. RAG 개요

## RAG란?

**RAG(Retrieval Augmented Generation)**는 LLM이 답변을 생성하기 전에 외부 지식소스에서 관련 정보를 **검색(Retrieve)**하여 프롬프트에 포함시키는 기법이다.

```
[사용자 질문] → [검색] → [관련 문서 추출] → [LLM + 문서 컨텍스트] → [답변]
```

---

## 왜 RAG가 필요한가?

| 문제 | RAG의 해결 |
|------|-----------|
| **할루시네이션** | 실제 문서 기반으로 답변 → 사실 기반 응답 |
| **지식 컷오프** | 최신 문서를 실시간 검색 → 최신 정보 반영 |
| **도메인 특화** | 사내 문서/매뉴얼 검색 → 기업 맞춤 답변 |
| **비용** | 파인튜닝 없이 지식 확장 → 비용 절감 |

---

## RAG 아키텍처

### 기본 구조

```
┌─────────────────────────────────────────────────┐
│                  RAG Pipeline                    │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │ Indexing  │   │Retrieval │   │ Generation   │ │
│  │          │   │          │   │              │ │
│  │ 문서→청크 │──▶│ 쿼리→검색 │──▶│ 컨텍스트+LLM │ │
│  │ →임베딩   │   │ →순위매김  │   │ →답변 생성   │ │
│  └──────────┘   └──────────┘   └──────────────┘ │
└─────────────────────────────────────────────────┘
```

### 3단계 프로세스

**1단계: 인덱싱 (Indexing)**

문서를 청크로 분할하고 임베딩 벡터로 변환하여 벡터 DB에 저장한다.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 문서 로드 & 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 임베딩 & 저장
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)
```

**2단계: 검색 (Retrieval)**

사용자 쿼리를 임베딩하고 유사한 청크를 검색한다.

```python
# 유사도 검색 (상위 3개)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke("RAG의 장점은?")
```

**3단계: 생성 (Generation)**

검색된 문서를 컨텍스트로 포함하여 LLM이 답변을 생성한다.

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model="gpt-4o")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain.invoke({"query": "RAG의 장점은?"})
print(result["result"])
```

---

## Naive RAG vs Advanced RAG

| 구분 | Naive RAG | Advanced RAG |
|------|-----------|-------------|
| 검색 | 단순 유사도 | 하이브리드 검색 + Reranking |
| 쿼리 | 원본 그대로 | 쿼리 변환/분해 |
| 청킹 | 고정 크기 | 의미 기반 분할 |
| 후처리 | 없음 | 답변 검증 + 출처 표시 |

---

## 핵심 요약

!!! summary "이것만 기억하자"
    1. RAG = **검색 + 생성** — LLM에게 문서를 "읽혀서" 답변하게 하는 것
    2. 파인튜닝 없이 도메인 지식을 LLM에 주입 가능
    3. 품질은 **검색 품질**에 크게 좌우됨 → 임베딩, 청킹, 검색 전략이 핵심
