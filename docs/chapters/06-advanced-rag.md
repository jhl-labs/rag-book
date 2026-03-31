# 6. 고급 RAG 기법

## Naive RAG의 한계

기본 RAG는 다음 상황에서 성능이 떨어진다:

- 질문이 모호하거나 복합적일 때
- 검색된 문서의 관련성이 낮을 때
- 여러 문서를 종합해야 할 때

→ **고급 기법**으로 각 단계를 개선한다.

---

## 1. 쿼리 변환 (Query Transformation)

### HyDE (Hypothetical Document Embeddings)

질문 대신 **가상 답변을 생성**하고, 그 답변으로 검색한다.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

hyde_prompt = ChatPromptTemplate.from_template("""
다음 질문에 대한 답변이 될 만한 문서 단락을 작성하세요.
실제 정보가 아니어도 됩니다. 형식과 내용의 스타일만 맞추세요.

질문: {question}

가상 문서:
""")

def hyde_search(question, vectorstore, k=3):
    # 1. 가상 답변 생성
    hypothetical_doc = llm.invoke(
        hyde_prompt.format(question=question)
    ).content

    # 2. 가상 답변으로 검색 (원래 질문보다 문서와 유사)
    results = vectorstore.similarity_search(hypothetical_doc, k=k)
    return results

# "RAG의 장점" 보다 "RAG는...한 장점이 있다" 형태가
# 실제 문서와 더 유사한 임베딩을 가짐
docs = hyde_search("RAG의 장점은?", vectorstore)
```

### Multi-Query (다중 쿼리)

하나의 질문을 **여러 관점의 쿼리로 변환**하여 검색 범위를 넓힌다.

```python
from langchain_core.prompts import ChatPromptTemplate

multi_query_prompt = ChatPromptTemplate.from_template("""
다음 질문에 대해 검색에 유용한 3개의 다른 버전을 만드세요.
각 버전은 다른 관점이나 키워드를 사용하세요.

원본 질문: {question}

3개의 대체 질문 (한 줄에 하나씩):
""")

def multi_query_search(question, vectorstore, k=3):
    # 다중 쿼리 생성
    response = llm.invoke(
        multi_query_prompt.format(question=question)
    ).content
    queries = [question] + response.strip().split("\n")

    # 각 쿼리로 검색 후 중복 제거
    all_docs = []
    seen = set()
    for q in queries:
        docs = vectorstore.similarity_search(q.strip(), k=k)
        for doc in docs:
            doc_id = doc.page_content[:100]
            if doc_id not in seen:
                seen.add(doc_id)
                all_docs.append(doc)

    return all_docs[:k * 2]  # 상위 결과 반환
```

### Query Decomposition (쿼리 분해)

복합 질문을 **하위 질문으로 분해**한다.

```python
decompose_prompt = ChatPromptTemplate.from_template("""
다음 복합 질문을 독립적으로 답할 수 있는 하위 질문들로 분해하세요.

질문: {question}

하위 질문들 (한 줄에 하나):
""")

def decompose_and_search(question, vectorstore, k=3):
    # 분해
    response = llm.invoke(
        decompose_prompt.format(question=question)
    ).content
    sub_questions = response.strip().split("\n")

    # 각 하위 질문에 대해 검색 & 답변
    sub_answers = []
    for sq in sub_questions:
        docs = vectorstore.similarity_search(sq.strip(), k=k)
        context = "\n".join(d.page_content for d in docs)
        answer = llm.invoke(
            f"컨텍스트: {context}\n\n질문: {sq}\n답변:"
        ).content
        sub_answers.append(f"Q: {sq}\nA: {answer}")

    # 최종 종합 답변
    final = llm.invoke(
        f"다음 하위 답변들을 종합하여 원래 질문에 답하세요.\n\n"
        f"원래 질문: {question}\n\n"
        f"{''.join(sub_answers)}\n\n종합 답변:"
    ).content

    return final

# "OpenSearch와 pgvector의 성능, 비용, 장단점을 비교해줘"
# → "OpenSearch 성능은?", "pgvector 성능은?", "비용 차이는?" 등으로 분해
```

---

## 2. Reranking (재순위매김)

벡터 검색 결과를 **교차 인코더(Cross-Encoder)**로 재평가하여 정밀도를 높인다.

```python
from sentence_transformers import CrossEncoder

# 재순위 모델 로드
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def search_with_rerank(question, vectorstore, initial_k=10, final_k=3):
    # 1단계: 넓게 검색 (recall 우선)
    initial_docs = vectorstore.similarity_search(question, k=initial_k)

    # 2단계: 재순위 (precision 우선)
    pairs = [(question, doc.page_content) for doc in initial_docs]
    scores = reranker.predict(pairs)

    # 점수 기준 정렬
    scored_docs = sorted(
        zip(initial_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in scored_docs[:final_k]]
```

### Cohere Rerank API

```python
import cohere

co = cohere.ClientV2()

def cohere_rerank(question, docs, top_n=3):
    results = co.rerank(
        model="rerank-v3.5",
        query=question,
        documents=[doc.page_content for doc in docs],
        top_n=top_n
    )

    reranked = [docs[r.index] for r in results.results]
    return reranked
```

!!! tip "Reranking 효과"
    벡터 검색(Bi-Encoder)은 빠르지만 정밀도가 낮고,
    Reranking(Cross-Encoder)은 느리지만 정밀도가 높다.
    → **넓게 검색 + 좁게 재순위** = 속도와 정확도 모두 확보

---

## 3. 하이브리드 검색 (Hybrid Search)

**벡터 검색 + 키워드 검색**을 결합. 고유명사, 코드, 에러 메시지 등은 키워드 검색이 더 정확하다.

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# BM25 (키워드 기반)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# 벡터 검색
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 앙상블 (가중 결합)
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 벡터 검색에 더 높은 가중치
)

docs = hybrid_retriever.invoke("에러 코드 NullPointerException 해결법")
```

---

## 4. Contextual Compression

검색된 문서에서 **질문에 관련된 부분만 추출**하여 LLM에 전달한다.

```python
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 긴 문서에서 관련 부분만 추출
compressed_docs = compression_retriever.invoke("RAG의 장점")
for doc in compressed_docs:
    print(f"압축된 내용: {doc.page_content[:100]}...")
```

---

## 5. Self-RAG (자기 반성 RAG)

LLM이 **자신의 답변을 평가**하고, 필요시 재검색한다.

```python
def self_rag(question, vectorstore, max_retries=2):
    for attempt in range(max_retries + 1):
        # 검색
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join(d.page_content for d in docs)

        # 생성
        answer = llm.invoke(
            f"컨텍스트: {context}\n\n질문: {question}\n답변:"
        ).content

        # 자기 평가
        evaluation = llm.invoke(f"""
답변의 품질을 평가하세요.

질문: {question}
컨텍스트: {context}
답변: {answer}

평가 기준:
1. 답변이 컨텍스트에 기반하는가? (grounded)
2. 질문에 충분히 답했는가? (sufficient)

결과를 JSON으로: {{"grounded": true/false, "sufficient": true/false}}
""").content

        if '"grounded": true' in evaluation and '"sufficient": true' in evaluation:
            return answer

        # 품질 불충분 → 쿼리 수정 후 재시도
        question = llm.invoke(
            f"'{question}'을 더 구체적으로 바꿔주세요"
        ).content

    return answer  # 최종 답변 반환
```

---

## 기법 조합 전략

| 상황 | 추천 조합 |
|------|-----------|
| 일반 QA | Hybrid Search + Reranking |
| 복합 질문 | Query Decomposition + Reranking |
| 모호한 질문 | HyDE + Multi-Query |
| 긴 문서 | Contextual Compression |
| 높은 정확도 요구 | Self-RAG + Reranking |

---

## 핵심 요약

!!! summary "이것만 기억하자"
    1. **HyDE**: 가상 답변으로 검색 → 질문-문서 갭 해소
    2. **Reranking**: 넓게 검색 후 Cross-Encoder로 정밀 재순위
    3. **하이브리드 검색**: 벡터 + 키워드 = 최고의 검색 품질
    4. **Query Decomposition**: 복합 질문을 나누어 각각 처리
    5. 실전에선 여러 기법을 **조합**하여 사용
