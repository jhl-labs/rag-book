# 6. 고급 RAG 기법

!!! abstract "이 챕터에서 배울 것"
    기본 RAG(Naive RAG)는 단순한 질문에는 잘 동작하지만, 복잡한 실전 환경에서는 한계를 드러냅니다.
    이 챕터에서는 그 한계를 극복하는 **7가지 핵심 고급 기법**을 깊이 있게 배웁니다.

    - HyDE, Multi-Query, Query Decomposition (쿼리 변환)
    - Reranking (재순위매김) - Bi-Encoder vs Cross-Encoder 완전 이해
    - Hybrid Search (하이브리드 검색) + BM25 원리
    - Contextual Compression (문맥 압축)
    - Self-RAG, Corrective RAG, Adaptive RAG (자기 반성 RAG)
    - 기법 조합 레시피 & 성능 비교

---

## Naive RAG의 한계 - 왜 고급 기법이 필요한가?

기본 RAG 파이프라인을 구현해 보면 생각보다 자주 이런 문제를 만납니다.

### 문제 시나리오

```
사용자 질문: "우리 제품의 환불 정책과 경쟁사 대비 가격 경쟁력에 대해 알려줘"

Naive RAG 결과:
- 검색된 문서: "환불 정책 안내" 문서 1개만 가져옴
- 경쟁사 비교 정보는 완전히 누락
- 질문이 복합적이라 단일 벡터 검색으로는 커버 불가
```

!!! warning "Naive RAG가 실패하는 3가지 상황"

    **1. 질문이 모호하거나 복합적일 때**
    "A와 B를 비교해줘" 같은 질문은 단일 쿼리 벡터로 표현하기 어렵습니다.

    **2. 검색된 문서의 관련성이 낮을 때**
    벡터 유사도 검색은 의미적으로 비슷한 단어를 찾지만, 진짜 필요한 문서를 놓치기도 합니다.

    **3. 여러 문서를 종합해야 할 때**
    답변이 여러 문서에 흩어져 있으면 단순 Top-K 검색으로는 부족합니다.

### 기본 RAG vs 고급 RAG 구조 비교

```
[Naive RAG]
사용자 질문 → 벡터 검색 → Top-K 문서 → LLM → 답변
              (1번만)

[Advanced RAG]
사용자 질문 → 쿼리 변환 → 다중 검색 → 재순위 → 압축 → LLM → 자기평가 → 답변
              (전처리)    (검색 개선)  (후처리)  (필터링)      (반성/재시도)
```

---

## 1. 쿼리 변환 (Query Transformation)

!!! info "핵심 아이디어"
    사용자가 입력한 질문 그대로 검색하지 말고, **더 잘 검색될 수 있는 형태로 바꿔서** 검색하자.

쿼리 변환은 "질문을 문서와 더 잘 매칭되게 변형"하는 전처리 단계입니다. 세 가지 주요 기법을 배웁니다.

---

### 1.1 HyDE (Hypothetical Document Embeddings)

#### 이 기법이 해결하는 문제

**문제**: 사용자의 질문과 데이터베이스의 문서 사이에는 **표현 방식의 차이**가 있습니다.

- 질문: "RAG가 뭐야?" → 짧고, 대화체, 의문형
- 문서: "RAG(Retrieval-Augmented Generation)는 외부 지식 베이스에서..." → 길고, 문서체, 설명형

이 두 텍스트를 벡터로 변환하면 의미는 같지만 **벡터 공간에서 거리가 멀어질 수 있습니다**.

#### 현실 세계 비유

> **HyDE는 마치 "참고 문헌 검색 전에 초안 먼저 써보기"와 같습니다.**
>
> 도서관에서 논문을 찾을 때, 찾으려는 내용을 먼저 대략 써보면 어떤 단어와 표현으로 검색해야 할지 감이 잡힙니다. HyDE는 LLM에게 "이런 답변이 있을 것 같다"는 가짜 문서를 먼저 생성하게 하고, 그 가짜 문서로 실제 문서를 검색합니다.

#### 동작 원리 (단계별)

```
[기존 방식]
"RAG의 장점은?" (짧은 질문 벡터) ──→ 문서 검색
                                  ↑
                           벡터 공간 거리가 멀 수 있음

[HyDE 방식]
"RAG의 장점은?"
      ↓  LLM이 가짜 답변 생성
"RAG는 실시간으로 외부 데이터를 참조하기 때문에
 hallucination을 줄이고 최신 정보를 반영할 수 있다는
 장점이 있습니다. 또한 파인튜닝 없이도..." (긴 문서 형태)
      ↓  이 가짜 문서로 검색
실제 유사한 문서 검색 (벡터 공간에서 더 가까움!)
```

#### 완전한 코드 예제

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# HyDE 프롬프트: 가짜 답변 문서를 생성하도록 유도
hyde_prompt = ChatPromptTemplate.from_template("""
당신은 문서 작성 전문가입니다.
다음 질문에 대한 답변이 담긴 문서 단락을 작성하세요.

중요: 실제 사실이 아니어도 됩니다.
문서의 형식, 스타일, 전문 용어 사용 방식만 맞추세요.
약 100-150단어로 작성하세요.

질문: {question}

가상 문서 단락:
""")

def hyde_search(question: str, vectorstore: Chroma, k: int = 3) -> list:
    """
    HyDE를 사용한 향상된 문서 검색

    Args:
        question: 사용자 질문
        vectorstore: 검색할 벡터 데이터베이스
        k: 반환할 문서 수

    Returns:
        검색된 문서 리스트
    """
    print(f"원본 질문: {question}")

    # Step 1: LLM으로 가상 답변 문서 생성
    hypothetical_doc = llm.invoke(
        hyde_prompt.format_messages(question=question)
    ).content
    print(f"\n생성된 가상 문서:\n{hypothetical_doc[:200]}...")

    # Step 2: 가상 문서로 실제 벡터 DB 검색
    # (원본 질문보다 실제 문서와 더 유사한 형태)
    results = vectorstore.similarity_search(hypothetical_doc, k=k)

    print(f"\n검색된 문서 수: {len(results)}")
    return results

# 사용 예시
# vectorstore = Chroma(...)  # 미리 구축된 벡터 DB
# docs = hyde_search("RAG의 장점은 무엇인가요?", vectorstore)
```

#### 예상 출력 예시

```
원본 질문: RAG의 장점은 무엇인가요?

생성된 가상 문서:
RAG(Retrieval-Augmented Generation)의 핵심 장점은 크게 세 가지입니다.
첫째, 할루시네이션(환각) 현상을 크게 줄일 수 있습니다. LLM이 학습 데이터에만
의존하지 않고 실시간으로 검증된 외부 문서를 참조하기 때문입니다...

검색된 문서 수: 3
```

#### 언제 사용하고, 언제 사용하지 말아야 하나?

| 상황 | HyDE 권장 여부 |
|------|---------------|
| 짧고 모호한 질문 ("RAG가 뭐야?") | ✅ 강력 권장 |
| 전문 기술 문서 검색 | ✅ 권장 |
| 단순 키워드 검색 ("에러코드 404") | ❌ 불필요, 오히려 노이즈 |
| 실시간 응답이 중요한 서비스 | ❌ LLM 호출 추가로 지연 발생 |
| 사실 오류가 치명적인 의료/법률 | ⚠️ 주의 필요 (가짜 문서의 내용이 영향줄 수 있음) |

#### 성능 영향

- **검색 품질**: 15-30% 향상 (특히 짧은 질문)
- **지연 시간**: LLM 호출 1회 추가 (약 0.5-2초 증가)
- **비용**: LLM 토큰 사용량 증가 (약 100-200 토큰/쿼리)

---

### 1.2 Multi-Query (다중 쿼리)

#### 이 기법이 해결하는 문제

**문제**: 하나의 질문을 하나의 벡터로만 표현하면 **관점의 편향**이 생깁니다.

```
질문: "파이썬으로 API 만드는 방법"
↓ 벡터 검색 시 주로 이런 문서만 찾음
"FastAPI를 이용한 REST API 구현..."

↓ 하지만 이런 유용한 문서는 놓칠 수 있음
"Flask 웹 서버 구축 가이드..."
"Django REST framework 사용법..."
"HTTP 엔드포인트 설계 패턴..."
```

#### 현실 세계 비유

> **Multi-Query는 마치 "여러 검색 전략을 동시에 쓰는 도서관 사서"와 같습니다.**
>
> 숙련된 사서는 "파이썬 API"라는 검색어로 찾을 때, "Flask", "FastAPI", "Django", "REST 서버", "웹 프레임워크" 등 여러 관련 키워드를 동시에 탐색합니다. Multi-Query는 LLM이 이 역할을 합니다.

#### 동작 원리

```
원본 질문: "파이썬으로 API 만드는 방법"
              ↓ LLM이 3-5개 변형 생성
쿼리 1: "Python REST API 개발 프레임워크 비교"
쿼리 2: "FastAPI Flask Django 웹 서버 구축"
쿼리 3: "Python HTTP 엔드포인트 구현 방법"
쿼리 4: "파이썬 백엔드 API 서버 만들기"
              ↓ 각각 독립적으로 검색
결과 1: [doc_a, doc_b, doc_c]
결과 2: [doc_b, doc_d, doc_e]
결과 3: [doc_c, doc_f, doc_g]
              ↓ 중복 제거 후 합산
최종 결과: [doc_a, doc_b, doc_c, doc_d, doc_e, doc_f, doc_g]
```

#### 완전한 코드 예제

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 다중 쿼리 생성 프롬프트
multi_query_prompt = ChatPromptTemplate.from_template("""
당신은 정보 검색 전문가입니다.
다음 질문에 대해 검색 범위를 넓히기 위한 4개의 다른 버전 질문을 만드세요.

각 버전은:
- 다른 키워드나 동의어 사용
- 다른 관점이나 각도에서 접근
- 검색 엔진에서 잘 매칭될 수 있는 형태

원본 질문: {question}

4개의 대체 질문 (번호 없이, 한 줄에 하나씩):
""")

def multi_query_search(
    question: str,
    vectorstore,
    k_per_query: int = 3
) -> list:
    """
    다중 쿼리를 사용한 확장 검색

    Args:
        question: 원본 사용자 질문
        vectorstore: 검색 대상 벡터 DB
        k_per_query: 쿼리당 검색 문서 수

    Returns:
        중복 제거된 문서 리스트
    """
    # Step 1: LLM으로 대체 질문들 생성
    chain = multi_query_prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})

    # 원본 질문 + 생성된 대체 질문들
    alternative_queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
    all_queries = [question] + alternative_queries

    print(f"총 {len(all_queries)}개 쿼리로 검색:")
    for i, q in enumerate(all_queries):
        print(f"  [{i}] {q}")

    # Step 2: 각 쿼리로 독립적으로 검색
    all_docs = []
    seen_content = set()  # 중복 제거용

    for query in all_queries:
        try:
            docs = vectorstore.similarity_search(query.strip(), k=k_per_query)
            for doc in docs:
                # 처음 100자를 기준으로 중복 판별
                doc_fingerprint = doc.page_content[:100]
                if doc_fingerprint not in seen_content:
                    seen_content.add(doc_fingerprint)
                    all_docs.append(doc)
        except Exception as e:
            print(f"쿼리 '{query}' 검색 실패: {e}")
            continue

    print(f"\n중복 제거 후 총 {len(all_docs)}개 문서 수집")
    return all_docs

# 사용 예시
# docs = multi_query_search("파이썬으로 API 만드는 방법", vectorstore)
```

#### 예상 출력 예시

```
총 5개 쿼리로 검색:
  [0] 파이썬으로 API 만드는 방법
  [1] Python REST API 개발 프레임워크 종류와 사용법
  [2] FastAPI Flask Django 웹 서버 구축 예제
  [3] 파이썬 HTTP 엔드포인트 백엔드 구현
  [4] Python API 서버 설계 패턴과 모범 사례

중복 제거 후 총 11개 문서 수집
```

#### 언제 사용하고, 언제 사용하지 말아야 하나?

| 상황 | Multi-Query 권장 여부 |
|------|----------------------|
| 포괄적인 정보가 필요한 질문 | ✅ 강력 권장 |
| 다양한 측면을 커버해야 하는 주제 | ✅ 권장 |
| 매우 구체적인 단일 사실 검색 | ❌ 오버킬, 비용 낭비 |
| 실시간 챗봇 (지연 민감) | ⚠️ 병렬 처리로 최적화 필요 |

---

### 1.3 Query Decomposition (쿼리 분해)

#### 이 기법이 해결하는 문제

**문제**: 복합 질문은 **단일 검색으로 답하기 불가능**합니다.

```
복합 질문: "OpenAI와 Anthropic의 API 가격, 성능, 안전성을 비교하고,
           우리 스타트업에 어떤 게 더 적합한지 알려줘"

이 질문은 사실 여러 개의 독립적인 질문입니다:
1. OpenAI API 가격은?
2. Anthropic API 가격은?
3. OpenAI 모델 성능 벤치마크는?
4. Anthropic 모델 성능 벤치마크는?
5. OpenAI 안전성 정책은?
6. Anthropic 안전성 정책은?
→ 각각 따로 검색하고 나중에 종합해야 함
```

#### 현실 세계 비유

> **Query Decomposition은 마치 "복잡한 요리를 레시피로 나누기"와 같습니다.**
>
> "파스타 만들어줘"라는 요청을 받은 요리사는 "물 끓이기", "면 삶기", "소스 만들기", "플레이팅" 등 세부 단계로 나눕니다. 복잡한 질문을 작은 질문들로 분해하면 각각을 정확하게 검색하고 답할 수 있습니다.

#### 두 가지 분해 전략

**전략 1: 병렬 분해 (독립적인 하위 질문)**
```
복합 질문 → [하위 질문 1, 하위 질문 2, 하위 질문 3]
                ↓              ↓              ↓
            검색+답변      검색+답변      검색+답변
                ↓              ↓              ↓
                    종합하여 최종 답변
```

**전략 2: 순차 분해 (이전 답변을 활용)**
```
복합 질문 → 하위 질문 1 → 답변 1
                              ↓
                         하위 질문 2 (답변 1 참고) → 답변 2
                                                        ↓
                                                   최종 종합
```

#### 완전한 코드 예제

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# 질문 분해 프롬프트
decompose_prompt = ChatPromptTemplate.from_template("""
다음 복합 질문을 각각 독립적으로 검색하고 답할 수 있는 하위 질문들로 분해하세요.

규칙:
- 각 하위 질문은 단독으로 완전한 의미를 가져야 함
- 3-5개의 하위 질문으로 분해
- 각 질문은 한 줄, 번호 없이

복합 질문: {question}

하위 질문들:
""")

# 하위 질문 답변 프롬프트
sub_answer_prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답하세요.
컨텍스트에 없는 내용은 "정보 없음"이라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:
""")

# 최종 종합 프롬프트
synthesis_prompt = ChatPromptTemplate.from_template("""
다음 하위 질문들과 답변들을 종합하여 원래 질문에 대한 완전한 답변을 작성하세요.

원래 질문: {original_question}

하위 질문/답변 모음:
{sub_qa_pairs}

종합 답변 (구조화하여 명확하게):
""")

def decompose_and_search(
    question: str,
    vectorstore,
    k: int = 3
) -> Dict:
    """
    쿼리 분해를 사용한 복합 질문 처리

    Returns:
        {"sub_questions": [...], "sub_answers": [...], "final_answer": "..."}
    """
    print(f"원본 질문: {question}\n")

    # Step 1: 질문 분해
    decompose_chain = decompose_prompt | llm | StrOutputParser()
    decomposed = decompose_chain.invoke({"question": question})
    sub_questions = [q.strip() for q in decomposed.strip().split("\n") if q.strip()]

    print(f"분해된 하위 질문 {len(sub_questions)}개:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq}")

    # Step 2: 각 하위 질문에 대해 검색 + 답변
    sub_qa_pairs = []
    for i, sq in enumerate(sub_questions, 1):
        print(f"\n[{i}/{len(sub_questions)}] 처리 중: {sq}")

        # 검색
        docs = vectorstore.similarity_search(sq, k=k)
        context = "\n\n".join(
            f"[문서 {j+1}]\n{doc.page_content}"
            for j, doc in enumerate(docs)
        )

        # 하위 답변 생성
        answer_chain = sub_answer_prompt | llm | StrOutputParser()
        sub_answer = answer_chain.invoke({
            "context": context,
            "question": sq
        })

        sub_qa_pairs.append({
            "question": sq,
            "answer": sub_answer,
            "source_docs": docs
        })
        print(f"  답변: {sub_answer[:100]}...")

    # Step 3: 모든 하위 답변을 종합
    sub_qa_text = "\n\n".join(
        f"Q{i+1}: {pair['question']}\nA{i+1}: {pair['answer']}"
        for i, pair in enumerate(sub_qa_pairs)
    )

    synthesis_chain = synthesis_prompt | llm | StrOutputParser()
    final_answer = synthesis_chain.invoke({
        "original_question": question,
        "sub_qa_pairs": sub_qa_text
    })

    print(f"\n최종 종합 답변:\n{final_answer}")

    return {
        "sub_questions": sub_questions,
        "sub_qa_pairs": sub_qa_pairs,
        "final_answer": final_answer
    }

# 사용 예시
# result = decompose_and_search(
#     "OpenAI와 Anthropic의 API 가격과 성능을 비교해줘",
#     vectorstore
# )
```

#### 예상 출력 예시

```
원본 질문: OpenAI와 Anthropic의 API 가격과 성능을 비교해줘

분해된 하위 질문 4개:
  1. OpenAI GPT-4o API 가격은 얼마인가?
  2. Anthropic Claude API 가격은 얼마인가?
  3. OpenAI GPT-4o의 주요 벤치마크 성능 지표는?
  4. Anthropic Claude의 주요 벤치마크 성능 지표는?

[1/4] 처리 중: OpenAI GPT-4o API 가격은 얼마인가?
  답변: GPT-4o는 입력 토큰 $2.50/1M, 출력 토큰 $10/1M...

...

최종 종합 답변:
## OpenAI vs Anthropic 비교

### 가격
- OpenAI GPT-4o: 입력 $2.50/1M 토큰...
- Anthropic Claude 3.5: 입력 $3.00/1M 토큰...
```

---

## 2. Reranking (재순위매김)

### 왜 이게 중요한가?

!!! question "벡터 검색의 치명적 약점"
    벡터 검색(Bi-Encoder)은 **빠르지만 부정확**할 수 있습니다.
    예를 들어:

    질문: "Python에서 리스트와 튜플의 차이점은?"

    벡터 검색 Top-3 결과:
    1. "Python 리스트 사용법" (관련성: 중간)
    2. "Python 데이터 타입 개요" (관련성: 낮음, 하지만 벡터상 가까움)
    3. "Python 튜플과 리스트 비교 가이드" (관련성: 매우 높음, 하지만 3위)

    Reranking 후:
    1. "Python 튜플과 리스트 비교 가이드" (관련성: 매우 높음 → 1위로 승격!)
    2. "Python 리스트 사용법"
    3. "Python 데이터 타입 개요"

### Bi-Encoder vs Cross-Encoder 완전 이해

이것이 Reranking을 이해하는 핵심입니다.

#### Bi-Encoder (양방향 인코더) - 벡터 검색에 사용

```
[Bi-Encoder 구조]

질문 → [인코더] → 질문 벡터 (384차원)
                        ↓
문서1 → [인코더] → 문서1 벡터 (384차원)  ← 코사인 유사도 계산
문서2 → [인코더] → 문서2 벡터 (384차원)  ← 코사인 유사도 계산
문서3 → [인코더] → 문서3 벡터 (384차원)  ← 코사인 유사도 계산

특징:
- 각 텍스트를 독립적으로 인코딩
- 벡터를 미리 저장해둘 수 있음 (사전 계산 가능)
- 수백만 개 문서도 빠르게 검색 (ANN 알고리즘)
- 단점: 질문-문서 간의 세밀한 상호작용을 포착 못함
```

#### Cross-Encoder (교차 인코더) - Reranking에 사용

```
[Cross-Encoder 구조]

[질문 + 문서1] → [인코더] → 관련성 점수 (0.95)
[질문 + 문서2] → [인코더] → 관련성 점수 (0.32)
[질문 + 문서3] → [인코더] → 관련성 점수 (0.78)

특징:
- 질문과 문서를 함께 입력하여 상호작용 포착
- 훨씬 정확한 관련성 점수 산출
- 단점: 모든 문서 쌍마다 실행해야 하므로 느림
        (100개 문서라면 100번의 인코더 실행 필요)
```

#### 현실 세계 비유

> **Bi-Encoder는 도서관 색인 카드**, **Cross-Encoder는 사서가 직접 읽어보는 것**입니다.
>
> - 색인 카드: 제목, 키워드만 보고 빠르게 후보 선별 (빠르지만 부정확)
> - 사서가 직접 읽기: 실제 내용을 보고 질문과 얼마나 맞는지 판단 (느리지만 정확)
>
> 최적 전략: **색인 카드로 100권을 추린 후, 사서가 직접 읽어서 Top-5 선별**

#### 2단계 검색 파이프라인

```
[1단계: 넓게 검색 (Recall 우선)]
Bi-Encoder로 수백만 문서 중 Top-50 후보 선별
→ 속도: 매우 빠름 (수밀리초)
→ 정밀도: 보통

[2단계: 좁게 재순위 (Precision 우선)]
Cross-Encoder로 50개 후보를 정밀 평가 → Top-5 선택
→ 속도: 느림 (수초)
→ 정밀도: 매우 높음

결과: 속도(Bi-Encoder 덕분) + 정확도(Cross-Encoder 덕분) 모두 확보!
```

### 완전한 코드 예제 - sentence-transformers 사용

```python
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from typing import List, Tuple
import numpy as np

# Cross-Encoder 모델 로드
# BAAI/bge-reranker-v2-m3: 한국어 포함 다국어 지원 (권장!)
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    max_length=512  # 입력 최대 길이
)

def search_with_rerank(
    question: str,
    vectorstore: Chroma,
    initial_k: int = 20,  # 1단계: 넓게 검색
    final_k: int = 5      # 2단계: 좁게 선별
) -> List[Tuple]:
    """
    2단계 검색 + 재순위매김

    Args:
        question: 사용자 질문
        vectorstore: 벡터 데이터베이스
        initial_k: 1차 검색 시 가져올 문서 수 (많을수록 재순위 효과 UP)
        final_k: 최종 반환할 문서 수

    Returns:
        (문서, 점수) 튜플 리스트
    """
    print(f"질문: {question}")
    print(f"1단계: {initial_k}개 문서 검색 중...")

    # 1단계: Bi-Encoder로 넓게 검색
    initial_docs = vectorstore.similarity_search(question, k=initial_k)
    print(f"1단계 완료: {len(initial_docs)}개 문서 수집")

    # 2단계: Cross-Encoder로 재순위
    print(f"2단계: Cross-Encoder로 재순위매김 중...")

    # (질문, 문서) 쌍 생성
    pairs = [(question, doc.page_content) for doc in initial_docs]

    # Cross-Encoder로 관련성 점수 계산
    scores = reranker.predict(pairs)

    # 점수 기준 내림차순 정렬
    scored_docs = sorted(
        zip(initial_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 결과 출력
    print(f"\n재순위 결과 Top-{final_k}:")
    for i, (doc, score) in enumerate(scored_docs[:final_k]):
        print(f"  [{i+1}] 점수: {score:.4f} | {doc.page_content[:80]}...")

    return scored_docs[:final_k]

# 점수만 추출하는 헬퍼 함수
def get_top_docs_only(question: str, vectorstore: Chroma, k: int = 5) -> list:
    """점수 없이 문서만 반환하는 간편 버전"""
    results = search_with_rerank(question, vectorstore, final_k=k)
    return [doc for doc, score in results]
```

### Cohere Rerank API (클라우드 서비스)

자체 모델을 돌리기 어려울 때는 Cohere의 재순위 API를 사용할 수 있습니다.

```python
import cohere
import os

# Cohere 클라이언트 초기화
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

def cohere_rerank(
    question: str,
    docs: list,
    top_n: int = 5
) -> list:
    """
    Cohere Rerank API를 사용한 재순위매김

    장점: 설치 없이 API로 바로 사용 가능
    단점: API 비용 발생, 인터넷 연결 필요
    """
    # 문서 텍스트 추출
    doc_texts = [doc.page_content for doc in docs]

    # Cohere API 호출
    results = co.rerank(
        model="rerank-v3.5",       # 최신 모델
        query=question,
        documents=doc_texts,
        top_n=top_n,
        return_documents=True      # 원본 문서도 반환
    )

    # 원래 문서 객체 순서 재정렬
    reranked_docs = [docs[r.index] for r in results.results]

    print(f"Cohere Rerank 완료: {len(reranked_docs)}개 문서")
    for i, r in enumerate(results.results):
        print(f"  [{i+1}] relevance_score: {r.relevance_score:.4f}")

    return reranked_docs
```

!!! tip "어떤 Reranker를 선택해야 할까?"
    | 옵션 | 장점 | 단점 | 추천 상황 |
    |------|------|------|----------|
    | `BAAI/bge-reranker-v2-m3` | 무료, 한국어 지원 | GPU 권장 | 자체 서버 운영 |
    | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 경량, 빠름 | 영어만 | 영어 전용 서비스 |
    | Cohere Rerank API | 설치 불필요, 최신 모델 | 비용 발생 | 빠른 프로토타입 |
    | Jina Reranker API | 무료 플랜 있음 | 사용량 제한 | 소규모 서비스 |

#### 이 기법이 해결하는 문제

| 지표 | 벡터 검색만 | 벡터 + Reranking |
|------|------------|-----------------|
| 검색 정밀도 (Precision@5) | 65% | 85% |
| 검색 속도 | 50ms | 500ms (10배 느림) |
| 비용 | 낮음 | 중간 |
| 복잡도 | 낮음 | 중간 |

---

## 3. 하이브리드 검색 (Hybrid Search)

### 이 기법이 해결하는 문제

**문제**: 벡터 검색은 "의미"를 잘 찾지만, "정확한 단어"를 찾는 데는 약합니다.

```
질문: "AttributeError: 'NoneType' object has no attribute 'split' 해결법"

벡터 검색 결과 (의미 기반):
→ "Python 문자열 처리 오류 해결 방법" (오류 코드가 정확히 매칭 안 됨)

키워드 검색 (BM25) 결과:
→ "AttributeError NoneType split 오류 원인과 해결" (정확한 에러 메시지 매칭!)
```

### BM25 알고리즘이란?

!!! info "BM25 (Best Match 25) 쉽게 이해하기"
    BM25는 **단어 빈도** 기반의 전통적인 정보 검색 알고리즘입니다.
    구글 검색이 나오기 전부터 사용된, 검증된 방법입니다.

    핵심 아이디어:
    1. **TF (단어 빈도)**: 문서에 검색어가 많이 나올수록 관련성 높음
    2. **IDF (역문서 빈도)**: 흔한 단어("의", "그", "은")는 중요도 낮음
    3. **문서 길이 정규화**: 긴 문서가 무조건 유리하지 않도록 보정

    예시:
    - "Python error" 검색 시
    - 문서 A: "Python error" 1번 등장, 길이 10단어 → 높은 점수
    - 문서 B: "Python error" 1번 등장, 길이 1000단어 → 낮은 점수 (희석됨)

#### 현실 세계 비유

> **하이브리드 검색은 "구글 검색 + 의미 검색"의 결합**입니다.
>
> - 구글 검색 (BM25): "NullPointerException"이라는 정확한 단어를 찾음
> - 의미 검색 (Vector): "null 포인터 예외", "참조 없는 객체 접근 오류"도 찾음
> - 결합: 두 방법이 모두 찾은 문서는 확실히 관련 있음!

### RRF (Reciprocal Rank Fusion) - 두 결과를 합치는 방법

```
[벡터 검색 순위]    [BM25 검색 순위]    [RRF 합산 점수]
1. 문서 A            1. 문서 C           문서 B: 0.48 ← 1위
2. 문서 B            2. 문서 B           문서 A: 0.45 ← 2위
3. 문서 D            3. 문서 A           문서 C: 0.32 ← 3위
4. 문서 E            4. 문서 F           ...

RRF 공식: score(d) = Σ 1/(k + rank(d))
  k = 60 (기본값), rank = 해당 검색 방법에서의 순위
```

### 완전한 코드 예제

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List

def build_hybrid_retriever(
    chunks: List[Document],
    vectorstore: Chroma,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
    k: int = 5
) -> EnsembleRetriever:
    """
    하이브리드 검색기 구축

    Args:
        chunks: 원본 문서 청크 (BM25용)
        vectorstore: 벡터 데이터베이스
        bm25_weight: BM25 검색 가중치 (합이 1.0이 되어야 함)
        vector_weight: 벡터 검색 가중치
        k: 각 검색기당 반환 문서 수

    Returns:
        앙상블 검색기
    """
    # BM25 검색기 설정 (키워드 기반)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # 벡터 검색기 설정 (의미 기반)
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    # 앙상블: RRF로 두 결과 결합
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight]
        # weights: 가중치 합이 1.0이 되어야 함
        # 기본값 [0.5, 0.5] = 동등한 가중치
        # [0.3, 0.7] = 벡터 검색 더 중요
    )

    return hybrid_retriever

# 사용 예시
# hybrid_retriever = build_hybrid_retriever(chunks, vectorstore)
# docs = hybrid_retriever.invoke("AttributeError NoneType 해결법")

# ──────────────────────────────────────────────────
# 직접 RRF 구현 (더 세밀한 제어가 필요할 때)
# ──────────────────────────────────────────────────

def reciprocal_rank_fusion(
    results_list: List[List[Document]],
    k: int = 60
) -> List[Document]:
    """
    여러 검색 결과를 RRF로 합산

    Args:
        results_list: 각 검색기의 결과 리스트 (예: [[docs_from_bm25], [docs_from_vector]])
        k: RRF 파라미터 (보통 60)

    Returns:
        합산 점수 기준으로 정렬된 문서 리스트
    """
    scores = {}  # {doc_content: rrf_score}
    doc_map = {}  # {doc_content: doc_object}

    for results in results_list:
        for rank, doc in enumerate(results):
            content = doc.page_content
            # RRF 공식: 1 / (k + rank + 1)
            rrf_score = 1.0 / (k + rank + 1)

            if content in scores:
                scores[content] += rrf_score  # 여러 결과에 있으면 합산
            else:
                scores[content] = rrf_score
                doc_map[content] = doc

    # 점수 내림차순 정렬
    sorted_contents = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[content] for content in sorted_contents]

# 사용 예시
# bm25_results = bm25_retriever.invoke(query)
# vector_results = vector_retriever.invoke(query)
# fused = reciprocal_rank_fusion([bm25_results, vector_results])
```

#### 언제 사용하고, 언제 사용하지 말아야 하나?

| 질문 유형 | BM25 강점 | Vector 강점 | 권장 비율 |
|----------|----------|------------|----------|
| 에러 코드, 고유명사 | ✅ 강함 | ❌ 약함 | BM25 70% |
| 개념, 의미 이해 | ❌ 약함 | ✅ 강함 | Vector 70% |
| 일반적인 질문 | 보통 | 보통 | 50:50 |
| 코드 검색 | ✅ 강함 | 보통 | BM25 60% |

---

## 4. Contextual Compression (문맥 압축)

### 이 기법이 해결하는 문제

**문제**: 검색된 문서는 보통 길고, **실제 질문과 관련된 부분은 일부**입니다.

```
질문: "RAG에서 청킹 전략은?"

검색된 문서 (2000 토큰):
"RAG(Retrieval-Augmented Generation)는 2020년 Facebook AI Research에서 제안한
기술입니다. 초기에는 오픈도메인 질문 답변에 사용되었으며... (중략)
...청킹 전략에는 여러 방법이 있습니다. 고정 크기 청킹은 512토큰씩 나누고...
...이후 RAG는 다양한 분야에 적용되었으며 의료, 법률 등에서도 활용됩니다..."

실제 필요한 부분 (200 토큰):
"청킹 전략에는 여러 방법이 있습니다. 고정 크기 청킹은 512토큰씩 나누고..."
```

LLM 컨텍스트 윈도우를 낭비하지 않고, 정확도를 높이기 위해 압축이 필요합니다.

#### 현실 세계 비유

> **Contextual Compression은 "사서가 관련 페이지만 복사해주기"와 같습니다.**
>
> 도서관에서 책 한 권을 빌려오는 대신, 사서가 질문과 관련된 페이지만 골라 복사본을 만들어줍니다. LLM이 300페이지 책 전체를 읽는 대신 5페이지 발췌문만 보면 됩니다.

### 완전한 코드 예제

```python
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMChainFilter,
    EmbeddingsFilter
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ──────────────────────────────────────────────────
# 방법 1: LLM 기반 추출 (가장 정확, 가장 비쌈)
# ──────────────────────────────────────────────────
def build_llm_extractor_retriever(vectorstore: Chroma):
    """LLM이 관련 부분만 추출하는 압축 검색기"""
    # LLMChainExtractor: LLM을 사용해 각 문서에서 관련 내용만 추출
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
    )
    return compression_retriever

# ──────────────────────────────────────────────────
# 방법 2: LLM 기반 필터링 (불관련 문서 제거)
# ──────────────────────────────────────────────────
def build_llm_filter_retriever(vectorstore: Chroma):
    """LLM이 관련 없는 문서를 통째로 제거하는 검색기"""
    # LLMChainFilter: 관련 없는 문서를 완전히 제거 (부분 추출 X)
    _filter = LLMChainFilter.from_llm(llm)

    filter_retriever = ContextualCompressionRetriever(
        base_compressor=_filter,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 8})
    )
    return filter_retriever

# ──────────────────────────────────────────────────
# 방법 3: 임베딩 기반 필터링 (LLM 없이, 빠름)
# ──────────────────────────────────────────────────
def build_embedding_filter_retriever(vectorstore: Chroma, threshold: float = 0.76):
    """
    임베딩 유사도로 관련 없는 문서를 제거하는 검색기

    threshold: 이 유사도 이하의 문서는 제거 (0~1, 높을수록 엄격)
    """
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=threshold
    )

    embedding_filter_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 8})
    )
    return embedding_filter_retriever

# 사용 및 비교 예시
def compare_compression_methods(question: str, vectorstore: Chroma):
    """세 가지 압축 방법 비교"""
    print(f"질문: {question}\n")
    print("="*60)

    # 압축 없는 기본 검색
    basic_docs = vectorstore.similarity_search(question, k=3)
    basic_total_tokens = sum(len(d.page_content.split()) for d in basic_docs)
    print(f"[기본 검색] 문서 수: {len(basic_docs)}, 총 단어 수: {basic_total_tokens}")

    # LLM 추출 압축
    extractor = build_llm_extractor_retriever(vectorstore)
    extracted_docs = extractor.invoke(question)
    extracted_tokens = sum(len(d.page_content.split()) for d in extracted_docs)
    print(f"[LLM 추출] 문서 수: {len(extracted_docs)}, 총 단어 수: {extracted_tokens}")
    print(f"  압축률: {(1 - extracted_tokens/basic_total_tokens)*100:.1f}% 감소")

    # 임베딩 필터
    emb_filter = build_embedding_filter_retriever(vectorstore)
    filtered_docs = emb_filter.invoke(question)
    filtered_tokens = sum(len(d.page_content.split()) for d in filtered_docs)
    print(f"[임베딩 필터] 문서 수: {len(filtered_docs)}, 총 단어 수: {filtered_tokens}")

    return extracted_docs
```

#### 성능 vs 비용 트레이드오프

| 방법 | 정확도 | 속도 | 비용 | 추천 상황 |
|------|--------|------|------|----------|
| LLMChainExtractor | ⭐⭐⭐⭐⭐ | 느림 | 높음 | 정밀도 최우선 |
| LLMChainFilter | ⭐⭐⭐⭐ | 보통 | 중간 | 불관련 문서 많을 때 |
| EmbeddingsFilter | ⭐⭐⭐ | 빠름 | 낮음 | 대량 처리, 비용 절감 |

---

## 5. Self-RAG (자기 반성 RAG)

### 이 기법이 해결하는 문제

**문제**: RAG 시스템이 잘못된 답변을 내도 그냥 반환합니다.

```
[기존 RAG]
검색 → 생성 → 답변 반환 (품질 검사 없음)

문제 상황:
- 검색된 문서가 질문과 관련 없음 → 엉뚱한 답변
- 컨텍스트 부족 → 불완전한 답변
- LLM이 컨텍스트 무시하고 환각 → 잘못된 답변
```

#### 현실 세계 비유

> **Self-RAG는 "답안지 제출 전에 스스로 검토하는 학생"과 같습니다.**
>
> 시험을 마친 학생이 제출 전에 "내가 이 질문에 정말 답했나?", "내 답변이 문제에서 벗어나진 않았나?"를 스스로 검토합니다. 문제가 있으면 다시 씁니다.

### Self-RAG 평가 기준

```
[Self-RAG 4단계 평가]

1. Retrieve? (검색 필요성 판단)
   "이 질문에 외부 문서가 필요한가?"
   → Yes: 검색 진행 / No: LLM 직접 답변

2. ISREL? (검색 결과 관련성)
   "검색된 문서가 질문에 관련 있는가?"
   → 관련 있음: 생성 진행 / 관련 없음: 재검색

3. ISSUP? (답변의 문서 근거)
   "답변 내용이 문서에 근거하는가?" (환각 탐지)
   → 근거 있음: OK / 근거 없음: 재생성

4. ISUSE? (최종 유용성)
   "답변이 질문을 충분히 해결하는가?"
   → 충분: 반환 / 불충분: 쿼리 개선 후 재시도
```

### 완전한 코드 예제

!!! warning "구현 참고"
    아래 코드는 Self-RAG의 핵심 아이디어(검색 필요성 판단 → 관련성 평가 → 근거 확인 → 유용성 평가)를 **일반 LLM 프롬프트로 근사 구현**한 것입니다. 원본 Self-RAG 논문(Asai et al., 2023)은 `[Retrieve]`, `[ISREL]`, `[ISSUP]`, `[ISUSE]` 같은 **특수 반성 토큰(reflection tokens)**을 학습한 파인튜닝 모델을 사용하며, 이 코드와 동작 방식이 다릅니다. 실제 원본 구현은 [selfrag/selfrag_llama2_7b](https://huggingface.co/selfrag/selfrag_llama2_7b) 모델을 참고하세요.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from typing import Dict, Optional

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 관련성 평가 프롬프트
relevance_prompt = ChatPromptTemplate.from_template("""
다음 문서가 질문에 관련이 있는지 평가하세요.

질문: {question}
문서: {document}

다음 JSON만 출력하세요 (다른 텍스트 없이):
{{"relevant": true/false, "reason": "이유 한 줄"}}
""")

# 근거 평가 프롬프트 (환각 탐지)
grounded_prompt = ChatPromptTemplate.from_template("""
다음 답변이 컨텍스트(문서)에 근거하는지 평가하세요.
컨텍스트에 없는 내용을 만들어냈다면 grounded=false입니다.

질문: {question}
컨텍스트: {context}
답변: {answer}

다음 JSON만 출력하세요:
{{"grounded": true/false, "hallucinated_parts": "환각된 내용 (없으면 null)"}}
""")

# 충분성 평가 프롬프트
sufficient_prompt = ChatPromptTemplate.from_template("""
다음 답변이 질문에 충분히 답했는지 평가하세요.

질문: {question}
답변: {answer}

다음 JSON만 출력하세요:
{{"sufficient": true/false, "missing": "부족한 부분 (없으면 null)"}}
""")

def parse_json_response(response: str) -> dict:
    """LLM JSON 응답 파싱 (오류 처리 포함)"""
    try:
        # JSON 블록 추출
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        return json.loads(response.strip())
    except Exception as e:
        # 파싱 실패 시 기본값
        import logging
        logging.warning(f"Self-RAG 응답 파싱 실패: {e}")
        return {"relevant": True, "grounded": True, "sufficient": True}

def self_rag(
    question: str,
    vectorstore,
    max_retries: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Self-RAG: 자기 평가 및 재시도를 포함한 RAG

    Args:
        question: 사용자 질문
        vectorstore: 벡터 데이터베이스
        max_retries: 최대 재시도 횟수
        verbose: 진행 상황 출력 여부

    Returns:
        {"answer": "...", "attempts": N, "quality": {...}}
    """
    current_question = question
    history = []

    for attempt in range(max_retries + 1):
        if verbose:
            print(f"\n{'='*50}")
            print(f"시도 {attempt + 1}/{max_retries + 1}")
            print(f"검색 쿼리: {current_question}")

        # Step 1: 문서 검색
        docs = vectorstore.similarity_search(current_question, k=4)

        # Step 2: 각 문서 관련성 평가 (ISREL)
        relevant_docs = []
        for doc in docs:
            rel_chain = relevance_prompt | llm | StrOutputParser()
            rel_response = rel_chain.invoke({
                "question": question,  # 원본 질문으로 평가
                "document": doc.page_content[:500]
            })
            rel_result = parse_json_response(rel_response)

            if rel_result.get("relevant", True):
                relevant_docs.append(doc)
                if verbose:
                    print(f"  ✓ 관련 문서: {doc.page_content[:60]}...")
            else:
                if verbose:
                    print(f"  ✗ 비관련 문서 제외: {rel_result.get('reason', '')}")

        if not relevant_docs:
            if verbose:
                print("  관련 문서 없음 → 쿼리 개선 후 재시도")
            # 쿼리 개선
            current_question = llm.invoke(
                f"'{current_question}'를 더 구체적이고 다른 표현으로 바꿔주세요. 한 문장만:"
            ).content
            history.append({"attempt": attempt, "issue": "no_relevant_docs"})
            continue

        # Step 3: 답변 생성
        context = "\n\n".join(
            f"[문서 {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        )

        answer = llm.invoke(
            f"다음 컨텍스트를 바탕으로 질문에 답하세요.\n\n"
            f"컨텍스트:\n{context}\n\n"
            f"질문: {question}\n\n"
            f"답변 (컨텍스트에 없는 내용은 포함하지 마세요):"
        ).content

        # Step 4: 근거 평가 (ISSUP) - 환각 탐지
        grounded_chain = grounded_prompt | llm | StrOutputParser()
        grounded_response = grounded_chain.invoke({
            "question": question,
            "context": context[:2000],
            "answer": answer
        })
        grounded_result = parse_json_response(grounded_response)

        if verbose:
            grounded_status = "✓" if grounded_result.get("grounded") else "✗"
            print(f"  근거 평가: {grounded_status} grounded={grounded_result.get('grounded')}")
            if not grounded_result.get("grounded"):
                print(f"  환각 탐지: {grounded_result.get('hallucinated_parts', '')}")

        # Step 5: 충분성 평가 (ISUSE)
        sufficient_chain = sufficient_prompt | llm | StrOutputParser()
        sufficient_response = sufficient_chain.invoke({
            "question": question,
            "answer": answer
        })
        sufficient_result = parse_json_response(sufficient_response)

        if verbose:
            sufficient_status = "✓" if sufficient_result.get("sufficient") else "✗"
            print(f"  충분성 평가: {sufficient_status} sufficient={sufficient_result.get('sufficient')}")

        # 품질 기준 통과 시 반환
        if (grounded_result.get("grounded", True) and
                sufficient_result.get("sufficient", True)):
            if verbose:
                print(f"\n✅ 품질 기준 통과! (시도 {attempt + 1}회)")
            return {
                "answer": answer,
                "attempts": attempt + 1,
                "quality": {
                    "grounded": grounded_result.get("grounded"),
                    "sufficient": sufficient_result.get("sufficient")
                },
                "source_docs": relevant_docs
            }

        # 품질 기준 미통과 → 쿼리 개선
        missing = sufficient_result.get("missing", "")
        if missing:
            current_question = llm.invoke(
                f"원래 질문: '{question}'\n"
                f"부족한 부분: '{missing}'\n"
                f"이 부족한 부분을 보완하는 검색 쿼리를 만들어주세요. 한 문장만:"
            ).content

        history.append({
            "attempt": attempt,
            "issue": "quality_failed",
            "grounded": grounded_result.get("grounded"),
            "sufficient": sufficient_result.get("sufficient")
        })

    # 최대 재시도 후 마지막 답변 반환
    if verbose:
        print(f"\n⚠️ 최대 재시도({max_retries}회) 도달. 마지막 답변 반환")
    return {
        "answer": answer,
        "attempts": max_retries + 1,
        "quality": {"grounded": False, "sufficient": False},
        "source_docs": relevant_docs if relevant_docs else []
    }
```

---

## 6. Corrective RAG (CRAG)

!!! info "CRAG이란?"
    **Corrective RAG**는 Self-RAG의 발전된 형태입니다. 검색 결과의 품질을 평가하고,
    품질이 낮으면 **웹 검색으로 보완**하는 전략입니다.

### CRAG 동작 원리

```
[CRAG 흐름도]

사용자 질문
    ↓
벡터 DB 검색
    ↓
검색 결과 품질 평가
    ↓
┌───────────────────────────────────────┐
│  HIGH (품질 좋음)    → 그대로 사용    │
│  AMBIGUOUS (불명확)  → 벡터 + 웹 보완 │
│  LOW (품질 나쁨)     → 웹 검색으로 대체│
└───────────────────────────────────────┘
    ↓
정제 및 필터링
    ↓
최종 답변 생성
```

### 완전한 코드 예제

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 웹 검색 도구 (Tavily API 필요)
web_search = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

# 검색 품질 평가 프롬프트
quality_eval_prompt = ChatPromptTemplate.from_template("""
다음 문서들이 질문에 답하기에 충분한지 평가하세요.

질문: {question}
검색된 문서들:
{documents}

평가 결과를 다음 중 하나로만 답하세요:
- HIGH: 문서들이 질문에 충분히 답할 수 있음
- AMBIGUOUS: 부분적으로 관련 있지만 불완전함
- LOW: 문서들이 질문과 관련 없거나 완전히 부족함

평가:
""")

def corrective_rag(
    question: str,
    vectorstore,
    use_web_search: bool = True
) -> dict:
    """
    Corrective RAG: 검색 품질에 따라 전략을 조정

    Args:
        question: 사용자 질문
        vectorstore: 로컬 벡터 DB
        use_web_search: 웹 검색 사용 여부

    Returns:
        {"answer": "...", "strategy": "...", "sources": [...]}
    """
    print(f"질문: {question}\n")

    # Step 1: 로컬 벡터 DB 검색
    local_docs = vectorstore.similarity_search(question, k=4)
    docs_text = "\n\n".join(
        f"[{i+1}] {doc.page_content[:300]}"
        for i, doc in enumerate(local_docs)
    )

    # Step 2: 검색 품질 평가
    quality_chain = quality_eval_prompt | llm | StrOutputParser()
    quality = quality_chain.invoke({
        "question": question,
        "documents": docs_text
    }).strip().upper()

    print(f"검색 품질 평가: {quality}")

    # Step 3: 품질에 따른 전략 선택
    final_docs = []
    strategy = ""

    if quality == "HIGH":
        # 로컬 검색 결과만 사용
        strategy = "local_only"
        final_docs = local_docs
        print("전략: 로컬 DB 결과만 사용")

    elif quality == "AMBIGUOUS":
        # 로컬 + 웹 검색 결합
        strategy = "hybrid_web"
        final_docs = local_docs  # 로컬 결과 포함

        if use_web_search:
            print("전략: 로컬 + 웹 검색 결합")
            try:
                web_results = web_search.invoke(question)
                for result in web_results:
                    from langchain_core.documents import Document
                    final_docs.append(Document(
                        page_content=result.get("content", ""),
                        metadata={"source": result.get("url", "web"), "type": "web"}
                    ))
                print(f"웹 검색 결과 {len(web_results)}개 추가")
            except Exception as e:
                print(f"웹 검색 실패: {e}")

    else:  # LOW
        # 웹 검색으로 완전 대체
        strategy = "web_only"
        print("전략: 웹 검색으로 완전 대체")

        if use_web_search:
            try:
                web_results = web_search.invoke(question)
                from langchain_core.documents import Document
                final_docs = [
                    Document(
                        page_content=result.get("content", ""),
                        metadata={"source": result.get("url", "web"), "type": "web"}
                    )
                    for result in web_results
                ]
                print(f"웹 검색 결과 {len(final_docs)}개 사용")
            except Exception as e:
                print(f"웹 검색 실패, 로컬 결과 사용: {e}")
                final_docs = local_docs
        else:
            final_docs = local_docs

    # Step 4: 최종 답변 생성
    context = "\n\n".join(
        f"[출처: {doc.metadata.get('source', 'local')}]\n{doc.page_content}"
        for doc in final_docs[:5]
    )

    answer = llm.invoke(
        f"다음 정보를 바탕으로 질문에 답하세요.\n\n"
        f"정보:\n{context}\n\n"
        f"질문: {question}\n\n답변:"
    ).content

    return {
        "answer": answer,
        "strategy": strategy,
        "quality_assessment": quality,
        "sources": [doc.metadata.get("source", "local") for doc in final_docs[:5]]
    }
```

---

## 7. Adaptive RAG

!!! info "Adaptive RAG란?"
    **Adaptive RAG**는 질문의 복잡도를 판단하여 **가장 적합한 처리 전략을 자동 선택**합니다.
    단순한 질문에는 가벼운 전략을, 복잡한 질문에는 무거운 전략을 씁니다.

### 질문 복잡도별 전략

```
[질문 복잡도 분류]

SIMPLE (단순)
  예: "RAG가 뭐야?", "Python 버전은?"
  → 직접 LLM 답변 or 단순 검색 1회
  비용: 낮음, 속도: 빠름

MODERATE (중간)
  예: "FastAPI vs Flask 비교해줘"
  → 기본 RAG + Reranking
  비용: 중간, 속도: 중간

COMPLEX (복잡)
  예: "우리 시스템에서 병목이 발생하는 이유와 최적화 방법 제시"
  → Query Decomposition + Multi-Query + Reranking + Self-RAG
  비용: 높음, 속도: 느림
```

### 완전한 코드 예제

```python
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 복잡도 분류 프롬프트
classify_prompt = ChatPromptTemplate.from_template("""
다음 질문의 복잡도를 분류하세요.

질문: {question}

분류 기준:
- simple: 단일 사실, 정의, 간단한 How-to
- moderate: 비교, 설명, 여러 개념 연관
- complex: 다단계 분석, 여러 도메인 통합, 추론 필요

다음 중 하나만 답하세요: simple / moderate / complex
""")

def adaptive_rag(
    question: str,
    vectorstore,
    reranker=None
) -> dict:
    """
    Adaptive RAG: 질문 복잡도에 따라 전략 자동 선택

    Args:
        question: 사용자 질문
        vectorstore: 벡터 데이터베이스
        reranker: CrossEncoder 재순위매김 모델 (선택)

    Returns:
        {"answer": "...", "strategy": "...", "complexity": "..."}
    """
    # Step 1: 복잡도 분류
    classify_chain = classify_prompt | llm | StrOutputParser()
    complexity_str = classify_chain.invoke({"question": question}).strip().lower()

    try:
        complexity = QueryComplexity(complexity_str)
    except ValueError:
        complexity = QueryComplexity.MODERATE  # 기본값

    print(f"질문 복잡도: {complexity.value.upper()}")

    # Step 2: 복잡도별 전략 실행
    if complexity == QueryComplexity.SIMPLE:
        print("전략: 단순 직접 답변")

        # 단순 검색 1회
        docs = vectorstore.similarity_search(question, k=2)
        context = "\n".join(d.page_content for d in docs)

        answer = llm.invoke(
            f"컨텍스트: {context}\n\n질문: {question}\n답변:"
        ).content

        return {
            "answer": answer,
            "strategy": "simple_rag",
            "complexity": complexity.value
        }

    elif complexity == QueryComplexity.MODERATE:
        print("전략: 기본 RAG + Reranking")

        # 넓게 검색
        docs = vectorstore.similarity_search(question, k=10)

        # Reranking (reranker가 있을 때)
        if reranker:
            pairs = [(question, doc.page_content) for doc in docs]
            scores = reranker.predict(pairs)
            docs = [doc for doc, score in
                    sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:4]]
        else:
            docs = docs[:4]

        context = "\n\n".join(d.page_content for d in docs)
        answer = llm.invoke(
            f"다음 문서들을 참고하여 답변하세요.\n\n{context}\n\n질문: {question}\n답변:"
        ).content

        return {
            "answer": answer,
            "strategy": "rag_with_reranking",
            "complexity": complexity.value
        }

    else:  # COMPLEX
        print("전략: Query Decomposition + Multi-Query + Reranking")

        # 쿼리 분해
        decompose_chain = ChatPromptTemplate.from_template(
            "다음 복합 질문을 3-4개 하위 질문으로 분해하세요 (한 줄에 하나):\n{question}"
        ) | llm | StrOutputParser()

        sub_questions_text = decompose_chain.invoke({"question": question})
        sub_questions = [q.strip() for q in sub_questions_text.split("\n") if q.strip()]

        # 각 하위 질문에 대해 검색
        all_docs = []
        seen = set()
        sub_answers = []

        for sq in sub_questions:
            sub_docs = vectorstore.similarity_search(sq, k=4)
            sub_context = "\n".join(d.page_content for d in sub_docs)

            sub_ans = llm.invoke(
                f"컨텍스트: {sub_context}\n\n질문: {sq}\n답변:"
            ).content
            sub_answers.append(f"Q: {sq}\nA: {sub_ans}")

            for doc in sub_docs:
                fp = doc.page_content[:80]
                if fp not in seen:
                    seen.add(fp)
                    all_docs.append(doc)

        # 종합 답변
        synthesis = "\n\n".join(sub_answers)
        answer = llm.invoke(
            f"다음 분석들을 종합하여 원래 질문에 완전히 답하세요.\n\n"
            f"원래 질문: {question}\n\n"
            f"분석 내용:\n{synthesis}\n\n"
            f"종합 답변:"
        ).content

        return {
            "answer": answer,
            "strategy": "decomposition_multiquery",
            "complexity": complexity.value,
            "sub_questions": sub_questions
        }

# 사용 예시
# result = adaptive_rag("RAG가 뭐야?", vectorstore)
# print(f"전략: {result['strategy']}")
# print(f"답변: {result['answer']}")
```

---

## 기법 조합 레시피

!!! success "실전에서 쓰는 조합 패턴"
    단일 기법보다 **조합**이 훨씬 강력합니다. 상황별 검증된 레시피를 소개합니다.

### 레시피 1: 엔터프라이즈 QA 시스템 (범용 최강 조합)

```
[구성 요소]
HyDE + Hybrid Search + Reranking + Contextual Compression

[파이프라인]
사용자 질문
    ↓ HyDE (가상 문서 생성)
    ↓ Hybrid Search (벡터 + BM25, 각 20개 = 총 40개 후보)
    ↓ Reranking (Cross-Encoder로 Top-8 선별)
    ↓ Contextual Compression (관련 부분만 추출)
    ↓ LLM 답변 생성

[적합한 서비스]
- 기업 내부 문서 QA
- 고객 지원 챗봇
- 기술 문서 검색
```

```python
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def enterprise_qa_pipeline(
    question: str,
    vectorstore: Chroma,
    chunks: list
) -> str:
    """
    엔터프라이즈급 QA 파이프라인
    HyDE + Hybrid Search + Reranking + Compression
    """

    # ① HyDE: 가상 답변 생성
    hyde_prompt = ChatPromptTemplate.from_template(
        "다음 질문에 대한 답변이 담긴 전문 문서 단락을 100단어로 작성하세요.\n"
        "질문: {question}\n가상 문서:"
    )
    hypothetical_doc = (hyde_prompt | llm).invoke(
        {"question": question}
    ).content

    # ② Hybrid Search: 가상 문서로 검색
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 15
    vector = vectorstore.as_retriever(search_kwargs={"k": 15})
    hybrid = EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.3, 0.7]
    )
    initial_docs = hybrid.invoke(hypothetical_doc)

    # ③ Reranking: Cross-Encoder로 Top-8 선별
    pairs = [(question, doc.page_content) for doc in initial_docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, _ in scored[:8]]

    # ④ Contextual Compression: 관련 부분 추출
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever()
    )

    # 압축은 재순위된 문서에 직접 적용
    from langchain_core.documents import Document
    compressed_context = []
    for doc in reranked_docs[:5]:
        try:
            compressed = compressor.compress_documents([doc], question)
            compressed_context.extend(compressed)
        except Exception as e:
            import logging
            logging.warning(f"문서 압축 실패, 원본 사용: {e}")
            compressed_context.append(doc)

    # ⑤ 최종 답변 생성
    context = "\n\n".join(
        f"[출처: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in compressed_context[:4]
    )

    answer = llm.invoke(
        f"다음 문서들을 바탕으로 질문에 답하세요.\n\n"
        f"{context}\n\n"
        f"질문: {question}\n\n"
        f"답변 (출처 포함):"
    ).content

    return answer
```

### 레시피 2: 복합 분석 시스템

```
[구성 요소]
Adaptive RAG → Complex 경로: Query Decomposition + Reranking + Self-RAG

[파이프라인]
복합 질문
    ↓ 복잡도 판단 (COMPLEX)
    ↓ Query Decomposition (3-5개 하위 질문)
    ↓ 각 하위 질문에 대해:
       └ Hybrid Search + Reranking
    ↓ 하위 답변들 종합
    ↓ Self-RAG로 품질 검증
    ↓ 최종 답변

[적합한 서비스]
- 리서치 보조 도구
- 컨설팅 보고서 생성
- 경쟁 분석 시스템
```

### 레시피 3: 경량 실시간 챗봇

```
[구성 요소]
Multi-Query + Hybrid Search (Reranking 생략으로 속도 최적화)

[파이프라인]
사용자 질문
    ↓ Multi-Query (3개 변형, 병렬 처리)
    ↓ Hybrid Search (각 쿼리, 병렬)
    ↓ RRF 합산 후 Top-5
    ↓ LLM 답변 (스트리밍)

[특징]
- Cross-Encoder 생략 → 응답 속도 빠름
- 병렬 처리로 Multi-Query 지연 최소화
- 스트리밍으로 체감 속도 개선
```

---

## 성능 비교

!!! note "실제 측정 기준 안내"
    아래 수치는 일반적인 RAG 벤치마크 연구 결과 기반 추정치입니다.
    실제 성능은 데이터셋, 언어, 도메인에 따라 다릅니다.

### 검색 품질 비교

| 기법 | Recall@10 | Precision@5 | MRR | 지연시간 |
|------|-----------|-------------|-----|---------|
| Naive RAG (기준선) | 70% | 58% | 0.52 | 200ms |
| + HyDE | 75% | 62% | 0.58 | 800ms |
| + Multi-Query | 82% | 64% | 0.61 | 600ms |
| + Hybrid Search | 80% | 68% | 0.64 | 300ms |
| + Reranking | 72% | 78% | 0.74 | 700ms |
| Hybrid + Reranking | 85% | 82% | 0.79 | 900ms |
| 전체 조합 | **88%** | **87%** | **0.85** | 2500ms |

> - **Recall@10**: 상위 10개 결과 중 관련 문서 포함 비율
> - **Precision@5**: 상위 5개 결과 중 실제 관련 문서 비율
> - **MRR**: Mean Reciprocal Rank (1위에 얼마나 관련 문서가 오는지)

### 답변 품질 비교

| 기법 | 정확도 | 완전성 | 환각율 |
|------|--------|--------|--------|
| Naive RAG | 72% | 65% | 18% |
| + Query Decomposition | 80% | 82% | 15% |
| + Reranking | 82% | 71% | 12% |
| + Self-RAG | 85% | 78% | 8% |
| + CRAG | 84% | 80% | 9% |
| 최적 조합 | **91%** | **88%** | **5%** |

### 비용 vs 성능 트레이드오프

```
[저비용 / 빠른 속도]
Naive RAG → Hybrid Search → +Reranking → +HyDE/Multi-Query → +Self-RAG/CRAG
    ↑                                                                    ↑
  최소 비용                                                           최고 성능
  (단순 서비스)                                                     (고정밀 필요 시)
```

---

## 자주 묻는 질문 (FAQ)

??? question "HyDE와 Multi-Query 중 어떤 걸 먼저 써야 하나요?"
    **HyDE**는 질문이 모호하거나 짧을 때 효과적입니다 (질문-문서 표현 갭 해소).
    **Multi-Query**는 질문이 특정 관점에 편향될 수 있을 때 효과적입니다 (검색 커버리지 확장).
    두 기법을 결합할 수도 있습니다: HyDE로 가상 문서를 만들고, 그 가상 문서의 변형 쿼리를 여러 개 생성하는 방식입니다.

??? question "Reranking 모델은 얼마나 자주 업데이트해야 하나요?"
    일반적으로 `BAAI/bge-reranker-v2-m3`처럼 공개 모델을 사용한다면 모델 자체는 자주 바꿀 필요가 없습니다. 하지만 도메인 특화 데이터로 파인튜닝하면 성능이 크게 향상됩니다. 처음에는 기존 모델로 시작하고, 사용자 피드백 데이터가 쌓이면 파인튜닝을 고려하세요.

??? question "BM25는 한국어에도 잘 동작하나요?"
    기본 BM25는 공백 기준 토크나이징을 사용하기 때문에 한국어에서 성능이 제한적입니다. 한국어 형태소 분석기(konlpy, kiwi 등)와 결합하면 성능이 크게 향상됩니다.

    ```python
    # 한국어 BM25 예시
    from kiwipiepy import Kiwi
    from langchain_community.retrievers import BM25Retriever

    kiwi = Kiwi()

    def korean_tokenizer(text: str) -> list:
        """한국어 형태소 분석 토크나이저"""
        tokens = kiwi.analyze(text)[0][0]
        return [token.form for token in tokens if token.tag not in ['SW', 'SB', 'SF']]

    # 한국어 형태소 분석 적용
    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=korean_tokenizer
    )
    ```

??? question "Self-RAG가 무한 루프에 빠질 수 있나요?"
    네, 그래서 `max_retries` 파라미터가 중요합니다. 실전에서는 2-3회로 제한하는 것이 좋습니다. 또한 재시도 시 쿼리를 **다른 방향으로** 개선하는 로직이 없으면 같은 문서를 반복 검색하게 됩니다. 위 코드 예제처럼 실패 원인을 분석하여 쿼리를 의미 있게 변형하는 것이 중요합니다.

??? question "컨텍스트 압축 시 중요한 정보가 잘릴 수 있지 않나요?"
    네, 가능합니다. `LLMChainExtractor`는 LLM이 판단하는 "관련 있는" 부분만 추출하므로, LLM의 판단이 틀리면 중요한 정보가 제거될 수 있습니다. 따라서 다음을 권장합니다:
    - 압축 전/후를 로깅하여 모니터링
    - `EmbeddingsFilter`처럼 덜 공격적인 방법부터 시작
    - 중요 문서는 압축 없이 전달하는 화이트리스트 구현

??? question "CRAG에서 Tavily 없이 웹 검색이 가능한가요?"
    네, 대안이 있습니다:
    - **SerpAPI**: Google 검색 결과 API
    - **DuckDuckGo**: 무료, `langchain-community`에 내장
    - **Bing Search API**: Microsoft Azure를 통해 제공

    ```python
    from langchain_community.tools import DuckDuckGoSearchRun
    web_search = DuckDuckGoSearchRun()
    results = web_search.run(question)
    ```

??? question "고급 RAG 기법들을 사용하면 비용이 얼마나 늘어나나요?"
    기법별 추가 LLM 호출 수를 기준으로:

    | 기법 | 추가 LLM 호출 | 추가 토큰 (대략) |
    |------|--------------|-----------------|
    | HyDE | +1회 | +200 토큰 |
    | Multi-Query | +1회 | +300 토큰 |
    | Query Decomposition | +N+1회 | +N×500 토큰 |
    | Self-RAG | +2-6회 | +1000-3000 토큰 |
    | CRAG | +1-2회 | +500 토큰 |

    비용 최적화 팁: `gpt-4o-mini`처럼 저렴한 모델을 평가/분해에 쓰고, 최종 답변 생성에만 강력한 모델을 사용하세요.

---

## 직접 해보기 (실습 과제)

### 실습 1: HyDE 효과 검증

아래 코드로 HyDE 전/후의 검색 결과를 직접 비교해보세요.

```python
def compare_hyde_vs_direct(question: str, vectorstore) -> None:
    """HyDE vs 직접 검색 비교 실험"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 직접 검색
    direct_docs = vectorstore.similarity_search(question, k=3)

    # HyDE 검색
    hypothetical = llm.invoke(
        f"다음 질문에 대한 답변 단락을 150단어로 작성하세요:\n{question}"
    ).content
    hyde_docs = vectorstore.similarity_search(hypothetical, k=3)

    # 비교 출력
    print("=== 직접 검색 결과 ===")
    for i, doc in enumerate(direct_docs):
        print(f"[{i+1}] {doc.page_content[:100]}...")

    print("\n=== HyDE 검색 결과 ===")
    for i, doc in enumerate(hyde_docs):
        print(f"[{i+1}] {doc.page_content[:100]}...")

    # 겹치는 문서 확인
    direct_set = {d.page_content[:50] for d in direct_docs}
    hyde_set = {d.page_content[:50] for d in hyde_docs}
    overlap = direct_set & hyde_set
    print(f"\n겹치는 문서: {len(overlap)}/3개")
    print("(겹치는 게 적을수록 HyDE가 새로운 관점을 찾아낸 것)")

# 실습: 여러 질문으로 테스트해보세요
# questions = [
#     "RAG의 장점은?",           # 짧은 질문
#     "청킹 전략 종류",           # 키워드형 질문
#     "대용량 문서에서 검색 성능을 최적화하려면 어떻게 해야 하나요?",  # 긴 질문
# ]
```

### 실습 2: Reranker 성능 측정

```python
def measure_reranker_improvement(
    test_queries: list,
    test_answers: list,  # 정답 문서 내용 (일부)
    vectorstore,
    reranker
) -> None:
    """
    Reranking 전/후 Precision@3 측정

    test_queries: 테스트 질문 목록
    test_answers: 각 질문에 대한 정답 문서의 키워드 (정답 판별용)
    """
    direct_hits = 0
    rerank_hits = 0
    total = len(test_queries)

    for question, answer_keyword in zip(test_queries, test_answers):
        # 직접 검색
        direct_docs = vectorstore.similarity_search(question, k=3)
        direct_relevant = any(
            answer_keyword.lower() in doc.page_content.lower()
            for doc in direct_docs
        )

        # Reranking 검색
        initial_docs = vectorstore.similarity_search(question, k=10)
        pairs = [(question, doc.page_content) for doc in initial_docs]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        top3_reranked = [doc for doc, _ in reranked[:3]]
        rerank_relevant = any(
            answer_keyword.lower() in doc.page_content.lower()
            for doc in top3_reranked
        )

        if direct_relevant:
            direct_hits += 1
        if rerank_relevant:
            rerank_hits += 1

    print(f"직접 검색 Precision@3: {direct_hits}/{total} = {direct_hits/total:.1%}")
    print(f"Reranking Precision@3: {rerank_hits}/{total} = {rerank_hits/total:.1%}")
    print(f"개선율: {(rerank_hits - direct_hits)/total:.1%}")
```

### 실습 3: 나만의 Adaptive RAG 설계

아래 질문에 답하며 자신의 서비스에 맞는 Adaptive RAG를 설계해보세요.

```
체크리스트:
□ 내 서비스의 주요 질문 유형은? (단순/중간/복잡 비율)
□ 응답 시간 요구사항은? (1초 이내? 5초 이내?)
□ 비용 예산은?
□ 사용자 피드백 수집이 가능한가? (Self-RAG 개선에 활용)
□ 웹 검색이 필요한 경우가 있는가? (CRAG 도입 여부)

설계 템플릿:
SIMPLE 질문 (전체의 ___%) → 전략: ___________
MODERATE 질문 (전체의 ___%) → 전략: ___________
COMPLEX 질문 (전체의 ___%) → 전략: ___________
```

---

## 핵심 요약

!!! abstract "핵심 요약"

    **쿼리 변환 (검색 전 개선)**
    - **HyDE**: 가상 답변으로 질문-문서 갭 해소 → 짧고 모호한 질문에 효과적
    - **Multi-Query**: 여러 관점으로 검색 범위 확장 → 편향된 검색 보완
    - **Query Decomposition**: 복합 질문 분해 → 여러 주제를 다루는 질문에 필수

    **검색 개선**
    - **Hybrid Search**: 벡터 + BM25 결합 → 의미 검색 + 키워드 검색의 장점 모두
    - **Reranking**: Bi-Encoder로 넓게 검색 → Cross-Encoder로 정밀 재순위

    **생성 개선**
    - **Contextual Compression**: 긴 문서에서 관련 부분만 추출 → 컨텍스트 낭비 방지
    - **Self-RAG**: 답변 자기 평가 + 재시도 → 환각 감소, 완전성 향상
    - **CRAG**: 검색 품질 평가 + 웹 검색 보완 → 로컬 DB 부족 시 대응
    - **Adaptive RAG**: 복잡도별 자동 전략 선택 → 비용 + 성능 균형

    **실전 원칙**
    1. 단일 기법보다 **조합**이 훨씬 강력하다
    2. 항상 **비용-성능 트레이드오프**를 고려하라
    3. 먼저 **Hybrid Search + Reranking**으로 시작하라 (대부분의 케이스 커버)
    4. 복잡한 질문이 많다면 **Query Decomposition** 추가
    5. 높은 정확도가 필요하면 **Self-RAG** 추가
