# 5. RAG 파이프라인 구현

## 전체 파이프라인 구조

```
[문서 수집] → [전처리] → [청킹] → [임베딩] → [벡터DB 저장]
                                                    ↓
[사용자 질문] → [쿼리 임베딩] → [벡터 검색] → [LLM 생성] → [답변]
```

---

## 완전한 RAG 파이프라인 (LangChain)

### 1단계: 문서 로드

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    WebBaseLoader
)

# PDF 로드
pdf_loader = PyPDFLoader("document.pdf")
pdf_docs = pdf_loader.load()

# 디렉토리 전체 로드
dir_loader = DirectoryLoader(
    "docs/",
    glob="**/*.md",
    loader_cls=TextLoader
)
all_docs = dir_loader.load()

# 웹페이지 로드
web_loader = WebBaseLoader("https://example.com/article")
web_docs = web_loader.load()

# 통합
documents = pdf_docs + all_docs + web_docs
print(f"총 {len(documents)}개 문서 로드됨")
```

### 2단계: 전처리 & 청킹

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def preprocess(text: str) -> str:
    """문서 전처리"""
    text = re.sub(r'\n{3,}', '\n\n', text)  # 과도한 줄바꿈 정리
    text = re.sub(r' {2,}', ' ', text)       # 과도한 공백 정리
    text = text.strip()
    return text

# 전처리 적용
for doc in documents:
    doc.page_content = preprocess(doc.page_content)

# 청킹
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"총 {len(chunks)}개 청크 생성됨")
```

### 3단계: 임베딩 & 벡터 DB 저장

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# ChromaDB에 저장 (로컬 개발용)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="rag_collection"
)
print("벡터 DB 저장 완료")
```

### 4단계: 검색 & 생성

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 검색기 설정
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 기반으로 질문에 답변하세요.
컨텍스트에 없는 내용은 "해당 정보를 찾을 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:
""")

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# LCEL 체인 구성
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 실행
answer = chain.invoke("RAG의 장점은 무엇인가요?")
print(answer)
```

---

## 출처 표시가 포함된 파이프라인

```python
from langchain_core.runnables import RunnableParallel

# 출처와 답변을 함께 반환
chain_with_sources = RunnableParallel(
    answer=chain,
    source_documents=retriever
)

result = chain_with_sources.invoke("RAG란?")

print("답변:", result["answer"])
print("\n출처:")
for doc in result["source_documents"]:
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "N/A")
    print(f"  - {source} (p.{page})")
```

---

## 스트리밍 응답

```python
# 실시간 스트리밍 (챗봇 UX)
for token in chain.stream("RAG 파이프라인을 설명해주세요"):
    print(token, end="", flush=True)
```

---

## 대화 기록이 있는 RAG (Conversational RAG)

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 대화 기록을 고려한 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", """컨텍스트를 기반으로 답변하세요.
컨텍스트: {context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

chat_history = []

def conversational_rag(question: str) -> str:
    """대화 기록을 유지하는 RAG"""
    # 검색
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # 생성
    response = llm.invoke(
        prompt.format_messages(
            context=context,
            chat_history=chat_history,
            question=question
        )
    )

    # 기록 저장
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))

    return response.content

# 대화
print(conversational_rag("RAG란 뭐야?"))
print(conversational_rag("그거 어떤 장점이 있어?"))  # "그거" = RAG를 이해
```

---

## 평가 (Evaluation)

```python
from langchain_openai import ChatOpenAI

eval_llm = ChatOpenAI(model="gpt-4o", temperature=0)

def evaluate_rag(question, generated_answer, reference_answer):
    """RAG 답변 품질 평가"""
    eval_prompt = f"""
    질문: {question}
    생성된 답변: {generated_answer}
    참조 답변: {reference_answer}

    다음 기준으로 1-5점 평가:
    1. 정확성: 참조 답변과 일치하는가?
    2. 완전성: 핵심 내용을 빠뜨리지 않았는가?
    3. 관련성: 질문에 적절히 답했는가?

    JSON 형식으로 답변: {{"accuracy": N, "completeness": N, "relevance": N}}
    """
    result = eval_llm.invoke(eval_prompt)
    return result.content

# 테스트 셋으로 평가
test_cases = [
    {"q": "RAG란?", "ref": "검색 증강 생성으로..."},
    {"q": "임베딩이란?", "ref": "텍스트를 벡터로..."},
]

for case in test_cases:
    answer = chain.invoke(case["q"])
    score = evaluate_rag(case["q"], answer, case["ref"])
    print(f"Q: {case['q']}\nScore: {score}\n")
```

---

## 핵심 요약

!!! summary "이것만 기억하자"
    1. 파이프라인: 로드 → 전처리 → 청킹 → 임베딩 → 저장 → 검색 → 생성
    2. LangChain LCEL로 체인을 간결하게 구성
    3. 출처 표시와 스트리밍은 프로덕션 필수 기능
    4. 대화형 RAG는 `chat_history`로 문맥 유지
    5. 반드시 평가 파이프라인을 구축하여 품질 측정
