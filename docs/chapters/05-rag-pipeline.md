# 5. RAG 파이프라인 구현

!!! abstract "이 챕터에서 배울 것"
    이 챕터를 마치면 여러분은:

    - RAG 파이프라인의 각 단계가 **왜** 필요한지 이해할 수 있습니다
    - LangChain의 **LCEL(LangChain Expression Language)**을 사용해 체인을 구성할 수 있습니다
    - 출처 표시, 스트리밍, 대화 기억이 있는 **완전한 RAG 시스템**을 직접 만들 수 있습니다
    - RAG 품질을 **측정하고 개선**하는 방법을 알 수 있습니다
    - 실제 서비스에 배포하기 위한 **프로덕션 체크리스트**를 알 수 있습니다

---

## 핵심 개념: 파이프라인이란?

!!! info "용어 설명: 파이프라인(Pipeline)"
    **파이프라인**이란 데이터가 여러 처리 단계를 순서대로 통과하는 구조를 말합니다.

    마치 자동차 공장의 **조립 라인**과 같습니다:

    - 1번 작업대: 차체 조립
    - 2번 작업대: 엔진 탑재
    - 3번 작업대: 도색
    - 4번 작업대: 품질 검사

    각 단계의 **출력**이 다음 단계의 **입력**이 됩니다.

    RAG 파이프라인도 동일합니다: 문서 → 청킹 → 임베딩 → 저장 → 검색 → 답변 생성

---

## 전체 파이프라인 구조

RAG 파이프라인은 크게 **두 개의 흐름**으로 나뉩니다.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [인덱싱 파이프라인] - 사전에 한번만 실행 (또는 주기적으로 업데이트)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [문서 수집]
      │  PDF, Word, 웹페이지, DB 등
      ▼
  [전처리]
      │  불필요한 문자, 공백 제거
      ▼
  [청킹 (Chunking)]
      │  긴 문서를 작은 조각으로 분할
      ▼
  [임베딩 (Embedding)]
      │  텍스트 → 숫자 벡터로 변환
      ▼
  [벡터 DB 저장]
      │  Chroma, Pinecone, Weaviate 등
      ▼
  [벡터 DB]  ◄────────────────────────────────────┐
                                                   │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┼━━━━━
 [질의응답 파이프라인] - 사용자 요청마다 실행              │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┼━━━━━

  [사용자 질문 입력]                                │
      │                                            │
      ▼                                            │
  [질문 임베딩]                                    │
      │  질문도 숫자 벡터로 변환                   │
      ▼                                            │
  [벡터 유사도 검색] ──────────────────────────────┘
      │  가장 관련 있는 청크 K개 검색
      ▼
  [프롬프트 구성]
      │  [검색된 문서] + [사용자 질문] 조합
      ▼
  [LLM 생성]
      │  GPT-4, Claude 등
      ▼
  [최종 답변]
```

!!! tip "왜 두 단계로 나누나요?"
    인덱싱은 문서가 변경될 때만 실행하면 됩니다. 매 질문마다 수백 개의 문서를 다시 처리하면 너무 느리고 비용이 많이 들기 때문입니다. 인덱싱을 미리 해두면 질의응답은 빠르게 처리할 수 있습니다.

---

## 환경 설정

시작하기 전에 필요한 라이브러리를 설치합니다.

```bash
pip install langchain langchain-openai langchain-community chromadb
pip install pypdf python-dotenv tiktoken
```

```python
# .env 파일을 만들고 API 키를 저장하세요 (절대로 코드에 직접 쓰지 마세요!)
# .env 파일 내용:
# OPENAI_API_KEY=sk-...

import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경 변수 로드

# API 키가 제대로 로드됐는지 확인
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다!")

print("환경 설정 완료!")
```

```
# 예상 출력:
환경 설정 완료!
```

---

## 1단계: 문서 로드 (Document Loading)

!!! info "용어 설명: 문서 로더(Document Loader)"
    **문서 로더**는 다양한 형식의 파일(PDF, Word, 웹페이지 등)을 읽어서 LangChain이 처리할 수 있는 표준 형식인 `Document` 객체로 변환해주는 도구입니다.

    `Document` 객체는 두 가지를 담고 있습니다:
    - `page_content`: 실제 텍스트 내용
    - `metadata`: 출처, 페이지 번호 등 부가 정보

### 왜 문서 로더가 필요한가?

각 파일 형식은 내부 구조가 전혀 다릅니다. PDF는 바이너리 포맷, Word는 XML 기반, 웹페이지는 HTML... 문서 로더가 이 복잡성을 숨겨주고, 우리는 텍스트 내용에만 집중할 수 있게 해줍니다.

### 스킵하면 어떻게 되나?

직접 `open("file.pdf", "r")`로 열면 깨진 문자만 보입니다. PDF는 단순한 텍스트 파일이 아니기 때문입니다.

```python
from langchain_community.document_loaders import (
    PyPDFLoader,      # PDF 파일용
    TextLoader,       # .txt, .md 파일용
    DirectoryLoader,  # 폴더 전체 로드용
    WebBaseLoader     # 웹페이지용
)

# ─────────────────────────────────────
# PDF 파일 로드
# ─────────────────────────────────────
pdf_loader = PyPDFLoader("document.pdf")
pdf_docs = pdf_loader.load()

# 결과 확인
print(f"PDF에서 {len(pdf_docs)}페이지 로드됨")
print(f"첫 페이지 내용 미리보기: {pdf_docs[0].page_content[:200]}")
print(f"메타데이터: {pdf_docs[0].metadata}")
# 출력 예시:
# PDF에서 15페이지 로드됨
# 첫 페이지 내용 미리보기: 인공지능의 발전과 RAG...
# 메타데이터: {'source': 'document.pdf', 'page': 0}

# ─────────────────────────────────────
# 폴더 전체 로드 (모든 .md 파일)
# ─────────────────────────────────────
dir_loader = DirectoryLoader(
    "docs/",          # 검색할 폴더
    glob="**/*.md",   # 패턴: 모든 하위 폴더의 .md 파일
    loader_cls=TextLoader,  # 텍스트 형식으로 로드
    loader_kwargs={"encoding": "utf-8"}  # 한글 깨짐 방지
)
all_docs = dir_loader.load()
print(f"폴더에서 {len(all_docs)}개 문서 로드됨")

# ─────────────────────────────────────
# 웹페이지 로드
# ─────────────────────────────────────
web_loader = WebBaseLoader(
    "https://python.langchain.com/docs/introduction/"
)
web_docs = web_loader.load()
print(f"웹에서 {len(web_docs)}개 문서 로드됨")
print(f"제목: {web_docs[0].metadata.get('title', 'N/A')}")

# ─────────────────────────────────────
# 모든 문서 합치기
# ─────────────────────────────────────
documents = pdf_docs + all_docs + web_docs
print(f"\n총 {len(documents)}개 문서 로드 완료")
```

```
# 예상 출력:
PDF에서 15페이지 로드됨
첫 페이지 내용 미리보기: 인공지능의 발전과 RAG...
메타데이터: {'source': 'document.pdf', 'page': 0}
폴더에서 8개 문서 로드됨
웹에서 1개 문서 로드됨
제목: Introduction | LangChain

총 24개 문서 로드 완료
```

!!! warning "자주 발생하는 오류"
    ```
    FileNotFoundError: [Errno 2] No such file or directory: 'document.pdf'
    ```
    파일 경로가 틀렸습니다. 절대 경로를 사용하거나, 현재 작업 디렉토리를 확인하세요:
    ```python
    import os
    print(os.getcwd())  # 현재 작업 디렉토리 확인
    ```

    ```
    UnicodeDecodeError: 'utf-8' codec can't decode
    ```
    한글 파일은 `loader_kwargs={"encoding": "utf-8"}` 또는 `"cp949"`를 추가하세요.

---

## 2단계: 전처리 & 청킹 (Preprocessing & Chunking)

### 전처리란?

!!! info "용어 설명: 전처리(Preprocessing)"
    **전처리**는 원본 텍스트에서 불필요한 부분을 제거하고 정리하는 과정입니다.

    예를 들어:
    - 연속된 빈 줄 제거: `\n\n\n\n` → `\n\n`
    - 중복 공백 제거: `안녕    하세요` → `안녕 하세요`
    - 헤더/푸터 제거: 모든 페이지에 반복되는 "회사명 | 페이지 번호" 제거

### 청킹이란?

!!! info "용어 설명: 청킹(Chunking)"
    **청킹**은 긴 문서를 작은 조각(chunk)으로 나누는 과정입니다.

    **왜 나눠야 할까요?**

    생각해보세요: 도서관에서 "파이썬 문법"을 찾을 때, 사서가 300페이지짜리 책 전체를 읽고 답해주는 게 빠를까요, 아니면 관련 페이지 2~3장만 찾아서 답해주는 게 빠를까요?

    LLM(언어 모델)도 마찬가지입니다. 컨텍스트(입력 가능한 텍스트 길이)가 제한되어 있고, 관련 내용만 넣어줄수록 더 정확한 답변을 생성합니다.

### 청킹을 스킵하면 어떻게 되나?

- 문서 전체를 벡터 하나로 만들면 **너무 일반적인** 벡터가 됩니다 (책 전체를 한 단어로 요약하는 것과 같음)
- 유사도 검색이 부정확해집니다
- LLM 컨텍스트 창을 초과할 수 있습니다

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# ─────────────────────────────────────
# 전처리 함수
# ─────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    문서 텍스트 전처리 함수

    Args:
        text: 원본 텍스트
    Returns:
        정리된 텍스트
    """
    # 3개 이상 연속된 줄바꿈을 2개로 줄임 (단락 구분은 유지)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 2개 이상 연속된 공백을 1개로 줄임
    text = re.sub(r' {2,}', ' ', text)

    # 탭 문자를 공백으로 변환
    text = text.replace('\t', ' ')

    # 앞뒤 공백 제거
    text = text.strip()

    return text

# 모든 문서에 전처리 적용
for doc in documents:
    doc.page_content = preprocess_text(doc.page_content)

print("전처리 완료!")

# 전처리 결과 확인
original = "안녕    하세요.\n\n\n\n\n파이썬을\t배워봅시다."
cleaned = preprocess_text(original)
print(f"원본: {repr(original)}")
print(f"정리: {repr(cleaned)}")
```

```
# 예상 출력:
전처리 완료!
원본: '안녕    하세요.\n\n\n\n\n파이썬을\t배워봅시다.'
정리: '안녕 하세요.\n\n파이썬을 배워봅시다.'
```

```python
# ─────────────────────────────────────
# 청킹 설정
# ─────────────────────────────────────

# RecursiveCharacterTextSplitter 설명:
# 이 스플리터는 우선순위 순서대로 분리 기준을 시도합니다:
# 1. 먼저 "\n\n" (단락)으로 나누려고 시도
# 2. 청크가 아직 크면 "\n" (줄)으로 나눔
# 3. 그래도 크면 ". " (문장)으로 나눔
# 4. 그래도 크면 " " (단어)로 나눔
# → 가능한 한 의미 있는 단위를 유지하면서 나눠줍니다

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 각 청크의 최대 글자 수
    chunk_overlap=50,    # 인접 청크 간 겹치는 글자 수 (문맥 연결성 유지)
    length_function=len, # 길이 측정 방법 (기본: 문자 수)
    separators=["\n\n", "\n", ". ", " ", ""]  # 분리 기준 (우선순위 순)
)

# 문서를 청크로 분할
chunks = splitter.split_documents(documents)

print(f"분할 전: {len(documents)}개 문서")
print(f"분할 후: {len(chunks)}개 청크")
print(f"평균 청크 크기: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f}글자")

# 첫 번째 청크 확인
print(f"\n첫 번째 청크:")
print(f"내용: {chunks[0].page_content[:200]}...")
print(f"메타데이터: {chunks[0].metadata}")

# chunk_overlap 시각화
print("\n[chunk_overlap 이해하기]")
print("청크 1: [━━━━━━━━━━━━━━━━━━━━━━━━]")
print("청크 2:                   [━━━━━━━━━━━━━━━━━━━━━━━━]")
print("         ← 500글자 →  ↑ 50글자 겹침 ↑ ← 500글자 →")
print("겹치는 부분 덕분에 청크 경계에서 문맥이 끊기지 않습니다!")
```

```
# 예상 출력:
분할 전: 24개 문서
분할 후: 187개 청크
평균 청크 크기: 423글자

첫 번째 청크:
내용: 인공지능의 발전과 RAG
최근 인공지능 기술의 발전으로...
메타데이터: {'source': 'document.pdf', 'page': 0}

[chunk_overlap 이해하기]
청크 1: [━━━━━━━━━━━━━━━━━━━━━━━━]
청크 2:                   [━━━━━━━━━━━━━━━━━━━━━━━━]
         ← 500글자 →  ↑ 50글자 겹침 ↑ ← 500글자 →
겹치는 부분 덕분에 청크 경계에서 문맥이 끊기지 않습니다!
```

!!! question "왜 왜 왜? chunk_overlap이 왜 필요한가요?"
    예를 들어 이런 문장이 있다고 해봅시다:

    ```
    ...삼성전자는 2024년에 반도체 사업에서 큰 성과를 거뒀습니다.   ← 청크 1의 끝
    이 성과는 3나노 공정 기술 덕분이었습니다...                      ← 청크 2의 시작
    ```

    청크 2만 보면 "이 성과"가 무엇인지 알 수 없습니다. 하지만 `chunk_overlap=50`으로 설정하면 청크 2가 청크 1의 마지막 50글자를 포함하므로 문맥이 자연스럽게 이어집니다.

!!! tip "최적의 청크 크기는?"
    | 용도 | 권장 chunk_size | 권장 overlap |
    |------|----------------|-------------|
    | 일반 문서 Q&A | 500~1000 | 50~100 |
    | 법률/기술 문서 | 1000~2000 | 100~200 |
    | 짧은 SNS/리뷰 | 200~300 | 20~30 |

    **정답은 없습니다!** 직접 실험해서 가장 좋은 결과를 내는 값을 찾으세요.

---

## 3단계: 임베딩 & 벡터 DB 저장

!!! info "용어 설명: 임베딩(Embedding)"
    **임베딩**은 텍스트를 숫자 배열(벡터)로 변환하는 과정입니다.

    예를 들어:
    - "고양이" → `[0.2, -0.5, 0.8, 0.1, ...]` (1536개 숫자)
    - "강아지" → `[0.3, -0.4, 0.7, 0.2, ...]` (비슷한 숫자!)
    - "자동차" → `[-0.1, 0.9, -0.3, 0.6, ...]` (많이 다른 숫자)

    **의미가 비슷한 텍스트는 비슷한 숫자 배열**을 가집니다. 이걸 이용해서 "관련 문서 찾기"를 수학적으로 처리할 수 있습니다.

!!! info "용어 설명: 벡터 DB(Vector Database)"
    **벡터 DB**는 이 숫자 배열들을 저장하고, "이 숫자 배열과 가장 비슷한 배열을 가진 문서를 찾아줘"라는 검색을 빠르게 해주는 특수한 데이터베이스입니다.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ─────────────────────────────────────
# 임베딩 모델 설정
# ─────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # dimensions=1024  # 선택사항: 벡터 차원 수 줄이기 (비용 절감)
)

# 임베딩이 어떻게 생겼는지 확인해봅시다
sample_text = "RAG는 검색 증강 생성입니다"
sample_vector = embeddings.embed_query(sample_text)

print(f"텍스트: '{sample_text}'")
print(f"벡터 차원 수: {len(sample_vector)}")
print(f"벡터 앞 5개 값: {sample_vector[:5]}")
print(f"벡터 타입: {type(sample_vector[0])}")
```

```
# 예상 출력:
텍스트: 'RAG는 검색 증강 생성입니다'
벡터 차원 수: 3072
벡터 앞 5개 값: [0.012847, -0.034521, 0.087634, -0.002134, 0.045678]
벡터 타입: <class 'float'>
```

```python
# ─────────────────────────────────────
# ChromaDB에 저장
# ─────────────────────────────────────
# ChromaDB는 로컬에 저장되는 무료 벡터 DB입니다. 개발/테스트에 아주 좋습니다.

print("벡터 DB에 저장 중... (시간이 걸릴 수 있습니다)")

vectorstore = Chroma.from_documents(
    documents=chunks,                   # 저장할 청크들
    embedding=embeddings,               # 사용할 임베딩 모델
    persist_directory="./chroma_db",    # 저장할 폴더 (영구 저장)
    collection_name="rag_collection"    # 컬렉션 이름 (테이블과 비슷한 개념)
)

print(f"벡터 DB 저장 완료!")
print(f"저장된 청크 수: {vectorstore._collection.count()}")

# ─────────────────────────────────────
# 기존 DB 불러오기 (이미 저장했다면)
# ─────────────────────────────────────
# 한번 저장한 후에는 다시 만들 필요 없이 아래 코드로 불러옵니다
vectorstore_loaded = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="rag_collection"
)
print(f"기존 DB 불러오기 완료: {vectorstore_loaded._collection.count()}개 청크")
```

```
# 예상 출력:
벡터 DB에 저장 중... (시간이 걸릴 수 있습니다)
벡터 DB 저장 완료!
저장된 청크 수: 187
기존 DB 불러오기 완료: 187개 청크
```

!!! warning "주의: API 비용"
    청크 수가 많을수록 임베딩 API 호출 비용이 늘어납니다. 개발 중에는:

    1. 소량의 문서로 테스트하세요
    2. 무료 로컬 임베딩 모델 사용을 고려하세요:
    ```python
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # 무료! 로컬에서 실행, 처음엔 모델 다운로드 필요
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"  # 한국어 지원 모델
    )
    ```

---

## 4단계: 검색기(Retriever) 설정

!!! info "용어 설명: 리트리버(Retriever)"
    **리트리버**는 질문을 받아서 벡터 DB에서 관련 문서를 찾아 반환하는 컴포넌트입니다.

    마치 도서관 사서와 같습니다: "파이썬 문법 알려줘" → 사서가 관련 책을 가져다 줌

    LangChain에서 리트리버는 `.invoke(질문)` 을 호출하면 관련 `Document` 목록을 반환합니다.

```python
# ─────────────────────────────────────
# 기본 유사도 검색 리트리버
# ─────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity",    # 코사인 유사도로 검색
    search_kwargs={
        "k": 3                   # 가장 유사한 청크 3개 반환
    }
)

# 리트리버 테스트
test_query = "RAG의 장점은 무엇인가요?"
retrieved_docs = retriever.invoke(test_query)

print(f"질문: {test_query}")
print(f"검색된 청크 수: {len(retrieved_docs)}")
print("\n" + "="*50)
for i, doc in enumerate(retrieved_docs):
    print(f"\n[청크 {i+1}]")
    print(f"출처: {doc.metadata.get('source', 'unknown')}")
    print(f"내용: {doc.page_content[:200]}...")
    print("-"*30)
```

```
# 예상 출력:
질문: RAG의 장점은 무엇인가요?
검색된 청크 수: 3

==================================================

[청크 1]
출처: document.pdf
내용: RAG(Retrieval-Augmented Generation)는 LLM의 한계를 극복하는 방법입니다.
주요 장점으로는 최신 정보 반영, 환각 현상 감소, 출처 추적 가능 등이 있습니다...
------------------------------

[청크 2]
출처: docs/intro.md
내용: LLM 단독 사용 대비 RAG의 가장 큰 장점은 지식의 업데이트가 용이하다는 점입니다...
------------------------------

[청크 3]
출처: document.pdf
내용: 특히 기업 환경에서 RAG는 내부 문서 기반의 정확한 답변 생성에 탁월한 성능을...
------------------------------
```

```python
# ─────────────────────────────────────
# MMR (Maximal Marginal Relevance) 검색
# ─────────────────────────────────────
# MMR은 관련성이 높으면서도 서로 다양한 내용의 청크를 가져옵니다
# "유사하지만 비슷비슷한 청크"를 여러 개 가져오는 문제를 해결합니다

retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,              # 최종 반환할 청크 수
        "fetch_k": 10,       # 처음 후보로 가져올 청크 수
        "lambda_mult": 0.7   # 1.0=관련성만, 0.0=다양성만 (0.7 권장)
    }
)

# similarity vs mmr 비교
print("[similarity 검색 결과 - 비슷한 내용이 중복될 수 있음]")
sim_docs = retriever.invoke("RAG 장점")
for doc in sim_docs:
    print(f"  → {doc.page_content[:80]}...")

print("\n[MMR 검색 결과 - 다양한 관점의 내용]")
mmr_docs = retriever_mmr.invoke("RAG 장점")
for doc in mmr_docs:
    print(f"  → {doc.page_content[:80]}...")
```

---

## 5단계: LCEL 체인 구성

이제 가장 핵심적인 부분입니다!

### LCEL이란?

!!! info "용어 설명: LCEL (LangChain Expression Language)"
    **LCEL**은 LangChain에서 만든 파이프라인 구성 방법입니다. `|` (파이프) 기호를 사용해서 여러 컴포넌트를 연결합니다.

    Unix 커맨드라인의 파이프와 비슷한 개념입니다:
    ```bash
    # Unix: 파일 내용 → 검색 → 정렬
    cat file.txt | grep "키워드" | sort

    # LCEL: 질문 → 프롬프트 → LLM → 파서
    chain = prompt | llm | output_parser
    ```

    LCEL의 장점:
    - 코드가 간결하고 읽기 쉬움
    - 스트리밍 자동 지원
    - 비동기 처리 자동 지원
    - 중간 단계 쉽게 추가/제거 가능

### 체인(Chain)이란?

!!! info "용어 설명: 체인(Chain)"
    **체인**은 여러 처리 단계를 이어 붙인 파이프라인을 말합니다. "체인처럼 연결되어 있다"는 의미에서 체인이라고 부릅니다.

### 프롬프트 템플릿이란?

!!! info "용어 설명: 프롬프트 템플릿(Prompt Template)"
    **프롬프트 템플릿**은 LLM에게 보낼 지시문의 틀입니다. `{변수명}` 부분이 실제 값으로 채워집니다.

    예를 들어:
    ```
    "당신은 {language} 전문가입니다. {question}에 답해주세요."
    ```
    → 실제 호출 시:
    ```
    "당신은 Python 전문가입니다. 리스트 컴프리헨션이 뭔가요?에 답해주세요."
    ```

### LCEL 시각적 구조 다이어그램

```
사용자 질문 ("RAG란?")
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  {"context": retriever | format_docs,                 │
│   "question": RunnablePassthrough()}                  │
│                                                       │
│  이 부분이 동시에 두 가지 처리를 합니다:              │
│  ┌─────────────────────┐  ┌──────────────────────┐   │
│  │ retriever           │  │ RunnablePassthrough() │   │
│  │ 질문으로 문서 검색  │  │ 질문을 그대로 통과   │   │
│  │        ↓            │  │                      │   │
│  │ format_docs         │  │                      │   │
│  │ 문서를 텍스트로 변환│  │                      │   │
│  └─────────┬───────────┘  └──────────┬───────────┘   │
│            │ context                 │ question       │
└────────────┼─────────────────────────┼───────────────┘
             └──────────────┬──────────┘
                            ▼
                   ┌────────────────┐
                   │ prompt         │
                   │ 프롬프트 구성  │
                   └───────┬────────┘
                            ▼
                   ┌────────────────┐
                   │ llm            │
                   │ GPT-4 호출     │
                   └───────┬────────┘
                            ▼
                   ┌────────────────┐
                   │ StrOutputParser│
                   │ 텍스트로 변환  │
                   └───────┬────────┘
                            ▼
              최종 답변 (문자열)
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────
# LLM 설정
# ─────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,  # 0 = 항상 같은 답변 (일관성), 1 = 창의적 (다양성)
)

# ─────────────────────────────────────
# 프롬프트 템플릿
# ─────────────────────────────────────
# {context}와 {question}은 나중에 채워질 빈칸입니다
prompt = ChatPromptTemplate.from_template("""
당신은 주어진 문서를 기반으로 정확하게 답변하는 AI 어시스턴트입니다.

아래 규칙을 반드시 따르세요:
1. 오직 주어진 컨텍스트(문서)에 있는 정보만 사용하세요
2. 컨텍스트에 없는 내용은 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요
3. 답변은 한국어로 작성하세요
4. 가능하면 구체적인 근거를 들어 설명하세요

===== 참고 문서 =====
{context}
====================

질문: {question}

답변:
""")

# ─────────────────────────────────────
# 문서 포맷 함수
# ─────────────────────────────────────
def format_docs(docs):
    """
    검색된 Document 목록을 하나의 텍스트로 합칩니다.
    각 청크는 '---' 구분선으로 나뉩니다.

    Args:
        docs: List[Document] - 검색된 문서 청크들
    Returns:
        str - 합쳐진 텍스트
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# ─────────────────────────────────────
# LCEL 체인 구성
# ─────────────────────────────────────
# 각 부분 설명:
# 1. {"context": ..., "question": ...}
#    → 딕셔너리를 만들어 prompt의 {context}와 {question}에 넣어줍니다
#
# 2. retriever | format_docs
#    → 질문을 받아 벡터DB에서 문서를 검색(retriever)하고
#      텍스트로 변환(format_docs)합니다
#
# 3. RunnablePassthrough()
#    → 입력을 그대로 통과시킵니다 (질문 문자열을 그대로 전달)
#
# 4. | prompt
#    → 딕셔너리를 프롬프트에 채워 넣습니다
#
# 5. | llm
#    → 프롬프트를 LLM에 전송하고 응답을 받습니다
#
# 6. | StrOutputParser()
#    → LLM 응답 객체에서 텍스트만 추출합니다

chain = (
    {
        "context": retriever | format_docs,  # 검색된 문서 → 텍스트
        "question": RunnablePassthrough()    # 질문 그대로 통과
    }
    | prompt           # 프롬프트 구성
    | llm              # LLM 호출
    | StrOutputParser() # 텍스트 추출
)

# ─────────────────────────────────────
# 체인 실행
# ─────────────────────────────────────
print("체인 실행 중...")
answer = chain.invoke("RAG의 장점은 무엇인가요?")
print("답변:")
print(answer)
```

```
# 예상 출력:
체인 실행 중...
답변:
RAG(Retrieval-Augmented Generation)의 주요 장점은 다음과 같습니다:

1. **최신 정보 반영**: LLM은 학습 데이터의 시점(지식 컷오프)이 고정되어 있지만,
   RAG는 실시간으로 업데이트된 문서를 검색하여 최신 정보를 활용할 수 있습니다.

2. **환각 현상 감소**: 검색된 실제 문서를 근거로 답변하므로, LLM이 없는 사실을
   만들어내는 "환각(hallucination)" 현상을 크게 줄일 수 있습니다.

3. **출처 추적 가능**: 어떤 문서를 참고했는지 추적할 수 있어 신뢰성 검증이 용이합니다.
```

!!! tip "왜 이게 중요한가?"
    LCEL의 `|` 연산자는 단순히 코드를 짧게 만들어주는 것이 아닙니다:

    - **스트리밍**: `chain.stream()`을 호출하면 모든 단계가 자동으로 스트리밍을 지원합니다
    - **배치 처리**: `chain.batch(["질문1", "질문2", ...])`로 여러 질문을 동시에 처리합니다
    - **비동기**: `await chain.ainvoke()`로 비동기 처리를 쉽게 구현합니다
    - **추적**: LangSmith와 연동하면 각 단계의 입출력을 자동으로 기록합니다

---

## 6단계: 출처 표시가 포함된 파이프라인

실제 서비스에서는 "이 답변은 어떤 문서에서 왔나요?"를 보여주는 것이 매우 중요합니다.

!!! info "용어 설명: RunnableParallel"
    **RunnableParallel**은 여러 작업을 동시에 실행하고 결과를 딕셔너리로 모아주는 컴포넌트입니다.

    ```
    입력
     │
     ├──→ [chain 실행] ──→ "답변 텍스트"
     │
     └──→ [retriever 실행] ──→ [Document, Document, Document]

    결과: {"answer": "답변 텍스트", "source_documents": [...]}
    ```

```python
from langchain_core.runnables import RunnableParallel

# ─────────────────────────────────────
# 답변 + 출처를 함께 반환하는 체인
# ─────────────────────────────────────
chain_with_sources = RunnableParallel(
    answer=chain,               # 기존 체인으로 답변 생성
    source_documents=retriever  # 검색된 원본 문서도 함께 반환
)

# 실행
result = chain_with_sources.invoke("RAG란 무엇인가요?")

# 결과는 딕셔너리 형태
print("=" * 60)
print("답변:")
print(result["answer"])

print("\n" + "=" * 60)
print("참고 문서 출처:")
for i, doc in enumerate(result["source_documents"], 1):
    source = doc.metadata.get("source", "출처 불명")
    page = doc.metadata.get("page", "N/A")
    print(f"  [{i}] {source} (페이지: {page})")
    print(f"      내용 미리보기: {doc.page_content[:100]}...")
```

```
# 예상 출력:
============================================================
답변:
RAG(Retrieval-Augmented Generation, 검색 증강 생성)는 대형 언어 모델(LLM)에
외부 지식 검색 기능을 결합한 AI 기법입니다...

============================================================
참고 문서 출처:
  [1] document.pdf (페이지: 2)
      내용 미리보기: RAG는 2020년 Facebook AI Research에서 처음 제안된 기법으로...
  [2] docs/rag-overview.md (페이지: N/A)
      내용 미리보기: 검색 증강 생성(RAG)의 핵심 아이디어는 LLM의 파라미터에 모든...
  [3] document.pdf (페이지: 5)
      내용 미리보기: RAG 시스템은 크게 인덱서(Indexer), 리트리버(Retriever), 생성기...
```

---

## 7단계: 스트리밍 응답

!!! info "용어 설명: 스트리밍(Streaming)"
    **스트리밍**은 답변이 완성될 때까지 기다리지 않고, 생성되는 단어(토큰)를 즉시즉시 화면에 표시하는 방식입니다.

    ChatGPT에서 답변이 한 글자씩 나오는 것처럼 보이는 게 바로 스트리밍입니다.

    왜 필요한가?
    - LLM이 긴 답변을 생성하는 데 수십 초가 걸릴 수 있습니다
    - 스트리밍 없이는 사용자가 화면이 멈춘 줄 알 수 있습니다
    - 스트리밍으로 첫 단어가 나오면 사용자는 "돌아가고 있구나"를 인식합니다

```python
# ─────────────────────────────────────
# 기본 스트리밍
# ─────────────────────────────────────
print("스트리밍 답변 (실시간으로 출력됩니다):")
print("-" * 40)

# chain.stream()은 generator를 반환합니다
# for 루프로 토큰이 생성될 때마다 출력합니다
for token in chain.stream("RAG 파이프라인의 각 단계를 설명해주세요"):
    print(token, end="", flush=True)
    # end=""   → 줄바꿈 없이 이어서 출력
    # flush=True → 버퍼 즉시 출력 (이게 없으면 한꺼번에 출력됨)

print("\n" + "-" * 40)
print("스트리밍 완료!")

# ─────────────────────────────────────
# 스트리밍 + 진행 표시
# ─────────────────────────────────────
import time

def stream_with_progress(query: str):
    """스트리밍과 함께 토큰 수를 표시하는 함수"""
    print(f"질문: {query}")
    print("답변: ", end="")

    token_count = 0
    start_time = time.time()

    for token in chain.stream(query):
        print(token, end="", flush=True)
        token_count += 1

    elapsed = time.time() - start_time
    print(f"\n\n[통계] 토큰 수: {token_count}, 소요 시간: {elapsed:.2f}초")

stream_with_progress("RAG가 LLM의 환각 현상을 어떻게 줄이나요?")
```

```
# 예상 출력:
스트리밍 답변 (실시간으로 출력됩니다):
----------------------------------------
RAG 파이프라인은 크게 두 단계로 구성됩니다.

**1단계: 인덱싱 파이프라인**
- 문서 수집: PDF, 웹페이지 등 다양한 소스에서 문서를 수집합니다
- 전처리: 불필요한 형식 정보를 제거하고 텍스트를 정리합니다
... (실시간으로 한 단어씩 출력됨)
----------------------------------------
스트리밍 완료!
```

---

## 8단계: 대화 기억이 있는 RAG (Conversational RAG)

!!! info "용어 설명: 대화 기억(Conversation Memory)"
    LLM은 기본적으로 **기억이 없습니다**. 매 요청은 독립적입니다.

    예를 들어:
    - 사용자: "RAG란 뭐야?"
    - AI: "RAG는 검색 증강 생성입니다..."
    - 사용자: "그거 어떤 장점이 있어?" ← **LLM은 "그거"가 뭔지 모름!**

    이를 해결하기 위해 이전 대화 내용을 매 요청에 함께 포함시킵니다.

    이것이 바로 **대화 기억(Conversation Memory)**입니다.

### 대화 흐름 다이어그램

```
1번째 질문: "RAG란 뭐야?"
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 시스템: 당신은 AI 어시스턴트입니다. 컨텍스트: {...}   │
│ 인간: RAG란 뭐야?                                   │
└─────────────────────────────────────────────────────┘
    │
    ▼ LLM 답변 → chat_history에 저장

2번째 질문: "그거 어떤 장점이 있어?"
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 시스템: 당신은 AI 어시스턴트입니다. 컨텍스트: {...}   │
│ 인간: RAG란 뭐야?                                   │  ← 이전 대화
│ AI: RAG는 검색 증강 생성으로...                     │  ← 이전 대화
│ 인간: 그거 어떤 장점이 있어?                         │  ← 현재 질문
└─────────────────────────────────────────────────────┘
    │
    ▼ LLM이 "그거" = "RAG"임을 이해하고 답변
```

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ─────────────────────────────────────
# 대화 기록을 지원하는 프롬프트 템플릿
# ─────────────────────────────────────
# MessagesPlaceholder는 대화 기록이 들어갈 자리입니다

conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 주어진 문서를 기반으로 답변하는 AI 어시스턴트입니다.

아래 규칙을 따르세요:
- 오직 컨텍스트(문서)에 있는 정보만 사용하세요
- 컨텍스트에 없으면 "해당 정보를 문서에서 찾을 수 없습니다"라고 답하세요
- 이전 대화 내용을 참고하여 문맥에 맞게 답변하세요

참고 문서:
{context}"""),
    MessagesPlaceholder("chat_history"),  # 이전 대화들이 여기에 들어감
    ("human", "{question}")               # 현재 질문
])

# ─────────────────────────────────────
# 대화 기록을 관리하는 RAG 클래스
# ─────────────────────────────────────
class ConversationalRAG:
    """대화 기억이 있는 RAG 시스템"""

    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.chat_history = []      # 대화 기록 저장 리스트
        self.max_history = 10       # 최대 저장할 대화 수 (메모리 관리)

    def ask(self, question: str) -> str:
        """
        질문을 받아 대화 기록을 고려한 답변을 반환합니다.

        Args:
            question: 사용자 질문
        Returns:
            str: AI 답변
        """
        # Step 1: 관련 문서 검색
        docs = self.retriever.invoke(question)
        context = format_docs(docs)

        # Step 2: 프롬프트 메시지 생성
        # (시스템 메시지 + 이전 대화 기록 + 현재 질문)
        messages = self.prompt.format_messages(
            context=context,
            chat_history=self.chat_history,  # 지금까지의 대화 전달
            question=question
        )

        # Step 3: LLM 호출
        response = self.llm.invoke(messages)
        answer = response.content

        # Step 4: 대화 기록에 추가
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        # 기록이 너무 길어지면 오래된 것 삭제 (메모리 관리)
        # 가장 최근 max_history 개만 유지 (1회 = 질문+답변 = 2개)
        if len(self.chat_history) > self.max_history * 2:
            self.chat_history = self.chat_history[-(self.max_history * 2):]

        return answer

    def clear_history(self):
        """대화 기록 초기화"""
        self.chat_history = []
        print("대화 기록이 초기화되었습니다.")

    def show_history(self):
        """현재 대화 기록 출력"""
        if not self.chat_history:
            print("대화 기록이 없습니다.")
            return

        for i, msg in enumerate(self.chat_history):
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            print(f"[{role}] {msg.content[:100]}...")

# ─────────────────────────────────────
# 사용 예시
# ─────────────────────────────────────
rag_bot = ConversationalRAG(retriever, llm, conversational_prompt)

# 1번째 질문
print("=" * 60)
print("사용자: RAG란 뭐야?")
answer1 = rag_bot.ask("RAG란 뭐야?")
print(f"AI: {answer1}\n")

# 2번째 질문 - "그거"가 RAG를 가리킴
print("=" * 60)
print("사용자: 그거 어떤 장점이 있어?")
answer2 = rag_bot.ask("그거 어떤 장점이 있어?")
print(f"AI: {answer2}\n")

# 3번째 질문 - 문맥 연속
print("=" * 60)
print("사용자: 구체적인 예시 들어줄 수 있어?")
answer3 = rag_bot.ask("구체적인 예시 들어줄 수 있어?")
print(f"AI: {answer3}\n")

# 대화 기록 확인
print("=" * 60)
print("현재 대화 기록:")
rag_bot.show_history()
```

```
# 예상 출력:
============================================================
사용자: RAG란 뭐야?
AI: RAG는 Retrieval-Augmented Generation(검색 증강 생성)의 줄임말입니다.
    LLM(대형 언어 모델)에 외부 문서 검색 기능을 결합한 기법으로...

============================================================
사용자: 그거 어떤 장점이 있어?
AI: RAG(검색 증강 생성)의 주요 장점은 다음과 같습니다:
    1. 최신 정보 활용: 학습 데이터 이후의 정보도 검색을 통해 활용...
    (AI가 "그거" = RAG임을 이해하고 답변합니다!)

============================================================
사용자: 구체적인 예시 들어줄 수 있어?
AI: 네! RAG의 구체적인 활용 예시를 들어드리겠습니다:
    1. 기업 내부 문서 Q&A: 직원들이 사내 정책이나 규정을...
```

---

## 완전한 엔드투엔드 프로젝트

!!! success "직접 해보기: 나만의 PDF 질문 봇 만들기"
    아래 코드는 **복사해서 바로 실행**할 수 있는 완전한 프로젝트입니다.

    **필요한 것:**
    1. OpenAI API 키 (`.env` 파일에 설정)
    2. 질문할 PDF 파일 하나 (없으면 샘플 텍스트 파일 사용)

```python
"""
완전한 PDF Q&A RAG 봇
=====================
이 파일 하나로 완전한 RAG 시스템을 실행할 수 있습니다.

사용법:
    python rag_bot.py

준비:
    1. pip install langchain langchain-openai langchain-community chromadb pypdf python-dotenv
    2. .env 파일에 OPENAI_API_KEY=sk-... 추가
    3. 같은 폴더에 PDF 파일 준비 (없으면 샘플 텍스트 사용)
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────
# 0. 환경 설정
# ─────────────────────────────────────
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY 환경 변수가 필요합니다. .env 파일을 확인하세요.")

print("환경 설정 완료")

# ─────────────────────────────────────
# 1. 샘플 문서 생성 (PDF가 없을 경우)
# ─────────────────────────────────────
SAMPLE_DOC_PATH = "sample_knowledge.txt"

if not Path(SAMPLE_DOC_PATH).exists():
    sample_content = """
RAG (Retrieval-Augmented Generation) 완전 가이드

## RAG란 무엇인가?

RAG(검색 증강 생성)는 대형 언어 모델(LLM)의 한계를 극복하기 위해 개발된 기법입니다.
2020년 Facebook AI Research(현 Meta AI)에서 처음 제안되었으며, 현재는 기업용 AI 솔루션의
핵심 기술로 자리잡았습니다.

기존 LLM의 주요 한계:
1. 지식 컷오프: 학습 데이터 이후의 정보를 모름
2. 환각 현상: 없는 사실을 만들어냄
3. 출처 불명: 어떤 근거로 답변하는지 알 수 없음
4. 도메인 지식 부족: 특정 분야(의료, 법률 등) 전문 지식 한계

## RAG의 핵심 원리

RAG는 "외부 문서를 검색해서 LLM에게 참고자료로 제공"하는 방식으로 작동합니다.

작동 과정:
1. 사전 처리: 문서를 작은 조각(청크)으로 분할하고 벡터로 변환해 데이터베이스에 저장
2. 질문 처리: 사용자 질문도 동일한 방법으로 벡터로 변환
3. 검색: 질문 벡터와 가장 유사한 문서 청크들을 찾아냄
4. 생성: 찾은 문서들을 참고자료로 LLM에게 제공하여 답변 생성

## RAG의 주요 장점

1. 최신 정보 활용
   - 언제든 새 문서를 추가하면 즉시 최신 정보를 활용 가능
   - LLM 재학습 불필요

2. 환각 현상 감소
   - 실제 문서에 근거한 답변 생성
   - "컨텍스트에 없으면 모른다고 답하라" 지시 가능

3. 출처 추적
   - 어떤 문서에서 답변을 생성했는지 확인 가능
   - 사용자가 원본 문서를 직접 확인할 수 있음

4. 비용 효율
   - 전체 재학습 없이 지식 업데이트
   - 특정 도메인에 맞춤화 가능

## RAG 구현 기술 스택

벡터 데이터베이스:
- Chroma: 로컬 개발에 적합, 무료
- Pinecone: 클라우드 서비스, 확장성 뛰어남
- Weaviate: 오픈소스, 다양한 기능
- pgvector: PostgreSQL 확장, 기존 DB 활용 가능

임베딩 모델:
- OpenAI text-embedding-3-large: 고성능, 유료
- BGE-M3: 오픈소스, 한국어 지원
- Cohere embed-multilingual: 다국어 지원

LLM:
- GPT-4o: OpenAI, 최고 성능
- Claude 3.5: Anthropic, 긴 컨텍스트
- Gemini: Google, 멀티모달

## 평가 지표

RAG 시스템 품질을 측정하는 주요 지표:

1. 충실도 (Faithfulness): 답변이 검색된 문서에 충실한가? (환각 정도)
2. 답변 관련성 (Answer Relevance): 답변이 질문에 적절한가?
3. 컨텍스트 관련성 (Context Relevance): 검색된 문서가 질문에 관련 있는가?
4. 컨텍스트 재현율 (Context Recall): 필요한 정보를 모두 검색했는가?
"""
    with open(SAMPLE_DOC_PATH, "w", encoding="utf-8") as f:
        f.write(sample_content)
    print(f"샘플 문서 생성됨: {SAMPLE_DOC_PATH}")

# ─────────────────────────────────────
# 2. 문서 로드
# ─────────────────────────────────────
print("\n문서 로드 중...")

# PDF 파일이 있으면 PDF를, 없으면 텍스트 파일 사용
pdf_files = list(Path(".").glob("*.pdf"))

if pdf_files:
    loader = PyPDFLoader(str(pdf_files[0]))
    print(f"  PDF 파일 사용: {pdf_files[0]}")
else:
    loader = TextLoader(SAMPLE_DOC_PATH, encoding="utf-8")
    print(f"  텍스트 파일 사용: {SAMPLE_DOC_PATH}")

documents = loader.load()
print(f"  {len(documents)}개 문서 로드 완료")

# ─────────────────────────────────────
# 3. 전처리 & 청킹
# ─────────────────────────────────────
print("\n청킹 중...")

def preprocess_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.replace('\t', ' ')
    return text.strip()

for doc in documents:
    doc.page_content = preprocess_text(doc.page_content)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(documents)
print(f"  {len(chunks)}개 청크 생성 완료")

# ─────────────────────────────────────
# 4. 임베딩 & 벡터 DB
# ─────────────────────────────────────
DB_DIR = "./rag_bot_db"

if Path(DB_DIR).exists():
    print(f"\n기존 벡터 DB 불러오는 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="rag_bot"
    )
    print(f"  {vectorstore._collection.count()}개 청크 로드됨")
else:
    print(f"\n벡터 DB 생성 중... (처음 한 번만 실행됩니다)")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="rag_bot"
    )
    print(f"  {vectorstore._collection.count()}개 청크 저장됨")

# ─────────────────────────────────────
# 5. RAG 체인 구성
# ─────────────────────────────────────
print("\nRAG 체인 구성 중...")

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 8, "lambda_mult": 0.7}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 주어진 문서를 기반으로 정확하고 친절하게 답변하는 AI 어시스턴트입니다.

규칙:
- 오직 아래 참고 문서의 내용만 사용하세요
- 문서에 없는 내용은 "해당 정보를 문서에서 찾을 수 없습니다"라고 명확히 말하세요
- 답변은 명확하고 구조적으로 작성하세요
- 이전 대화 맥락을 고려하여 자연스럽게 답변하세요

참고 문서:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# ─────────────────────────────────────
# 6. 대화형 RAG 봇
# ─────────────────────────────────────
class RAGBot:
    def __init__(self):
        self.chat_history = []

    def ask(self, question: str, stream: bool = True) -> str:
        # 관련 문서 검색
        docs = retriever.invoke(question)
        context = format_docs(docs)
        sources = [doc.metadata.get("source", "N/A") for doc in docs]

        # 메시지 구성
        messages = prompt.format_messages(
            context=context,
            chat_history=self.chat_history,
            question=question
        )

        # LLM 답변 생성
        if stream:
            print("AI: ", end="")
            answer_parts = []
            for token in llm.stream(messages):
                print(token.content, end="", flush=True)
                answer_parts.append(token.content)
            print()  # 줄바꿈
            answer = "".join(answer_parts)
        else:
            response = llm.invoke(messages)
            answer = response.content
            print(f"AI: {answer}")

        # 출처 표시
        unique_sources = list(set(sources))
        print(f"\n[출처: {', '.join(unique_sources)}]\n")

        # 대화 기록 저장
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        # 최근 20개 대화만 유지
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        return answer

    def reset(self):
        self.chat_history = []
        print("대화 기록이 초기화되었습니다.\n")

# ─────────────────────────────────────
# 7. 대화 루프 시작
# ─────────────────────────────────────
print("\nRAG 봇 준비 완료!")
print("=" * 60)
print("문서 기반 Q&A 봇입니다. 질문을 입력하세요.")
print("명령어: 'quit' 또는 'q' = 종료, 'reset' = 대화 초기화")
print("=" * 60)

bot = RAGBot()

while True:
    print()
    user_input = input("사용자: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ("quit", "q", "exit", "종료"):
        print("봇을 종료합니다. 감사합니다!")
        break

    if user_input.lower() in ("reset", "초기화"):
        bot.reset()
        continue

    bot.ask(user_input)
```

```
# 예상 실행 화면:
환경 설정 완료
문서 로드 중...
  텍스트 파일 사용: sample_knowledge.txt
  1개 문서 로드 완료
청킹 중...
  12개 청크 생성 완료
벡터 DB 생성 중... (처음 한 번만 실행됩니다)
  12개 청크 저장됨
RAG 체인 구성 중...
RAG 봇 준비 완료!
============================================================
문서 기반 Q&A 봇입니다. 질문을 입력하세요.
명령어: 'quit' 또는 'q' = 종료, 'reset' = 대화 초기화
============================================================

사용자: RAG의 장점이 뭐야?

AI: 문서에 따르면 RAG의 주요 장점은 네 가지입니다:

1. **최신 정보 활용**: 새 문서를 추가하면 즉시 최신 정보 반영 가능, LLM 재학습 불필요
2. **환각 현상 감소**: 실제 문서에 근거한 답변 생성
3. **출처 추적**: 어떤 문서에서 답변했는지 확인 가능
4. **비용 효율**: 전체 재학습 없이 도메인 지식 업데이트 가능

[출처: sample_knowledge.txt]

사용자: 그 중에서 가장 중요한 건 뭐야?

AI: 개인적 견해보다는 문서에서 언급된 맥락을 바탕으로 말씀드리면,
    **환각 현상 감소**가 가장 중요한 장점으로 볼 수 있습니다...
    (AI가 "그 중에서" = "RAG 장점들 중에서"를 이해합니다!)

[출처: sample_knowledge.txt]
```

---

## 평가 (Evaluation) - RAG 품질 측정하기

!!! tip "왜 이게 중요한가?"
    RAG 시스템을 만들었다고 끝이 아닙니다. **얼마나 잘 작동하는지 측정**해야 합니다.

    측정 없이는:
    - 어떤 설정이 더 좋은지 비교할 수 없습니다
    - 서비스 배포 후 품질이 나빠지고 있는지 모릅니다
    - 개선 방향을 잡기 어렵습니다

### 핵심 평가 지표 설명

```
┌─────────────────────────────────────────────────────────┐
│                RAG 평가 지표 관계도                      │
│                                                         │
│  사용자 질문 ──→ [검색] ──→ 참고 문서들 ──→ [생성] ──→ 답변  │
│                    │              │              │      │
│                    │              │              │      │
│              컨텍스트          컨텍스트        답변    │
│              관련성             재현율         관련성   │
│           (관련 없는         (필요한 내용이   (질문에  │
│           문서를 가져         모두 검색됐나?)  맞는    │
│           오진 않았나?)                       답변인가?)│
│                                    │                   │
│                              충실도(Faithfulness)       │
│                         (답변이 문서에 근거하는가?       │
│                          환각이 없는가?)                │
└─────────────────────────────────────────────────────────┘
```

| 지표 | 설명 | 이상적인 값 | 낮으면? |
|------|------|------------|---------|
| **충실도 (Faithfulness)** | 답변이 검색된 문서에 근거하는가 | 1.0 (완전히 근거) | 환각 발생 |
| **답변 관련성 (Answer Relevance)** | 답변이 질문과 관련 있는가 | 1.0 (완전히 관련) | 동문서답 |
| **컨텍스트 관련성 (Context Relevance)** | 검색된 문서가 질문과 관련 있는가 | 1.0 (모두 관련) | 검색 품질 문제 |
| **컨텍스트 재현율 (Context Recall)** | 필요한 정보를 모두 검색했는가 | 1.0 (완전 검색) | 답변 불완전 |

### LLM-as-Judge 평가 방법

!!! info "LLM-as-Judge란?"
    LLM(GPT-4 등)이 다른 LLM의 답변을 평가하는 방법입니다. 인간 평가자가 모든 답변을 검토하기 어려울 때 사용합니다.

```python
from langchain_openai import ChatOpenAI
import json

# ─────────────────────────────────────
# 평가 전용 LLM (temperature=0 으로 일관성 유지)
# ─────────────────────────────────────
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ─────────────────────────────────────
# JSON 응답 파싱 헬퍼
# ─────────────────────────────────────
def parse_json_response(response_text: str) -> dict:
    """LLM의 JSON 응답을 파싱합니다."""
    raw = response_text.strip()
    # 코드 블록 형태로 감싸져 있으면 제거
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    return json.loads(raw)

# ─────────────────────────────────────
# 1. 충실도 (Faithfulness) 평가
# ─────────────────────────────────────
def check_faithfulness(
    question: str,
    answer: str,
    context: str
) -> dict:
    """
    답변이 주어진 컨텍스트(문서)에 충실한지 평가합니다.

    충실도 = 답변의 각 주장이 컨텍스트에 근거하는 비율
    환각이 없을수록 1.0에 가깝습니다.
    """
    prompt_text = f"""당신은 RAG 시스템의 품질을 평가하는 전문가입니다.

다음 정보를 바탕으로 '충실도(Faithfulness)'를 평가하세요.

충실도 정의: 생성된 답변의 각 주장/사실이 참고 문서(컨텍스트)에서 직접 근거하는 정도

[질문]
{question}

[참고 문서 (컨텍스트)]
{context}

[생성된 답변]
{answer}

평가 방법:
1. 답변에서 각 사실적 주장을 식별하세요
2. 각 주장이 컨텍스트에 근거하는지 확인하세요
3. (근거 있는 주장 수) / (전체 주장 수) = 충실도 점수

반드시 아래 JSON 형식으로만 답하세요:
{{
    "score": 0.0에서 1.0 사이의 숫자,
    "explanation": "평가 이유 (한국어)",
    "unsupported_claims": ["문서에 근거 없는 주장 목록"]
}}"""

    response = judge_llm.invoke(prompt_text)

    try:
        return parse_json_response(response.content)
    except (json.JSONDecodeError, KeyError):
        return {
            "score": 0.0,
            "explanation": "파싱 오류",
            "unsupported_claims": []
        }

# ─────────────────────────────────────
# 2. 답변 관련성 (Answer Relevance) 평가
# ─────────────────────────────────────
def check_answer_relevance(question: str, answer: str) -> dict:
    """
    생성된 답변이 질문에 얼마나 관련 있는지 평가합니다.

    핵심: 질문에 실제로 답하고 있는가? 동문서답은 아닌가?
    """
    prompt_text = f"""당신은 RAG 시스템의 품질을 평가하는 전문가입니다.

다음 질문과 답변을 보고 '답변 관련성(Answer Relevance)'을 평가하세요.

답변 관련성 정의: 생성된 답변이 질문의 의도를 얼마나 잘 파악하고 답하는가

[질문]
{question}

[생성된 답변]
{answer}

반드시 아래 JSON 형식으로만 답하세요:
{{
    "score": 0.0에서 1.0 사이의 숫자,
    "explanation": "평가 이유 (한국어)"
}}"""

    response = judge_llm.invoke(prompt_text)
    try:
        return parse_json_response(response.content)
    except (json.JSONDecodeError, KeyError):
        return {"score": 0.0, "explanation": "파싱 오류"}

# ─────────────────────────────────────
# 3. 컨텍스트 관련성 (Context Relevance) 평가
# ─────────────────────────────────────
def check_context_relevance(question: str, context: str) -> dict:
    """
    검색된 문서가 질문에 얼마나 관련 있는지 평가합니다.

    컨텍스트 관련성 = 관련 있는 문장 수 / 전체 컨텍스트 문장 수
    이 값이 낮으면 리트리버가 잘못된 문서를 가져오는 것입니다.
    """
    prompt_text = f"""당신은 RAG 시스템의 품질을 평가하는 전문가입니다.

다음 질문과 컨텍스트(검색된 참고 문서)를 보고 '컨텍스트 관련성'을 평가하세요.

컨텍스트 관련성 정의: 검색된 문서에서 실제로 질문에 답하는 데 필요한 정보의 비율

[질문]
{question}

[검색된 컨텍스트]
{context}

반드시 아래 JSON 형식으로만 답하세요:
{{
    "score": 0.0에서 1.0 사이의 숫자,
    "explanation": "평가 이유 (한국어)",
    "relevant_sentences": ["질문과 관련된 문장들"]
}}"""

    response = judge_llm.invoke(prompt_text)
    try:
        return parse_json_response(response.content)
    except (json.JSONDecodeError, KeyError):
        return {"score": 0.0, "explanation": "파싱 오류", "relevant_sentences": []}

# ─────────────────────────────────────
# 4. 종합 평가 실행
# ─────────────────────────────────────
def run_evaluation(test_cases: list, rag_chain, rag_retriever) -> dict:
    """
    테스트 케이스 목록으로 RAG 시스템을 종합 평가합니다.

    Args:
        test_cases: [{"question": "...", "expected": "..."}] 형식의 리스트
        rag_chain: 평가할 RAG 체인
        rag_retriever: 평가할 리트리버
    Returns:
        dict: 평가 결과 및 통계
    """
    results = []

    print("=" * 60)
    print("RAG 시스템 품질 평가 시작")
    print("=" * 60)

    for i, case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}/{len(test_cases)}]")
        question = case["question"]
        print(f"질문: {question}")

        # RAG 실행
        retrieved_docs = rag_retriever.invoke(question)
        context = format_docs(retrieved_docs)
        answer = rag_chain.invoke(question)

        print(f"답변: {answer[:100]}...")

        # 각 지표 측정
        print("  충실도 평가 중...")
        faithfulness = check_faithfulness(question, answer, context)

        print("  답변 관련성 평가 중...")
        answer_relevance = check_answer_relevance(question, answer)

        print("  컨텍스트 관련성 평가 중...")
        context_relevance = check_context_relevance(question, context)

        result = {
            "question": question,
            "answer": answer,
            "faithfulness": faithfulness["score"],
            "answer_relevance": answer_relevance["score"],
            "context_relevance": context_relevance["score"],
            "avg_score": (
                faithfulness["score"] +
                answer_relevance["score"] +
                context_relevance["score"]
            ) / 3
        }

        results.append(result)

        print(f"  충실도: {faithfulness['score']:.2f} | "
              f"답변관련성: {answer_relevance['score']:.2f} | "
              f"컨텍스트관련성: {context_relevance['score']:.2f}")

    # 통계 계산
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_answer_rel = sum(r["answer_relevance"] for r in results) / len(results)
    avg_context_rel = sum(r["context_relevance"] for r in results) / len(results)
    overall_avg = sum(r["avg_score"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("평가 결과 요약")
    print("=" * 60)
    good = lambda v: "OK" if v > 0.8 else "WARNING"
    print(f"충실도 평균:           {avg_faithfulness:.3f} [{good(avg_faithfulness)}]")
    print(f"답변 관련성 평균:      {avg_answer_rel:.3f} [{good(avg_answer_rel)}]")
    print(f"컨텍스트 관련성 평균: {avg_context_rel:.3f} [{good(avg_context_rel)}]")
    print(f"-" * 40)
    print(f"전체 평균 점수:        {overall_avg:.3f} [{good(overall_avg)}]")

    return {
        "results": results,
        "summary": {
            "faithfulness": avg_faithfulness,
            "answer_relevance": avg_answer_rel,
            "context_relevance": avg_context_rel,
            "overall": overall_avg
        }
    }

# ─────────────────────────────────────
# 테스트 케이스 정의 및 실행
# ─────────────────────────────────────
test_cases = [
    {
        "question": "RAG란 무엇인가요?",
        "expected": "검색 증강 생성으로 LLM에 외부 검색 기능을 결합한 기법"
    },
    {
        "question": "RAG의 주요 장점은 무엇인가요?",
        "expected": "최신 정보 활용, 환각 감소, 출처 추적, 비용 효율"
    },
    {
        "question": "RAG에서 사용하는 벡터 데이터베이스 종류는?",
        "expected": "Chroma, Pinecone, Weaviate, pgvector"
    },
    {
        "question": "RAG 시스템의 평가 지표는?",
        "expected": "충실도, 답변 관련성, 컨텍스트 관련성, 컨텍스트 재현율"
    }
]

# 평가 실행
evaluation_results = run_evaluation(test_cases, chain, retriever)
```

```
# 예상 출력:
============================================================
RAG 시스템 품질 평가 시작
============================================================

[테스트 1/4]
질문: RAG란 무엇인가요?
답변: RAG(Retrieval-Augmented Generation, 검색 증강 생성)는 대형 언어 모델(LLM)의 한계...
  충실도 평가 중...
  답변 관련성 평가 중...
  컨텍스트 관련성 평가 중...
  충실도: 0.95 | 답변관련성: 0.98 | 컨텍스트관련성: 0.87

[테스트 2/4]
...

============================================================
평가 결과 요약
============================================================
충실도 평균:           0.923 [OK]
답변 관련성 평균:      0.945 [OK]
컨텍스트 관련성 평균: 0.867 [OK]
----------------------------------------
전체 평균 점수:        0.912 [OK]
```

---

## 오류 처리 및 디버깅 가이드

### 자주 발생하는 오류와 해결 방법

!!! failure "오류 1: RateLimitError"
    ```
    openai.RateLimitError: Rate limit exceeded
    ```

    **원인**: API 요청을 너무 빠르게 보냄

    **해결**: 재시도 로직 추가
    ```python
    import time
    from openai import RateLimitError

    def safe_invoke(chain, query: str, max_retries: int = 3):
        """RateLimit 오류 시 자동 재시도"""
        for attempt in range(max_retries):
            try:
                return chain.invoke(query)
            except RateLimitError:
                wait_time = 2 ** attempt  # 지수 백오프: 1초, 2초, 4초
                print(f"Rate limit 초과. {wait_time}초 후 재시도... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
        raise RuntimeError(f"{max_retries}번 시도 후 실패")
    ```

!!! failure "오류 2: 검색 결과가 비어있음"
    ```python
    # 디버깅: 검색이 제대로 되는지 확인
    def debug_retrieval(query: str):
        """검색 과정을 단계별로 디버깅"""
        print(f"[DEBUG] 쿼리: {query}")

        # 임베딩 확인
        embedding = embeddings.embed_query(query)
        print(f"[DEBUG] 임베딩 차원: {len(embedding)}")
        print(f"[DEBUG] 임베딩 값 범위: {min(embedding):.4f} ~ {max(embedding):.4f}")

        # 검색 결과 확인
        docs = retriever.invoke(query)
        print(f"[DEBUG] 검색된 문서 수: {len(docs)}")

        if not docs:
            print("[경고] 검색 결과가 없습니다!")
            print("  벡터 DB에 데이터가 있는지 확인하세요")
            print("  검색 쿼리를 더 구체적으로 해보세요")
        else:
            for i, doc in enumerate(docs):
                print(f"[DEBUG] 문서 {i+1}:")
                print(f"  출처: {doc.metadata}")
                print(f"  내용: {doc.page_content[:150]}...")

        return docs

    # 사용
    debug_retrieval("RAG 장점")
    ```

!!! failure "오류 3: 답변 품질이 낮음"
    ```python
    # 단계별 체인 디버깅 - 각 단계의 입출력 확인

    question = "RAG의 장점은?"

    # Step 1: 검색 단계 확인
    docs = retriever.invoke(question)
    print("=== 검색된 문서 ===")
    print(format_docs(docs))

    # Step 2: 프롬프트 확인
    formatted_prompt = prompt.format_messages(
        context=format_docs(docs),
        chat_history=[],
        question=question
    )
    print("\n=== 최종 프롬프트 ===")
    for msg in formatted_prompt:
        print(f"[{msg.type}]: {msg.content[:300]}...")

    # Step 3: LLM 응답 확인
    response = llm.invoke(formatted_prompt)
    print("\n=== LLM 원시 응답 ===")
    print(response)
    print(f"\n응답 타입: {type(response)}")
    print(f"응답 내용: {response.content}")
    ```

!!! failure "오류 4: 한국어가 깨짐"
    ```python
    # 원인 1: 파일 인코딩 문제
    loader = TextLoader("문서.txt", encoding="utf-8")  # encoding 명시
    # 또는
    loader = TextLoader("문서.txt", encoding="cp949")  # 윈도우 한국어

    # 원인 2: ChromaDB 한국어 처리
    # ChromaDB는 UTF-8을 기본 지원합니다 - 별도 설정 불필요

    # 확인 방법
    print(chunks[0].page_content.encode("utf-8").decode("utf-8"))
    ```

### 성능 모니터링

```python
import time
from functools import wraps

def measure_performance(func):
    """함수 실행 시간을 측정하는 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[성능] {func.__name__}: {elapsed:.2f}초")
        return result
    return wrapper

@measure_performance
def timed_retrieval(query: str):
    return retriever.invoke(query)

@measure_performance
def timed_chain_invoke(query: str):
    return chain.invoke(query)

# 성능 측정
docs = timed_retrieval("RAG란?")
answer = timed_chain_invoke("RAG란?")
```

```
# 예상 출력:
[성능] timed_retrieval: 0.34초
[성능] timed_chain_invoke: 2.87초
```

---

## 자주 묻는 질문 (FAQ)

??? question "Q1. chunk_size를 얼마로 설정해야 하나요?"
    정답은 없지만, 일반적인 가이드라인:

    - **500~1000글자**: 일반적인 문서 Q&A에 적합
    - **200~500글자**: 짧고 구체적인 사실 기반 Q&A
    - **1000~2000글자**: 복잡한 추론이 필요한 질문

    **팁**: 여러 값을 테스트하고 평가 지표로 비교해보세요.

??? question "Q2. k값(검색할 청크 수)을 얼마로 해야 하나요?"
    일반적으로 **3~5개**가 적당합니다.

    - **너무 작으면 (k=1)**: 정보가 부족해 답변이 불완전할 수 있음
    - **너무 크면 (k=10+)**: LLM 컨텍스트가 길어져 비용 증가, 관련 없는 정보 포함될 수 있음

    LLM의 컨텍스트 창 크기도 고려하세요. (GPT-4o: 128K 토큰)

??? question "Q3. 임베딩 비용을 줄이는 방법은?"
    1. **무료 로컬 모델 사용**: `HuggingFaceEmbeddings(model_name="BAAI/bge-m3")`
    2. **캐싱**: 이미 임베딩한 문서는 다시 임베딩하지 않도록 저장
    3. **배치 처리**: 청크를 한 번에 모아서 임베딩 (API 호출 횟수 감소)
    4. **작은 차원 사용**: `text-embedding-3-large`의 경우 `dimensions=256`으로 줄일 수 있음

??? question "Q4. 답변에서 '문서에 없는 내용'이 계속 나옵니다 (환각)"
    프롬프트를 강화하세요:

    ```python
    prompt = ChatPromptTemplate.from_template("""
    당신은 주어진 문서만을 기반으로 답변하는 시스템입니다.

    중요 규칙:
    - 문서에 없는 내용은 절대 말하지 마세요
    - 확실하지 않으면 "문서에서 확인할 수 없습니다"라고 하세요
    - 문서의 내용을 그대로 인용하세요

    문서: {context}
    질문: {question}
    답변:
    """)
    ```

    또한 `temperature=0`으로 설정하면 창의적 답변을 줄일 수 있습니다.

??? question "Q5. 한국어 문서에 적합한 임베딩 모델은?"
    | 모델 | 특징 | 한국어 성능 |
    |------|------|------------|
    | `text-embedding-3-large` | OpenAI, 유료 | 최고 수준 |
    | `BAAI/bge-m3` | 오픈소스, 무료 | 매우 우수 |
    | `intfloat/multilingual-e5-large` | 오픈소스, 무료 | 우수 |
    | `jhgan/ko-sroberta-multitask` | 한국어 특화, 무료 | 우수 (한국어 전용) |

??? question "Q6. 벡터 DB를 업데이트하는 방법은?"
    ```python
    # 새 문서만 추가 (기존 데이터 유지)
    new_chunks = splitter.split_documents(new_documents)
    vectorstore.add_documents(new_chunks)
    print(f"새로 추가됨: {len(new_chunks)}개 청크")

    # 특정 문서 삭제 후 재추가
    # (ChromaDB는 문서 ID로 삭제 가능)
    ids_to_delete = vectorstore.get(where={"source": "old_doc.pdf"})["ids"]
    vectorstore.delete(ids=ids_to_delete)
    ```

??? question "Q7. 프로덕션에서 ChromaDB 대신 무엇을 써야 하나요?"
    | 서비스 | 특징 | 적합한 경우 |
    |--------|------|------------|
    | **Chroma** | 로컬, 무료 | 개발/테스트, 소규모 |
    | **Pinecone** | 클라우드 SaaS | 스케일업이 필요한 서비스 |
    | **Weaviate** | 오픈소스/클라우드 | 자체 운영 원하는 경우 |
    | **pgvector** | PostgreSQL 확장 | 기존 PostgreSQL DB 활용 |
    | **Qdrant** | 오픈소스/클라우드 | 고성능 필요 |

??? question "Q8. LCEL 없이 기존 방식으로도 만들 수 있나요?"
    네! LCEL은 편의를 위한 것이고, 아래처럼 직접 함수로 만들 수도 있습니다:

    ```python
    def rag_pipeline(question: str) -> str:
        # 1. 검색
        docs = retriever.invoke(question)
        context = format_docs(docs)

        # 2. 프롬프트 구성
        messages = prompt.format_messages(
            context=context,
            question=question
        )

        # 3. LLM 호출
        response = llm.invoke(messages)

        # 4. 출력 파싱
        return response.content

    answer = rag_pipeline("RAG란?")
    ```

    LCEL이 더 권장되는 이유: 스트리밍, 비동기, 배치 처리가 자동 지원됩니다.

---

## 프로덕션 체크리스트

실제 서비스에 배포하기 전에 확인해야 할 항목들입니다.

!!! success "프로덕션 체크리스트"

    ### 기능 완성도
    - [ ] **기본 Q&A**: 질문에 적절한 답변을 생성하는가?
    - [ ] **출처 표시**: 답변의 근거가 되는 문서를 함께 반환하는가?
    - [ ] **"모른다" 처리**: 문서에 없는 질문에 적절히 대응하는가?
    - [ ] **대화 기억**: 연속 대화에서 문맥이 유지되는가?
    - [ ] **스트리밍**: 실시간 응답이 가능한가?

    ### 성능
    - [ ] **응답 시간**: 평균 3~5초 이내인가?
    - [ ] **검색 시간**: 평균 0.5초 이내인가?
    - [ ] **동시 요청**: 여러 사용자가 동시에 사용할 수 있는가?
    - [ ] **메모리 사용량**: 서버 메모리 한계 내에 있는가?

    ### 보안
    - [ ] **API 키 보호**: 환경 변수로 관리, 코드에 직접 기재 금지
    - [ ] **입력 검증**: 악의적인 프롬프트 인젝션 방어
    - [ ] **접근 제어**: 인증된 사용자만 사용 가능한가?
    - [ ] **요청 제한**: Rate limiting으로 남용 방지

    ### 안정성
    - [ ] **에러 핸들링**: API 오류, 타임아웃 등 예외 처리
    - [ ] **재시도 로직**: 일시적 오류 시 자동 재시도
    - [ ] **폴백 처리**: 검색 실패 시 대안 응답
    - [ ] **로깅**: 오류와 이상 동작 기록

    ### 품질
    - [ ] **평가 데이터셋**: 테스트 케이스 최소 50개 이상
    - [ ] **충실도 점수**: 0.8 이상
    - [ ] **답변 관련성**: 0.8 이상
    - [ ] **정기 평가**: 주기적인 품질 모니터링 계획

    ### 운영
    - [ ] **문서 업데이트 프로세스**: 새 문서 추가 절차 수립
    - [ ] **모니터링**: 서비스 상태 실시간 확인
    - [ ] **백업**: 벡터 DB 정기 백업
    - [ ] **비용 모니터링**: API 비용 추적 및 예산 알림

### 프로덕션 권장 코드 패턴

```python
import logging
from typing import Optional
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_pipeline")

class ProductionRAGPipeline:
    """프로덕션 환경을 위한 RAG 파이프라인"""

    def __init__(
        self,
        retriever,
        chain,
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.retriever = retriever
        self.chain = chain
        self.max_retries = max_retries
        self.timeout = timeout
        self._request_count = 0
        self._error_count = 0

    def query(
        self,
        question: str,
        user_id: Optional[str] = None
    ) -> dict:
        """
        안전하고 안정적인 RAG 쿼리 실행

        Args:
            question: 사용자 질문
            user_id: 사용자 식별자 (로깅용)
        Returns:
            dict: {"answer": str, "sources": list, "latency_ms": int}
        """
        self._request_count += 1
        start_time = time.time()

        # 입력 검증
        if not question or not question.strip():
            return {
                "answer": "질문을 입력해주세요.",
                "sources": [],
                "latency_ms": 0,
                "error": "empty_question"
            }

        if len(question) > 1000:
            return {
                "answer": "질문이 너무 깁니다. 1000자 이내로 입력해주세요.",
                "sources": [],
                "latency_ms": 0,
                "error": "question_too_long"
            }

        # 재시도 로직
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"쿼리 실행 (시도 {attempt+1}/{self.max_retries}): {question[:50]}...")

                # 검색
                docs = self.retriever.invoke(question)
                sources = [
                    {
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", "N/A")
                    }
                    for doc in docs
                ]

                # 생성
                answer = self.chain.invoke(question)

                latency_ms = int((time.time() - start_time) * 1000)
                logger.info(f"쿼리 완료: {latency_ms}ms")

                return {
                    "answer": answer,
                    "sources": sources,
                    "latency_ms": latency_ms,
                    "error": None
                }

            except Exception as e:
                last_error = e
                self._error_count += 1
                logger.warning(f"오류 발생 (시도 {attempt+1}): {type(e).__name__}: {e}")

                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"{wait_time}초 후 재시도...")
                    time.sleep(wait_time)

        # 모든 재시도 실패
        logger.error(f"모든 재시도 실패: {last_error}")
        return {
            "answer": "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "sources": [],
            "latency_ms": int((time.time() - start_time) * 1000),
            "error": str(last_error)
        }

    def get_stats(self) -> dict:
        """시스템 통계 반환"""
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": (
                self._error_count / self._request_count
                if self._request_count > 0 else 0
            )
        }

# 사용 예시
prod_pipeline = ProductionRAGPipeline(
    retriever=retriever,
    chain=chain,
    max_retries=3
)

result = prod_pipeline.query("RAG란 무엇인가요?", user_id="user_123")
print(f"답변: {result['answer'][:100]}...")
print(f"출처: {result['sources']}")
print(f"응답 시간: {result['latency_ms']}ms")
print(f"오류: {result['error']}")

# 통계 확인
stats = prod_pipeline.get_stats()
print(f"\n시스템 통계: {stats}")
```

```
# 예상 출력:
답변: RAG(Retrieval-Augmented Generation, 검색 증강 생성)는...
출처: [{'source': 'sample_knowledge.txt', 'page': 'N/A'}]
응답 시간: 2847ms
오류: None

시스템 통계: {'total_requests': 1, 'total_errors': 0, 'error_rate': 0.0}
```

---

## 핵심 요약

!!! abstract "이것만 기억하자"

    **파이프라인의 7단계:**

    1. **문서 로드**: 다양한 형식의 파일을 `Document` 객체로 변환
    2. **전처리**: 불필요한 노이즈 제거
    3. **청킹**: 긴 문서를 검색 가능한 작은 조각으로 분할
    4. **임베딩 & 저장**: 텍스트를 벡터로 변환하여 벡터 DB에 저장
    5. **검색**: 질문과 유사한 청크를 벡터 DB에서 검색
    6. **생성**: 검색된 문서 + 질문을 LLM에 전달하여 답변 생성
    7. **평가**: 충실도, 관련성 등 지표로 품질 측정

    **핵심 개념:**

    - `LCEL`: `|` 연산자로 컴포넌트를 연결하는 LangChain의 파이프라인 문법
    - `Retriever`: 질문을 받아 관련 문서를 반환하는 컴포넌트
    - `RunnablePassthrough()`: 입력을 그대로 통과시키는 패스스루
    - `StrOutputParser()`: LLM 응답에서 텍스트만 추출
    - `MessagesPlaceholder`: 대화 기록이 들어갈 자리 표시자

    **실전 팁:**

    - chunk_overlap을 설정해 청크 경계에서 문맥이 끊기지 않도록 하세요
    - MMR 검색으로 다양한 관점의 문서를 검색하세요
    - 충실도(Faithfulness)를 정기적으로 측정하여 환각을 모니터링하세요
    - 프로덕션에는 반드시 오류 처리와 재시도 로직을 추가하세요
