# 8. Context7과 MCP 기반 RAG

## 이 챕터에서 배우는 것

- MCP(Model Context Protocol)가 무엇인지
- Context7이 RAG를 어떻게 구현하는지
- 전통적인 RAG와 MCP 기반 RAG의 차이
- 직접 MCP 기반 Documentation RAG를 만드는 방법

---

## Context7이란?

**Context7**은 Upstash에서 개발한 **MCP 서버**로, LLM이 **최신 라이브러리 문서와 코드 예제**를 실시간으로 검색하여 컨텍스트에 주입하는 시스템이다.

!!! note "한줄 요약"
    Context7 = **개발 문서 전용 RAG 시스템**을 MCP 프로토콜로 제공하는 서비스

### Context7이 해결하는 문제

LLM은 학습 데이터의 **지식 컷오프(knowledge cutoff)** 때문에 최신 API를 모른다.

```
❌ LLM에게 "Next.js App Router 사용법" 물어보면:
   → 이미 deprecated된 Pages Router 코드를 답변할 수 있음

✅ Context7을 사용하면:
   → 최신 Next.js 공식 문서를 먼저 검색
   → 검색된 문서를 LLM 컨텍스트에 주입
   → 최신 App Router 코드를 정확하게 답변
```

!!! example "실생활 비유"
    - **LLM 단독** = 3년 전 교과서만 암기한 학생. 시험에서 최신 내용이 나오면 옛날 답을 씀
    - **LLM + Context7** = 시험 중에 **최신 공식 교재를 찾아볼 수 있는** 학생. 항상 정확한 답을 씀

### 이것이 RAG인 이유

Context7의 동작 방식을 분해하면 **전형적인 RAG 패턴**이다:

```
┌─────────────────────────────────────────────────────────┐
│            Context7 = Documentation RAG                  │
│                                                          │
│  [사용자 질문]                                            │
│       ↓                                                  │
│  [1. Retrieval] Context7 서버에서 관련 문서 검색           │
│       ↓                                                  │
│  [2. Augmentation] 검색된 문서를 LLM 컨텍스트에 주입       │
│       ↓                                                  │
│  [3. Generation] LLM이 문서 기반으로 정확한 답변 생성      │
│                                                          │
│  → Retrieval-Augmented Generation의 정확한 구현!          │
└─────────────────────────────────────────────────────────┘
```

---

## MCP(Model Context Protocol)란?

### 기본 개념

**MCP**는 Anthropic이 개발한 **오픈 프로토콜**로, LLM이 외부 도구와 데이터 소스에 **표준화된 방식으로** 연결할 수 있게 해준다.

!!! note "한줄 요약"
    MCP = LLM과 외부 세계를 연결하는 **USB-C 같은 표준 인터페이스**

!!! example "실생활 비유"
    - MCP 이전: 각 도구마다 다른 충전기 필요 (Lightning, Micro USB, USB-C...)
    - MCP 이후: 하나의 표준(USB-C)으로 모든 기기를 연결

    마찬가지로:
    - MCP 이전: LLM이 DB, API, 파일시스템 각각 다른 방식으로 연결
    - MCP 이후: **하나의 프로토콜**로 모든 외부 도구에 연결

### MCP 아키텍처

```
┌──────────────────┐     MCP 프로토콜      ┌──────────────────┐
│   MCP 호스트      │ ◄──────────────────► │   MCP 서버        │
│                  │                       │                  │
│  Claude Code     │     JSON-RPC          │  Context7        │
│  Cursor          │ ◄──────────────────► │  GitHub MCP      │
│  Claude Desktop  │                       │  Slack MCP       │
│  VS Code         │                       │  DB MCP          │
│                  │                       │  커스텀 서버      │
└──────────────────┘                       └──────────────────┘
     (클라이언트)                              (서버/도구)
```

### MCP의 3가지 핵심 요소

| 요소 | 설명 | Context7 예시 |
|------|------|--------------|
| **Tools** | LLM이 호출할 수 있는 함수 | `resolve-library-id`, `query-docs` |
| **Resources** | LLM이 읽을 수 있는 데이터 | 라이브러리 문서, 코드 스니펫 |
| **Prompts** | 미리 정의된 프롬프트 템플릿 | (Context7에선 미사용) |

---

## Context7의 내부 동작 원리

### 2단계 파이프라인

Context7은 **2개의 MCP Tool**을 순차적으로 호출하여 동작한다.

```
┌─────────────────────────────────────────────────────────────┐
│                  Context7 파이프라인                          │
│                                                              │
│  Step 1: resolve-library-id                                  │
│  ┌───────────────────────────────────────────────────┐       │
│  │ 입력: "langchain"                                  │       │
│  │                                                    │       │
│  │ 처리: 9,000+ 라이브러리 DB에서 의미 검색            │       │
│  │       → 관련성/품질/신뢰도 기준 랭킹                │       │
│  │                                                    │       │
│  │ 출력: "/websites/langchain" (라이브러리 ID)          │       │
│  │       + 코드 스니펫 수, 벤치마크 점수 등 메타데이터   │       │
│  └───────────────────────────────────────────────────┘       │
│           ↓                                                  │
│  Step 2: query-docs                                          │
│  ┌───────────────────────────────────────────────────┐       │
│  │ 입력: 라이브러리 ID + 구체적 질문                    │       │
│  │                                                    │       │
│  │ 처리: 벡터 검색 → 시맨틱 리랭킹 → 필터링            │       │
│  │       (서버 사이드에서 토큰 65% 절감)                │       │
│  │                                                    │       │
│  │ 출력: 관련 코드 예제 + 설명 + 출처 URL               │       │
│  └───────────────────────────────────────────────────┘       │
│           ↓                                                  │
│  LLM이 검색된 문서를 기반으로 답변 생성                       │
└─────────────────────────────────────────────────────────────┘
```

### 내부 인덱싱 파이프라인

Context7 서버는 다음과 같은 과정으로 문서를 사전 처리한다:

```
[공식 문서 수집]
      ↓
[1. Parse] 코드 스니펫과 설명 텍스트 분리 추출
      ↓
[2. Enrich] LLM으로 각 스니펫에 짧은 설명 자동 생성
      ↓
[3. Vectorize] 임베딩 벡터로 변환하여 벡터 DB 저장
      ↓
[4. Rerank] 커스텀 랭킹 알고리즘으로 품질 점수 부여
      ↓
[5. Cache] Redis 캐시로 빠른 응답 보장
```

!!! info "왜 이게 중요한가?"
    이 파이프라인은 우리가 앞 챕터에서 배운 RAG 파이프라인과 **정확히 동일한 구조**이다:

    - Parse = **문서 로드** (1장)
    - Enrich = **메타데이터 추가** (3장 청킹 팁)
    - Vectorize = **임베딩** (2장)
    - Rerank = **리랭킹** (6장 고급 기법)
    - Cache = **성능 최적화** (7장 프로덕션 팁)

---

## 실제 사용 예시

### Claude Code에서 Context7 사용

Claude Code(이 도구)에서 Context7은 MCP 서버로 연결되어 있다. 실제 동작 예시:

**1단계: 라이브러리 ID 검색**

LLM이 `resolve-library-id`를 호출하면:

```json
// 입력
{
  "libraryName": "langchain",
  "query": "RAG pipeline retrieval augmented generation"
}

// 출력 (실제 결과)
{
  "libraries": [
    {
      "id": "/websites/langchain",
      "name": "LangChain",
      "description": "LangChain is a platform for agent engineering...",
      "codeSnippets": 21066,
      "sourceReputation": "High",
      "benchmarkScore": 81.91
    }
  ]
}
```

!!! tip "주목할 점"
    - **21,066개**의 코드 스니펫이 인덱싱되어 있음
    - **벤치마크 점수**(81.91)로 문서 품질을 정량화
    - **신뢰도**(High/Medium/Low)로 출처의 권위성을 표시
    - 이 모든 메타데이터가 3장에서 배운 "풍부한 메타데이터" 개념과 동일

**2단계: 문서 검색**

LLM이 `query-docs`를 호출하면:

```json
// 입력
{
  "libraryId": "/websites/langchain",
  "query": "RAG pipeline LCEL chain example"
}

// 출력 (실제 결과 요약)
{
  "results": [
    {
      "title": "Build RAG Chain with LangChain",
      "source": "https://docs.langchain.com/...",
      "code": "from langchain_core.output_parsers import StrOutputParser\n..."
    }
  ]
}
```

검색된 **최신 공식 문서의 코드 예제**가 LLM의 컨텍스트에 주입되어, deprecated된 API 대신 최신 LCEL 패턴을 정확하게 답변할 수 있게 된다.

### Python으로 Context7 API 직접 호출

Context7은 REST API도 제공한다. 자체 앱에서 활용할 수 있다:

```python
import httpx

CONTEXT7_BASE = "https://api.context7.com/v1"

async def search_library_docs(library_name: str, query: str) -> dict:
    """Context7 API로 라이브러리 문서 검색"""
    async with httpx.AsyncClient() as client:
        # 1단계: 라이브러리 ID 검색
        resolve_resp = await client.get(
            f"{CONTEXT7_BASE}/resolve",
            params={"library": library_name, "query": query}
        )
        libraries = resolve_resp.json()
        library_id = libraries[0]["id"]

        # 2단계: 문서 검색
        docs_resp = await client.get(
            f"{CONTEXT7_BASE}/docs",
            params={"libraryId": library_id, "query": query}
        )
        return docs_resp.json()

# 사용 예시
import asyncio

result = asyncio.run(
    search_library_docs("react", "useEffect cleanup function")
)
print(result)
```

---

## 전통적 RAG vs Context7(MCP 기반 RAG) 비교

### 아키텍처 비교

```
전통적 RAG:
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ 문서   │ →  │ 청킹   │ →  │ 임베딩  │ →  │벡터 DB │
│ 수집   │    │ 분할   │    │ 변환   │    │ 저장   │
└────────┘    └────────┘    └────────┘    └────────┘
                                              ↓
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ 사용자  │ →  │ 쿼리   │ →  │ 벡터   │ →  │ LLM   │
│ 질문   │    │ 임베딩  │    │ 검색   │    │ 생성   │
└────────┘    └────────┘    └────────┘    └────────┘

→ 모든 단계를 직접 구축해야 함


Context7 (MCP 기반 RAG):
┌────────┐    ┌───────────────────────┐    ┌────────┐
│ 사용자  │ →  │  MCP Tool 호출        │ →  │ LLM   │
│ 질문   │    │  (Context7가 내부에서  │    │ 생성   │
│        │    │   검색/리랭킹 전부 수행)│    │        │
└────────┘    └───────────────────────┘    └────────┘

→ 인프라 구축 없이 MCP Tool 연결만으로 RAG 구현
```

### 상세 비교표

| 특성 | 전통적 RAG | Context7 (MCP 기반) |
|------|-----------|-------------------|
| **데이터 소스** | 자체 문서, 사내 자료 | 공식 라이브러리 문서 9,000+ |
| **인프라** | 벡터 DB, 임베딩 모델 직접 운영 | Context7 서버가 전부 처리 |
| **인덱싱** | 직접 구축 (청킹, 임베딩, 저장) | Context7이 사전 처리 완료 |
| **검색 방식** | 벡터 유사도 + 키워드 | 벡터 + 시맨틱 리랭킹 (서버사이드) |
| **문서 최신성** | 수동/주기적 동기화 | 실시간 공식 문서 반영 |
| **비용** | 임베딩 API + 벡터 DB 운영비 | 무료 (Upstash 제공) |
| **커스터마이징** | 완전 자유 | 제한적 (공식 문서만) |
| **적합한 용도** | 사내 지식베이스, 도메인 특화 | 개발 문서 참조, 코딩 어시스턴트 |

!!! warning "Context7은 전통 RAG를 대체하지 않는다"
    Context7은 **공식 라이브러리 문서** 검색에 특화되어 있다.
    사내 문서, 비즈니스 데이터, 도메인 특화 지식에는 여전히 **전통적 RAG**가 필요하다.
    두 방식은 **상호보완적**이다.

---

## MCP 기반 RAG의 핵심 패턴

Context7이 보여주는 "MCP 기반 RAG" 패턴은 다른 영역에도 적용할 수 있다.

### 패턴 1: Tool-as-Retriever

LLM이 **MCP Tool을 검색기(Retriever)로 사용**하는 패턴.

```python
# 전통적 RAG: 검색기를 코드에 하드코딩
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("질문")

# MCP 기반 RAG: LLM이 어떤 도구로 검색할지 스스로 결정
# LLM이 Context7, GitHub MCP, Slack MCP 중 적절한 것을 선택
```

!!! example "실생활 비유"
    - 전통적 RAG: 도서관에서 **한 서가**만 찾아보는 사람
    - MCP 기반 RAG: 도서관, 인터넷, 동료에게 물어보기 중 **가장 적절한 방법을 선택**하는 사람

### 패턴 2: Multi-Source RAG via MCP

여러 MCP 서버를 연결하여 **다중 소스 RAG**를 구현한다.

```
┌─────────────────────────────────────────────────────────┐
│                  Multi-Source MCP RAG                     │
│                                                          │
│  LLM이 질문의 성격에 따라 적절한 MCP 서버 선택:           │
│                                                          │
│  "React의 useEffect 사용법"                              │
│       → Context7 MCP (공식 문서 검색)                     │
│                                                          │
│  "이 PR의 변경 내역 요약해줘"                             │
│       → GitHub MCP (PR 데이터 조회)                       │
│                                                          │
│  "지난주 팀 논의 내용 찾아줘"                              │
│       → Slack MCP (메시지 검색)                           │
│                                                          │
│  "사내 온보딩 문서 찾아줘"                                │
│       → 자체 RAG 서버 (사내 문서 검색)                     │
└─────────────────────────────────────────────────────────┘
```

이 패턴은 7장에서 다룬 엔터프라이즈 RAG의 **MCP 버전**이다. 각 데이터 소스를 MCP 서버로 구현하면, LLM이 자율적으로 적절한 소스를 선택할 수 있다.

### 패턴 3: 버전 인식 RAG (Version-Aware RAG)

Context7의 독특한 기능 중 하나는 **버전별 문서 검색**이다.

```python
# 특정 버전의 문서만 검색
# Context7 라이브러리 ID에 버전을 포함
library_id = "/vercel/next.js/v14.3.0"  # Next.js 14.3.0 문서만 검색

# 이렇게 하면 해당 버전에만 존재하는 API를 정확히 답변 가능
```

!!! tip "왜 버전 인식이 중요한가?"
    - React 18 vs React 19는 API가 크게 다르다
    - 프로젝트가 특정 버전에 잠겨 있으면, 최신 버전 문서가 오히려 해로울 수 있다
    - 버전 인식 RAG는 **정확한 버전의 문서만** 검색하여 이 문제를 해결한다

---

## 직접 만들어보기: MCP Documentation RAG 서버

Context7의 핵심 아이디어를 활용하여 **자체 문서용 MCP RAG 서버**를 만들어보자.

### 프로젝트 구조

```
my-docs-mcp-server/
├── server.py         # MCP 서버 메인
├── indexer.py        # 문서 인덱싱
├── requirements.txt
└── docs/             # 검색할 문서들
```

### MCP 서버 구현

```python
# server.py
"""자체 문서를 검색하는 MCP RAG 서버"""

from mcp.server.fastmcp import FastMCP
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path

# ── MCP 서버 초기화 ──
mcp = FastMCP("my-docs-rag")

# ── 벡터 스토어 초기화 ──
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_docs"
)


@mcp.tool()
def index_documents(docs_directory: str) -> str:
    """디렉토리의 문서를 벡터 DB에 인덱싱합니다.

    Args:
        docs_directory: 마크다운 문서가 있는 디렉토리 경로
    """
    docs_path = Path(docs_directory)
    documents = []

    # 마크다운 파일 로드
    for md_file in docs_path.glob("**/*.md"):
        content = md_file.read_text(encoding="utf-8")
        documents.append(Document(
            page_content=content,
            metadata={
                "source": str(md_file),
                "filename": md_file.name,
                "category": md_file.parent.name
            }
        ))

    # 청킹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # 벡터 DB에 저장
    vectorstore.add_documents(chunks)

    return f"{len(documents)}개 문서에서 {len(chunks)}개 청크 인덱싱 완료"


@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> str:
    """인덱싱된 문서에서 관련 내용을 검색합니다.

    Args:
        query: 검색할 질문이나 키워드
        top_k: 반환할 결과 수 (기본값: 5)
    """
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    if not results:
        return "관련 문서를 찾을 수 없습니다."

    output_parts = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        similarity = 1 - score  # Chroma는 거리를 반환하므로 변환
        output_parts.append(
            f"**[{source}]** (유사도: {similarity:.2f})\n"
            f"{doc.page_content}\n"
        )

    return "\n---\n".join(output_parts)


@mcp.tool()
def list_indexed_docs() -> str:
    """현재 인덱싱된 문서 목록을 반환합니다."""
    collection = vectorstore._collection
    count = collection.count()

    if count == 0:
        return "인덱싱된 문서가 없습니다. index_documents 도구를 먼저 실행하세요."

    # 고유 소스 파일 목록
    all_data = collection.get(include=["metadatas"])
    sources = set()
    for metadata in all_data["metadatas"]:
        if metadata and "source" in metadata:
            sources.add(metadata["source"])

    return (
        f"총 {count}개 청크 인덱싱됨\n"
        f"소스 파일 {len(sources)}개:\n"
        + "\n".join(f"  - {s}" for s in sorted(sources))
    )


if __name__ == "__main__":
    mcp.run()
```

### requirements.txt

```
mcp>=1.0
langchain-openai>=0.3
langchain-text-splitters>=0.3
langchain-chroma>=0.2
chromadb>=0.5
```

### 설정 & 실행

**1. Claude Code에 MCP 서버 등록**

```json
// ~/.claude/settings.json 또는 프로젝트의 .mcp.json
{
  "mcpServers": {
    "my-docs-rag": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/my-docs-mcp-server",
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

**2. 사용 흐름**

```
사용자: "사내 API 가이드에서 인증 방법 찾아줘"
     ↓
LLM: search_docs("사내 API 인증 방법") 호출
     ↓
MCP 서버: 벡터 검색 → 관련 문서 반환
     ↓
LLM: 검색된 문서를 기반으로 정확한 답변 생성
```

!!! tip "Context7과의 차이점"
    Context7은 **공식 라이브러리 문서**를 검색하고,
    이 서버는 **자체 문서**를 검색한다.
    둘을 함께 사용하면 "공식 문서 + 사내 문서"를 동시에 검색하는
    **Multi-Source RAG**를 구현할 수 있다.

---

## Context7의 성능 최적화 기법

Context7이 프로덕션에서 사용하는 최적화 기법들을 분석해보자. 이 기법들은 자체 RAG 시스템에도 적용할 수 있다.

### 1. 서버사이드 리랭킹

```
일반적인 RAG:
  벡터 검색 → 20개 결과 반환 → LLM이 20개 전부 읽음 (토큰 낭비)

Context7 방식:
  벡터 검색 → 20개 결과 → 서버에서 리랭킹 → 상위 5개만 반환
  → 토큰 65% 절감 (9,700 → 3,300 토큰)
```

!!! info "왜 서버사이드인가?"
    리랭킹을 LLM에게 맡기면 불필요한 토큰을 소비한다.
    서버에서 미리 걸러내면 LLM은 **관련성 높은 문서만** 읽으면 된다.
    이것은 6장에서 배운 Reranking의 실전 적용이다.

### 2. 벤치마크 기반 품질 점수

```python
# Context7의 라이브러리 품질 평가 기준
quality_metrics = {
    "source_reputation": "High/Medium/Low",  # 출처 신뢰도
    "benchmark_score": 81.91,                 # 문서 품질 점수 (0-100)
    "code_snippets": 21066,                   # 코드 예제 수
}

# 이 점수를 기반으로 동일한 라이브러리의 여러 문서 소스 중
# 가장 품질이 좋은 것을 자동 선택
```

### 3. 코드 스니펫 특화 파싱

일반적인 RAG는 텍스트를 통째로 청킹하지만, Context7은 **코드 블록을 인식**하여 처리한다:

```
일반 청킹:
  "이 함수는 다음과 같다. ```python\ndef foo():\n    return" | "bar\n```"
  → 코드가 잘림!

Context7 방식:
  코드 블록을 하나의 단위로 보존 + 설명 텍스트를 메타데이터로 연결
  → 완전한 코드 예제가 항상 보장됨
```

이것은 3장에서 배운 "코드 문서는 함수/클래스 단위로 청킹" 원칙의 실전 적용이다.

---

## MCP 기반 RAG vs 전통 RAG: 언제 어떤 것을?

```
어떤 RAG 방식을 선택할까?

검색 대상이 공개 라이브러리 문서인가?
├─ Yes → Context7 사용 (이미 9,000+ 라이브러리 인덱싱됨)
└─ No
    ├─ 검색 대상이 자체/사내 문서인가?
    │   ├─ Yes, 규모가 작음 → MCP RAG 서버 직접 구축
    │   └─ Yes, 규모가 큼 → 전통 RAG (OpenSearch/pgvector 등)
    │
    └─ 여러 소스를 통합 검색해야 하는가?
        ├─ Yes → Multi-Source MCP RAG
        │        (Context7 + GitHub MCP + 자체 MCP 서버)
        └─ No → 단일 소스 전통 RAG로 충분
```

---

## 자주 묻는 질문

??? question "Q1. Context7은 무료인가요?"
    네, 현재 무료입니다. Upstash에서 컴퓨팅 비용을 부담하고 있습니다.
    다만 인프라 비용이 포함되지 않은 것일 뿐, 자체 RAG를 구축하면
    임베딩 API 비용, 벡터 DB 운영비 등이 발생합니다.

??? question "Q2. Context7이 지원하지 않는 라이브러리는 어떻게 하나요?"
    9,000+ 라이브러리를 지원하지만, 모든 라이브러리를 커버하지는 않습니다.
    지원하지 않는 경우:

    1. Context7에 라이브러리 추가 요청 (GitHub Issues)
    2. 자체 MCP RAG 서버를 구축하여 해당 문서를 인덱싱
    3. `llms.txt` 파일을 제공하는 라이브러리는 자동 인덱싱 가능

??? question "Q3. MCP 없이 Context7을 사용할 수 있나요?"
    네, REST API로도 사용할 수 있습니다. 하지만 MCP를 통해 사용하면
    LLM이 **자동으로** 적절한 시점에 문서를 검색하므로 훨씬 자연스럽습니다.

??? question "Q4. Context7과 웹 검색의 차이는 뭔가요?"
    - **웹 검색**: 블로그, StackOverflow 등 다양한 출처 → 품질 보장 어려움
    - **Context7**: **공식 문서만** 인덱싱 → 정확성 보장, 신뢰할 수 있는 출처

    웹 검색은 "어딘가에 답이 있을 것"이고,
    Context7은 "공식 문서에서 정확한 답을 찾아줌"입니다.

??? question "Q5. 한국어 문서도 검색되나요?"
    Context7은 주로 영어 공식 문서를 인덱싱합니다. 한국어 문서 검색이 필요하면
    자체 MCP RAG 서버를 구축하여 한국어 문서를 인덱싱하는 것이 좋습니다.
    다국어 임베딩 모델(bge-m3 등)을 사용하면 됩니다.

??? question "Q6. 전통 RAG를 이미 구축했는데, MCP로 전환해야 하나요?"
    전환이 아니라 **확장**으로 생각하세요. 기존 RAG 시스템을 MCP 서버로
    감싸면(wrapping), Claude Code 같은 MCP 호스트에서 자연스럽게 사용할 수 있습니다.
    기존 투자를 버리지 않고 MCP 생태계에 참여할 수 있습니다.

---

## 핵심 요약

!!! abstract "핵심 요약"
    1. **Context7 = Documentation RAG** — 공식 라이브러리 문서를 검색하여 LLM에 주입하는 특화된 RAG 시스템
    2. **MCP** = LLM과 외부 도구를 연결하는 표준 프로토콜 (USB-C 같은 것)
    3. Context7의 내부는 우리가 배운 RAG와 동일: **임베딩 → 벡터 검색 → 리랭킹 → 생성**
    4. **서버사이드 리랭킹**으로 토큰 65% 절감 — 자체 RAG에도 적용 가능
    5. 전통 RAG(사내 문서)와 MCP RAG(공식 문서)는 **상호보완적** — 함께 사용이 최적
    6. 자체 MCP RAG 서버를 만들면 Claude Code에서 **사내 문서 검색**을 자연스럽게 사용 가능
