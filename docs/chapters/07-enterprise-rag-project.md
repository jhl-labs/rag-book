# 7. 실전 프로젝트: 업무 AI 검색 에이전트

## 프로젝트 개요

수천 개의 Jira 이슈, GitHub PR/이슈, Confluence 문서를 **하나의 대화형 AI 검색 엔진**으로 통합하여 업무 생산성을 높이는 시스템을 구축한다.

```
┌─────────────────────────────────────────────────────────┐
│                    사용자 (ChatGPT 스타일 UI)              │
│         "지난달 결제 모듈 관련 장애 이력 알려줘"              │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│                   AI Agent (LangChain)                   │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │
│  │쿼리 분석  │→│ 소스 선택  │→│ 멀티소스 RAG 검색       │ │
│  │& 의도 파악│  │(Jira/GH/ │  │+ Reranking             │ │
│  │          │  │Confluence)│  │+ 답변 생성 & 의견 제시  │ │
│  └──────────┘  └──────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│                    벡터 DB (OpenSearch)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐          │
│  │  Jira    │  │  GitHub  │  │  Confluence  │          │
│  │ 이슈/댓글 │  │ PR/이슈  │  │   문서/페이지  │          │
│  └──────────┘  └──────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────┘
```

---

## 이런 질문에 답할 수 있다

| 질문 유형 | 예시 |
|-----------|------|
| **장애 이력** | "결제 시스템 관련 최근 3개월 장애 이력과 해결 방법" |
| **의사 결정** | "마이크로서비스 전환 관련 논의된 내용 정리해줘" |
| **담당자 찾기** | "인증 모듈 관련 가장 많이 작업한 사람이 누구야?" |
| **코드 리뷰** | "이 API 변경과 관련된 이전 PR이나 이슈 찾아줘" |
| **온보딩** | "신규 개발자가 알아야 할 배포 프로세스 문서" |
| **분석/의견** | "현재 백로그에서 가장 시급한 기술부채가 뭐야?" |

---

## Step 1: 데이터 수집기 구현

### Jira 수집기

```python
from jira import JIRA
from langchain.schema import Document
from datetime import datetime

class JiraCollector:
    def __init__(self, server, email, api_token):
        self.jira = JIRA(
            server=server,
            basic_auth=(email, api_token)
        )

    def collect(self, project_key, max_results=1000) -> list[Document]:
        documents = []
        issues = self.jira.search_issues(
            f"project={project_key} ORDER BY updated DESC",
            maxResults=max_results,
            fields="summary,description,comment,status,"
                   "assignee,reporter,labels,created,updated"
        )

        for issue in issues:
            # 이슈 본문
            content = f"""
[{issue.key}] {issue.fields.summary}
상태: {issue.fields.status.name}
담당자: {getattr(issue.fields.assignee, 'displayName', '미지정')}
보고자: {issue.fields.reporter.displayName}
라벨: {', '.join(issue.fields.labels) if issue.fields.labels else '없음'}

{issue.fields.description or '(설명 없음)'}
""".strip()

            # 댓글 포함
            comments = self.jira.comments(issue)
            if comments:
                content += "\n\n--- 댓글 ---\n"
                for c in comments:
                    content += f"\n[{c.author.displayName}] {c.body}\n"

            documents.append(Document(
                page_content=content,
                metadata={
                    "source": "jira",
                    "issue_key": issue.key,
                    "project": project_key,
                    "status": issue.fields.status.name,
                    "assignee": getattr(issue.fields.assignee, 'displayName', None),
                    "labels": issue.fields.labels or [],
                    "created": str(issue.fields.created),
                    "updated": str(issue.fields.updated),
                    "url": f"{self.jira.server_url}/browse/{issue.key}"
                }
            ))

        return documents
```

### GitHub 수집기

```python
from github import Github
from langchain.schema import Document

class GitHubCollector:
    def __init__(self, token):
        self.gh = Github(token)

    def collect(self, repo_name, max_items=500) -> list[Document]:
        documents = []
        repo = self.gh.get_repo(repo_name)

        # 이슈 수집
        for issue in repo.get_issues(state="all", sort="updated")[:max_items]:
            content = f"""
[Issue #{issue.number}] {issue.title}
상태: {issue.state}
작성자: {issue.user.login}
라벨: {', '.join(l.name for l in issue.labels)}

{issue.body or '(내용 없음)'}
""".strip()

            # 댓글 포함
            comments = list(issue.get_comments())
            if comments:
                content += "\n\n--- 댓글 ---\n"
                for c in comments[:10]:  # 상위 10개
                    content += f"\n[{c.user.login}] {c.body}\n"

            doc_type = "pull_request" if issue.pull_request else "issue"
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": "github",
                    "type": doc_type,
                    "number": issue.number,
                    "repo": repo_name,
                    "state": issue.state,
                    "author": issue.user.login,
                    "labels": [l.name for l in issue.labels],
                    "created": str(issue.created_at),
                    "updated": str(issue.updated_at),
                    "url": issue.html_url
                }
            ))

        return documents
```

### Confluence 수집기

```python
from atlassian import Confluence
from langchain.schema import Document
import re

class ConfluenceCollector:
    def __init__(self, url, username, api_token):
        self.confluence = Confluence(
            url=url,
            username=username,
            password=api_token
        )

    def collect(self, space_key, limit=500) -> list[Document]:
        documents = []
        pages = self.confluence.get_all_pages_from_space(
            space_key,
            start=0,
            limit=limit,
            expand="body.storage,version,ancestors"
        )

        for page in pages:
            # HTML → 텍스트 변환
            html_content = page.get("body", {}).get("storage", {}).get("value", "")
            text_content = re.sub(r'<[^>]+>', '', html_content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()

            # 경로 (상위 페이지 구조)
            ancestors = page.get("ancestors", [])
            path = " > ".join(a.get("title", "") for a in ancestors)

            content = f"""
[Confluence] {page['title']}
경로: {path}

{text_content}
""".strip()

            documents.append(Document(
                page_content=content,
                metadata={
                    "source": "confluence",
                    "page_id": page["id"],
                    "title": page["title"],
                    "space": space_key,
                    "path": path,
                    "version": page.get("version", {}).get("number", 1),
                    "updated": page.get("version", {}).get("when", ""),
                    "url": f"{self.confluence.url}/wiki/spaces/{space_key}/pages/{page['id']}"
                }
            ))

        return documents
```

---

## Step 2: 통합 인덱서

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch

class EnterpriseIndexer:
    def __init__(self, opensearch_url, index_name="enterprise-docs"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        self.opensearch_url = opensearch_url
        self.index_name = index_name

    def index_all(self, jira_docs, github_docs, confluence_docs):
        """모든 소스의 문서를 통합 인덱싱"""
        all_docs = jira_docs + github_docs + confluence_docs
        print(f"총 {len(all_docs)}개 문서 수집됨")

        # 청킹 (소스별 메타데이터 유지)
        chunks = self.splitter.split_documents(all_docs)
        print(f"총 {len(chunks)}개 청크 생성됨")

        # 청크에 컨텍스트 헤더 추가
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            header = f"[소스: {source}] "
            if source == "jira":
                header += f"[이슈: {chunk.metadata.get('issue_key', '')}] "
            elif source == "github":
                header += f"[{chunk.metadata.get('type', '')} #{chunk.metadata.get('number', '')}] "
            elif source == "confluence":
                header += f"[문서: {chunk.metadata.get('title', '')}] "
            chunk.page_content = header + chunk.page_content

        # OpenSearch에 저장
        vectorstore = OpenSearchVectorSearch.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            opensearch_url=self.opensearch_url,
            index_name=self.index_name
        )
        print(f"인덱싱 완료: {self.index_name}")
        return vectorstore
```

---

## Step 3: AI 검색 에이전트

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import OpenSearchVectorSearch

class EnterpriseRAGAgent:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.chat_history = []

        self.system_prompt = """당신은 팀의 업무 지식을 검색하고 분석하는 AI 어시스턴트입니다.

역할:
1. Jira 이슈, GitHub PR/이슈, Confluence 문서에서 관련 정보를 검색합니다
2. 검색 결과를 종합하여 명확하게 답변합니다
3. 출처(이슈 번호, 문서 링크)를 항상 표시합니다
4. 데이터 기반의 의견과 제안을 제공합니다

규칙:
- 검색 결과에 없는 내용은 추측하지 마세요
- 여러 소스의 정보를 교차 검증하세요
- 시간순으로 정리할 때는 최신 정보를 우선 표시하세요

컨텍스트:
{context}"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])

    def _retrieve(self, query, k=5, source_filter=None):
        """소스 필터링이 가능한 검색"""
        search_kwargs = {"k": k * 2}  # 필터링 후 k개 남기기 위해 여유분

        docs = self.vectorstore.similarity_search(query, **search_kwargs)

        if source_filter:
            docs = [d for d in docs if d.metadata.get("source") in source_filter]

        return docs[:k]

    def _detect_sources(self, question):
        """질문에서 관련 소스 자동 감지"""
        source_keywords = {
            "jira": ["이슈", "티켓", "백로그", "스프린트", "jira"],
            "github": ["PR", "풀리퀘", "커밋", "코드", "github", "리뷰"],
            "confluence": ["문서", "위키", "가이드", "매뉴얼", "confluence"]
        }

        detected = []
        q_lower = question.lower()
        for source, keywords in source_keywords.items():
            if any(kw in q_lower for kw in keywords):
                detected.append(source)

        return detected if detected else None  # None = 전체 검색

    def chat(self, question: str) -> str:
        """대화형 검색"""
        # 소스 감지
        sources = self._detect_sources(question)
        source_info = f" (검색 대상: {', '.join(sources)})" if sources else ""

        # 검색
        docs = self._retrieve(question, k=5, source_filter=sources)

        # 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            url = doc.metadata.get("url", "")
            context_parts.append(
                f"[{i+1}] ({source}) {doc.page_content}\n출처: {url}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # LLM 생성
        response = self.llm.invoke(
            self.prompt.format_messages(
                context=context,
                chat_history=self.chat_history,
                question=question
            )
        )

        # 기록 저장
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response.content))

        return response.content

    def reset(self):
        """대화 기록 초기화"""
        self.chat_history = []
```

---

## Step 4: 웹 UI (Streamlit)

```python
# app.py
import streamlit as st
from enterprise_rag import EnterpriseRAGAgent

st.set_page_config(page_title="업무 AI 검색", page_icon="🔍", layout="wide")
st.title("업무 AI 검색 에이전트")
st.caption("Jira · GitHub · Confluence 통합 검색")

# 세션 초기화
if "agent" not in st.session_state:
    st.session_state.agent = EnterpriseRAGAgent(vectorstore)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바: 필터 옵션
with st.sidebar:
    st.header("검색 옵션")
    sources = st.multiselect(
        "검색 소스",
        ["jira", "github", "confluence"],
        default=["jira", "github", "confluence"]
    )
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.agent.reset()

# 대화 기록 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
if prompt := st.chat_input("무엇이든 물어보세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            response = st.session_state.agent.chat(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

---

## Step 5: 주기적 동기화

```python
# sync.py — 크론잡으로 실행
from apscheduler.schedulers.blocking import BlockingScheduler
from enterprise_rag import JiraCollector, GitHubCollector, ConfluenceCollector, EnterpriseIndexer

def sync_all():
    """모든 소스에서 최신 데이터 수집 & 재인덱싱"""
    jira = JiraCollector(server="https://your-domain.atlassian.net",
                         email="you@example.com",
                         api_token="your-jira-token")
    github = GitHubCollector(token="your-github-token")
    confluence = ConfluenceCollector(url="https://your-domain.atlassian.net",
                                     username="you@example.com",
                                     api_token="your-confluence-token")

    jira_docs = jira.collect("PROJECT_KEY")
    github_docs = github.collect("org/repo")
    confluence_docs = confluence.collect("SPACE_KEY")

    indexer = EnterpriseIndexer(opensearch_url="http://localhost:9200")
    indexer.index_all(jira_docs, github_docs, confluence_docs)
    print("동기화 완료")

# 매 6시간마다 실행
scheduler = BlockingScheduler()
scheduler.add_job(sync_all, "interval", hours=6)
scheduler.start()
```

---

## 프로젝트 구조

```
enterprise-rag-agent/
├── collectors/
│   ├── jira_collector.py
│   ├── github_collector.py
│   └── confluence_collector.py
├── core/
│   ├── indexer.py
│   └── agent.py
├── app.py                # Streamlit 웹 UI
├── sync.py               # 주기적 동기화
├── docker-compose.yml    # OpenSearch + 앱
├── requirements.txt
└── .env.example
```

### docker-compose.yml

```yaml
services:
  opensearch:
    image: opensearchproject/opensearch:latest
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - DISABLE_SECURITY_PLUGIN=true
    volumes:
      - opensearch-data:/usr/share/opensearch/data

  app:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - opensearch
    env_file:
      - .env

volumes:
  opensearch-data:
```

### requirements.txt

```
langchain>=0.3
langchain-openai>=0.2
langchain-community>=0.3
opensearch-py>=2.4
sentence-transformers>=3.0
jira>=3.8
PyGithub>=2.1
atlassian-python-api>=3.41
streamlit>=1.38
apscheduler>=3.10
```

---

## 생산성 향상 효과

| 기존 방식 | AI 검색 에이전트 |
|-----------|----------------|
| Jira, GitHub, Confluence 각각 검색 | **한 곳에서 대화로 통합 검색** |
| 키워드 정확히 기억해야 검색 가능 | **자연어로 의미 검색** |
| 검색 결과 직접 읽고 종합 | **AI가 요약 + 의견 제시** |
| 장애 이력 수동 추적 | **"결제 장애 이력" 한마디로 조회** |
| 신규 입사자 온보딩 수일 소요 | **AI에게 물어보며 빠르게 파악** |

---

## 핵심 요약

!!! summary "이것만 기억하자"
    1. Jira/GitHub/Confluence를 **하나의 벡터 DB에 통합 인덱싱**
    2. 소스별 메타데이터(이슈 키, PR 번호, 문서 경로)가 검색 품질의 핵심
    3. **대화형 인터페이스**로 자연어 질문 → AI가 검색 + 종합 + 의견 제시
    4. 주기적 동기화로 항상 최신 데이터 유지
    5. 실무에선 **접근 권한 관리**와 **민감 정보 필터링**이 반드시 필요
