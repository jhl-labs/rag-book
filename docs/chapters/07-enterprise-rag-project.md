# 7. 실전 프로젝트: 업무 AI 검색 에이전트

!!! abstract "이 챕터에서 배우는 것"
    수천 개의 Jira 이슈, GitHub PR/이슈, Confluence 문서를 **하나의 대화형 AI 검색 엔진**으로 통합하는 실전 엔터프라이즈 프로젝트를 처음부터 끝까지 구축한다. 단순히 코드를 붙여넣는 것이 아니라, **왜 이렇게 설계했는지**, **어떤 문제를 해결하는지**를 함께 이해한다.

---

## 왜 이 프로젝트가 필요한가?

현대 소프트웨어 팀은 지식이 여러 곳에 흩어져 있다는 문제를 공통적으로 겪는다.

- **Jira**: 버그 리포트, 기능 요청, 스프린트 계획
- **GitHub**: 코드 변경 이력, PR 토론, 기술적 결정
- **Confluence**: 아키텍처 문서, 온보딩 가이드, 회의록

"6개월 전에 결제 모듈에서 발생했던 그 버그, 어떻게 해결했었지?"라는 질문에 답하려면 세 곳을 각각 뒤져야 한다. 이 프로젝트는 그 모든 곳을 **한 번의 자연어 질문**으로 검색하게 해준다.

!!! tip "왜 이게 중요한가?"
    소프트웨어 개발자는 하루 업무 시간의 약 **19%를 정보 검색**에 소비한다는 연구 결과가 있다(Stack Overflow Developer Survey). AI 검색 에이전트는 이 시간을 대폭 줄여줄 수 있다.

---

## 프로젝트 전체 아키텍처

### 시스템 구조도

```
┌─────────────────────────────────────────────────────────────────────┐
│                        사용자 인터페이스 레이어                          │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              Streamlit 웹 앱  (app.py)                       │   │
│   │   채팅 UI | 사이드바 필터 | 출처 배지 | 피드백 버튼              │   │
│   └─────────────────────────┬───────────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────────────┘
                              │ 질문 전달
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AI 에이전트 레이어                              │
│                                                                       │
│   ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│   │  의도 분류기  │──▶│  소스 선택기  │──▶│   멀티소스 RAG 검색     │  │
│   │ (Intent     │   │ (어느 DB를   │   │   + Reranking          │  │
│   │  Classifier)│   │  검색할지)   │   │   + 신뢰도 스코어링     │  │
│   └──────────────┘   └──────────────┘   └──────────┬─────────────┘  │
│                                                     │                │
│   ┌──────────────────────────────────────────────┐  │                │
│   │  LLM (GPT-4o) - 답변 생성 + 출처 인용         │◀─┘                │
│   └──────────────────────────────────────────────┘                  │
│                                                                       │
│   ┌──────────────────────────────────────────────┐                   │
│   │  대화 기록 관리 (멀티턴 컨텍스트 유지)            │                   │
│   └──────────────────────────────────────────────┘                  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ 벡터 검색
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        벡터 DB 레이어 (OpenSearch)                     │
│                                                                       │
│   ┌─────────────┐   ┌─────────────┐   ┌──────────────────────────┐  │
│   │    Jira     │   │   GitHub    │   │       Confluence         │  │
│   │  이슈/댓글  │   │  PR/이슈    │   │      문서/페이지          │  │
│   │  인덱스     │   │  인덱스     │   │      인덱스              │  │
│   └─────────────┘   └─────────────┘   └──────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ 인덱싱 (주기적 동기화)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        데이터 수집 레이어                               │
│                                                                       │
│   ┌─────────────┐   ┌─────────────┐   ┌──────────────────────────┐  │
│   │    Jira     │   │   GitHub    │   │       Confluence         │  │
│   │  Collector  │   │  Collector  │   │       Collector          │  │
│   │  (REST API) │   │  (REST API) │   │      (REST API)          │  │
│   └──────┬──────┘   └──────┬──────┘   └────────────┬─────────────┘  │
│          │                │                        │                │
│          ▼                ▼                        ▼                │
│    Jira Cloud       GitHub.com /          Confluence Cloud         │
│    (atlassian.net)  GitHub Enterprise     (atlassian.net)          │
└─────────────────────────────────────────────────────────────────────┘
```

### 각 컴포넌트 역할 설명

| 컴포넌트 | 역할 | 비유 |
|----------|------|------|
| **데이터 수집기 (Collector)** | API를 호출해서 원본 데이터를 가져옴 | 도서관에서 책을 가져오는 사서 |
| **인덱서 (Indexer)** | 문서를 청크로 나누고 벡터로 변환해 저장 | 책의 색인(Index)을 만드는 작업 |
| **벡터 DB (OpenSearch)** | 벡터 형태로 저장된 지식 창고 | 의미 기반으로 정리된 도서관 서가 |
| **AI 에이전트 (Agent)** | 질문을 이해하고, 검색하고, 답변 생성 | 도서관 전문 사서 |
| **웹 UI (Streamlit)** | 사용자가 대화하는 인터페이스 | 도서관 데스크 |
| **동기화 스케줄러** | 주기적으로 새 데이터를 가져와 인덱스 갱신 | 매일 새 책을 입고하는 작업 |

---

## 이런 질문에 답할 수 있다

| 질문 유형 | 예시 질문 | 어떤 소스를 검색하나? |
|-----------|----------|---------------------|
| **장애 이력** | "결제 시스템 관련 최근 3개월 장애 이력과 해결 방법" | Jira + GitHub |
| **의사 결정** | "마이크로서비스 전환 관련 논의된 내용 정리해줘" | Confluence + GitHub |
| **담당자 찾기** | "인증 모듈 관련 가장 많이 작업한 사람이 누구야?" | Jira + GitHub |
| **코드 리뷰** | "이 API 변경과 관련된 이전 PR이나 이슈 찾아줘" | GitHub + Jira |
| **온보딩** | "신규 개발자가 알아야 할 배포 프로세스 문서" | Confluence |
| **분석/의견** | "현재 백로그에서 가장 시급한 기술부채가 뭐야?" | Jira + Confluence |

---

## 프로젝트 구조

```
enterprise-rag-agent/
├── collectors/
│   ├── __init__.py
│   ├── jira_collector.py       # Jira 데이터 수집
│   ├── github_collector.py     # GitHub 데이터 수집
│   └── confluence_collector.py # Confluence 데이터 수집
├── core/
│   ├── __init__.py
│   ├── indexer.py              # 통합 인덱서
│   ├── agent.py                # AI 검색 에이전트
│   └── intent_classifier.py   # 질문 의도 분류기
├── ui/
│   └── app.py                  # Streamlit 웹 UI
├── sync.py                     # 주기적 동기화 스크립트
├── docker-compose.yml          # OpenSearch + 앱 컨테이너
├── requirements.txt
├── .env.example                # 환경 변수 템플릿
└── README.md
```

---

## Step 0: 환경 설정

### API 토큰 발급 방법

프로젝트를 시작하기 전에 세 가지 서비스의 API 토큰이 필요하다.

#### Jira API 토큰 발급

!!! info "Jira API 토큰이란?"
    Jira에 프로그래밍 방식으로 접근할 때 사용하는 비밀번호 역할을 하는 문자열이다. 실제 비밀번호 대신 API 토큰을 사용하면 보안상 더 안전하다.

1. [https://id.atlassian.com/manage-profile/security/api-tokens](https://id.atlassian.com/manage-profile/security/api-tokens) 접속
2. "Create API token" 버튼 클릭
3. 토큰 이름 입력 (예: `enterprise-rag-agent`)
4. 생성된 토큰 복사 (다시 볼 수 없으므로 즉시 저장)

#### GitHub Personal Access Token 발급

!!! info "Personal Access Token이란?"
    GitHub API를 사용할 때 신원을 증명하는 토큰이다. 세밀한 권한 설정이 가능하다.

1. GitHub 우측 상단 프로필 → Settings
2. Developer settings → Personal access tokens → Fine-grained tokens
3. "Generate new token" 클릭
4. Repository access: 검색 대상 레포지토리 선택
5. Permissions: `Issues: Read-only`, `Pull requests: Read-only` 선택
6. 생성된 토큰 저장

#### Confluence API 토큰

Confluence는 Jira와 같은 Atlassian 계정을 사용하므로 **Jira API 토큰과 동일한 토큰**을 사용한다.

### .env 파일 설정

```bash
# .env.example 을 복사해서 .env 로 만든다
cp .env.example .env
```

```bash
# .env 파일 내용
# ─────────────────────────────────────────
# Jira / Confluence 설정
ATLASSIAN_URL=https://your-company.atlassian.net
ATLASSIAN_EMAIL=your-email@company.com
ATLASSIAN_API_TOKEN=your-atlassian-api-token

# Jira 프로젝트 키 (예: PROJ, BACKEND, INFRA)
JIRA_PROJECT_KEYS=PROJ,BACKEND

# Confluence 스페이스 키 (예: ENG, DOCS)
CONFLUENCE_SPACE_KEYS=ENG,DOCS

# GitHub 설정
GITHUB_TOKEN=your-github-personal-access-token
GITHUB_REPOS=your-org/repo1,your-org/repo2

# OpenAI 설정
OPENAI_API_KEY=your-openai-api-key

# OpenSearch 설정
OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_INDEX=enterprise-docs
```

!!! warning "보안 주의사항"
    `.env` 파일은 절대로 Git에 커밋하지 않는다. `.gitignore`에 반드시 추가해야 한다.
    ```bash
    echo ".env" >> .gitignore
    ```

---

## Step 1: 데이터 수집기 구현

### 1-1. Jira 수집기 (완전판)

!!! info "용어 설명: JQL (Jira Query Language)"
    JQL은 Jira에서 이슈를 검색할 때 사용하는 쿼리 언어다. SQL과 비슷하게 조건을 지정해서 원하는 이슈만 가져올 수 있다.
    예: `project = PROJ AND updated >= -7d ORDER BY updated DESC`

```python
# collectors/jira_collector.py

from jira import JIRA
from langchain_core.documents import Document
from datetime import datetime, timezone
from typing import Optional
import time
import logging
import re

# 로깅 설정: 실행 중 어떤 작업을 하는지 콘솔에 출력
logger = logging.getLogger(__name__)


class JiraCollector:
    """
    Jira에서 이슈와 댓글을 수집해서 LangChain Document 형태로 반환하는 클래스.

    비유: Jira라는 거대한 서류 창고에서 필요한 서류를 골라 정리해 가져오는 직원.
    """

    def __init__(self, server: str, email: str, api_token: str):
        """
        Args:
            server: Jira 서버 URL (예: https://your-company.atlassian.net)
            email: Atlassian 계정 이메일
            api_token: Atlassian API 토큰
        """
        self.jira = JIRA(
            server=server,
            basic_auth=(email, api_token),
            # 연결 실패 시 최대 3번 재시도
            options={"max_retries": 3}
        )
        self.server_url = server
        logger.info(f"Jira 연결 성공: {server}")

    # ──────────────────────────────────────────
    # 공개 메서드: 외부에서 호출하는 메인 메서드
    # ──────────────────────────────────────────

    def collect(
        self,
        project_key: str,
        max_results: int = 1000,
        since: Optional[datetime] = None  # 증분 동기화용: 이 날짜 이후 변경된 것만
    ) -> list[Document]:
        """
        Jira 프로젝트에서 이슈를 수집한다.

        Args:
            project_key: Jira 프로젝트 키 (예: "PROJ", "BACKEND")
            max_results: 최대 수집 건수 (API 부하를 막기 위해 제한)
            since: 이 날짜 이후에 업데이트된 이슈만 가져옴 (증분 동기화)

        Returns:
            LangChain Document 리스트
        """
        documents = []

        # ── JQL 쿼리 구성 ──
        # JQL(Jira Query Language): SQL처럼 조건을 지정해서 이슈를 검색하는 언어
        jql = f"project = {project_key} ORDER BY updated DESC"
        if since:
            # 증분 동기화: 마지막 동기화 이후 변경된 이슈만 가져옴
            since_str = since.strftime("%Y-%m-%d %H:%M")
            jql = f'project = {project_key} AND updated >= "{since_str}" ORDER BY updated DESC'
            logger.info(f"증분 동기화 모드: {since_str} 이후 변경된 이슈만 수집")

        # ── 페이지네이션(Pagination) 처리 ──
        # Jira API는 한 번에 최대 100개까지만 반환한다.
        # 1000개를 가져오려면 100개씩 10번 요청해야 한다. 이를 페이지네이션이라 한다.
        start_at = 0
        page_size = 100  # Jira API 최대 페이지 크기

        while len(documents) < max_results:
            # 현재 페이지의 이슈 가져오기
            try:
                issues = self.jira.search_issues(
                    jql,
                    startAt=start_at,
                    maxResults=min(page_size, max_results - len(documents)),
                    fields="summary,description,comment,status,"
                           "assignee,reporter,labels,priority,"
                           "issuetype,created,updated,resolution"
                )
            except Exception as e:
                logger.error(f"Jira API 오류 (start={start_at}): {e}")
                # 일시적인 오류면 잠시 대기 후 재시도
                time.sleep(5)
                continue

            if not issues:
                break  # 더 이상 가져올 이슈가 없음

            for issue in issues:
                doc = self._issue_to_document(issue, project_key)
                if doc:
                    documents.append(doc)

            logger.info(f"수집 진행: {len(documents)}개 / 요청 최대: {max_results}개")

            # 다음 페이지로 이동
            start_at += len(issues)

            # Rate Limiting 방지: 너무 빠르게 API를 호출하면 차단될 수 있다
            # 100개 요청 후 0.5초 대기
            time.sleep(0.5)

            # 마지막 페이지인 경우 종료
            if len(issues) < page_size:
                break

        logger.info(f"Jira 수집 완료: 프로젝트={project_key}, 총 {len(documents)}개")
        return documents

    # ──────────────────────────────────────────
    # 내부 메서드: 이슈 하나를 Document로 변환
    # ──────────────────────────────────────────

    def _issue_to_document(self, issue, project_key: str) -> Optional[Document]:
        """
        Jira 이슈 객체 하나를 LangChain Document로 변환한다.

        비유: 서류 한 장을 표준 양식으로 재작성하는 작업.
        """
        try:
            # 기본 정보 추출
            assignee = getattr(issue.fields.assignee, 'displayName', '미지정')
            reporter = getattr(issue.fields.reporter, 'displayName', '알 수 없음')
            labels = ', '.join(issue.fields.labels) if issue.fields.labels else '없음'
            priority = getattr(issue.fields.priority, 'name', '알 수 없음')
            issue_type = getattr(issue.fields.issuetype, 'name', '알 수 없음')

            # HTML 태그 제거 (Jira 설명에 HTML이 포함될 수 있음)
            description = self._clean_text(issue.fields.description or '(설명 없음)')

            # 본문 구성
            content = f"""[{issue.key}] {issue.fields.summary}
유형: {issue_type}
우선순위: {priority}
상태: {issue.fields.status.name}
담당자: {assignee}
보고자: {reporter}
라벨: {labels}
생성일: {issue.fields.created}
수정일: {issue.fields.updated}

{description}"""

            # 댓글 포함 (중요한 컨텍스트 정보가 많음)
            try:
                comments = self.jira.comments(issue)
                if comments:
                    content += "\n\n--- 댓글 ---"
                    for c in comments[:20]:  # 최대 20개 댓글만 포함
                        author = getattr(c.author, 'displayName', '알 수 없음')
                        body = self._clean_text(c.body)
                        content += f"\n\n[{author}] {body}"
            except Exception as e:
                logger.warning(f"{issue.key} 댓글 로드 실패: {e}")

            return Document(
                page_content=content.strip(),
                metadata={
                    "source": "jira",
                    "issue_key": issue.key,
                    "project": project_key,
                    "type": issue_type,
                    "priority": priority,
                    "status": issue.fields.status.name,
                    "assignee": assignee,
                    "reporter": reporter,
                    "labels": issue.fields.labels or [],
                    "created": str(issue.fields.created),
                    "updated": str(issue.fields.updated),
                    "url": f"{self.server_url}/browse/{issue.key}",
                    # 중복 제거를 위한 고유 ID
                    "doc_id": f"jira_{issue.key}",
                    # 최신 여부 판단을 위한 타임스탬프
                    "updated_ts": issue.fields.updated
                }
            )
        except Exception as e:
            logger.error(f"이슈 변환 실패 ({getattr(issue, 'key', '?')}): {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """
        Jira 본문에서 불필요한 마크업을 제거한다.

        Jira는 자체 마크업 언어(Jira Markup)나 HTML을 사용하는데,
        이를 그대로 임베딩하면 노이즈가 될 수 있다.
        """
        if not text:
            return ""
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # Jira 코드 블록 표시 제거
        text = re.sub(r'\{code[^}]*\}|\{noformat[^}]*\}|\{[^}]+\}', '', text)
        # 연속된 빈 줄 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
```

### 1-2. GitHub 수집기 (완전판)

!!! info "용어 설명: GitHub REST API"
    GitHub는 이슈, PR, 커밋 등의 데이터를 프로그래밍 방식으로 읽을 수 있는 REST API를 제공한다. `PyGithub` 라이브러리를 사용하면 Python에서 쉽게 접근할 수 있다.

```python
# collectors/github_collector.py

from github import Github, RateLimitExceededException, GithubException
from langchain_core.documents import Document
from datetime import datetime
from typing import Optional
import time
import logging
import re

logger = logging.getLogger(__name__)


class GitHubCollector:
    """
    GitHub 레포지토리에서 이슈와 PR을 수집하는 클래스.

    PR(Pull Request): 코드 변경 사항을 메인 브랜치에 합치자고 요청하는 것.
    Issue: 버그 신고, 기능 요청 등을 기록하는 것.
    """

    def __init__(self, token: str):
        """
        Args:
            token: GitHub Personal Access Token
        """
        self.gh = Github(token)
        # Rate Limit 상태 확인 (GitHub API는 시간당 최대 5000번 호출 가능)
        rate_limit = self.gh.get_rate_limit()
        logger.info(f"GitHub 연결 성공. API 남은 요청 수: {rate_limit.core.remaining}/5000")

    def collect(
        self,
        repo_name: str,
        max_items: int = 500,
        since: Optional[datetime] = None,
        include_prs: bool = True,
        include_issues: bool = True
    ) -> list[Document]:
        """
        GitHub 레포지토리에서 이슈와 PR을 수집한다.

        Args:
            repo_name: "조직명/레포명" 형식 (예: "my-company/backend-api")
            max_items: 최대 수집 건수
            since: 이 날짜 이후 업데이트된 것만 수집 (증분 동기화)
            include_prs: PR 포함 여부
            include_issues: 일반 이슈 포함 여부
        """
        documents = []

        try:
            repo = self.gh.get_repo(repo_name)
            logger.info(f"레포지토리 접근 성공: {repo_name}")
        except GithubException as e:
            logger.error(f"레포지토리 접근 실패 ({repo_name}): {e}")
            return []

        # GitHub API에서 이슈와 PR을 함께 가져옴
        # (GitHub에서는 PR도 이슈의 한 종류로 취급됨)
        kwargs = {"state": "all", "sort": "updated", "direction": "desc"}
        if since:
            kwargs["since"] = since
            logger.info(f"증분 동기화 모드: {since} 이후 항목만 수집")

        count = 0
        for item in repo.get_issues(**kwargs):
            if count >= max_items:
                break

            is_pr = item.pull_request is not None

            # 필터링: PR/이슈 선택적 수집
            if is_pr and not include_prs:
                continue
            if not is_pr and not include_issues:
                continue

            # Rate Limit 처리
            # GitHub API를 너무 빨리 호출하면 일시적으로 차단된다
            try:
                doc = self._item_to_document(item, repo_name, is_pr)
                if doc:
                    documents.append(doc)
                    count += 1
            except RateLimitExceededException:
                # Rate Limit 초과 시 1분 대기 후 재시도
                logger.warning("GitHub API Rate Limit 초과. 60초 대기 중...")
                time.sleep(60)
                doc = self._item_to_document(item, repo_name, is_pr)
                if doc:
                    documents.append(doc)
                    count += 1

            # API 부하 분산: 10개마다 0.2초 대기
            if count % 10 == 0:
                time.sleep(0.2)
                logger.info(f"GitHub 수집 진행: {count}개")

        logger.info(f"GitHub 수집 완료: {repo_name}, 총 {len(documents)}개")
        return documents

    def _item_to_document(self, item, repo_name: str, is_pr: bool) -> Optional[Document]:
        """이슈/PR 하나를 Document로 변환"""
        try:
            item_type = "pull_request" if is_pr else "issue"
            labels = [label.name for label in item.labels]
            body = self._clean_markdown(item.body or "(내용 없음)")

            content = f"""[{'PR' if is_pr else 'Issue'} #{item.number}] {item.title}
타입: {item_type}
상태: {item.state}
작성자: {item.user.login}
라벨: {', '.join(labels) if labels else '없음'}
생성일: {item.created_at}
수정일: {item.updated_at}

{body}"""

            # 댓글 포함 (최대 15개)
            try:
                comments = list(item.get_comments())[:15]
                if comments:
                    content += "\n\n--- 댓글 ---"
                    for c in comments:
                        c_body = self._clean_markdown(c.body or "")
                        content += f"\n\n[{c.user.login}] {c_body}"
            except Exception as e:
                logger.warning(f"#{item.number} 댓글 로드 실패: {e}")

            # PR인 경우 리뷰 코멘트도 포함
            if is_pr:
                try:
                    pr = item.repository.get_pull(item.number)
                    review_comments = list(pr.get_review_comments())[:10]
                    if review_comments:
                        content += "\n\n--- 코드 리뷰 ---"
                        for rc in review_comments:
                            content += f"\n\n[{rc.user.login}] 파일: {rc.path}\n{rc.body}"
                except Exception as e:
                    logger.warning(f"PR #{item.number} 리뷰 로드 실패: {e}")

            return Document(
                page_content=content.strip(),
                metadata={
                    "source": "github",
                    "type": item_type,
                    "number": item.number,
                    "repo": repo_name,
                    "state": item.state,
                    "author": item.user.login,
                    "labels": labels,
                    "created": str(item.created_at),
                    "updated": str(item.updated_at),
                    "url": item.html_url,
                    "doc_id": f"github_{repo_name}_{item.number}",
                    "updated_ts": str(item.updated_at)
                }
            )
        except Exception as e:
            logger.error(f"GitHub 항목 변환 실패 (#{getattr(item, 'number', '?')}): {e}")
            return None

    def _clean_markdown(self, text: str) -> str:
        """마크다운에서 렌더링에 불필요한 요소 정리"""
        if not text:
            return ""
        # 이미지 링크 제거 (텍스트 검색에 불필요)
        text = re.sub(r'!\[.*?\]\(.*?\)', '[이미지]', text)
        # 긴 코드 블록은 요약 (토큰 절약)
        text = re.sub(r'```[\s\S]{500,}?```', '[코드 블록 생략]', text)
        # 연속 빈 줄 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
```

### 1-3. Confluence 수집기 (완전판)

!!! info "용어 설명: Confluence Space"
    Confluence에서 Space(스페이스)는 페이지들의 그룹이다. 팀별, 프로젝트별로 스페이스를 나눠서 관리한다. 예: 엔지니어링 팀 스페이스(`ENG`), 제품 팀 스페이스(`PROD`).

```python
# collectors/confluence_collector.py

from atlassian import Confluence
from langchain_core.documents import Document
from bs4 import BeautifulSoup  # HTML 파싱 라이브러리
from datetime import datetime
from typing import Optional
import time
import logging
import re

logger = logging.getLogger(__name__)


class ConfluenceCollector:
    """
    Confluence에서 페이지를 수집하는 클래스.

    Confluence는 팀 위키이자 문서화 도구다.
    아키텍처 문서, 온보딩 가이드, 회의록 등이 여기에 있다.
    """

    def __init__(self, url: str, username: str, api_token: str):
        """
        Args:
            url: Confluence URL (예: https://your-company.atlassian.net)
            username: Atlassian 계정 이메일
            api_token: Atlassian API 토큰
        """
        self.confluence = Confluence(
            url=url,
            username=username,
            password=api_token,
            cloud=True  # Confluence Cloud 사용 여부
        )
        self.base_url = url
        logger.info(f"Confluence 연결 성공: {url}")

    def collect(
        self,
        space_key: str,
        limit: int = 500,
        since: Optional[datetime] = None
    ) -> list[Document]:
        """
        Confluence 스페이스에서 모든 페이지를 수집한다.

        Args:
            space_key: 스페이스 키 (예: "ENG", "DOCS")
            limit: 최대 수집 페이지 수
            since: 이 날짜 이후 수정된 페이지만 수집
        """
        documents = []
        start = 0
        page_size = 50  # Confluence API 기본 페이지 크기

        while len(documents) < limit:
            try:
                pages = self.confluence.get_all_pages_from_space(
                    space_key,
                    start=start,
                    limit=min(page_size, limit - len(documents)),
                    expand="body.storage,version,ancestors,metadata.labels"
                )
            except Exception as e:
                logger.error(f"Confluence API 오류 (start={start}): {e}")
                time.sleep(5)
                break

            if not pages:
                break

            for page in pages:
                # 증분 동기화: since 이후 수정된 페이지만 처리
                if since:
                    updated_str = page.get("version", {}).get("when", "")
                    if updated_str:
                        try:
                            updated = datetime.fromisoformat(
                                updated_str.replace("Z", "+00:00")
                            )
                            if updated <= since.replace(tzinfo=updated.tzinfo):
                                continue
                        except ValueError:
                            pass

                doc = self._page_to_document(page, space_key)
                if doc:
                    documents.append(doc)

            logger.info(f"Confluence 수집 진행: {len(documents)}개")
            start += len(pages)

            # API 부하 분산
            time.sleep(0.3)

            if len(pages) < page_size:
                break

        logger.info(f"Confluence 수집 완료: space={space_key}, 총 {len(documents)}개")
        return documents

    def _page_to_document(self, page: dict, space_key: str) -> Optional[Document]:
        """Confluence 페이지 딕셔너리를 Document로 변환"""
        try:
            page_id = page["id"]
            title = page.get("title", "제목 없음")

            # HTML → 텍스트 변환
            # Confluence는 페이지 본문을 HTML 형태로 저장한다.
            # BeautifulSoup을 사용해 HTML 태그를 제거하고 순수 텍스트만 추출한다.
            html_content = page.get("body", {}).get("storage", {}).get("value", "")
            text_content = self._html_to_text(html_content)

            if not text_content.strip():
                logger.debug(f"빈 페이지 건너뜀: {title}")
                return None

            # 페이지 계층 경로 구성 (예: 엔지니어링 > 백엔드 > API 설계)
            # ancestors: 상위 페이지들의 목록
            ancestors = page.get("ancestors", [])
            path = " > ".join(a.get("title", "") for a in ancestors)
            if path:
                path = f"{path} > {title}"
            else:
                path = title

            # 라벨 추출
            labels = []
            label_data = page.get("metadata", {}).get("labels", {}).get("results", [])
            labels = [label.get("name", "") for label in label_data]

            # 버전 정보
            version = page.get("version", {}).get("number", 1)
            updated = page.get("version", {}).get("when", "")
            author = page.get("version", {}).get("by", {}).get("displayName", "알 수 없음")

            content = f"""[Confluence] {title}
경로: {path}
스페이스: {space_key}
작성자: {author}
버전: {version}
수정일: {updated}
라벨: {', '.join(labels) if labels else '없음'}

{text_content}"""

            return Document(
                page_content=content.strip(),
                metadata={
                    "source": "confluence",
                    "page_id": page_id,
                    "title": title,
                    "space": space_key,
                    "path": path,
                    "labels": labels,
                    "version": version,
                    "updated": updated,
                    "author": author,
                    "url": f"{self.base_url}/wiki/spaces/{space_key}/pages/{page_id}",
                    "doc_id": f"confluence_{page_id}",
                    "updated_ts": updated
                }
            )
        except Exception as e:
            logger.error(f"Confluence 페이지 변환 실패 ({page.get('id', '?')}): {e}")
            return None

    def _html_to_text(self, html: str) -> str:
        """
        HTML을 읽기 좋은 텍스트로 변환한다.

        단순히 태그를 제거하는 것보다 BeautifulSoup을 사용하면
        테이블, 목록 등의 구조를 더 자연스럽게 변환할 수 있다.
        """
        if not html:
            return ""

        soup = BeautifulSoup(html, "html.parser")

        # 테이블을 텍스트 형식으로 변환
        for table in soup.find_all("table"):
            rows = []
            for row in table.find_all("tr"):
                cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
                rows.append(" | ".join(cells))
            table.replace_with("\n".join(rows))

        # 코드 블록 표시
        for code in soup.find_all(["code", "pre"]):
            code.replace_with(f"\n```\n{code.get_text()}\n```\n")

        text = soup.get_text(separator="\n")
        # 연속 빈 줄 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
```

---

## Step 2: 통합 인덱서 (배치 처리 + 중복 제거)

!!! info "용어 설명: 임베딩(Embedding)"
    텍스트를 숫자 벡터로 변환하는 과정이다. 예를 들어 "결제 오류"라는 텍스트를 [0.23, -0.87, 0.45, ...] 같은 숫자 배열로 바꾼다. 의미가 비슷한 텍스트는 비슷한 벡터가 된다.

!!! info "용어 설명: 청킹(Chunking)"
    긴 문서를 작은 조각으로 나누는 과정이다. LLM이 한 번에 처리할 수 있는 텍스트 길이(컨텍스트 윈도우)에 한계가 있기 때문이다. 마치 두꺼운 책을 챕터별로 나눠서 색인을 만드는 것과 같다.

```python
# core/indexer.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from typing import Optional
import hashlib
import logging
import time

logger = logging.getLogger(__name__)


class EnterpriseIndexer:
    """
    여러 소스에서 수집한 문서를 벡터 DB에 인덱싱하는 클래스.

    주요 기능:
    1. 문서 청킹 (큰 문서를 작은 조각으로 분할)
    2. 중복 문서 제거 (같은 문서를 두 번 인덱싱하지 않음)
    3. 배치 처리 (메모리 효율적으로 대량 인덱싱)
    4. 진행 상황 추적
    """

    def __init__(
        self,
        opensearch_url: str,
        index_name: str = "enterprise-docs",
        embedding_model: str = "text-embedding-3-large",
        batch_size: int = 100  # 한 번에 처리할 문서 수
    ):
        """
        Args:
            opensearch_url: OpenSearch 서버 URL (예: http://localhost:9200)
            index_name: 인덱스 이름 (DB의 테이블 이름에 해당)
            embedding_model: 사용할 임베딩 모델
            batch_size: 배치 처리 크기 (메모리 부족 시 줄이면 됨)
        """
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,       # 청크 최대 크기 (문자 수)
            chunk_overlap=100,    # 청크 간 겹침 크기 (문맥 연속성 유지)
            separators=["\n\n", "\n", ".", " ", ""]  # 분할 기준 순서
        )
        self.opensearch_url = opensearch_url
        self.index_name = index_name
        self.batch_size = batch_size

    def index_all(
        self,
        jira_docs: list[Document],
        github_docs: list[Document],
        confluence_docs: list[Document],
        incremental: bool = False  # True면 기존 문서 유지하고 새/변경된 것만 추가
    ) -> OpenSearchVectorSearch:
        """
        모든 소스의 문서를 통합 인덱싱한다.

        Args:
            incremental: True면 증분 업데이트 (기존 인덱스 유지)
                         False면 전체 재인덱싱 (기존 인덱스 삭제 후 재생성)
        """
        all_docs = jira_docs + github_docs + confluence_docs
        logger.info(f"총 수집 문서: {len(all_docs)}개 "
                    f"(Jira: {len(jira_docs)}, "
                    f"GitHub: {len(github_docs)}, "
                    f"Confluence: {len(confluence_docs)})")

        # ── 1단계: 중복 제거 ──
        # 같은 문서가 여러 번 수집됐을 경우 하나만 남긴다.
        unique_docs = self._deduplicate(all_docs)
        logger.info(f"중복 제거 후: {len(unique_docs)}개 "
                    f"({len(all_docs) - len(unique_docs)}개 중복 제거)")

        # ── 2단계: 데이터 정제 ──
        clean_docs = self._normalize_documents(unique_docs)

        # ── 3단계: 청킹 ──
        chunks = self.splitter.split_documents(clean_docs)
        logger.info(f"청킹 완료: {len(chunks)}개 청크 생성")

        # ── 4단계: 컨텍스트 헤더 추가 ──
        # 각 청크에 출처 정보를 앞에 붙여서 검색 품질을 높인다
        self._add_context_headers(chunks)

        # ── 5단계: 배치 임베딩 & 인덱싱 ──
        vectorstore = self._batch_index(chunks, incremental)

        logger.info("인덱싱 완료!")
        return vectorstore

    def _deduplicate(self, documents: list[Document]) -> list[Document]:
        """
        중복 문서를 제거한다.

        같은 Jira 이슈를 두 번 수집했거나, 증분 동기화에서
        이미 있는 문서가 다시 들어오는 경우를 처리한다.

        방법: doc_id 메타데이터를 해시해서 고유 ID로 사용한다.
        """
        seen_ids = set()
        unique = []

        for doc in documents:
            # doc_id가 있으면 사용, 없으면 내용 기반으로 해시 생성
            doc_id = doc.metadata.get("doc_id")
            if not doc_id:
                # 내용의 처음 200자를 해시해서 ID 생성
                content_hash = hashlib.md5(
                    doc.page_content[:200].encode()
                ).hexdigest()
                doc_id = f"hash_{content_hash}"

            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique.append(doc)

        return unique

    def _normalize_documents(self, documents: list[Document]) -> list[Document]:
        """
        문서를 정규화한다 (인코딩 문제, 특수문자 등 처리).
        """
        normalized = []
        for doc in documents:
            # 너무 짧은 문서 제거 (50자 미만은 의미 없는 내용일 가능성이 높음)
            if len(doc.page_content.strip()) < 50:
                continue

            # 너무 긴 문서는 청킹 전에 일부 잘라냄 (OpenAI 토큰 한도 대비)
            # text-embedding-3-large는 최대 8191 토큰
            if len(doc.page_content) > 30000:
                doc.page_content = doc.page_content[:30000] + "\n\n[이하 내용 생략]"

            normalized.append(doc)

        return normalized

    def _add_context_headers(self, chunks: list[Document]) -> None:
        """
        각 청크 앞에 출처 정보 헤더를 추가한다.

        헤더 예시:
        [소스: jira] [이슈: PROJ-123] [상태: Done]
        """
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")

            if source == "jira":
                issue_key = chunk.metadata.get("issue_key", "")
                status = chunk.metadata.get("status", "")
                header = f"[소스: Jira] [이슈: {issue_key}] [상태: {status}] "
            elif source == "github":
                item_type = chunk.metadata.get("type", "")
                number = chunk.metadata.get("number", "")
                repo = chunk.metadata.get("repo", "")
                header = f"[소스: GitHub] [{item_type} #{number}] [레포: {repo}] "
            elif source == "confluence":
                title = chunk.metadata.get("title", "")
                path = chunk.metadata.get("path", "")
                header = f"[소스: Confluence] [문서: {title}] [경로: {path}] "
            else:
                header = f"[소스: {source}] "

            chunk.page_content = header + chunk.page_content

    def _batch_index(
        self,
        chunks: list[Document],
        incremental: bool
    ) -> OpenSearchVectorSearch:
        """
        청크를 배치(batch) 단위로 나눠서 OpenSearch에 인덱싱한다.

        배치 처리가 필요한 이유:
        - 수천 개의 청크를 한 번에 임베딩하면 API 비용이 많이 들거나 실패할 수 있다.
        - 배치로 나누면 중간에 실패해도 처리된 부분은 저장된다.
        - 진행 상황을 추적할 수 있다.
        """
        total = len(chunks)
        vectorstore = None

        for i in range(0, total, self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size

            logger.info(f"배치 {batch_num}/{total_batches} 처리 중... "
                        f"({i}~{min(i + self.batch_size, total)}/{total})")

            try:
                if vectorstore is None and not incremental:
                    # 첫 번째 배치: 새 인덱스 생성
                    vectorstore = OpenSearchVectorSearch.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        opensearch_url=self.opensearch_url,
                        index_name=self.index_name,
                        engine="faiss",  # 벡터 검색 엔진
                        space_type="innerproduct"  # 유사도 계산 방식
                    )
                else:
                    # 이후 배치: 기존 인덱스에 추가
                    if vectorstore is None:
                        # 증분 모드에서 기존 인덱스에 연결
                        vectorstore = OpenSearchVectorSearch(
                            opensearch_url=self.opensearch_url,
                            index_name=self.index_name,
                            embedding_function=self.embeddings
                        )
                    vectorstore.add_documents(batch)

                # API 부하 분산: 배치 사이에 잠시 대기
                time.sleep(1)

            except Exception as e:
                logger.error(f"배치 {batch_num} 인덱싱 실패: {e}")
                # 실패해도 계속 진행 (일부 배치 실패는 허용)
                time.sleep(5)
                continue

        logger.info(f"전체 {total}개 청크 인덱싱 완료")
        return vectorstore
```

---

## Step 3: 의도 분류기

!!! info "왜 의도 분류가 필요한가?"
    "결제 버그 알려줘"와 "결제 관련 문서 찾아줘"는 비슷해 보이지만, 전자는 Jira를 검색해야 하고 후자는 Confluence를 검색해야 한다. 의도 분류기는 질문의 종류를 파악해서 최적의 소스를 선택하게 해준다.

```python
# core/intent_classifier.py

from enum import Enum
from dataclasses import dataclass
import re


class QueryIntent(Enum):
    """질문의 의도(유형) 분류"""
    BUG_INVESTIGATION = "버그_조사"       # 버그, 오류, 장애 관련
    FEATURE_REQUEST = "기능_요청"         # 기능, 요구사항 관련
    CODE_REVIEW = "코드_리뷰"             # PR, 코드 변경 관련
    DOCUMENTATION = "문서화"              # 가이드, 위키, 매뉴얼 관련
    ONBOARDING = "온보딩"                 # 신규 입사자용 정보
    INCIDENT_HISTORY = "장애_이력"        # 과거 장애 분석
    PERSON_SEARCH = "담당자_찾기"         # 특정 작업 담당자 조회
    GENERAL = "일반"                      # 분류 불가 일반 질문


@dataclass
class ClassificationResult:
    """의도 분류 결과"""
    intent: QueryIntent
    confidence: float          # 신뢰도 (0.0 ~ 1.0)
    recommended_sources: list  # 검색 권장 소스 목록
    search_keywords: list      # 검색에 사용할 키워드


class IntentClassifier:
    """
    사용자 질문의 의도를 분석해서 최적의 검색 전략을 결정하는 클래스.

    규칙 기반(Rule-based) 분류를 사용한다.
    LLM 기반 분류도 가능하지만, 규칙 기반이 더 빠르고 비용이 적다.
    """

    # 의도별 키워드 매핑
    INTENT_PATTERNS = {
        QueryIntent.BUG_INVESTIGATION: {
            "keywords": [
                "버그", "오류", "에러", "error", "bug", "실패", "broken",
                "안 됨", "안됨", "이상", "문제", "크래시", "crash", "exception"
            ],
            "sources": ["jira", "github"],
            "weight": 1.0
        },
        QueryIntent.INCIDENT_HISTORY: {
            "keywords": [
                "장애", "인시던트", "incident", "downtime", "다운", "서비스 중단",
                "장애 이력", "이전 장애", "지난 장애", "복구", "RCA", "post-mortem"
            ],
            "sources": ["jira", "confluence"],
            "weight": 1.2  # 높은 가중치 = 더 강한 신호
        },
        QueryIntent.CODE_REVIEW: {
            "keywords": [
                "PR", "풀리퀘", "pull request", "코드 리뷰", "변경", "수정",
                "커밋", "merge", "머지", "diff", "코드 변경"
            ],
            "sources": ["github"],
            "weight": 1.0
        },
        QueryIntent.DOCUMENTATION: {
            "keywords": [
                "문서", "가이드", "매뉴얼", "위키", "wiki", "설명",
                "어떻게", "방법", "절차", "프로세스", "confluenc"
            ],
            "sources": ["confluence"],
            "weight": 1.0
        },
        QueryIntent.ONBOARDING: {
            "keywords": [
                "온보딩", "신규", "입사", "처음", "시작", "beginner",
                "튜토리얼", "tutorial", "시작하는 법"
            ],
            "sources": ["confluence"],
            "weight": 1.1
        },
        QueryIntent.PERSON_SEARCH: {
            "keywords": [
                "담당자", "누가", "작성자", "만든 사람", "소유자", "owner",
                "responsible", "담당", "팀원"
            ],
            "sources": ["jira", "github"],
            "weight": 1.0
        },
        QueryIntent.FEATURE_REQUEST: {
            "keywords": [
                "기능", "feature", "요구사항", "requirement", "스펙", "spec",
                "기획", "신규 기능", "추가"
            ],
            "sources": ["jira", "confluence"],
            "weight": 0.9
        }
    }

    def classify(self, question: str) -> ClassificationResult:
        """
        질문을 분석해서 의도를 분류한다.

        Args:
            question: 사용자 질문

        Returns:
            ClassificationResult: 의도, 신뢰도, 권장 소스, 키워드
        """
        q_lower = question.lower()
        scores = {}

        # 각 의도별로 점수 계산
        for intent, config in self.INTENT_PATTERNS.items():
            score = 0
            matched_keywords = []
            for keyword in config["keywords"]:
                if keyword in q_lower:
                    score += config["weight"]
                    matched_keywords.append(keyword)
            if score > 0:
                scores[intent] = {
                    "score": score,
                    "sources": config["sources"],
                    "keywords": matched_keywords
                }

        if not scores:
            # 매칭되는 의도가 없으면 일반 검색 (모든 소스 대상)
            return ClassificationResult(
                intent=QueryIntent.GENERAL,
                confidence=0.3,
                recommended_sources=["jira", "github", "confluence"],
                search_keywords=self._extract_keywords(question)
            )

        # 가장 높은 점수의 의도 선택
        best_intent = max(scores, key=lambda k: scores[k]["score"])
        best_score = scores[best_intent]["score"]

        # 신뢰도 계산 (0~1 사이로 정규화)
        max_possible = 5.0
        confidence = min(best_score / max_possible, 1.0)

        return ClassificationResult(
            intent=best_intent,
            confidence=confidence,
            recommended_sources=scores[best_intent]["sources"],
            search_keywords=scores[best_intent]["keywords"]
        )

    def _extract_keywords(self, text: str) -> list:
        """질문에서 핵심 키워드를 추출한다"""
        # 불용어(stop words) 제거 후 2자 이상 단어 추출
        stop_words = {"을", "를", "이", "가", "은", "는", "에", "의", "도", "로",
                      "과", "와", "에서", "으로", "이야", "야", "알려줘", "줘"}
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text)
        return [w for w in words if w not in stop_words and len(w) >= 2]
```

---

## Step 4: AI 검색 에이전트 (멀티턴 + 출처 인용 + 신뢰도)

```python
# core/agent.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import OpenSearchVectorSearch
from core.intent_classifier import IntentClassifier, QueryIntent
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """에이전트 응답 구조체"""
    answer: str                  # 생성된 답변
    sources: list[dict]          # 참조한 출처 목록
    intent: str                  # 분류된 의도
    confidence: float            # 전체 신뢰도 점수
    searched_sources: list[str]  # 실제로 검색한 소스 목록


class EnterpriseRAGAgent:
    """
    기업 내 문서를 검색하고 답변을 생성하는 AI 에이전트.

    핵심 기능:
    1. 의도 분류: 질문이 버그 관련인지, 문서 요청인지 파악
    2. 스마트 소스 선택: 의도에 맞는 소스만 검색
    3. 멀티턴 대화: 이전 대화를 기억하고 문맥을 유지
    4. 출처 인용: 클릭 가능한 링크와 함께 근거 제시
    5. 신뢰도 스코어링: 얼마나 확신할 수 있는 답변인지 표시
    """

    SYSTEM_PROMPT = """당신은 팀의 업무 지식을 검색하고 분석하는 전문 AI 어시스턴트입니다.

역할:
- Jira 이슈, GitHub PR/이슈, Confluence 문서에서 관련 정보를 검색합니다.
- 검색 결과를 바탕으로 명확하고 구체적인 답변을 생성합니다.
- 반드시 출처를 [1], [2] 형식으로 표시하고, 답변 마지막에 출처 목록을 정리합니다.
- 불확실한 정보는 "확실하지 않지만" 또는 "추가 확인이 필요합니다"라고 명시합니다.

답변 형식:
1. 핵심 답변 (2-3줄 요약)
2. 상세 내용 (필요 시)
3. 출처 목록 (이슈 번호 또는 문서 링크)

규칙:
- 검색 결과에 없는 내용은 추측하지 마세요.
- 여러 출처의 정보를 교차 검증하세요.
- 시간 관련 질문은 최신 정보를 우선 표시하세요.
- 기술적 용어는 간단히 설명을 추가하세요.

검색된 컨텍스트:
{context}"""

    def __init__(self, vectorstore: OpenSearchVectorSearch):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.classifier = IntentClassifier()
        # ⚠️ chat_history는 인스턴스에 저장하지 않는다.
        # 멀티유저 환경에서 여러 사용자가 같은 에이전트 인스턴스를 공유하면
        # 대화 기록이 섞이는 버그가 발생한다.
        # 대신 chat() 메서드에 session_history를 인자로 받는 방식을 사용한다.

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])

    def chat(self, question: str, session_history: list | None = None) -> AgentResponse:
        """
        사용자 질문에 대한 답변을 생성한다.

        Args:
            question: 사용자의 자연어 질문
            session_history: 세션별 대화 기록 (멀티유저 세션 격리를 위해 외부에서 전달)
                            Streamlit에서는 st.session_state["chat_history"]를 사용한다.

        Returns:
            AgentResponse: 답변, 출처, 신뢰도 등을 포함한 구조화된 응답
        """
        # 세션 격리: 각 사용자의 대화 기록을 분리해서 관리
        if session_history is None:
            session_history = []
        # ── 1단계: 의도 분류 ──
        classification = self.classifier.classify(question)
        logger.info(f"의도 분류: {classification.intent.value} "
                    f"(신뢰도: {classification.confidence:.2f})")
        logger.info(f"검색 소스: {classification.recommended_sources}")

        # ── 2단계: 문서 검색 ──
        docs = self._smart_retrieve(
            question=question,
            sources=classification.recommended_sources,
            k=6
        )

        if not docs:
            # 추천 소스에서 못 찾으면 전체 검색
            logger.info("추천 소스에서 결과 없음. 전체 소스 검색으로 전환")
            docs = self._smart_retrieve(question=question, sources=None, k=6)

        # ── 3단계: 컨텍스트 + 출처 구성 ──
        context, sources = self._build_context(docs)

        # ── 4단계: 신뢰도 계산 ──
        # 검색된 문서의 품질과 수를 기반으로 신뢰도를 계산한다
        confidence = self._calculate_confidence(docs, classification.confidence)

        # ── 5단계: LLM 답변 생성 ──
        try:
            response = self.llm.invoke(
                self.prompt.format_messages(
                    context=context,
                    chat_history=session_history,
                    question=question
                )
            )
            answer = response.content
        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
            answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

        # ── 6단계: 대화 기록 저장 (멀티턴 대화를 위해) ──
        # session_history는 호출자(Streamlit)가 st.session_state에서 관리한다.
        session_history.append(HumanMessage(content=question))
        session_history.append(AIMessage(content=answer))

        # 대화 기록이 너무 길어지면 오래된 것부터 제거
        # (LLM의 컨텍스트 윈도우 한도 대비)
        if len(session_history) > 20:
            del session_history[:-20]

        return AgentResponse(
            answer=answer,
            sources=sources,
            intent=classification.intent.value,
            confidence=confidence,
            searched_sources=classification.recommended_sources
        )

    def _smart_retrieve(
        self,
        question: str,
        sources: list | None,
        k: int = 6
    ) -> list:
        """
        소스 필터링을 적용한 스마트 검색.

        sources가 None이면 전체 검색,
        특정 소스 목록이 주어지면 해당 소스만 검색한다.
        """
        # 여유분을 두고 검색 후 필터링
        fetch_k = k * 3 if sources else k

        # ⚠️ 프로덕션 권장: Hybrid Search (BM25 + Vector)
        # 현재는 벡터 검색만 사용하지만, 프로덕션에서는 BM25(키워드) + 벡터(의미)를
        # 결합한 하이브리드 검색을 강력히 권장한다.
        # 특히 "PROJ-1234" 같은 Jira 이슈 키를 정확히 찾으려면
        # BM25 키워드 검색이 벡터 검색보다 훨씬 정확하다.
        # OpenSearch는 기본적으로 하이브리드 검색을 지원한다:
        #   vectorstore.similarity_search(query, search_type="hybrid", ...)
        docs = self.vectorstore.similarity_search(
            question,
            k=fetch_k
        )

        if sources:
            docs = [d for d in docs if d.metadata.get("source") in sources]

        return docs[:k]

    def _build_context(self, docs: list) -> tuple[str, list[dict]]:
        """
        검색된 문서에서 LLM에 넘길 컨텍스트와 출처 목록을 생성한다.
        """
        context_parts = []
        sources = []

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            url = doc.metadata.get("url", "")
            title = self._get_doc_title(doc)

            context_parts.append(
                f"[{i}] 출처: {title}\nURL: {url}\n\n{doc.page_content}"
            )

            sources.append({
                "index": i,
                "title": title,
                "source": source,
                "url": url,
                "snippet": doc.page_content[:200] + "..."
            })

        return "\n\n---\n\n".join(context_parts), sources

    def _get_doc_title(self, doc) -> str:
        """문서 메타데이터에서 제목을 추출한다"""
        source = doc.metadata.get("source", "")
        if source == "jira":
            return f"Jira {doc.metadata.get('issue_key', '')}"
        elif source == "github":
            t = doc.metadata.get("type", "issue")
            n = doc.metadata.get("number", "")
            repo = doc.metadata.get("repo", "")
            return f"GitHub {t} #{n} ({repo})"
        elif source == "confluence":
            return f"Confluence: {doc.metadata.get('title', '')}"
        return "알 수 없는 소스"

    def _calculate_confidence(self, docs: list, intent_confidence: float) -> float:
        """
        전체 응답 신뢰도를 계산한다.

        신뢰도 계산 공식:
        - 관련 문서 수가 많을수록 높음 (최대 가중치 40%)
        - 의도 분류 신뢰도 (가중치 30%)
        - 검색 결과의 다양성 (가중치 30%)
        """
        if not docs:
            return 0.1

        # 문서 수 점수 (0~1)
        doc_score = min(len(docs) / 5.0, 1.0)

        # 소스 다양성 점수 (여러 소스에서 결과가 나올수록 신뢰도 높음)
        unique_sources = len(set(d.metadata.get("source") for d in docs))
        diversity_score = min(unique_sources / 3.0, 1.0)

        # 종합 신뢰도
        confidence = (
            doc_score * 0.4 +
            intent_confidence * 0.3 +
            diversity_score * 0.3
        )

        return round(confidence, 2)

    def reset(self, session_history: list | None = None):
        """대화 기록을 초기화한다. session_history 리스트를 직접 비운다."""
        if session_history is not None:
            session_history.clear()
        logger.info("대화 기록 초기화")
```

!!! tip "프로덕션 팁: Structured Output"
    실제 프로덕션에서는 LLM의 답변을 자유 텍스트가 아닌 구조화된 형식으로 받는 것이 좋습니다:
    ```python
    from pydantic import BaseModel

    class RAGResponse(BaseModel):
        answer: str
        citations: list[str]
        confidence: float
        needs_clarification: bool

    structured_llm = llm.with_structured_output(RAGResponse)
    ```

!!! info "2025년 권장: Agentic RAG"
    위 구현은 규칙 기반 소스 선택을 사용합니다. 프로덕션에서는
    **LangGraph** 기반 Agentic RAG 패턴으로 LLM이 직접 검색 도구를
    선택하고, 검색 결과를 평가하여 필요시 재검색하는 방식이 권장됩니다.
    이를 통해 복합 질문에 대한 답변 품질이 크게 향상됩니다.

!!! tip "프로덕션 팁: 임베딩 캐싱 (Redis)"
    동일한 질문이 반복될 때마다 OpenAI 임베딩 API를 호출하면 비용이 낭비됩니다.
    Redis를 활용해 임베딩 결과를 캐싱하면 API 호출을 크게 줄일 수 있습니다:
    ```python
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain.storage import RedisStore

    store = RedisStore(redis_url="redis://localhost:6379")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
        document_embedding_cache=store,
        namespace="enterprise-rag-embeddings"
    )
    # 동일 텍스트는 Redis에서 즉시 반환, API 호출 없음
    ```

---

## Step 5: 웹 UI — 완전한 Streamlit 앱

!!! info "용어 설명: Streamlit"
    Python 코드만으로 웹 앱을 만들 수 있는 프레임워크다. HTML/CSS/JavaScript를 몰라도 데이터 앱, 챗봇 UI 등을 빠르게 만들 수 있다. `pip install streamlit` 후 `streamlit run app.py`로 실행한다.

```python
# ui/app.py

import streamlit as st
import os
from datetime import datetime
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_openai import OpenAIEmbeddings

# 에이전트 임포트
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.agent import EnterpriseRAGAgent

# ─────────────────────────────────
# 페이지 설정
# ─────────────────────────────────
st.set_page_config(
    page_title="업무 AI 검색",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────
# CSS 스타일 (출처 배지 등)
# ─────────────────────────────────
st.markdown("""
<style>
.source-badge-jira {
    background-color: #0052CC;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    margin-right: 4px;
}
.source-badge-github {
    background-color: #24292e;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    margin-right: 4px;
}
.source-badge-confluence {
    background-color: #0747A6;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    margin-right: 4px;
}
.confidence-bar {
    height: 4px;
    border-radius: 2px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────
# 벡터스토어 & 에이전트 초기화
# ─────────────────────────────────
@st.cache_resource  # 앱이 재실행돼도 한 번만 초기화
def load_agent():
    """에이전트를 초기화한다. @st.cache_resource로 한 번만 실행됨."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorstore = OpenSearchVectorSearch(
        opensearch_url=os.getenv("OPENSEARCH_URL", "http://localhost:9200"),
        index_name=os.getenv("OPENSEARCH_INDEX", "enterprise-docs"),
        embedding_function=embeddings
    )
    return EnterpriseRAGAgent(vectorstore)

# ─────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────
# st.session_state: 사용자가 페이지를 새로고침해도 유지되는 상태 저장소
if "agent" not in st.session_state:
    st.session_state.agent = load_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ 세션 격리: 각 사용자(브라우저 탭)마다 별도의 chat_history를 유지한다.
# @st.cache_resource로 공유되는 agent 인스턴스와 달리,
# st.session_state는 사용자별로 분리되어 대화가 섞이지 않는다.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # LangChain HumanMessage/AIMessage 리스트

if "feedback" not in st.session_state:
    st.session_state.feedback = {}  # {message_index: "up" or "down"}

# ─────────────────────────────────
# 사이드바
# ─────────────────────────────────
with st.sidebar:
    st.title("⚙️ 검색 설정")
    st.divider()

    # 소스 필터
    st.subheader("검색 소스")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_jira = st.checkbox("Jira", value=True)
    with col2:
        use_github = st.checkbox("GitHub", value=True)
    with col3:
        use_confluence = st.checkbox("Confluence", value=True)

    st.divider()

    # 예시 질문들
    st.subheader("질문 예시")
    example_questions = [
        "결제 시스템 관련 최근 장애 이력 알려줘",
        "인증 모듈에서 가장 많이 작업한 사람은?",
        "신규 개발자 온보딩 가이드 찾아줘",
        "마이크로서비스 전환 관련 결정 사항들",
        "현재 가장 우선순위 높은 버그는?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    # 대화 초기화
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.session_state.agent.reset(session_history=st.session_state.chat_history)
        st.success("대화 기록이 초기화됐습니다.")
        st.rerun()

    st.divider()
    st.caption("🔄 마지막 동기화: 오늘")

# ─────────────────────────────────
# 메인 화면
# ─────────────────────────────────
st.title("🔍 업무 AI 검색 에이전트")
st.caption("Jira · GitHub · Confluence 통합 검색 | 자연어로 무엇이든 물어보세요")

# ─────────────────────────────────
# 대화 기록 표시
# ─────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 에이전트 응답이면 출처 및 피드백 표시
        if msg["role"] == "assistant" and "sources" in msg:
            # 신뢰도 표시
            confidence = msg.get("confidence", 0)
            confidence_color = (
                "green" if confidence > 0.7
                else "orange" if confidence > 0.4
                else "red"
            )
            st.progress(confidence, text=f"신뢰도: {confidence:.0%}")

            # 출처 배지 & 링크
            if msg["sources"]:
                with st.expander(f"📎 참조 출처 {len(msg['sources'])}개 보기"):
                    for src in msg["sources"]:
                        badge_class = f"source-badge-{src['source']}"
                        badge_label = {
                            "jira": "Jira",
                            "github": "GitHub",
                            "confluence": "Confluence"
                        }.get(src["source"], src["source"])

                        st.markdown(
                            f'<span class="{badge_class}">{badge_label}</span> '
                            f'**[{src["title"]}]({src["url"]})**',
                            unsafe_allow_html=True
                        )
                        st.caption(src["snippet"])
                        st.divider()

            # 피드백 버튼 (엄지 위/아래)
            feedback_key = f"feedback_{idx}"
            current_feedback = st.session_state.feedback.get(idx)

            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button(
                    "👍" if current_feedback != "up" else "✅",
                    key=f"up_{idx}",
                    help="도움이 됐어요"
                ):
                    st.session_state.feedback[idx] = "up"
                    st.toast("피드백 감사합니다! 👍")
            with col2:
                if st.button(
                    "👎" if current_feedback != "down" else "✅",
                    key=f"down_{idx}",
                    help="도움이 안 됐어요"
                ):
                    st.session_state.feedback[idx] = "down"
                    st.toast("피드백 감사합니다. 개선하겠습니다! 🙏")

# ─────────────────────────────────
# 사용자 입력 처리
# ─────────────────────────────────
def process_question(question: str):
    """질문을 처리하고 응답을 생성한다"""
    # 사용자 메시지 추가
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # 응답 생성 (사용자별 세션 history 전달 → 멀티유저 격리)
    with st.chat_message("assistant"):
        with st.spinner("🔍 검색 중..."):
            response = st.session_state.agent.chat(
                question,
                session_history=st.session_state.chat_history
            )

        st.markdown(response.answer)

        # 신뢰도 바
        st.progress(response.confidence, text=f"신뢰도: {response.confidence:.0%}")

        # 출처 표시
        if response.sources:
            with st.expander(f"📎 참조 출처 {len(response.sources)}개 보기"):
                for src in response.sources:
                    badge_class = f"source-badge-{src['source']}"
                    badge_label = {
                        "jira": "Jira",
                        "github": "GitHub",
                        "confluence": "Confluence"
                    }.get(src["source"], src["source"])
                    st.markdown(
                        f'<span class="{badge_class}">{badge_label}</span> '
                        f'**[{src["title"]}]({src["url"]})**',
                        unsafe_allow_html=True
                    )
                    st.caption(src["snippet"])
                    st.divider()

    # 메시지 기록에 추가
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "intent": response.intent
    })

# 예시 질문 클릭 처리
if hasattr(st.session_state, "pending_question"):
    q = st.session_state.pending_question
    del st.session_state.pending_question
    process_question(q)
    st.rerun()

# 채팅 입력
if prompt := st.chat_input("무엇이든 물어보세요... (예: 결제 장애 이력, 인증 담당자, 배포 가이드)"):
    with st.chat_message("user"):
        st.markdown(prompt)
    process_question(prompt)
    st.rerun()
```

---

## Step 6: 주기적 동기화 (증분 업데이트)

```python
# sync.py

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

from collectors.jira_collector import JiraCollector
from collectors.github_collector import GitHubCollector
from collectors.confluence_collector import ConfluenceCollector
from core.indexer import EnterpriseIndexer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sync")

# 마지막 동기화 시각을 저장하는 파일
SYNC_STATE_FILE = Path(".sync_state.json")


def load_sync_state() -> dict:
    """마지막 동기화 상태를 읽어온다"""
    if SYNC_STATE_FILE.exists():
        with open(SYNC_STATE_FILE) as f:
            return json.load(f)
    return {}


def save_sync_state(state: dict):
    """동기화 상태를 저장한다"""
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def sync_all(full_sync: bool = False):
    """
    모든 소스에서 데이터를 수집하고 인덱싱한다.

    Args:
        full_sync: True면 전체 재인덱싱,
                   False면 마지막 동기화 이후 변경된 것만 업데이트 (증분 동기화)
    """
    start_time = datetime.now(timezone.utc)
    logger.info(f"동기화 시작: {'전체' if full_sync else '증분'} 모드")

    # 이전 동기화 상태 로드
    state = load_sync_state()
    last_sync = None

    if not full_sync and "last_sync" in state:
        last_sync = datetime.fromisoformat(state["last_sync"])
        logger.info(f"마지막 동기화: {last_sync}")
    else:
        logger.info("전체 동기화 모드 (이전 상태 없음)")

    # ── 수집기 초기화 ──
    jira = JiraCollector(
        server=os.getenv("ATLASSIAN_URL"),
        email=os.getenv("ATLASSIAN_EMAIL"),
        api_token=os.getenv("ATLASSIAN_API_TOKEN")
    )
    github = GitHubCollector(token=os.getenv("GITHUB_TOKEN"))
    confluence = ConfluenceCollector(
        url=os.getenv("ATLASSIAN_URL"),
        username=os.getenv("ATLASSIAN_EMAIL"),
        api_token=os.getenv("ATLASSIAN_API_TOKEN")
    )

    # ── 데이터 수집 ──
    all_jira_docs = []
    for project_key in os.getenv("JIRA_PROJECT_KEYS", "").split(","):
        if project_key.strip():
            try:
                docs = jira.collect(
                    project_key=project_key.strip(),
                    max_results=1000,
                    since=last_sync
                )
                all_jira_docs.extend(docs)
            except Exception as e:
                logger.error(f"Jira 수집 실패 (project={project_key.strip()}): {e}")
                # 한 프로젝트 실패해도 나머지는 계속 진행

    all_github_docs = []
    for repo in os.getenv("GITHUB_REPOS", "").split(","):
        if repo.strip():
            try:
                docs = github.collect(
                    repo_name=repo.strip(),
                    max_items=500,
                    since=last_sync
                )
                all_github_docs.extend(docs)
            except Exception as e:
                logger.error(f"GitHub 수집 실패 (repo={repo.strip()}): {e}")

    all_confluence_docs = []
    for space_key in os.getenv("CONFLUENCE_SPACE_KEYS", "").split(","):
        if space_key.strip():
            try:
                docs = confluence.collect(
                    space_key=space_key.strip(),
                    limit=500,
                    since=last_sync
                )
                all_confluence_docs.extend(docs)
            except Exception as e:
                logger.error(f"Confluence 수집 실패 (space={space_key.strip()}): {e}")

    # ── 인덱싱 ──
    indexer = EnterpriseIndexer(
        opensearch_url=os.getenv("OPENSEARCH_URL", "http://localhost:9200"),
        index_name=os.getenv("OPENSEARCH_INDEX", "enterprise-docs"),
        batch_size=100
    )

    indexer.index_all(
        jira_docs=all_jira_docs,
        github_docs=all_github_docs,
        confluence_docs=all_confluence_docs,
        incremental=not full_sync
    )

    # ── 동기화 상태 저장 ──
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    state = {
        "last_sync": start_time.isoformat(),
        "duration_seconds": duration,
        "counts": {
            "jira": len(all_jira_docs),
            "github": len(all_github_docs),
            "confluence": len(all_confluence_docs),
            "total": len(all_jira_docs) + len(all_github_docs) + len(all_confluence_docs)
        }
    }
    save_sync_state(state)
    logger.info(f"동기화 완료: {state['counts']['total']}개 문서, {duration:.1f}초 소요")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="전체 재인덱싱")
    parser.add_argument("--once", action="store_true", help="한 번만 실행 (스케줄러 없이)")
    args = parser.parse_args()

    if args.once:
        # 한 번만 실행
        sync_all(full_sync=args.full)
    else:
        # 스케줄러: 매 6시간마다 증분 동기화, 매주 일요일 새벽 2시에 전체 동기화
        scheduler = BlockingScheduler()

        # 증분 동기화: 6시간마다
        scheduler.add_job(sync_all, "interval", hours=6, args=[False])

        # 전체 동기화: 매주 일요일 새벽 2시
        scheduler.add_job(
            sync_all,
            "cron",
            day_of_week="sun",
            hour=2,
            args=[True]
        )

        logger.info("동기화 스케줄러 시작 (6시간 간격 증분, 매주 일요일 전체)")
        scheduler.start()
```

---

## Step 7: Docker Compose로 인프라 구성

```yaml
# docker-compose.yml

version: "3.8"

services:
  # ── OpenSearch: 벡터 DB ──
  opensearch:
    image: opensearchproject/opensearch:2.11.0
    container_name: enterprise-rag-opensearch
    environment:
      - discovery.type=single-node          # 단일 노드 모드 (개발용)
      - DISABLE_SECURITY_PLUGIN=true        # 보안 플러그인 비활성화 (개발용)
      - OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g # JVM 메모리 설정
    ports:
      - "9200:9200"   # REST API 포트
      - "9600:9600"   # 성능 분석 포트
    volumes:
      - opensearch-data:/usr/share/opensearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -s http://localhost:9200/_cluster/health | grep -q green"]
      interval: 30s
      timeout: 10s
      retries: 5

  # ── OpenSearch Dashboards: 데이터 시각화 (선택사항) ──
  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:2.11.0
    container_name: enterprise-rag-dashboards
    ports:
      - "5601:5601"
    environment:
      - OPENSEARCH_HOSTS=["http://opensearch:9200"]
      - DISABLE_SECURITY_DASHBOARDS_PLUGIN=true
    depends_on:
      - opensearch

  # ── Streamlit 웹 앱 ──
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: enterprise-rag-app
    ports:
      - "8501:8501"
    environment:
      - OPENSEARCH_URL=http://opensearch:9200
    env_file:
      - .env
    depends_on:
      opensearch:
        condition: service_healthy
    restart: unless-stopped

  # ── 동기화 스케줄러 ──
  sync:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: enterprise-rag-sync
    command: ["python", "sync.py"]
    env_file:
      - .env
    environment:
      - OPENSEARCH_URL=http://opensearch:9200
    depends_on:
      opensearch:
        condition: service_healthy
    restart: unless-stopped

volumes:
  opensearch-data:
    driver: local
```

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY . .

# Streamlit 실행 명령
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

### requirements.txt

```
# LangChain 핵심
langchain>=0.3
langchain-openai>=0.2
langchain-community>=0.3

# 벡터 DB
opensearch-py>=2.4

# 임베딩
openai>=1.40

# 데이터 수집
jira>=3.8
PyGithub>=2.1
atlassian-python-api>=3.41

# HTML 파싱
beautifulsoup4>=4.12

# 웹 UI
streamlit>=1.38

# 스케줄러
apscheduler>=3.10

# 환경 변수
python-dotenv>=1.0

# 로깅/유틸
tqdm>=4.66
```

---

## 실행 방법

### 1단계: 첫 실행 (전체 인덱싱)

```bash
# 환경 변수 설정
cp .env.example .env
# .env 파일을 열어서 실제 값 입력

# Docker로 인프라 시작
docker compose up -d opensearch

# OpenSearch 준비 대기 (약 30초)
sleep 30

# 처음 한 번만 전체 인덱싱 실행
python sync.py --once --full
# 예상 시간: 소스 규모에 따라 10분~1시간
```

### 2단계: 웹 앱 실행

```bash
# 방법 1: 로컬 실행
streamlit run ui/app.py

# 방법 2: Docker 전체 스택 실행
docker compose up -d
```

### 3단계: 접속

| 서비스 | URL |
|--------|-----|
| **웹 앱** | http://localhost:8501 |
| **OpenSearch** | http://localhost:9200 |
| **OpenSearch Dashboards** | http://localhost:5601 |

---

## 생산성 향상 효과

| 기존 방식 | AI 검색 에이전트 |
|-----------|----------------|
| Jira, GitHub, Confluence 각각 검색 | **한 곳에서 대화로 통합 검색** |
| 키워드를 정확히 기억해야 검색 가능 | **자연어로 의미 검색** |
| 검색 결과 직접 읽고 종합 | **AI가 요약 + 출처 인용** |
| 장애 이력 수동 추적 | **"결제 장애 이력" 한마디로 조회** |
| 신규 입사자 온보딩 수일 소요 | **AI에게 물어보며 빠르게 파악** |
| 담당자 찾기 위해 여러 사람에게 문의 | **AI가 이력 기반으로 즉시 추천** |

---

## 비용 추정 가이드

!!! info "왜 이게 중요한가?"
    API 비용을 미리 예측하지 않으면 월말에 예상치 못한 청구서를 받을 수 있다. 규모에 따라 비용을 미리 계산해보자.

### 초기 인덱싱 비용 (1회)

| 항목 | 규모 예시 | 토큰 수 | 비용 (USD) |
|------|----------|---------|-----------|
| Jira 이슈 1,000개 | 이슈당 ~500토큰 | 500K | ~$0.05 |
| GitHub 이슈/PR 500개 | 항목당 ~800토큰 | 400K | ~$0.04 |
| Confluence 페이지 300개 | 페이지당 ~1,000토큰 | 300K | ~$0.03 |
| **합계 (초기 인덱싱)** | | ~1.2M | **~$0.12** |

> `text-embedding-3-large` 기준: $0.00013/1K tokens

### 월간 운영 비용

| 항목 | 예시 | 비용/월 |
|------|------|---------|
| 증분 동기화 | 하루 100개 신규/변경 문서 × 30일 | ~$0.12 |
| 사용자 질문 | 하루 50개 질문 × GPT-4o | ~$15 |
| OpenSearch 인프라 (AWS) | t3.medium 인스턴스 | ~$30 |
| **예상 총 월간 비용** | | **~$45/월** |

!!! tip "비용 절약 팁"
    - 임베딩은 `text-embedding-3-small`($0.00002/1K)을 사용하면 6배 저렴 (품질 약간 저하)
    - 질문 답변은 `gpt-4o-mini`를 사용하면 15배 저렴
    - OpenSearch는 자체 서버에 구축하면 인프라 비용 절약 가능

---

## 보안 고려사항

!!! danger "반드시 지켜야 할 보안 규칙"

### 1. API 토큰 보호

```python
# 나쁜 예 ❌ - 코드에 직접 토큰 입력
jira = JiraCollector(
    server="https://company.atlassian.net",
    email="admin@company.com",
    api_token="ATATT3..."  # 절대 하드코딩 금지!
)

# 좋은 예 ✅ - 환경 변수 사용
import os
jira = JiraCollector(
    server=os.getenv("ATLASSIAN_URL"),
    email=os.getenv("ATLASSIAN_EMAIL"),
    api_token=os.getenv("ATLASSIAN_API_TOKEN")
)
```

### 2. 민감 정보 필터링

인덱싱 전에 민감한 정보를 제거하는 필터를 적용해야 한다.

```python
import re

def filter_sensitive_data(text: str) -> str:
    """
    텍스트에서 민감한 정보를 마스킹한다.
    인덱싱 전 반드시 적용해야 한다.
    """
    # 신용카드 번호 마스킹
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                  '[카드번호 마스킹]', text)

    # 주민등록번호 마스킹
    text = re.sub(r'\b\d{6}[-]\d{7}\b', '[주민번호 마스킹]', text)

    # API 키 형태 문자열 마스킹
    text = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[토큰 마스킹]', text)

    # 이메일 (옵션: 보존 또는 마스킹)
    # text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    #               '[이메일 마스킹]', text)

    return text
```

### 3. 접근 권한 관리

```python
# 사용자별 접근 가능한 소스 제한
ROLE_PERMISSIONS = {
    "developer": ["jira", "github", "confluence"],
    "intern": ["confluence"],         # 인턴은 Confluence만 접근
    "pm": ["jira", "confluence"],     # PM은 GitHub 코드 제외
    "readonly": ["confluence"],
}

def get_allowed_sources(user_role: str) -> list[str]:
    """사용자 역할에 따라 허용된 소스 목록을 반환한다"""
    return ROLE_PERMISSIONS.get(user_role, ["confluence"])
```

### 4. OpenSearch 프로덕션 보안 설정

```yaml
# docker-compose.prod.yml (프로덕션용)
services:
  opensearch:
    environment:
      # 개발 환경과 달리 보안 플러그인 활성화
      - DISABLE_SECURITY_PLUGIN=false
      - plugins.security.ssl.transport.pemcert_filepath=esnode.pem
      - plugins.security.ssl.http.enabled=true
      - OPENSEARCH_INITIAL_ADMIN_PASSWORD=${OPENSEARCH_ADMIN_PASSWORD}
```

---

## Step 6: 평가 & 모니터링

### RAGAS 기반 품질 평가

```python
# pip install ragas
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate

# 테스트 데이터셋으로 평가
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
print(result)
```

### LangSmith 연동 (Observability)

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "enterprise-rag"

# 이후 모든 LangChain 호출이 자동으로 LangSmith에 기록됨
# → 프롬프트, 검색 결과, 응답, 지연시간, 토큰 사용량 추적 가능
```

---

## 자주 묻는 질문 (FAQ)

??? question "Q: OpenSearch 대신 다른 벡터 DB를 사용할 수 있나요?"
    네, LangChain이 지원하는 모든 벡터 스토어로 교체 가능합니다.

    ```python
    # Pinecone 사용
    from langchain_pinecone import PineconeVectorStore
    vectorstore = PineconeVectorStore.from_documents(...)

    # Chroma 사용 (로컬, 무료)
    from langchain_chroma import Chroma
    vectorstore = Chroma.from_documents(...)

    # Weaviate 사용
    from langchain_weaviate import WeaviateVectorStore
    vectorstore = WeaviateVectorStore.from_documents(...)
    ```

    개발 환경에서는 Chroma(로컬 무료), 프로덕션에서는 OpenSearch나 Pinecone을 추천합니다.

??? question "Q: Jira나 GitHub가 없어도 되나요?"
    네, 있는 소스만 사용하면 됩니다. 예를 들어 Confluence만 있다면 ConfluenceCollector만 사용하고 나머지는 빈 리스트를 전달하면 됩니다.

    ```python
    indexer.index_all(
        jira_docs=[],      # Jira 없으면 빈 리스트
        github_docs=[],    # GitHub 없으면 빈 리스트
        confluence_docs=confluence_docs
    )
    ```

??? question "Q: GPT-4o 대신 다른 LLM을 사용할 수 있나요?"
    네, LangChain은 다양한 LLM을 지원합니다.

    ```python
    # Claude 사용
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

    # 로컬 Ollama 사용 (무료)
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.1")

    # Azure OpenAI 사용
    from langchain_openai import AzureChatOpenAI
    llm = AzureChatOpenAI(azure_deployment="gpt-4o")
    ```

??? question "Q: 인덱싱이 너무 오래 걸립니다. 어떻게 빠르게 할 수 있나요?"
    1. **병렬 처리**: 각 수집기를 `concurrent.futures.ThreadPoolExecutor`로 병렬 실행
    2. **배치 크기 조정**: `batch_size`를 200으로 늘려 임베딩 API 호출 횟수 감소
    3. **증분 동기화**: 첫 전체 인덱싱 후에는 변경된 것만 업데이트 (`since` 파라미터 사용)
    4. **로컬 임베딩**: `sentence-transformers`로 로컬 임베딩 사용 (API 비용 제로)

    ```python
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # 무료 로컬 임베딩 (품질은 OpenAI보다 약간 낮음)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    ```

??? question "Q: 한국어 검색 품질이 낮습니다."
    한국어 특화 임베딩 모델을 사용하면 개선됩니다.

    ```python
    # 한국어 특화 임베딩
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="snunlp/KR-ELECTRA-discriminator"
        # 또는: "jhgan/ko-sroberta-multitask"
    )
    ```

??? question "Q: 답변에 환각(Hallucination)이 있습니다."
    1. **시스템 프롬프트 강화**: "검색 결과에 없는 내용은 절대 추측하지 말 것" 명시
    2. **온도(temperature) 낮추기**: `temperature=0`으로 설정 (결정론적 응답)
    3. **청크 품질 개선**: 청크 크기를 줄이고 오버랩을 늘려 문맥 연결 강화
    4. **Reranking 추가**: Cohere Rerank API나 BGE-Reranker로 검색 정밀도 향상

---

## 직접 해보기 — 간소화 스타터 프로젝트

실제 Jira/GitHub/Confluence API 없이도 이 프로젝트의 핵심 개념을 실습할 수 있다.

### 샘플 데이터로 시작하기

```python
# starter_project.py
# 실행: python starter_project.py

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma  # 로컬 벡터 DB (설치 불필요)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import os

# ── 샘플 데이터 (실제 API 없이 테스트용) ──
SAMPLE_DOCUMENTS = [
    Document(
        page_content="""[PROJ-1234] 결제 페이지 로딩 무한 반복 버그
상태: Resolved
담당자: 김철수
우선순위: Critical

사용자가 결제 완료 후 로딩 화면이 무한으로 돌아가는 버그 발생.
원인: payment_callback 함수에서 redirect URL이 잘못 설정됨.
해결: `return_url` 파라미터를 `/payment/success`로 수정.
배포일: 2024-12-15""",
        metadata={"source": "jira", "issue_key": "PROJ-1234",
                  "url": "https://example.atlassian.net/browse/PROJ-1234"}
    ),
    Document(
        page_content="""[PR #892] Fix: payment callback redirect URL
작성자: chulsoo.kim
상태: merged

결제 콜백 URL 버그 수정 PR.
변경 파일: payment/views.py

diff:
- return_url = "/home"
+ return_url = "/payment/success"

리뷰어 코멘트:
[reviewer] 이 변경으로 PROJ-1234 이슈 해결됨. LGTM!""",
        metadata={"source": "github", "type": "pull_request", "number": 892,
                  "url": "https://github.com/example/backend/pull/892"}
    ),
    Document(
        page_content="""결제 시스템 장애 대응 가이드
경로: 엔지니어링 > 운영 가이드 > 장애 대응

결제 시스템에서 장애 발생 시 대응 절차:
1. PagerDuty로 온콜 담당자에게 알림
2. payment-service 로그 확인: kubectl logs -f payment-service
3. Stripe 대시보드에서 결제 실패율 확인
4. 필요 시 결제 페이지를 점검 모드로 전환

과거 주요 장애:
- 2024-12-15: 결제 콜백 URL 오류 (PROJ-1234) → PR #892로 핫픽스
- 2024-10-01: Stripe API 레이트 리밋 초과 → 재시도 로직 추가""",
        metadata={"source": "confluence", "title": "결제 시스템 장애 대응 가이드",
                  "url": "https://example.atlassian.net/wiki/spaces/ENG/pages/123"}
    ),
]

# ── 인덱싱 ──
print("문서 인덱싱 중...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(SAMPLE_DOCUMENTS)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"인덱싱 완료: {len(chunks)}개 청크")

# ── 검색 + 답변 생성 ──
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 팀의 업무 지식을 검색하는 AI 어시스턴트입니다.
아래 컨텍스트를 바탕으로 답변하고, 반드시 출처를 명시하세요.

컨텍스트:
{context}"""),
    ("human", "{question}")
])

def ask(question: str) -> str:
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n---\n\n".join([
        f"[출처: {d.metadata.get('url', '알 수 없음')}]\n{d.page_content}"
        for d in docs
    ])
    response = llm.invoke(prompt.format_messages(
        context=context,
        question=question
    ))
    print(f"\n질문: {question}")
    print(f"답변: {response.content}")
    print(f"참조 소스: {[d.metadata.get('source') for d in docs]}")
    return response.content

# 테스트
ask("결제 페이지 버그 이력 알려줘")
ask("결제 장애 발생 시 어떻게 대응해야 해?")
ask("PR #892는 무슨 내용이야?")
```

실행 방법:
```bash
pip install langchain langchain-openai langchain-chroma chromadb
export OPENAI_API_KEY="your-key"
python starter_project.py
```

---

## 확장 아이디어

### 1. Slack 봇 연동

```python
# slack_bot.py — Slack에서 /ask 커맨드로 검색
from slack_bolt import App
from core.agent import EnterpriseRAGAgent

app = App(token=os.environ["SLACK_BOT_TOKEN"])

@app.command("/ask")
def handle_ask(ack, command, say):
    ack()
    question = command["text"]
    say(f"🔍 '{question}' 검색 중...")

    response = agent.chat(question)
    # 출처 링크 포함해서 Slack 메시지 전송
    say({
        "text": response.answer,
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": response.answer}},
            {"type": "divider"},
            {"type": "section", "text": {
                "type": "mrkdwn",
                "text": "*참조 출처:*\n" + "\n".join(
                    f"• <{s['url']}|{s['title']}>" for s in response.sources[:3]
                )
            }}
        ]
    })
```

### 2. 일간 이메일 다이제스트

```python
# digest.py — 매일 아침 팀에게 중요 이슈 요약 이메일 발송
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def generate_daily_digest():
    """오늘 주목할 이슈/변경사항 요약"""
    agent = EnterpriseRAGAgent(vectorstore)

    questions = [
        "오늘 새로 생성되거나 업데이트된 Critical/High 이슈 요약해줘",
        "이번 주 머지된 주요 PR 요약해줘",
        "현재 해결이 시급한 버그 Top 5"
    ]

    digest = []
    for q in questions:
        response = agent.chat(q)
        digest.append(f"## {q}\n\n{response.answer}\n")
        agent.reset()

    return "\n\n---\n\n".join(digest)
```

### 3. 자동 이슈 연결

```python
# linker.py — 새 Jira 이슈 생성 시 관련 과거 이슈/PR 자동으로 연결
def find_related_issues(new_issue_description: str) -> list[dict]:
    """
    새 이슈와 유사한 과거 이슈/PR을 찾아서 자동으로 링크한다.
    중복 이슈 생성을 방지하고 과거 해결책을 빠르게 재활용한다.
    """
    docs = vectorstore.similarity_search(new_issue_description, k=5)
    related = []
    for doc in docs:
        if doc.metadata.get("source") in ["jira", "github"]:
            related.append({
                "url": doc.metadata["url"],
                "title": doc.metadata.get("issue_key") or
                         f"PR #{doc.metadata.get('number')}",
                "snippet": doc.page_content[:200]
            })
    return related
```

### 4. 온콜 어시스턴트

```python
# oncall_assistant.py — 장애 발생 시 과거 유사 장애 이력과 해결책 즉시 제공
def handle_incident(incident_description: str) -> str:
    """
    장애 발생 시 자동으로 관련 이력을 검색하고
    대응 가이드를 제시한다.
    """
    response = agent.chat(
        f"다음 장애와 유사한 과거 이력과 해결 방법을 찾아줘: {incident_description}"
    )
    return response.answer
```

---

## 핵심 요약

!!! abstract "핵심 요약"
    1. **데이터 수집**: Jira/GitHub/Confluence API로 문서를 가져올 때 페이지네이션, Rate Limit, 증분 동기화를 반드시 처리한다.
    2. **인덱싱 품질**: 중복 제거, 텍스트 정제, 컨텍스트 헤더 추가가 검색 품질을 크게 좌우한다.
    3. **의도 분류**: 질문의 종류를 파악해서 적합한 소스만 검색하면 정확도가 올라간다.
    4. **멀티턴 대화**: 이전 대화 기록을 유지하면 더 자연스러운 후속 질문이 가능하다.
    5. **출처 인용**: 모든 답변에 클릭 가능한 출처 링크를 포함시켜야 신뢰성이 높아진다.
    6. **보안 필수**: API 토큰은 환경 변수로, 민감 정보는 인덱싱 전에 마스킹, 접근 권한은 역할 기반으로 관리한다.
    7. **비용 관리**: 초기 전체 인덱싱 후에는 증분 동기화로 비용을 최소화한다.
