# RAG 핵심 가이드북

> **Retrieval Augmented Generation** — LLM의 한계를 극복하는 핵심 기술

---

## 이 교재에서 다루는 내용

| 챕터 | 주제 | 핵심 내용 |
|------|------|-----------|
| 1 | [RAG 개요](chapters/01-rag-overview.md) | RAG란 무엇인가, 왜 필요한가, 아키텍처 |
| 2 | [임베딩 모델](chapters/02-embedding-models.md) | 최신 임베딩 모델 비교 및 선택 가이드 |
| 3 | [문서 청킹 전략](chapters/03-chunking-strategies.md) | 청크 분할, 관리, 문서화 꿀팁 |
| 4 | [벡터 데이터베이스](chapters/04-vector-databases.md) | OpenSearch, pgvector, Meilisearch |
| 5 | [RAG 파이프라인 구현](chapters/05-rag-pipeline.md) | Python으로 E2E 파이프라인 구축 |
| 6 | [고급 RAG 기법](chapters/06-advanced-rag.md) | HyDE, Reranking, Query Decomposition |
| 7 | [실전 프로젝트](chapters/07-enterprise-rag-project.md) | Jira/GitHub/Confluence 통합 AI 검색 에이전트 |
| 8 | [Context7과 MCP 기반 RAG](chapters/08-context7-mcp-rag.md) | MCP 프로토콜, Documentation RAG, 자체 MCP 서버 구축 |

---

## 빠른 시작

```bash
# 의존성 설치
pip install langchain langchain-community langchain-openai \
    chromadb sentence-transformers opensearch-py \
    pgvector psycopg2-binary meilisearch

# OpenAI API 키 설정
export OPENAI_API_KEY="your-api-key"
```

## 대상 독자

- LLM 기반 애플리케이션을 만들고 싶은 개발자
- RAG의 핵심 개념을 빠르게 이해하고 싶은 분
- 벡터 DB 선택에 고민하는 분

---

!!! tip "학습 방법"
    각 챕터는 독립적으로 읽을 수 있지만, 1→8 순서로 읽으면 체계적으로 이해할 수 있습니다.
