# 3. 문서 청킹 전략

## 왜 청킹이 중요한가?

청킹은 RAG 품질의 **70%를 결정**한다. 아무리 좋은 임베딩 모델과 LLM을 써도 청크가 잘못 나뉘면 검색 결과가 엉망이 된다.

```
나쁜 청킹: "...의 특징은 다음과 같다." | "1. 빠르다 2. 안정적이다..."
→ 질문과 답변이 서로 다른 청크에 분리됨!

좋은 청킹: "...의 특징은 다음과 같다. 1. 빠르다 2. 안정적이다..."
→ 질문과 답변이 하나의 청크에 포함됨
```

---

## 청킹 전략 비교

### 1. 고정 크기 청킹 (Fixed-size)

가장 단순. 일정 문자/토큰 수로 자른다.

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)
chunks = splitter.split_text(document)
```

| 장점 | 단점 |
|------|------|
| 구현 간단 | 의미 단위 무시 |
| 예측 가능한 크기 | 문맥 단절 발생 |

### 2. 재귀 분할 (Recursive) ⭐ 가장 많이 사용

여러 구분자를 우선순위로 적용. 의미 단위를 최대한 보존한다.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]  # 우선순위
)
chunks = splitter.split_documents(documents)
```

!!! tip "재귀 분할 작동 원리"
    1. 먼저 `\n\n`(단락)으로 나눈다
    2. 너무 크면 `\n`(줄바꿈)으로 나눈다
    3. 그래도 크면 `. `(문장)으로 나눈다
    4. 계속 작은 단위로 분할

### 3. 의미 기반 청킹 (Semantic)

임베딩 유사도로 의미가 바뀌는 지점을 감지하여 분할한다.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90  # 유사도 하위 10%에서 분할
)
chunks = splitter.split_text(document)
```

### 4. 문서 구조 기반 청킹

마크다운, HTML 등의 구조를 활용한다.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers = [
    ("#", "제목"),
    ("##", "소제목"),
    ("###", "세부제목"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)

# 각 청크에 헤더 메타데이터가 자동 포함됨
for chunk in chunks:
    print(f"메타: {chunk.metadata}, 내용: {chunk.page_content[:50]}...")
```

---

## 청크 크기 선정 가이드

| 용도 | 권장 크기 | 오버랩 | 이유 |
|------|-----------|--------|------|
| QA (짧은 답변) | 200-500 토큰 | 20-50 | 정확한 답변 추출 |
| 요약/분석 | 500-1000 토큰 | 50-100 | 넓은 문맥 필요 |
| 코드 문서 | 함수/클래스 단위 | 0 | 코드 구조 보존 |
| 법률/계약서 | 조항 단위 | 50-100 | 법적 의미 보존 |

---

## 🍯 청크 관리 & 문서화 꿀팁

### 팁 1: 메타데이터를 풍부하게

```python
from langchain.schema import Document

# 나쁜 예: 내용만 저장
bad_chunk = Document(page_content="RAG는 검색 기반 생성 기법이다")

# 좋은 예: 풍부한 메타데이터
good_chunk = Document(
    page_content="RAG는 검색 기반 생성 기법이다",
    metadata={
        "source": "rag_guide.pdf",
        "page": 3,
        "chapter": "1. RAG 개요",
        "doc_type": "technical_guide",
        "created_at": "2025-01-15",
        "chunk_index": 5,
        "total_chunks": 42
    }
)
```

### 팁 2: 부모-자식 청킹 (Parent-Child)

검색은 작은 청크로, LLM에는 큰 청크를 전달한다.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 부모 청크 (큰 단위 - LLM에 전달)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
parent_chunks = parent_splitter.split_documents(documents)

# 자식 청크 (작은 단위 - 검색용)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
for i, parent in enumerate(parent_chunks):
    children = child_splitter.split_documents([parent])
    for child in children:
        child.metadata["parent_id"] = i  # 부모 참조
```

### 팁 3: 청크에 컨텍스트 헤더 추가

```python
def add_context_header(chunks, document_title):
    """각 청크 앞에 문서 제목과 위치 정보를 추가"""
    for i, chunk in enumerate(chunks):
        header = f"[문서: {document_title} | 섹션 {i+1}/{len(chunks)}]\n\n"
        chunk.page_content = header + chunk.page_content
    return chunks

# "이 내용이 어디서 왔는지" LLM이 바로 파악 가능
```

### 팁 4: 청크 품질 검증

```python
def validate_chunks(chunks, min_length=50, max_length=2000):
    """청크 품질을 검증하고 통계를 출력"""
    issues = []
    lengths = [len(c.page_content) for c in chunks]

    for i, chunk in enumerate(chunks):
        if len(chunk.page_content) < min_length:
            issues.append(f"청크 {i}: 너무 짧음 ({len(chunk.page_content)}자)")
        if len(chunk.page_content) > max_length:
            issues.append(f"청크 {i}: 너무 김 ({len(chunk.page_content)}자)")
        if not chunk.metadata.get("source"):
            issues.append(f"청크 {i}: source 메타데이터 누락")

    print(f"총 {len(chunks)}개 청크")
    print(f"평균 길이: {sum(lengths)/len(lengths):.0f}자")
    print(f"최소: {min(lengths)}자, 최대: {max(lengths)}자")

    if issues:
        print(f"\n⚠️ {len(issues)}개 이슈 발견:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ 모든 청크 정상")
```

### 팁 5: 문서 전처리 체크리스트

```
✅ 불필요한 헤더/푸터 제거 (페이지 번호, 워터마크 등)
✅ 표를 텍스트로 변환하거나 별도 처리
✅ 이미지/차트의 캡션이나 설명 텍스트 추출
✅ 인코딩 통일 (UTF-8)
✅ 중복 문서 제거
✅ 메타데이터 스키마 사전 정의
✅ 청크 ID 체계 수립 (재인덱싱 시 중복 방지)
```

---

## 핵심 요약

!!! summary "이것만 기억하자"
    1. **RecursiveCharacterTextSplitter**가 범용적으로 가장 무난
    2. 청크 크기는 용도에 따라 200-1000 토큰 사이로 조절
    3. **메타데이터가 핵심** — source, page, chapter 등 반드시 포함
    4. 부모-자식 청킹으로 검색 정밀도와 컨텍스트를 동시에 확보
    5. 전처리 → 분할 → 검증 파이프라인을 반드시 구축
