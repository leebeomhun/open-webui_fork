KCD MCPO (MCP Server + LangGraph)

개요
- 의료 쿼리를 LangGraph 플로우(확장 → 분류 → 검색 → 집계)로 처리하고, MCP(Model Context Protocol) 서버로 노출합니다.
- open-webui의 mcpo를 통해 OpenAPI 서버로 프록시하여 쉽게 연동할 수 있습니다.
- 카테고리별 데이터 파일(`data/`):
  - `pathogen`: `data/kcd_kb_pathogen.txt`
  - `resistance`: `data/kcd_kb_resistance.txt`
  - `external`: `data/kcd_kb_external.txt`

설치
- Python 패키지 설치:
  - `pip install -r requirements.txt`
- 선택 사항(네트워크/키 필요):
  - OpenAI 사용 시 `OPENAI_API_KEY` 설정
  - SentenceTransformers 임베딩 사용 시 모델 다운로드 필요

환경변수 (.env)
- `DATA_DIR` (기본: `data`)
- `MODEL` (기본: `gpt-5-mini` — 환경에 맞게 조정)
- `OPENAI_API_KEY`, `OPENAI_BASE_URL` (선택)
- `EMBEDDING_MODEL` (기본: `dragonkue/multilingual-e5-small-ko-v2`)
- `MCPO_REQUIRE_LLM` / `MCPO_REQUIRE_EMBEDDINGS` (0/1) — 1이면 해당 의존성 없을 때 실패
- `VECTOR_STORE` (`chroma` | `qdrant`, 기본 `chroma`) — 벡터스토어 선택
- `CHROMA_PERSIST_DIR` (기본: `data/chroma`) — ChromaDB 저장 경로 (VECTOR_STORE=chroma 시)
- `QDRANT_URL`, `QDRANT_API_KEY` — Qdrant 접속 설정 (VECTOR_STORE=qdrant 시)
- `CHUNK_SIZE` (기본: 800) — 인덱싱 시 청크 크기(문자 단위)
- `CHUNK_OVERLAP` (기본: 150) — 인접 청크 오버랩(문자 단위)
- `CHROMA_REBUILD` (0/1) — 1이면 다음 실행에서 컬렉션 재생성
- `EMBEDDING_CACHE_SIZE` (기본: 256) — 동일/유사 질의 반복 시 쿼리 임베딩 LRU 캐시 크기
- `ENABLE_HYBRID` (0/1) — BM25 + 벡터 하이브리드 검색 활성화(기본 1)
- `HYBRID_ALPHA` (기본: 0.6) — 하이브리드 가중치(벡터 가중치; 1-알파는 BM25 가중치)

벡터 인덱싱 동작
- 앱 시작 시 `data/kcd_kb_*.txt`를 읽어 청크로 분할한 뒤, 카테고리별 컬렉션(`kcd_pathogen`, `kcd_resistance`, `kcd_external`)에 저장합니다.
- Chroma 사용 시 `CHROMA_PERSIST_DIR`에 퍼시스턴스가 남아 있으면 재사용하고, `CHROMA_REBUILD=1`이면 드롭 후 재빌드합니다.
- Qdrant 사용 시 `QDRANT_URL`로 접속하여 동일한 컬렉션 이름으로 upsert 합니다.
- 임베딩/벡터스토어 사용이 불가하면 자동으로 BM25/키워드 기반 검색으로 폴백합니다.

성능 최적화
- 서버 시작 시 컬렉션을 미리 로딩하려면 `MCPO_EAGER_INIT=1`을 권장합니다.
- 한 번의 요청에서 확장된 질의들이 여러 카테고리에 재사용될 때, 쿼리 임베딩을 1회만 계산하고(`query_embeddings`) 재사용합니다.
- 최근 질의 임베딩은 LRU 캐시(`EMBEDDING_CACHE_SIZE`)로 재사용하여 반복 질의 응답속도를 개선합니다.
- 하이브리드: 동일 청크 집합에 대해 BM25 인덱스를 구축하고, 쿼리 시 벡터(sim)와 BM25(norm)를 `HYBRID_ALPHA`로 가중 결합하여 순위를 산출합니다.

실행: mcpo 프록시 사용
1) mcpo 설치 (외부): `pip install mcpo` 또는 `uvx mcpo`
2) 이 저장소에서 MCP 서버 준비 완료 후:
   - `mcpo --config ./config.json`
3) 브라우저 접속:
   - `http://localhost:8000/kcd/docs` (OpenAPI 문서)

config.json
```
{
  "mcpServers": {
    "kcd": {
      "command": "python",
      "args": ["-m", "mcpo_kcd.mcp_server"]
    }
  }
}
```

구현 파일
- `mcpo_kcd/mcp_server.py`: MCP stdio 서버(FastMCP). `kcd_query(query, top_k)` 툴 제공.
- `mcpo_kcd/graph.py`: LangGraph 기반 실행 플로우.
- `mcpo_kcd/pipeline.py`: 폴백 파이프라인(그래프 미설치 시 사용).
- `mcpo_kcd/retriever.py`: 벡터스토어(ChromaDB 또는 Qdrant) 기반 임베딩 검색(청크/오버랩), 키워드 폴백 포함.
- `mcpo_kcd/llm.py`: LLM 기반 확장/분류(폴백 포함).
- `mcpo_kcd/utils.py`: 텍스트 유틸.

주의사항
- 네트워크/모델 설치가 불가한 환경에서도 키워드 기반 검색으로 동작하도록 폴백이 포함되어 있습니다.
- OpenWebUI 통합은 mcpo가 제공하는 OpenAPI 엔드포인트를 등록하면 됩니다.
