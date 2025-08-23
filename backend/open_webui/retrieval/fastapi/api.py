import uvicorn
import asyncio
import logging
from fastapi import FastAPI, Body, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from cachetools import TTLCache

# agent.py에서 LangGraph 앱을 가져옵니다.
from agent import app as query_agent_app

# --- 로깅 설정 ---
# 루트 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Uvicorn 및 FastAPI 로거 가져오기
uvicorn_logger = logging.getLogger("uvicorn.access")
fastapi_logger = logging.getLogger("fastapi")

# 애플리케이션 로거 생성
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --- FastAPI 앱 및 캐시 설정 ---

api = FastAPI(
    title="의학용어 쿼리 처리 API",
    description="쿼리를 입력받아 의학 약어 확장 및 검색어 확장을 수행하는 API. 결과는 캐시됩니다.",
    version="2.2.0", # 로깅 기능 추가로 버전 업데이트
)

# 7일 동안 유지되는 TTL 캐시 (최대 10,000개 항목)
cache = TTLCache(maxsize=10000, ttl=7 * 24 * 3600)


# --- Pydantic 모델 정의 ---

class QueryRequest(BaseModel):
    query: str = Field(..., description="사용자가 입력한 원본 텍스트 쿼리")

class ProcessedQueryResponse(BaseModel):
    """API의 최종 응답 모델"""
    original_query: str
    query_type: Literal["KCD", "CANCER_REG", "UNKNOWN"]
    expanded_terms: List[str]


# --- 에이전트 호출 로직 ---

def invoke_agent_sync(original_query: str) -> dict:
    """
    동기적으로 LangGraph를 호출하는 함수.
    """
    logger.info(f"Cache Miss: Executing LangGraph for query: '{original_query}'")
    inputs = {"original_query": original_query}
    
    # agent.py의 app은 동기/비동기를 모두 지원하지만, 여기서는 동기 invoke를 사용합니다.
    final_state = query_agent_app.invoke(inputs)
    
    # 결과를 JSON 직렬화 가능한 dict로 반환합니다.
    result = {
        "original_query": final_state.get("original_query"),
        "query_type": final_state.get("query_type"),
        "expanded_terms": final_state.get("expanded_terms"),
    }
    # 성공적인 실행 후 결과를 캐시에 저장
    cache[original_query] = result
    return result


# --- API 엔드포인트 ---

@api.post("/process-query", response_model=ProcessedQueryResponse)
async def process_query(request: QueryRequest = Body(...)):
    """
    사용자 쿼리를 받아 의도를 분류하고 검색에 적합한 용어로 확장합니다.
    결과는 캐시되어 동일한 쿼리에 대한 반복 호출 시 빠른 응답을 제공합니다.
    """
    query = request.query
    logger.info(f"Received query: '{query}'")

    # 1. 캐시 확인
    if query in cache:
        logger.info(f"Cache Hit for query: '{query}'")
        return cache[query]

    # 2. 캐시 미스 시, 에이전트 실행
    # 캐시된 동기 함수를 이벤트 루프의 기본 실행기에서 실행하여 I/O 블로킹을 방지합니다.
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, invoke_agent_sync, query
    )
    return result

@api.get("/cache-stats", summary="Get Cache Statistics")
async def get_cache_stats():
    """현재 캐시의 통계 정보(현재 크기, 최대 크기, TTL)를 반환합니다."""
    stats = {
        "current_size": cache.currsize,
        "max_size": cache.maxsize,
        "ttl_seconds": cache.ttl,
    }
    logger.info(f"Cache stats requested: {stats}")
    return stats

@api.post("/clear-cache", summary="Clear Cache")
async def clear_cache():
    """API의 인메모리 캐시를 모두 지웁니다."""
    logger.info("Clearing query expansion cache.")
    cache.clear()
    message = "Query expansion cache cleared successfully."
    logger.info(message)
    return {"message": message}


# --- 서버 실행 ---

if __name__ == "__main__":
    # 서버 실행 (예: 8001번 포트 사용)
    # 명령어: uvicorn backend.open_webui.retrieval.fastapi.api:api --host 0.0.0.0 --port 8001 --reload
    uvicorn.run("api:api", host="0.0.0.0", port=8001, reload=True)