#25.5.21 upstage solar llm으로 교체
#25.5.27 기존 버전에 6.11에서 추가된 BM25 변수 추가
#25.5.29 gemini 2.5 flash 모델로 수정
#25.5.30 test
#25.5.30 0.6.13 업데이트내용 추가
#25.6.10 llm reranking에서 gemini api호출 오류발생시 openai api로 호출하도록 수정
#25.6.18 gemini model 쿼리확장, 리랭킹 gemini-2.5-flash model name 변경
#25.6.19 expand_medical_abbreviation 함수 수정 - 의학약어 처리 규칙 추가, 예외 처리 추가
#25.6.22 쿼리향상, 리랭킹 api 호출로 변경
#25.6.26 임베딩 생략을 통한 직접 파일 업로드 최적화, v0.6.16버전 업데이트대비 호환성 추가(context 대신 query_result 사용)
import httpx
import asyncio
import hashlib
import json
import logging
import os
import operator
import re
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from urllib.parse import quote
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from open_webui.config import VECTOR_DB
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT

from open_webui.models.users import UserModel
from open_webui.models.files import Files
from open_webui.models.knowledge import Knowledges
from open_webui.models.notes import Notes

from open_webui.retrieval.vector.main import GetResult
from open_webui.utils.access_control import has_access

from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    Callbacks,
)
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.retrievers import BaseRetriever

from open_webui.config import (
    VECTOR_DB,
    RAG_EMBEDDING_QUERY_PREFIX,
    RAG_EMBEDDING_CONTENT_PREFIX,
    RAG_EMBEDDING_PREFIX_FIELD_NAME,
)
from open_webui.env import (
    SRC_LOG_LEVELS,
    OFFLINE_MODE,
    ENABLE_FORWARD_USER_INFO_HEADERS,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

load_dotenv()
QUERY_EXPANSION_API_URL = os.getenv("QUERY_EXPANSION_API_URL", "http://localhost:8001/process-query")
RERANK_API_URL = os.getenv("RERANK_API_URL", "http://localhost:8002/rerank")
GEMINI_API_KEY = os.getenv("GEMINIAPIKEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Configuration constants
TOP_K_PER_QUERY = int(os.getenv("TOP_K_PER_QUERY", "3"))
DEFAULT_BM25_WEIGHT = float(os.getenv("DEFAULT_BM25_WEIGHT", "0.3"))
DEFAULT_VECTOR_WEIGHT = float(os.getenv("DEFAULT_VECTOR_WEIGHT", "0.7"))
ASYNC_TIMEOUT_SECONDS = int(os.getenv("ASYNC_TIMEOUT_SECONDS", "30"))
MMR_DIVERSITY_THRESHOLD = float(os.getenv("MMR_DIVERSITY_THRESHOLD", "0.8"))
CANDIDATE_MULTIPLIER_LOW_K = 2.0
CANDIDATE_MULTIPLIER_HIGH_K = 1.5

# Cache configuration
ENABLE_CACHING = os.getenv("ENABLE_RAG_CACHING", "true").lower() == "true"
CACHE_TTL_MEDICAL_ABBREV = int(os.getenv("CACHE_TTL_MEDICAL_ABBREV", str(30 * 24 * 3600)))  # 30 days
CACHE_TTL_QUERY_ENHANCE = int(os.getenv("CACHE_TTL_QUERY_ENHANCE", str(7 * 24 * 3600))) # 7 days
CACHE_TTL_BM25 = int(os.getenv("CACHE_TTL_BM25", str(24 * 3600)))  # 2 hours
MAX_CACHE_SIZE = int(os.getenv("MAX_RAG_CACHE_SIZE", "10000"))
MAX_CANDIDATE_MULTIPLIER = float(os.getenv("MAX_CANDIDATE_MULTIPLIER", "3.0"))

# Simple in-memory cache with TTL support
class SimpleCache:
    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if not ENABLE_CACHING:
            return None
            
        if key in self.cache:
            value, expiry = self.cache[key]
            if datetime.now() < expiry:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        if not ENABLE_CACHING:
            return
            
        # Clean up expired entries if cache is getting full
        if len(self.cache) >= self.max_size:
            self._cleanup_expired()
            
        # If still full, remove oldest entries
        if len(self.cache) >= self.max_size:
            oldest_keys = sorted(self.cache.keys(), 
                               key=lambda k: self.cache[k][1])[:len(self.cache) // 4]
            for old_key in oldest_keys:
                del self.cache[old_key]
        
        expiry = datetime.now() + timedelta(seconds=ttl_seconds)
        self.cache[key] = (value, expiry)
    
    def _cleanup_expired(self) -> None:
        now = datetime.now()
        expired_keys = [k for k, (_, expiry) in self.cache.items() if now >= expiry]
        for key in expired_keys:
            del self.cache[key]
    
    def clear(self) -> None:
        self.cache.clear()
    
    def size(self) -> int:
        return len(self.cache)

# Global cache instances
medical_abbrev_cache = SimpleCache()
query_enhance_cache = SimpleCache()
bm25_retriever_cache = SimpleCache() # BM25 캐시 추가

def cache_key_hash(data: str) -> str:
    """Generate a consistent hash for cache keys"""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]

async def call_llm_api(prompt: str, system_prompt: str, api_key: str, use_gemini: bool = True) -> str:
    """Unified LLM API call function with fallback support"""
    if use_gemini and api_key:
        try:
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": GEMINI_MODEL,
                    "reasoning_effort": "none",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": API_TEMPERATURE
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning(f"Gemini API call failed: {type(e).__name__}, trying OpenAI fallback")
    
    # OpenAI fallback
    if OPENAI_API_KEY:
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": API_TEMPERATURE
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.error(f"OpenAI API call also failed: {type(e).__name__}")
            raise
    else:
        raise ValueError("No valid API key available for LLM calls")

def async_cached(cache: SimpleCache, ttl_seconds: int, key_prefix: str):
    """Decorator for caching async function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            cache_data = f"{key_prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = cache_key_hash(cache_data)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                log.debug(f"Cache hit for {func.__name__}: {cache_key}")
                return cached_result
            
            # Execute function and cache result
            log.debug(f"Cache miss for {func.__name__}: {cache_key}")
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl_seconds)
            
            return result
        return wrapper
    return decorator

def sync_cached(cache: SimpleCache, ttl_seconds: int, key_prefix: str):
    """Decorator for caching sync function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            cache_data = f"{key_prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = cache_key_hash(cache_data)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                log.debug(f"Cache hit for {func.__name__}: {cache_key}")
                return cached_result
            
            # Execute function and cache result
            log.debug(f"Cache miss for {func.__name__}: {cache_key}")
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl_seconds)
            
            return result
        return wrapper
    return decorator

# RAG 코드에서
@async_cached(query_enhance_cache, CACHE_TTL_QUERY_ENHANCE, "query_api")
async def call_query_expansion_api(query: str) -> Optional[List[str]]:
    """
    쿼리 확장 API 서버를 호출하여 확장된 쿼리 목록을 가져옵니다.
    """
    log.info(f"API 호출 시작: 쿼리='{query}', URL='{QUERY_EXPANSION_API_URL}'") # [디버깅 로그 추가 1]
    try:
        # 비동기 환경에서 requests를 사용하려면 run_in_executor를 사용하는 것이 좋습니다.
        # 또는 httpx와 같은 비동기 HTTP 클라이언트를 사용해야 합니다.
        # 현재 코드에서는 requests를 직접 호출하고 있으므로, 블로킹 호출이 될 수 있습니다.
        # 먼저 httpx를 설치합니다: uv pip install httpx
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                QUERY_EXPANSION_API_URL,
                json={"query": query},
                timeout=10
            )
        
        log.info(f"API 응답 수신: 상태 코드={response.status_code}") # [디버깅 로그 추가 2]
        response.raise_for_status()
        
        data = response.json()
        log.debug(f"API로부터 받은 원본 데이터: {data}") # [디버깅 로그 추가 3]
        
        expanded_terms = data.get("expanded_terms")
        if expanded_terms and isinstance(expanded_terms, list):
            log.info(f"API로부터 쿼리 '{query}' 확장 결과 성공적으로 수신: {expanded_terms}")
            return expanded_terms
        else:
            log.warning(f"API 응답에 'expanded_terms'가 없거나 형식이 올바르지 않습니다. 원본 쿼리 사용: {query}")
            return [query]
            
    except httpx.RequestError as e: # httpx 예외 처리로 변경
        log.error(f"쿼리 확장 API 네트워크 호출 실패 ({type(e).__name__}): {e}. 원본 쿼리를 사용합니다.")
        return [query]
    except Exception as e:
        log.error(f"쿼리 확장 처리 중 예상치 못한 예외 발생: {e.__class__.__name__}: {e}", exc_info=True) # [디버깅 로그 추가 4]
        return [query]

# `process_queries_async` 함수를 API를 사용하도록 대폭 수정합니다.
async def process_queries_async(queries: List[str], openai_key: Optional[str] = None) -> List[str]:
    """
    여러 쿼리를 병렬로 처리하는 비동기 함수.
    내부적으로 쿼리 확장 API를 호출합니다.
    """
    # openai_key는 이제 사용되지 않지만, 함수 시그니처 유지를 위해 남겨둡니다.
    if not queries:
        return []

    # 각 쿼리에 대해 API 호출 태스크를 생성
    tasks = [call_query_expansion_api(query) for query in queries]
    
    try:
        # 모든 API 호출을 병렬로 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        log.error(f"쿼리 확장 API 병렬 처리 중 심각한 오류 발생: {e}")
        return queries # 실패 시 원본 쿼리 반환

    all_queries = []
    for i, result in enumerate(results):
        original_query = queries[i]
        
        # 원본 쿼리는 항상 포함
        all_queries.append(original_query)

        if isinstance(result, Exception):
            log.error(f"쿼리 '{original_query}' 처리 중 예외 발생: {result}")
        elif isinstance(result, list):
            # API로부터 받은 확장된 쿼리들을 추가 (중복 방지를 위해 set 사용)
            all_queries.extend(result)
        else:
            log.warning(f"쿼리 '{original_query}'에 대한 API 결과가 비정상적입니다: {result}")

    # 최종적으로 순서를 유지하며 중복을 제거
    seen = set()
    unique_queries = []
    for q in all_queries:
        if q not in seen:
            unique_queries.append(q)
            seen.add(q)
            
    log.info(f"최종 확장 쿼리 목록: {unique_queries}")
    return unique_queries

async def call_rerank_api_async(
    combined_results: dict,
    original_query: list,
    k: int,
    r: float,
    api_key: str,
) -> dict:
    """새로운 리랭킹 API를 비동기적으로 호출합니다."""
    log.info(f"LLM 리랭킹을 위해 API 호출: {RERANK_API_URL}")
    
    payload = {
        "combined_results": combined_results,
        "original_query": original_query,
        "k": k,
        "r": r,
        "api_key": api_key,
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(RERANK_API_URL, json=payload)
            response.raise_for_status() # 오류가 있으면 예외 발생
            log.info("리랭킹 API로부터 성공적으로 응답 받음.")
            return response.json()
    except httpx.RequestError as e:
        log.error(f"리랭킹 API 호출 중 네트워크 오류 발생: {e}. 리랭킹 없이 진행합니다.")
        return combined_results # 실패 시 원본 결과 반환
    except httpx.HTTPStatusError as e:
        log.error(f"리랭킹 API가 오류를 반환했습니다 (상태 코드: {e.response.status_code}): {e.response.text}. 리랭킹 없이 진행합니다.")
        return combined_results # 실패 시 원본 결과 반환
    except Exception as e:
        log.error(f"리랭킹 API 호출 중 예상치 못한 오류 발생: {e}", exc_info=True)
        return combined_results # 실패 시 원본 결과 반환

class VectorSearchRetriever(BaseRetriever):
    collection_name: str
    embedding_function: Any
    top_k: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        result = VECTOR_DB_CLIENT.search(
            collection_name=self.collection_name,
            vectors=[self.embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)],
            limit=self.top_k,
        )

        ids = result.ids[0]
        metadatas = result.metadatas[0]
        documents = result.documents[0]

        results = []
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results


def query_doc(
    collection_name: str, 
    query_embedding: List[float], 
    k: int, 
    user: Optional[UserModel] = None
) -> Any:
    try:
        log.debug(f"query_doc:doc {collection_name}")
        result = VECTOR_DB_CLIENT.search(
            collection_name=collection_name,
            vectors=[query_embedding],
            limit=k,
        )

        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with limit {k}: {e}")
        raise e


def get_doc(
    collection_name: str, 
    user: Optional[UserModel] = None
) -> Any:
    try:
        log.debug(f"get_doc:doc {collection_name}")
        result = VECTOR_DB_CLIENT.get(collection_name=collection_name)

        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error getting doc {collection_name}: {e}")
        raise e

def query_doc_with_hybrid_search(
    collection_name: str,
    collection_result: GetResult,
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    hybrid_bm25_weight: float,
    openai_key: Optional[str] = None,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
) -> dict:
    
    api_key = openai_key or GEMINI_API_KEY
    
    try:
        # BM_25 required only if weight is greater than 0
        if hybrid_bm25_weight > 0:
            log.debug(f"query_doc_with_hybrid_search:doc {collection_name}")
            adjusted_weights = adjust_search_weights(query, bm25_weight, vector_weight)
            
            bm25_retriever = BM25Retriever.from_texts(
                texts=collection_result.documents[0],
                metadatas=collection_result.metadatas[0],
            )
            bm25_retriever.k = k

        vector_search_retriever = VectorSearchRetriever(
            collection_name=collection_name,
            embedding_function=embedding_function,
            top_k=k,
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_search_retriever], 
            weights=[adjusted_weights["bm25"], adjusted_weights["vector"]]
        )
        compressor = RerankCompressor(
            embedding_function=embedding_function,
            top_n=k_reranker,
            reranking_function=None,
            r_score=r,
            openai_key=api_key,
            use_llm_reranking=True,
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )

        result = compression_retriever.invoke(query)

        distances = [d.metadata.get("score") for d in result]
        documents = [d.page_content for d in result]
        metadatas = [d.metadata for d in result]

        # retrieve only min(k, k_reranker) items, sort and cut by distance if k < k_reranker
        if k < k_reranker:
            sorted_items = sorted(
                zip(distances, metadatas, documents), key=lambda x: x[0], reverse=True
            )
            sorted_items = sorted_items[:k]
            distances, documents, metadatas = map(list, zip(*sorted_items))

        result = {
            "distances": [distances],
            "documents": [documents],
            "metadatas": [metadatas],
        }

        log.info(
            "query_doc_with_hybrid_search:result "
            + f'{result["metadatas"]} {result["distances"]}'
        )
        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with hybrid search: {e}")
        raise e

def adjust_search_weights(
    query: str, 
    default_bm25_weight: float, 
    default_vector_weight: float
) -> Dict[str, float]:
    """쿼리 특성에 따라 검색 가중치를 동적으로 조정하는 함수"""
    # 기본 가중치
    weights = {
        "bm25": default_bm25_weight,
        "vector": default_vector_weight
    }
    
    # 쿼리 길이
    query_length = len(query.split())
    
    # 특수 문자 포함 여부
    has_special_chars = any(char in query for char in "{}[]()\"':;,.<>/?\\|!@#$%^&*-_=+~`")
    
    # 숫자 포함 여부
    has_numbers = any(char.isdigit() for char in query)
    
    # 쿼리 특성에 따른 가중치 조정
    if query_length <= 3:  # 짧은 쿼리: 정확한 키워드 매칭이 중요하므로 BM25 가중치 증가
        weights["bm25"] = min(default_bm25_weight + 0.2, 0.6)
        weights["vector"] = 1.0 - weights["bm25"]
    elif query_length >= 8:  # 긴 쿼리: 의미적 유사성이 중요하므로 벡터 가중치 증가
        weights["vector"] = min(default_vector_weight + 0.1, 0.8)
        weights["bm25"] = 1.0 - weights["vector"]
    
    # 특수 문자나 숫자가 있으면 정확한 매칭이 중요하므로 BM25 가중치 증가
    if has_special_chars or has_numbers:
        weights["bm25"] = min(weights["bm25"] + 0.15, 0.7)
        weights["vector"] = 1.0 - weights["bm25"]
    
    # 가중치 합이 1이 되도록 정규화
    total = weights["bm25"] + weights["vector"]
    weights["bm25"] /= total
    weights["vector"] /= total
    
    log.info(f"Adjusted search weights for query '{query}': {weights}")
    return weights

def merge_get_results(get_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Initialize lists to store combined data
    combined_documents = []
    combined_metadatas = []
    combined_ids = []

    for data in get_results:
        combined_documents.extend(data["documents"][0])
        combined_metadatas.extend(data["metadatas"][0])
        combined_ids.extend(data["ids"][0])

    # Create the output dictionary
    result = {
        "documents": [combined_documents],
        "metadatas": [combined_metadatas],
        "ids": [combined_ids],
    }

    return result


def merge_and_sort_query_results(
    query_results: List[Dict[str, Any]], 
    k: int, 
    diversity_threshold: float = MMR_DIVERSITY_THRESHOLD
) -> Dict[str, Any]:
    # Initialize lists to store combined data
    combined = dict()  # To store documents with unique document hashes

    for data in query_results:
        distances = data["distances"][0]
        documents = data["documents"][0]
        metadatas = data["metadatas"][0]

        for distance, document, metadata in zip(distances, documents, metadatas):
            if isinstance(document, str):
                doc_hash = hashlib.sha256(
                    document.encode()
                ).hexdigest()  # Compute a hash for uniqueness

                if doc_hash not in combined.keys():
                    combined[doc_hash] = (distance, document, metadata)
                    continue  # if doc is new, no further comparison is needed

                # if doc is alredy in, but new distance is better, update
                if distance > combined[doc_hash][0]:
                    combined[doc_hash] = (distance, document, metadata)

    combined = list(combined.values())
    # Sort the list based on distances
    combined.sort(key=lambda x: x[0], reverse=True)

    if not combined:
        sorted_distances = []
        sorted_documents = []
        sorted_metadatas = []
    else:
        # 결과 다양성 향상을 위한 처리
        diversified_results = apply_maximal_marginal_relevance(combined, k, diversity_threshold)
        
        # Unzip the diversified list
        sorted_distances, sorted_documents, sorted_metadatas = zip(*diversified_results)

        # 리스트로 변환
        sorted_distances = list(sorted_distances)
        sorted_documents = list(sorted_documents)
        sorted_metadatas = list(sorted_metadatas)

    # Create the output dictionary
    result = {
        "distances": [sorted_distances],
        "documents": [sorted_documents],
        "metadatas": [sorted_metadatas],
    }
    
    return result

def apply_maximal_marginal_relevance(
    combined_results: List[Tuple[float, str, Dict[str, Any]]], 
    k: int, 
    diversity_threshold: float = MMR_DIVERSITY_THRESHOLD
) -> List[Tuple[float, str, Dict[str, Any]]]:
    """
    결과의 다양성을 높이기 위해 개선된 Maximal Marginal Relevance 알고리즘 적용
    
    Args:
        combined_results: (score, document, metadata) 튜플 리스트
        k: 반환할 최대 결과 수
        diversity_threshold: 다양성 임계값 (0에 가까울수록 다양성 증가, 1에 가까울수록 원래 순위 유지)
    
    Returns:
        다양성이 개선된 결과 리스트
    """
    if len(combined_results) <= 1 or k <= 1:
        return combined_results[:k]
    
    # 이미 선택된 결과와 후보 결과 분리
    selected_results = [combined_results[0]]  # 첫 번째 결과는 항상 포함
    candidates = combined_results[1:]
    
    # 문서 텍스트 추출
    doc_contents = [doc for _, doc, _ in combined_results]
    
    # 벡터화 프로세스
    doc_vectors = []
    
    try:
        # 먼저 내장된 간단한 벡터화 방법 시도 (TF-IDF 벡터화 없이)
        for doc in doc_contents:
            # 간단한 BoW(Bag of Words) 벡터 계산
            words = doc.lower().split()
            word_counts = {}
            for word in words:
                if len(word) > 1:  # 짧은 단어 무시
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # 정규화된 벡터 생성
            total_words = sum(word_counts.values())
            if total_words > 0:
                normalized_vector = {word: count/total_words for word, count in word_counts.items()}
            else:
                normalized_vector = {}
                
            doc_vectors.append(normalized_vector)
            
    except Exception as e:
        log.error(f"Simple vectorization failed: {e}")
        # 실패하면 대체 벡터화
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # 벡터화 인스턴스 사용
            tfidf_vectorizer = TfidfVectorizer()
            doc_vectors = tfidf_vectorizer.fit_transform(doc_contents).toarray()
        except Exception as e:
            log.error(f"TfidfVectorizer failed too: {e}")
            # 마지막 대안: 직접 문자열 유사도 계산 (자카드 유사도)
            for doc in doc_contents:
                words = set(doc.lower().split())
                doc_vectors.append(words)
    
    # MMR 알고리즘으로 다양성 있는 결과 선택
    while len(selected_results) < k and candidates:
        best_score = float('-inf')
        best_idx = -1
        
        for i, candidate in enumerate(candidates):
            candidate_idx = combined_results.index(candidate)
            original_score = candidate[0]  # 원래 유사도 점수
            
            # 이미 선택된 문서와의 최대 유사도 계산
            max_similarity = 0
            for selected in selected_results:
                selected_idx = combined_results.index(selected)
                
                # 벡터 유형에 따른 유사도 계산
                similarity = 0
                
                if isinstance(doc_vectors[0], dict):
                    # BoW 벡터인 경우 (dict)
                    v1 = doc_vectors[candidate_idx]
                    v2 = doc_vectors[selected_idx]
                    
                    # 코사인 유사도 계산
                    common_words = set(v1.keys()) & set(v2.keys())
                    if not common_words:
                        similarity = 0
                    else:
                        numerator = sum(v1[word] * v2[word] for word in common_words)
                        sum1 = sum(val**2 for val in v1.values())
                        sum2 = sum(val**2 for val in v2.values())
                        denominator = (sum1**0.5) * (sum2**0.5)
                        if denominator > 0:
                            similarity = numerator / denominator
                        else:
                            similarity = 0
                
                elif isinstance(doc_vectors[0], set):
                    # 집합인 경우 (자카드 유사도)
                    v1 = doc_vectors[candidate_idx]
                    v2 = doc_vectors[selected_idx]
                    if not v1 or not v2:
                        similarity = 0
                    else:
                        similarity = len(v1 & v2) / len(v1 | v2)
                
                else:
                    try:
                        # sklearn 배열인 경우
                        from sklearn.metrics.pairwise import cosine_similarity
                        similarity = cosine_similarity(
                            doc_vectors[candidate_idx].reshape(1, -1),
                            doc_vectors[selected_idx].reshape(1, -1)
                        )[0][0]
                    except Exception as e:
                        # 마지막 대안: 문자열 직접 비교
                        text1 = doc_contents[candidate_idx]
                        text2 = doc_contents[selected_idx]
                        common_words = set(text1.lower().split()) & set(text2.lower().split())
                        all_words = set(text1.lower().split()) | set(text2.lower().split())
                        if all_words:
                            similarity = len(common_words) / len(all_words)
                        else:
                            similarity = 0
                
                max_similarity = max(max_similarity, similarity)
            
            # MMR 점수 계산: λ * 원래점수 - (1-λ) * 최대유사도
            mmr_score = diversity_threshold * original_score - (1 - diversity_threshold) * max_similarity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        if best_idx != -1:
            selected_results.append(candidates[best_idx])
            candidates.pop(best_idx)
        else:
            break
            
    return selected_results

def get_all_items_from_collections(
    collection_names: List[str]
) -> Dict[str, Any]:
    results = []

    for collection_name in collection_names:
        if collection_name:
            try:
                result = get_doc(collection_name=collection_name)
                if result is not None:
                    results.append(result.model_dump())
            except Exception as e:
                log.exception(f"Error when querying the collection: {e}")
        else:
            pass

    return merge_get_results(results)


def query_collection(
    collection_names: List[str],
    queries: List[str],
    embedding_function: Any,
    k: int,
) -> Dict[str, Any]:
    results = []
    error = False

    def process_query_collection(collection_name, query_embedding):
        try:
            if collection_name:
                result = query_doc(
                    collection_name=collection_name,
                    k=k,
                    query_embedding=query_embedding,
                )
                if result is not None:
                    return result.model_dump(), None
            return None, None
        except Exception as e:
            log.exception(f"Error when querying the collection: {e}")
            return None, e

    # Generate all query embeddings (in one call)
    query_embeddings = embedding_function(queries, prefix=RAG_EMBEDDING_QUERY_PREFIX)
    log.debug(
        f"query_collection: processing {len(queries)} queries across {len(collection_names)} collections"
    )

    with ThreadPoolExecutor(max_workers=min(len(collection_names) * len(query_embeddings), 10)) as executor:
        # Submit all tasks
        future_to_params = {}
        for query_embedding in query_embeddings:
            for collection_name in collection_names:
                future = executor.submit(process_query_collection, collection_name, query_embedding)
                future_to_params[future] = (collection_name, query_embedding)
        
        # Collect results as they complete
        task_results = []
        from concurrent.futures import as_completed
        
        for future in as_completed(future_to_params):
            try:
                result, err = future.result(timeout=30)  # 30초 타임아웃
                task_results.append((result, err))
            except Exception as e:
                collection_name, _ = future_to_params[future]
                log.error(f"Task failed for collection {collection_name}: {e}")
                task_results.append((None, e))

    for result, err in task_results:
        if err is not None:
            error = True
        elif result is not None:
            results.append(result)

    if error and not results:
        log.warning("All collection queries failed. No results returned.")

    return merge_and_sort_query_results(results, k=k)


def query_collection_with_hybrid_search(
    collection_names: List[str],
    queries: List[str],
    embedding_function: Any,
    k: int,
    reranking_function: Any,
    k_reranker: int,
    r: float,
    hybrid_bm25_weight: float,
    openai_key: Optional[str] = None,
) -> Dict[str, Any]:
    results = []
    error = False
    # Fetch collection data once per collection sequentially
    # Avoid fetching the same data multiple times later
    collection_results = {}
    # Only retrieve entire collection if bm_25 calculation is required
    if hybrid_bm25_weight > 0:
        for collection_name in collection_names:
            try:
                log.debug(
                    f"query_collection_with_hybrid_search:VECTOR_DB_CLIENT.get:collection {collection_name}"
                )
                collection_results[collection_name] = VECTOR_DB_CLIENT.get(
                    collection_name=collection_name
                )
            except Exception as e:
                log.exception(f"Failed to fetch collection {collection_name}: {e}")
                collection_results[collection_name] = None
    else:
        for collection_name in collection_names:
            collection_results[collection_name] = []
    log.info(
        f"Starting hybrid search for {len(queries)} queries in {len(collection_names)} collections..."
    )

    def process_query(collection_name, query):
        try:
            result = query_doc_with_hybrid_search(
                collection_name=collection_name,
                collection_result=collection_results[collection_name],
                query=query,
                embedding_function=embedding_function,
                k=k,
                reranking_function=reranking_function,
                k_reranker=k_reranker,
                r=r,
                hybrid_bm25_weight=hybrid_bm25_weight,
            )
            return result, None
        except Exception as e:
            log.exception(f"Error when querying the collection with hybrid_search: {e}")
            return None, e
    
    api_key = openai_key or GEMINI_API_KEY
    
    try:
        # 비동기 코드를 한 번의 호출로 처리
        expanded_queries = []
        if api_key:
            # 이벤트 루프가 이미 실행 중인지 확인하고 적절히 처리
            try:
                # 현재 이벤트 루프 가져오기 시도
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 실행 중인 루프가 있으면 새 루프를 만들지 않고 future로 처리
                    future = asyncio.run_coroutine_threadsafe(
                        process_queries_async(queries, api_key), loop
                    )
                    expanded_queries = future.result(timeout=ASYNC_TIMEOUT_SECONDS)
                else:
                    # 루프가 실행 중이 아니면 run_until_complete 사용
                    expanded_queries = loop.run_until_complete(
                        process_queries_async(queries, api_key)
                    )
            except RuntimeError:
                # 이벤트 루프가 없으면 새로 생성
                expanded_queries = asyncio.run(process_queries_async(queries, api_key))
        else:
            log.warning("Gemini API 키가 설정되지 않았습니다. 쿼리 향상을 건너뜁니다.")
            expanded_queries = queries
            
        log.info(f"Final expanded queries: {expanded_queries}")
        
        # 각 쿼리의 결과를 모두 수집
        all_results = []
        for collection_name in collection_names:
            try:
                log.debug(
                     f"query_collection_with_hybrid_search:VECTOR_DB_CLIENT.get:collection {collection_name}"
                )
                collection_result = VECTOR_DB_CLIENT.get(collection_name=collection_name)
                for query in expanded_queries:
                    result = get_hybrid_search_results_without_reranking(
                        collection_name=collection_name,
                        collection_result=collection_result,
                        query=query,
                        embedding_function=embedding_function,
                        k=k,
                    )
                    all_results.append(result)
            except Exception as e:
                log.exception(f"Error when querying the collection with hybrid_search: {e}")
                error = True

        if error and not all_results:
            raise Exception(
                "Hybrid search failed for all collections. Using Non-hybrid search as fallback."
            )
            
        # 중복 제거 및 결과 통합 (모든 쿼리 결과를 병합)
        combined_results = merge_and_deduplicate_results(all_results)
        
        # LLM 리랭킹 수행 부분을 API 호출로 변경
        if api_key and combined_results and combined_results["documents"][0]:
            log.info(f"API를 통해 LLM 기반 리랭킹 수행: 문서 수={len(combined_results['documents'][0])}")
            
            # ############################################################### #
            # ##               여기가 수정된 핵심 부분입니다                ## #
            # ############################################################### #
            
            try:
                # 동기 스레드 컨텍스트에서 비동기 API 호출을 위한 가장 간단하고 안전한 방법
                final_results = asyncio.run(
                    call_rerank_api_async(
                        combined_results=combined_results,
                        original_query=expanded_queries,
                        k=k,
                        r=r,
                        api_key=api_key,
                    )
                )
                results = [final_results]
            except Exception as e:
                # asyncio.run()은 이미 실행 중인 루프에서 호출되면 RuntimeError를 발생시킬 수 있습니다.
                # 이는 이 함수가 다른 비동기 컨텍스트에서 호출될 경우에 대한 안전장치입니다.
                if "cannot run event loop while another loop is running" in str(e):
                    log.warning("리랭킹을 중첩된 이벤트 루프에서 호출하려고 시도했습니다. 이 시나리오는 현재 지원되지 않습니다.")
                
                log.error(f"리랭킹 API 호출 중 오류 발생: {e}. 리랭킹 없이 진행합니다.", exc_info=True)
                results = [combined_results] # 실패 시 리랭킹 전 결과 사용

        else:
            log.info("리랭킹을 건너뜁니다 (API 키 또는 문서 없음).")
            results = [combined_results]

        if VECTOR_DB == "chroma":
            return merge_and_sort_query_results(results, k=k)
        else:
            return merge_and_sort_query_results(results, k=k)
    except Exception as e:
        raise e


def get_embedding_function(
    embedding_engine,
    embedding_model,
    embedding_function,
    url,
    key,
    embedding_batch_size,
    azure_api_version=None,
):
    if embedding_engine == "":
        return lambda query, prefix=None, user=None: embedding_function.encode(
            query, prompt=prefix if prefix else None
        ).tolist()
    elif embedding_engine in ["ollama", "openai", "azure_openai"]:
        func = lambda query, prefix=None, user=None: generate_embeddings(
            engine=embedding_engine,
            model=embedding_model,
            text=query,
            prefix=prefix,
            url=url,
            key=key,
            user=user,
            azure_api_version=azure_api_version,
        )

        def generate_multiple(query, prefix, user, func):
            if isinstance(query, list):
                embeddings = []
                for i in range(0, len(query), embedding_batch_size):
                    embeddings.extend(
                        func(
                            query[i : i + embedding_batch_size],
                            prefix=prefix,
                            user=user,
                        )
                    )
                return embeddings
            else:
                return func(query, prefix, user)

        return lambda query, prefix=None, user=None: generate_multiple(
            query, prefix, user, func
        )
    else:
        raise ValueError(f"Unknown embedding engine: {embedding_engine}")


def get_reranking_function(reranking_engine, reranking_model, reranking_function):
    if reranking_function is None:
        return None
    if reranking_engine == "external":
        return lambda sentences, user=None: reranking_function.predict(
            sentences, user=user
        )
    else:
        return lambda sentences, user=None: reranking_function.predict(sentences)


def get_sources_from_items(
    request,
    items,
    queries,
    embedding_function,
    k,
    reranking_function,
    k_reranker,
    r,
    hybrid_bm25_weight,
    hybrid_search,
    full_context=False,
    user: Optional[UserModel] = None,
):
    log.debug(
        f"items: {items} {queries} {embedding_function} {reranking_function} {full_context}"
    )

    extracted_collections = []
    query_results = []

    for item in items:
        query_result = None
        collection_names = []

        if item.get("type") == "text":
            # Raw Text
            # Used during temporary chat file uploads

            if item.get("file"):
                # if item has file data, use it
                query_result = {
                    "documents": [
                        [item.get("file", {}).get("data", {}).get("content")]
                    ],
                    "metadatas": [
                        [item.get("file", {}).get("data", {}).get("meta", {})]
                    ],
                }
            else:
                # Fallback to item content
                query_result = {
                    "documents": [[item.get("content")]],
                    "metadatas": [
                        [{"file_id": item.get("id"), "name": item.get("name")}]
                    ],
                }

        elif item.get("type") == "note":
            # Note Attached
            note = Notes.get_note_by_id(item.get("id"))

            if note and (
                user.role == "admin"
                or note.user_id == user.id
                or has_access(user.id, "read", note.access_control)
            ):
                # User has access to the note
                query_result = {
                    "documents": [[note.data.get("content", {}).get("md", "")]],
                    "metadatas": [[{"file_id": note.id, "name": note.title}]],
                }

        elif item.get("type") == "file":
            if (
                item.get("context") == "full"
                or request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
            ):
                if item.get("file", {}).get("data", {}).get("content", ""):
                    # Manual Full Mode Toggle
                    # Used from chat file modal, we can assume that the file content will be available from item.get("file").get("data", {}).get("content")
                    query_result = {
                        "documents": [
                            [item.get("file", {}).get("data", {}).get("content", "")]
                        ],
                        "metadatas": [
                            [
                                {
                                    "file_id": item.get("id"),
                                    "name": item.get("name"),
                                    **item.get("file")
                                    .get("data", {})
                                    .get("metadata", {}),
                                }
                            ]
                        ],
                    }
<<<<<<< HEAD
                elif item.get("id"):
                    file_object = Files.get_file_by_id(item.get("id"))
                    if file_object:
=======
            elif file.get("file").get("data"):
                query_result = {
                    "documents": [[file.get("file").get("data", {}).get("content")]],
                    "metadatas": [
                        [file.get("file").get("data", {}).get("metadata", {})]
                    ],
                }
        else:
            # [최종 수정] 파일 타입에 따라 RAG 사용 여부를 명확히 결정합니다.
            # Case 1: 직접 업로드된 파일 (컬렉션의 일부가 아님)
            if file.get("type") != "collection":
                log.info(f"Direct file upload '{file.get('name')}' detected. Using full content.")
                file_id = file.get("id")
                if file_id:
                    file_object = Files.get_file_by_id(file_id)
                    if file_object and file_object.data and 'content' in file_object.data:
>>>>>>> 8388961c2 (v0.6.16버전 업데이트대비 호환성 추가(context 대신 query_result 사용))
                        query_result = {
                            "documents": [[file_object.data.get("content", "")]],
                            "metadatas": [
                                [
                                    {
                                        "file_id": item.get("id"),
                                        "name": file_object.filename,
                                        "source": file_object.filename,
                                    }
                                ]
                            ],
                        }
<<<<<<< HEAD
            else:
                # Fallback to collection names
                if item.get("legacy"):
                    collection_names.append(f"{item['id']}")
                else:
                    collection_names.append(f"file-{item['id']}")

        elif item.get("type") == "collection":
            if (
                item.get("context") == "full"
                or request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
            ):
                # Manual Full Mode Toggle for Collection
                knowledge_base = Knowledges.get_knowledge_by_id(item.get("id"))
=======
                    else:
                        log.warning(f"Could not retrieve content for direct file upload: {file_id}")
                        query_result = None
                else:
                    query_result = None
            
            # Case 2: 컬렉션의 일부인 파일 (RAG 수행)
            else:
                collection_names = []
                if file.get("type") == "collection":
                    if file.get("legacy"):
                        collection_names = file.get("collection_names", [])
                    else:
                        collection_names.append(file["id"])
                elif file.get("collection_name"):
                    collection_names.append(file["collection_name"])
                elif file.get("id"):
                    if file.get("legacy"):
                        collection_names.append(f"{file['id']}")
                    else:
                        collection_names.append(f"file-{file['id']}")

                collection_names = set(collection_names).difference(extracted_collections)
                if not collection_names:
                    log.debug(f"skipping {file} as it has already been extracted")
                    continue

                if full_context:
                    try:
                        query_result = get_all_items_from_collections(collection_names)
                    except Exception as e:
                        log.exception(e)
                else:
                    try:
                        query_result = None
                        if file.get("type") == "text":
                            # Not sure when this is used, but it seems to be a fallback
                            query_result = {
                                "documents": [
                                    [file.get("file").get("data", {}).get("content")]
                                ],
                                "metadatas": [
                                    [file.get("file").get("data", {}).get("meta", {})]
                                ],
                            }
                        else:
                            if hybrid_search:
                                try:
                                    query_result = query_collection_with_hybrid_search(
                                        collection_names=collection_names,
                                        queries=queries,
                                        embedding_function=embedding_function,
                                        k=k,
                                        reranking_function=reranking_function,
                                        k_reranker=k_reranker,
                                        r=r,
                                        hybrid_bm25_weight=hybrid_bm25_weight,
                                    )
                                except Exception as e:
                                    log.debug(
                                        "Error when using hybrid search, using"
                                        " non hybrid search as fallback."
                                    )

                            if (not hybrid_search) or (query_result is None):
                                query_result = query_collection(
                                    collection_names=collection_names,
                                    queries=queries,
                                    embedding_function=embedding_function,
                                    k=k,
                                )
                    except Exception as e:
                        log.exception(e)
>>>>>>> 8388961c2 (v0.6.16버전 업데이트대비 호환성 추가(context 대신 query_result 사용))

                if knowledge_base and (
                    user.role == "admin"
                    or has_access(user.id, "read", knowledge_base.access_control)
                ):

                    file_ids = knowledge_base.data.get("file_ids", [])

                    documents = []
                    metadatas = []
                    for file_id in file_ids:
                        file_object = Files.get_file_by_id(file_id)

                        if file_object:
                            documents.append(file_object.data.get("content", ""))
                            metadatas.append(
                                {
                                    "file_id": file_id,
                                    "name": file_object.filename,
                                    "source": file_object.filename,
                                }
                            )

                    query_result = {
                        "documents": [documents],
                        "metadatas": [metadatas],
                    }
            else:
                # Fallback to collection names
                if item.get("legacy"):
                    collection_names = item.get("collection_names", [])
                else:
                    collection_names.append(item["id"])

        elif item.get("docs"):
            # BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
            query_result = {
                "documents": [[doc.get("content") for doc in item.get("docs")]],
                "metadatas": [[doc.get("metadata") for doc in item.get("docs")]],
            }
        elif item.get("collection_name"):
            # Direct Collection Name
            collection_names.append(item["collection_name"])
        elif item.get("collection_names"):
            # Collection Names List
            collection_names.extend(item["collection_names"])

        # If query_result is None
        # Fallback to collection names and vector search the collections
        if query_result is None and collection_names:
            collection_names = set(collection_names).difference(extracted_collections)
            if not collection_names:
                log.debug(f"skipping {item} as it has already been extracted")
                continue

            try:
                if full_context:
                    query_result = get_all_items_from_collections(collection_names)
                else:
                    query_result = None  # Initialize to None
                    if hybrid_search:
                        try:
                            query_result = query_collection_with_hybrid_search(
                                collection_names=collection_names,
                                queries=queries,
                                embedding_function=embedding_function,
                                k=k,
                                reranking_function=reranking_function,
                                k_reranker=k_reranker,
                                r=r,
                                hybrid_bm25_weight=hybrid_bm25_weight,
                            )
                        except Exception as e:
                            log.debug(
                                "Error when using hybrid search, using non hybrid search as fallback."
                            )

                    # fallback to non-hybrid search
                    if not hybrid_search and query_result is None:
                        query_result = query_collection(
                            collection_names=collection_names,
                            queries=queries,
                            embedding_function=embedding_function,
                            k=k,
                        )
            except Exception as e:
                log.exception(e)

            extracted_collections.extend(collection_names)

        if query_result:
            if "data" in item:
                del item["data"]
            query_results.append({**query_result, "file": item})

    sources = []
    for query_result in query_results:
        try:
            if "documents" in query_result:
                if "metadatas" in query_result:
                    source = {
                        "source": query_result["file"],
                        "document": query_result["documents"][0],
                        "metadata": query_result["metadatas"][0],
                    }
                    if "distances" in query_result and query_result["distances"]:
                        source["distances"] = query_result["distances"][0]

                    sources.append(source)
        except Exception as e:
            log.exception(e)

    return sources


def get_model_path(model: str, update_model: bool = False):
    # Construct huggingface_hub kwargs with local_files_only to return the snapshot path
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

    local_files_only = not update_model

    if OFFLINE_MODE:
        local_files_only = True

    snapshot_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    log.debug(f"model: {model}")
    log.debug(f"snapshot_kwargs: {snapshot_kwargs}")

    # Inspiration from upstream sentence_transformers
    if (
        os.path.exists(model)
        or ("\\" in model or model.count("/") > 1)
        and local_files_only
    ):
        # If fully qualified path exists, return input, else set repo_id
        return model
    elif "/" not in model:
        # Set valid repo_id for model short-name
        model = "sentence-transformers" + "/" + model

    snapshot_kwargs["repo_id"] = model

    # Attempt to query the huggingface_hub library to determine the local path and/or to update
    try:
        model_repo_path = snapshot_download(**snapshot_kwargs)
        log.debug(f"model_repo_path: {model_repo_path}")
        return model_repo_path
    except Exception as e:
        log.exception(f"Cannot determine model snapshot path: {e}")
        return model


def generate_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str = "https://api.openai.com/v1",
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
             f"generate_openai_batch_embeddings:model {model} batch size: {len(texts)}"
         )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        r = requests.post(
            f"{url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                **(
                    {
                        "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS and user
                    else {}
                ),
            },
            json=json_data,
        )
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return [elem["embedding"] for elem in data["data"]]
        else:
            raise Exception("Something went wrong :/")
    except Exception as e:
        log.exception(f"Error generating openai batch embeddings: {e}")
        return None

def generate_azure_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    version: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_azure_openai_batch_embeddings:deployment {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        url = f"{url}/openai/deployments/{model}/embeddings?api-version={version}"

        for _ in range(5):
            r = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "api-key": key,
                    **(
                        {
                            "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                            "X-OpenWebUI-User-Id": user.id,
                            "X-OpenWebUI-User-Email": user.email,
                            "X-OpenWebUI-User-Role": user.role,
                        }
                        if ENABLE_FORWARD_USER_INFO_HEADERS and user
                        else {}
                    ),
                },
                json=json_data,
            )
            if r.status_code == 429:
                retry = float(r.headers.get("Retry-After", "1"))
                time.sleep(retry)
                continue
            r.raise_for_status()
            data = r.json()
            if "data" in data:
                return [elem["embedding"] for elem in data["data"]]
            else:
                raise Exception("Something went wrong :/")
        return None
    except Exception as e:
        log.exception(f"Error generating azure openai batch embeddings: {e}")
        return None


def generate_ollama_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
             f"generate_ollama_batch_embeddings:model {model} batch size: {len(texts)}"
         )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        r = requests.post(
            f"{url}/api/embed",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                **(
                    {
                        "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS
                    else {}
                ),
            },
            json=json_data,
        )
        r.raise_for_status()
        data = r.json()

        if "embeddings" in data:
            return data["embeddings"]
        else:
            raise Exception("Something went wrong :/")
    except Exception as e:
        log.exception(f"Error generating ollama batch embeddings: {e}")
        return None


def generate_embeddings(
    engine: str,
    model: str,
    text: Union[str, list[str]],
    prefix: Union[str, None] = None,
    **kwargs,
):
    url = kwargs.get("url", "")
    key = kwargs.get("key", "")
    user = kwargs.get("user")

    if prefix is not None and RAG_EMBEDDING_PREFIX_FIELD_NAME is None:
        if isinstance(text, list):
            text = [f"{prefix}{text_element}" for text_element in text]
        else:
            text = f"{prefix}{text}"

    if engine == "ollama":
        embeddings = generate_ollama_batch_embeddings(
            **{
                "model": model,
                "texts": text if isinstance(text, list) else [text],
                "url": url,
                "key": key,
                "prefix": prefix,
                "user": user,
            }
        )
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "openai":
        embeddings = generate_openai_batch_embeddings(
            model, text if isinstance(text, list) else [text], url, key, prefix, user
        )
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "azure_openai":
        azure_api_version = kwargs.get("azure_api_version", "")
        embeddings = generate_azure_openai_batch_embeddings(
            model,
            text if isinstance(text, list) else [text],
            url,
            key,
            azure_api_version,
            prefix,
            user,
        )
        return embeddings[0] if isinstance(text, str) else embeddings




class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None

        scores = None
        if reranking:
            scores = self.reranking_function(
                [(query, doc.page_content) for doc in documents]
            )
        else:
            from sentence_transformers import util

            query_embedding = self.embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)
            document_embedding = self.embedding_function(
                [doc.page_content for doc in documents], RAG_EMBEDDING_CONTENT_PREFIX
            )
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        if scores is not None:
            docs_with_scores = list(
                zip(
                    documents,
                    scores.tolist() if not isinstance(scores, list) else scores,
                )
            )
            if self.r_score:
                docs_with_scores = [
                    (d, s) for d, s in docs_with_scores if s >= self.r_score
                ]

            result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
            final_results = []
            for doc, doc_score in result[: self.top_n]:
                metadata = doc.metadata
                metadata["score"] = doc_score
                doc = Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
                final_results.append(doc)
            return final_results
        else:
            log.warning(
                "No valid scores found, check your reranking function. Returning original documents."
            )
            return documents

def _update_rrf_scores(
    ranked_results: dict,
    retrieved_docs: list,
    original_texts: list,
    rrf_k: int,
    weight: float,
    is_bm25: bool = False,
):
    """RRF 점수를 계산하고 ranked_results 딕셔너리를 업데이트하는 헬퍼 함수"""
    for rank, doc in enumerate(retrieved_docs):
        if is_bm25:
            original_idx = doc.metadata.get("_original_index")
            if original_idx is None or original_idx >= len(original_texts):
                continue
            content = original_texts[original_idx]
            metadata = doc.metadata.copy()
            del metadata["_original_index"]
        else:
            # 벡터 검색 결과 (doc은 (content, metadata) 튜플)
            content, metadata = doc

        # 더 효율적인 해싱을 위해 처음 64자만 사용
        doc_hash = hashlib.sha256(content[:64].encode("utf-8")).hexdigest()
        rrf_score = 1 / (rrf_k + rank + 1)

        if doc_hash not in ranked_results:
            ranked_results[doc_hash] = {
                "content": content,
                "metadata": metadata,
                "score": 0,
            }
        ranked_results[doc_hash]["score"] += weight * rrf_score


def get_hybrid_search_results_without_reranking(
    collection_name: str,
    collection_result: GetResult,
    query: str,
    embedding_function,
    k: int,
    rrf_k: int = 60,
) -> dict:
    """
    BM25와 벡터 검색을 사용하여 하이브리드 검색을 수행하고,
    Reciprocal Rank Fusion (RRF)로 결과를 결합합니다.

    이 함수는 키워드 검색과 의미적 검색을 병렬로 수행하며,
    캐시된 BM25 검색기를 사용하여 성능을 향상시킵니다.
    결과는 RRF 알고리즘을 사용하여 최적 순위로 결합됩니다.

    Args:
        collection_name: 검색할 컬렉션 이름
        collection_result: 컬렉션의 전체 내용 (문서, 메타데이터)
        query: 사용자 검색 쿼리
        embedding_function: 쿼리 임베딩 생성 함수
        k: 반환할 최종 문서 수
        rrf_k: RRF 계산에 사용되는 상수 (기본값: 60)

    Returns:
        거리(점수), 문서, 메타데이터를 포함한 병합된 순위 검색 결과 딕셔너리
    """
    try:
        log.debug(f"컬렉션 '{collection_name}'에 대한 하이브리드 검색 실행")

        # 1. 쿼리 특성에 따른 동적 가중치와 후보 수 계산
        weights = adjust_search_weights(query, DEFAULT_BM25_WEIGHT, DEFAULT_VECTOR_WEIGHT)
        candidate_multiplier = min(
            CANDIDATE_MULTIPLIER_LOW_K if k <= 5 else CANDIDATE_MULTIPLIER_HIGH_K,
            MAX_CANDIDATE_MULTIPLIER
        )
        candidate_k = max(k, min(int(k * candidate_multiplier), k * 5))  # 최대 5배로 제한

        # 2. 설정 가능한 TTL로 캐시된 BM25 검색기 가져오기 또는 생성
        bm25_retriever = bm25_retriever_cache.get(collection_name)
        if not bm25_retriever:
            log.info(f"'{collection_name}'에 대한 BM25 캐시 미스. 새 검색기 생성.")
            original_texts = collection_result.documents[0]
            lowercase_texts = [text.lower() for text in original_texts]
            enhanced_metadatas = [
                {**(meta or {}), "_original_index": i}
                for i, meta in enumerate(collection_result.metadatas[0])
            ]
            bm25_retriever = BM25Retriever.from_texts(
                texts=lowercase_texts, metadatas=enhanced_metadatas
            )
            bm25_retriever_cache.set(
                collection_name, bm25_retriever, ttl_seconds=CACHE_TTL_BM25
            )
        else:
            log.debug(f"'{collection_name}'에 대한 BM25 캐시 히트.")
        bm25_retriever.k = candidate_k

        # 3. BM25와 벡터 검색을 병렬로 수행
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(bm25_retriever.invoke, query.lower())
            vector_future = executor.submit(
                VECTOR_DB_CLIENT.search,
                collection_name=collection_name,
                vectors=[embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)],
                limit=candidate_k,
            )
            bm25_results_raw = bm25_future.result()
            vector_search_results = vector_future.result()

        # 4. Reciprocal Rank Fusion (RRF)로 결과 결합
        ranked_results = {}

        # BM25 결과 처리
        _update_rrf_scores(
            ranked_results,
            bm25_results_raw,
            collection_result.documents[0],
            rrf_k,
            weight=weights["bm25"],
            is_bm25=True,
        )

        # 벡터 검색 결과 처리
        if vector_search_results and vector_search_results.documents:
            vector_docs = zip(
                vector_search_results.documents[0], vector_search_results.metadatas[0]
            )
            _update_rrf_scores(
                ranked_results,
                list(vector_docs),
                [],
                rrf_k,
                weight=weights["vector"],
                is_bm25=False,
            )

        # 5. RRF 점수로 최종 결과 정렬
        sorted_results = sorted(
            ranked_results.values(), key=lambda x: x["score"], reverse=True
        )
        top_k_results = sorted_results[:k]

        # 6. 결과 포맷팅 및 반환
        if not top_k_results:
            return {
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "query": query,
            }

        scores = [res["score"] for res in top_k_results]
        documents = [res["content"] for res in top_k_results]
        metadatas = [res["metadata"] for res in top_k_results]

        result = {
            "distances": [scores],
            "documents": [documents],
            "metadatas": [metadatas],
            "query": query,
        }

        log.debug(
            f"하이브리드 검색 완료: {len(top_k_results)}개 결과. "
            f"최고 점수: {scores[0] if scores else 'N/A'}. "
            f"가중치 (BM25/벡터): {weights['bm25']:.2f}/{weights['vector']:.2f}. "
            f"후보 수: {candidate_k}"
        )
        return result

    except Exception as e:
        log.error(
            f"get_hybrid_search_results_without_reranking에서 오류 발생: {e}", exc_info=True
        )
        raise e
        
def merge_and_deduplicate_results(all_results: list[dict]) -> dict:
    """
    여러 쿼리의 모든 결과를 병합하고, 내용 기반으로 중복을 제거하며, 각 문서의 최고 점수를 보존합니다.

    Args:
        all_results: 각 쿼리에 대한 결과 딕셔너리의 리스트.
                     각 딕셔너리는 'documents', 'metadatas', 'distances' 키를 포함할 수 있습니다.

    Returns:
        병합되고 정렬된 결과 딕셔너리.
    """
    if not all_results:
        return {"distances": [[]], "documents": [[]], "metadatas": [[]]}

    # 문서 해시를 키로 사용하여 최고 점수, 문서 내용, 메타데이터를 저장
    # 형식: {doc_hash: (score, doc_content, metadata)}
    best_docs = {}
    total_docs_before = 0

    for result in all_results:
        # 결과에 문서가 없으면 건너뛰
        if not (docs := result.get("documents", [[]])[0]):
            continue

        total_docs_before += len(docs)

        # 메타데이터와 점수가 없으면 기본값으로 채움
        metas = result.get("metadatas", [[]])[0] or [{}] * len(docs)
        dists = result.get("distances", [[]])[0] or [0.5] * len(docs)
        query = result.get("query", "unknown_query")

        for doc, meta, dist in zip(docs, metas, dists):
            if not isinstance(doc, str):
                log.warning(f"문서 내용이 문자열이 아니므로 건너뛰니다: {type(doc)}")
                continue

            # Use first 64 chars for more efficient hashing
            doc_hash = hashlib.sha256(doc[:64].encode('utf-8')).hexdigest()

            # 기존에 없거나, 새 점수가 더 높으면 정보 업데이트
            if doc_hash not in best_docs or dist > best_docs[doc_hash][0]:
                # 원본 메탄데이터를 수정하지 않도록 복사
                updated_meta = meta.copy() if meta else {}
                updated_meta["original_query"] = query
                best_docs[doc_hash] = (dist, doc, updated_meta)

    if not best_docs:
        return {"distances": [[]], "documents": [[]], "metadatas": [[]]}

    # 점수(distance) 기준으로 내림차순 정렬
    sorted_results = sorted(best_docs.values(), key=lambda item: item[0], reverse=True)

    # 결과 분리
    final_dists, final_docs, final_metas = zip(*sorted_results)

    log.info(
        f"병합 및 중복 제거 완료: "
        f"총 {total_docs_before}개 문서에서 {len(final_docs)}개의 고유한 문서 선택 "
        f"({total_docs_before - len(final_docs)}개 중복 제거)"
    )

    return {
        "distances": [list(final_dists)],
        "documents": [list(final_docs)],
        "metadatas": [list(final_metas)],
    }

# 캐시 관리 함수들
def get_cache_stats() -> Dict[str, Any]:
    """캐시 통계 정보 반환"""
    return {
        "enabled": ENABLE_CACHING,
        "medical_abbrev": {
            "size": medical_abbrev_cache.size(),
            "max_size": medical_abbrev_cache.max_size,
            "ttl_seconds": CACHE_TTL_MEDICAL_ABBREV
        },
        "query_enhance": {
            "size": query_enhance_cache.size(),
            "max_size": query_enhance_cache.max_size,
            "ttl_seconds": CACHE_TTL_QUERY_ENHANCE
        },
        "bm25_retriever": {
            "size": bm25_retriever_cache.size(),
            "max_size": bm25_retriever_cache.max_size,
            "ttl_seconds": CACHE_TTL_BM25
        }
    }

def clear_all_caches() -> None:
    """모든 캐시 클리어"""
    medical_abbrev_cache.clear()
    query_enhance_cache.clear()
    bm25_retriever_cache.clear()
    log.info("모든 RAG 캐시가 클리어되었습니다.")

def clear_cache_by_type(cache_type: str) -> bool:
    """특정 타입의 캐시만 클리어"""
    if cache_type == "medical_abbrev":
        medical_abbrev_cache.clear()
        log.info("의학약어 캐시가 클리어되었습니다.")
        return True
    elif cache_type == "query_enhance":
        query_enhance_cache.clear()
        log.info("쿼리 확장 캐시가 클리어되었습니다.")
        return True
    elif cache_type == "bm25":
        bm25_retriever_cache.clear()
        log.info("BM25 검색기 캐시가 클리어되었습니다.")
        return True
    else:
        log.warning(f"알 수 없는 캐시 타입: {cache_type}")
        return False

def cleanup_expired_caches() -> None:
    """만료된 캐시 엔트리들 정리"""
    medical_abbrev_cache._cleanup_expired()
    query_enhance_cache._cleanup_expired()
    bm25_retriever_cache._cleanup_expired()
    log.info("만료된 캐시 엔트리들이 정리되었습니다.")

