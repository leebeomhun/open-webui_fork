# rerank_api.py
import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# --- 프롬프트 및 설정 로드 ---
load_dotenv()
log = logging.getLogger("rerank_api")
logging.basicConfig(level=logging.INFO)

# 시스템 프롬프트 로드
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    log.error("system_prompt.txt 파일을 찾을 수 없습니다. 애플리케이션을 종료합니다.")
    exit(1) # 또는 적절한 오류 처리

# Configuration constants from your original script
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
API_TEMPERATURE = float(os.getenv("API_TEMPERATURE", "0"))
SAFE_MAX_DOCS_FOR_RERANKING = int(os.getenv("SAFE_MAX_DOCS_FOR_RERANKING", "20"))
MAX_CONTENT_PREVIEW_LENGTH = int(os.getenv("MAX_CONTENT_PREVIEW_LENGTH", "500"))
INITIAL_SCORE_WEIGHT = float(os.getenv("INITIAL_SCORE_WEIGHT", "0.4"))
LLM_SCORE_WEIGHT = float(os.getenv("LLM_SCORE_WEIGHT", "0.6"))

# Cache configuration
ENABLE_CACHING = os.getenv("ENABLE_RAG_CACHING", "true").lower() == "true"
CACHE_TTL_LLM_RERANK = int(os.getenv("CACHE_TTL_LLM_RERANK", str(7 * 24 * 3600)))  # 7 days
MAX_CACHE_SIZE = int(os.getenv("MAX_RAG_CACHE_SIZE", "10000"))

# --- 캐시 설정 ---
# NOTE: 이 SimpleCache는 단일 프로세스 환경에 적합합니다.
# Gunicorn 등 다중 워커를 사용하는 프로덕션 환경에서는
# 모든 워커가 캐시를 공유할 수 있도록 Redis, Memcached 같은 외부 캐시 사용을 권장합니다.
class InMemoryCache:
    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if not ENABLE_CACHING: return None
        if key in self.cache:
            value, expiry = self.cache[key]
            if datetime.now() < expiry: return value
            else: del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        if not ENABLE_CACHING: return
        if len(self.cache) >= self.max_size: self._cleanup_expired()
        if len(self.cache) >= self.max_size:
            oldest_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])[:len(self.cache) // 4]
            for old_key in oldest_keys: del self.cache[old_key]
        expiry = datetime.now() + timedelta(seconds=ttl_seconds)
        self.cache[key] = (value, expiry)

    def _cleanup_expired(self) -> None:
        now = datetime.now()
        expired_keys = [k for k, (_, expiry) in self.cache.items() if now >= expiry]
        for key in expired_keys: del self.cache[key]


in_memory_cache = InMemoryCache()

def cache_key_hash(data: str) -> str:
    return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]

# --- Pydantic 모델 정의 (API 요청/응답 형식) ---
class RerankRequest(BaseModel):
    combined_results: Dict[str, List[List[Any]]]
    original_query: List[str]
    k: int
    r: float
    api_key: Optional[str] = Field(None, description="API key for LLM services (Gemini/OpenAI)")

# LLM의 구조화된 출력을 위한 Pydantic 모델
class RerankOutput(BaseModel):
    rank: int = Field(description="문서의 새로운 순위")
    doc_num: int = Field(description="원본 문서 번호 (1부터 시작)")
    score: float = Field(description="관련성 점수 (0.0-10.0)")
    reason: str = Field(description="순위 및 점수에 대한 간략한 설명")

class RerankResults(BaseModel):
    reranked_documents: List[RerankOutput] = Field(description="재정렬된 문서 목록")


# --- 리랭킹 관련 함수들 (원본 코드에서 복사 및 수정) ---
def _parse_llm_output(llm_output: str, num_docs: int, log) -> List[Dict[str, Any]]:
    """LLM의 출력을 파싱하여 구조화된 랭킹 정보를 추출합니다."""
    parsed_rankings = []
    try:
        if llm_output.strip().startswith("```json"):
            llm_output = re.sub(r"```json\s*(.*)\s*```", r"\1", llm_output, flags=re.DOTALL)
        data = json.loads(llm_output)
        parsed_rankings = data.get("reranked_documents", [])
        log.info(f"LLM으로부터 구조화된 JSON 응답을 성공적으로 파싱했습니다. {len(parsed_rankings)}개의 문서 순위 확인.")
        return parsed_rankings
    except (json.JSONDecodeError, AttributeError) as e:
        log.warning(f"LLM JSON 출력 파싱 실패: {e}. Regex 기반 파싱으로 대체합니다.")
        log.debug(f"파싱 실패한 LLM 출력: {llm_output}")

    # Regex-based fallback
    for line in llm_output.split('\n'):
        match = re.match(r'(\d+)[.,]\s*(\d+)[.,]\s*(\d+\.?\d*)[.,]?(.*)', line)
        if match:
            rank, doc_num, llm_score, reason = match.groups()
            parsed_rankings.append({
                'rank': int(rank),
                'doc_num': int(doc_num),
                'score': float(llm_score),
                'reason': reason.strip()
            })
    
    if parsed_rankings:
        log.info(f"Regex 기반 파싱으로 {len(parsed_rankings)}개의 순위를 추출했습니다.")
        return parsed_rankings

    # Final fallback: extract document numbers
    log.warning("JSON 및 Regex 파싱 모두 실패. 문서 번호 추출을 시도합니다.")
    doc_nums = re.findall(r'(?:^|\D)(\d+)(?:\D|$)', llm_output)
    if doc_nums:
        seen = set()
        for num_str in doc_nums:
            num = int(num_str)
            if 1 <= num <= num_docs and num not in seen:
                seen.add(num)
                parsed_rankings.append({
                    'rank': len(parsed_rankings) + 1,
                    'doc_num': num,
                    'score': 10.0 - (len(parsed_rankings) * 0.5),
                    'reason': "LLM이 선택한 문서 (대체 파싱)"
                })
        log.info(f"대체 파싱으로 {len(parsed_rankings)}개의 문서 번호를 추출했습니다.")

    return parsed_rankings


def _generate_rerank_log_details(reranked_docs_subset: List[Document], reranking_query_context: str) -> str:
    """리랭킹 결과에 대한 상세 로그 메시지를 생성합니다."""
    rerank_details = []
    for i, doc in enumerate(reranked_docs_subset):
        content_preview = doc.page_content[:30].replace('\n', ' ')
        if len(doc.page_content) > 30:
            content_preview += "..."
        score = doc.metadata.get("score", "N/A")
        normalized_rrf_score = doc.metadata.get("normalized_rrf_score", "N/A")
        llm_score = doc.metadata.get("llm_score", "N/A")
        llm_rank = doc.metadata.get("llm_rank", "N/A")
        llm_reason = doc.metadata.get("llm_reason", "N/A")
        original_index = doc.metadata.get("original_index", "N/A")
        query_info = f" (쿼리 컨텍스트: {reranking_query_context[:30]}...)" if reranking_query_context else ""
        score_info = f"{f'{score:.3f}' if isinstance(score, float) else score}"
        if isinstance(normalized_rrf_score, float) and isinstance(llm_score, float):
            score_info += f" (Norm_RRF: {normalized_rrf_score:.3f}, LLM: {llm_score:.3f})"
        rerank_details.append(
            f"\n  {i+1}. [원래순위: {original_index + 1 if isinstance(original_index, int) else original_index}, "
            f"LLM순위: {llm_rank}, 최종점수: {score_info}] {query_info}\n"
            f"     내용: {content_preview}\n"
            f"     이유: {llm_reason[:80] + '...' if len(llm_reason) > 80 else llm_reason}"
        )
    return "".join(rerank_details)


def parse_and_rerank(
    llm_output: str,
    document_objects: List[Document],
    reranking_query_context: str,
    k: int,
    log
) -> Optional[Dict[str, Any]]:
    """
    LLM 출력을 파싱하고, 점수를 재계산하여 문서를 리랭킹합니다.
    """
    parsed_rankings = _parse_llm_output(llm_output, len(document_objects), log)

    if not parsed_rankings:
        log.warning("LLM 출력 파싱에 완전히 실패하여 리랭킹을 건너뜁니다. 원본 결과를 반환합니다.")
        return None

    reranked_docs = []
    
    # 1. 정규화를 위해 LLM에 전달된 *모든* 문서의 초기 RRF 점수를 수집하고,
    #    원래 순위를 이용해 동점 점수에 대한 tie-breaking을 적용합니다.
    adjusted_initial_scores = [
        doc.metadata.get('initial_score', 0.0) - (doc.metadata.get('original_index', 100) * 1e-9)
        for doc in document_objects
    ]

    # 2. 조정된 점수들로 최대/최소 점수를 찾습니다.
    min_rrf_score = min(adjusted_initial_scores) if adjusted_initial_scores else 0.0
    max_rrf_score = max(adjusted_initial_scores) if adjusted_initial_scores else 1.0
    rrf_score_range = max_rrf_score - min_rrf_score

    # 3. 파싱된 결과에 따라 문서 점수 재계산
    for ranking in sorted(parsed_rankings, key=lambda x: x['rank']):
        doc_idx = ranking['doc_num'] - 1
        if not (0 <= doc_idx < len(document_objects)):
            continue

        doc = document_objects[doc_idx]
        initial_score = doc.metadata.get('initial_score', 0.5)
        original_index = doc.metadata.get('original_index', 100)
        adjusted_initial_score = initial_score - (original_index * 1e-9)
        llm_score = min(ranking['score'] / 10.0, 1.0)

        # 3a. RRF 점수를 0-1 스케일로 정규화합니다.
        if rrf_score_range > 1e-10:
            rank_position = original_index + 1
            total_docs = len(document_objects)
            rank_score = max(0.3, 1.0 - (rank_position - 1) / max(1, total_docs - 1) * 0.7)
            rrf_normalized = 0.2 + 0.6 * (adjusted_initial_score - min_rrf_score) / rrf_score_range
            normalized_rrf_score = 0.4 * rank_score + 0.6 * rrf_normalized
        else:
            rank_position = original_index + 1
            total_docs = len(document_objects)
            normalized_rrf_score = max(0.3, 1.0 - (rank_position - 1) / max(1, total_docs - 1) * 0.7)

        # 3b. 정규화된 RRF 점수와 LLM 점수를 결합하여 최종 점수 계산
        final_score = (normalized_rrf_score * INITIAL_SCORE_WEIGHT) + (llm_score * LLM_SCORE_WEIGHT)

        if doc.metadata is None: doc.metadata = {}
        doc.metadata.update({
            'score': final_score,
            'initial_score': initial_score,
            'normalized_rrf_score': normalized_rrf_score,
            'llm_score': llm_score,
            'llm_reason': ranking['reason'],
            'llm_rank': ranking['rank']
        })
        reranked_docs.append(doc)

    # 4. LLM이 언급하지 않은 문서 처리
    mentioned_indices = {r['doc_num'] - 1 for r in parsed_rankings}
    for i, doc in enumerate(document_objects):
        if i not in mentioned_indices:
            if doc.metadata is None: doc.metadata = {}
            doc.metadata['score'] = 0.3
            doc.metadata['llm_reason'] = "LLM이 관련성이 낮다고 판단한 문서"
            reranked_docs.append(doc)

    # 5. 최종 점수 기준 정렬
    reranked_docs.sort(key=lambda d: d.metadata.get('score', 0.0), reverse=True)
    
    total_reranked_docs = len(reranked_docs)
    final_docs = reranked_docs[:k]

    # 6. 로그 생성 및 출력
    log_details = _generate_rerank_log_details(final_docs, reranking_query_context)
    log.info(f"LLM 리랭킹 완료: 총 {total_reranked_docs}개 문서 중 상위 {len(final_docs)}개 반환 - {log_details}")

    # 7. 최종 결과 포맷팅
    return {
        "distances": [[d.metadata.get("score", 0.5) for d in final_docs]],
        "documents": [[d.page_content for d in final_docs]],
        "metadatas": [[d.metadata for d in final_docs]],
    }


# perform_llm_reranking 함수는 여기에 그대로 복사
async def perform_llm_reranking(
    combined_results: dict,
    original_query: list,
    k: int,
    r: float,
    api_key: str,
    log=None,  # log 인자 추가 (기본값 None)
) -> dict:
    """통합된 검색 결과에 대해 LLM 기반 리랭킹 수행"""
    try:
        if log is None:
            import logging
            log = logging.getLogger("llm_rerank")

        # 캐싱을 위한 키 생성
        docs = combined_results["documents"][0]
        metas = combined_results["metadatas"][0]
        
        # 문서 내용의 해시를 생성 (처음 100자씩만 사용해서 키 길이 제한)
        docs_sample = [doc[:100] for doc in docs[:10]]  # 상위 10개 문서의 처음 100자만
        cache_data = f"llm_rerank:{str(original_query)}:{str(docs_sample)}:{k}:{r}"
        cache_key = cache_key_hash(cache_data)
        
        # 캐시 확인
        cached_result = in_memory_cache.get(cache_key)
        if cached_result is not None:
            log.debug(f"Cache hit for LLM reranking: {cache_key}")
            return cached_result

        log.debug(f"Cache miss for LLM reranking: {cache_key}")
        
        # 필요한 데이터 추출

        max_docs_for_reranking = min(SAFE_MAX_DOCS_FOR_RERANKING, k*3, len(docs))
        log.info(f"리랭킹을 위한 문서 수: {max_docs_for_reranking}개 (요청: {k}, 안전 최대값: {SAFE_MAX_DOCS_FOR_RERANKING})")
        selected_docs = docs[:max_docs_for_reranking]
        selected_metas = metas[:max_docs_for_reranking]

        # Document 객체 생성
        document_objects = []
        initial_scores = combined_results.get("distances", [[]])[0]
        
        for i, (doc_content, meta) in enumerate(zip(selected_docs, selected_metas)):
            if meta is None:
                meta = {}
            meta['original_index'] = i
            
            # RRF 점수를 메타데이터에 추가
            if i < len(initial_scores):
                meta['initial_score'] = initial_scores[i]
            else:
                meta['initial_score'] = 0.5  # Fallback score
                
            document_objects.append(
                Document(
                    page_content=doc_content,
                    metadata=meta
                )
            )

        # Pydantic 모델을 사용하여 JSON 스키마 생성
        schema = RerankResults.model_json_schema()
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            schema=json.dumps(schema, indent=2, ensure_ascii=False)
        )


        reranking_query_context = " | ".join(original_query)
        # 1. 문서 목록 부분을 먼저 만듭니다.
        document_list_str = ""
        for i, doc in enumerate(document_objects):
            content = doc.page_content
            # 문서 내용을 자르는 것은 그대로 유지합니다. (토큰 길이 제한 때문에 중요)
            if len(content) > MAX_CONTENT_PREVIEW_LENGTH:
                content = content[:MAX_CONTENT_PREVIEW_LENGTH] + "..."
            document_list_str += f"문서 {i+1}:\n---\n{content}\n---\n\n"

        # 2. 구조화된 프롬프트를 최종적으로 조합합니다.
        user_prompt =f"""
[쿼리]
{reranking_query_context}

[문서 목록]
{document_list_str}

[작업 지시]
위 [쿼리]와 관련된 [문서 목록]을 당신의 역할과 평가 기준에 따라 분석하여, 관련성이 높은 순서대로 순위를 매겨주세요.
반드시 지정된 '출력 형식'과 JSON 스키마를 정확히 지켜서 답변해야 합니다.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        llm_output = None
        async with httpx.AsyncClient() as client:
            # 1. Gemini API 시도
            gemini_payload = {
                "model": GEMINI_MODEL,
                "reasoning_effort": "none",
                "messages": messages,
                "temperature": API_TEMPERATURE,
                "response_format": {"type": "json_object"},
            }
            gemini_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            llm_output = await _call_llm_api(
                client, 
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", 
                gemini_headers, 
                gemini_payload, 
                "Gemini", 
                log
            )

            # 2. Gemini 실패 시 OpenAI로 폴백
            if llm_output is None and OPENAI_API_KEY:
                log.info("Gemini 호출 실패, OpenAI로 폴백합니다.")
                openai_payload = {
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "temperature": API_TEMPERATURE,
                    "response_format": {"type": "json_object"},
                }
                openai_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
                llm_output = await _call_llm_api(
                    client, 
                    "https://api.openai.com/v1/chat/completions", 
                    openai_headers, 
                    openai_payload, 
                    "OpenAI", 
                    log
                )

        # API 호출이 모두 실패한 경우
        if llm_output is None:
            log.error("모든 LLM API 호출에 실패하여 리랭킹을 건너뜁니다.")
            return combined_results

        # 공통 파싱/리랭킹/로그출력 (이 부분은 동일)
        result = parse_and_rerank(llm_output, document_objects, reranking_query_context, k, log)
        if result is not None:
            # 성공한 결과를 캐시에 저장
            in_memory_cache.set(cache_key, result, CACHE_TTL_LLM_RERANK)
            return result
        else:
            return combined_results
            
    except Exception as e:
        if log:
            log.error(f"LLM 리랭킹 중 예외 발생: {e}", exc_info=True)
        return combined_results


async def _call_llm_api(client: httpx.AsyncClient, url: str, headers: dict, json_payload: dict, model_name: str, log) -> Optional[str]:
    """지정된 LLM API를 호출하고 결과를 반환하는 헬퍼 함수"""
    try:
        log.debug(f"{model_name} API 호출 시도...")
        response = await client.post(url, headers=headers, json=json_payload, timeout=30)
        response.raise_for_status()
        llm_output = response.json()["choices"][0]["message"]["content"].strip()
        log.debug(f"{model_name} 응답 수신 성공")
        return llm_output
    except httpx.RequestError as e:
        log.warning(f"{model_name} API 요청 실패: {e}")
    except httpx.HTTPStatusError as e:
        log.warning(f"{model_name} API가 비정상 상태 코드를 반환했습니다: {e.response.status_code} - {e.response.text}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        log.warning(f"{model_name} API 응답 파싱 중 오류 발생: {e}")
    return None


# --- FastAPI 애플리케이션 정의 ---
app = FastAPI(
    title="LLM Reranking Service",
    description="Provides LLM-based reranking for search results.",
)

@app.post("/rerank", response_model=Dict[str, Any])
async def rerank_endpoint(request: RerankRequest):
    """
    Receives search results and reranks them using an LLM.
    """
    # 요청에서 받은 api_key를 사용하고, 없으면 환경변수의 GEMINI_API_KEY 사용
    api_key_to_use = request.api_key or GEMINI_API_KEY
    if not api_key_to_use:
        raise HTTPException(
            status_code=400,
            detail="API key is missing. Provide it in the request body or set GEMINI_API_KEY environment variable."
        )

    # 문서가 비어 있는지 확인
    if not request.combined_results.get("documents") or not request.combined_results["documents"][0]:
        log.warning("Received rerank request with no documents. Returning original empty results.")
        return request.combined_results

    try:
        reranked_result = await perform_llm_reranking(
            combined_results=request.combined_results,
            original_query=request.original_query,
            k=request.k,
            r=request.r,
            api_key=api_key_to_use,
            log=log,
        )
        return reranked_result
    except Exception as e:
        log.error(f"An error occurred during reranking: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rerank documents due to an internal error: {e}"
        )

# API 서버 실행을 위한 uvicorn 설정
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("RERANK_API_PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)