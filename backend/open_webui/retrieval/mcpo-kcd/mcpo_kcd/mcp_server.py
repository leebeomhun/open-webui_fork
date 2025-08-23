from __future__ import annotations

import os
from typing import Dict, Any, Annotated, Optional
import logging


def _build_runner():
    from .graph import GraphRunner

    # Prefer repo-local data path by default for portability
    default_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    data_dir = os.getenv("DATA_DIR", default_data_dir)
    model = os.getenv("MODEL", "gpt-5-mini")
    return GraphRunner(data_dir=data_dir, model=model)


RUNNER = None


def _run_fastmcp() -> None:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency: mcp. Please install with `pip install mcp`.\n"
            f"Original error: {e}"
        )

    # Load .env if available (CWD and repo-relative)
    try:
        from dotenv import load_dotenv  # type: ignore
        # 1) Try current working directory
        load_dotenv()
        # 2) Try repo root relative to this file
        try:
            from pathlib import Path
            repo_env = Path(__file__).resolve().parents[1] / ".env"
            if repo_env.exists():
                load_dotenv(repo_env)
        except Exception:
            pass
    except Exception:
        pass

    # Configure logging
    level_name = os.getenv("MCPO_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=os.getenv("MCPO_LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger("mcpo_kcd.server")

    app = FastMCP("kcd-mcpo")
    logger.info("Starting FastMCP app: kcd-mcpo")

    # Optional eager GraphRunner init to warm embeddings before first request
    try:
        if bool(int(os.getenv("MCPO_EAGER_INIT", "0"))):
            global RUNNER
            logger.info("Eagerly initializing GraphRunner...")
            RUNNER = _build_runner()
            logger.info("GraphRunner eagerly initialized")
    except Exception as e:
        logger.warning("Eager init skipped due to error: %s", e)

    # Default top_k from environment
    try:
        DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", os.getenv("TOP_K", "3")))
    except Exception:
        DEFAULT_TOP_K = 5

    logger.info("Tool default top_k=%s (from env DEFAULT_TOP_K/TOP_K)", DEFAULT_TOP_K)

    @app.tool(
        name="kcd_query",
        description="모든 쿼리에 대해 이 tool을 실행합니다 (top_k 기본값은 환경변수로 설정 가능)"
    )
    def kcd_query(
        query: Annotated[str, "검색할 원문 질의(약어/의학용어/문장)"] ,
        top_k: Annotated[int, "카테고리별 최대 결과 수"] = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """의학·보건(KCD 맥락) 질의를 LangGraph 파이프라인으로 처리합니다."""
        global RUNNER
        if top_k is None:
            top_k = DEFAULT_TOP_K
        if RUNNER is None:
            logger.info("Initializing GraphRunner (lazy)")
            RUNNER = _build_runner()
            logger.info("GraphRunner ready")
        logger.debug("kcd_query called: top_k=%s, query_len=%d", top_k, len(query or ""))
        try:
            res = RUNNER.run(query, top_k=top_k)
            engine = res.get("engine", "unknown")
            sizes = {k: len(v or []) for k, v in (res.get("results") or {}).items()}
            logger.debug("kcd_query done: engine=%s, results=%s", engine, sizes)
            return res
        except Exception as e:
            # Return structured error instead of raising to avoid HTTP 500
            logger.exception("kcd_query failed: %s", e)
            return {
                "error": "ToolExecutionError",
                "message": str(e),
                "hint": "Check OPENAI_API_KEY/OPENAI_BASE_URL connectivity or relax MCPO_REQUIRE_LLM.",
            }

    app.run()


if __name__ == "__main__":
    _run_fastmcp()
