from __future__ import annotations

from typing import Any, Dict, List, TypedDict
import logging

from .llm import LLMClient, rule_based_classify
from .retriever import Retriever
from .utils import has_abbreviation, normalize_text


class State(TypedDict, total=False):
    query: str
    top_k: int
    expanded_queries: List[str]
    classifications: Dict[str, List[str]]
    results: Dict[str, List[Dict[str, Any]]]
    external_matches: List[str]


class GraphRunner:
    """LangGraph-based execution flow.

    Falls back to direct pipeline logic if langgraph is not installed.
    """

    def __init__(self, data_dir: str | None = None, model: str = "gpt-5-mini"):
        self.logger = logging.getLogger(__name__)
        import os
        if not data_dir:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        require_llm = bool(int(os.getenv("MCPO_REQUIRE_LLM", "0")))
        require_embeddings = bool(int(os.getenv("MCPO_REQUIRE_EMBEDDINGS", "0")))
        warm_embeddings = bool(int(os.getenv("MCPO_WARM_EMBEDDINGS", "0")))
        self.llm = LLMClient(model=model, require_llm=require_llm)
        self.retriever = Retriever(data_dir=data_dir, require_embeddings=require_embeddings)
        # Simple in-memory LRU cache for end-to-end results
        try:
            self.cache_size = int(os.getenv("MCPO_CACHE_SIZE", "128"))
        except Exception:
            self.cache_size = 128
        from collections import OrderedDict
        self._result_cache: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
        if warm_embeddings:
            try:
                self.logger.info("Warming up embeddings at startup...")
                self.retriever.warm_embeddings()
                self.logger.info("Embeddings warmed up")
            except Exception as e:
                self.logger.warning("Embedding warm-up failed: %s", e)
        self._graph = None
        self._langgraph_available = False
        self.logger.info("GraphRunner init (data_dir=%s, model=%s, require_llm=%s, require_embeddings=%s)", data_dir, model, require_llm, require_embeddings)
        self._build_graph_if_possible()

        # External category note (configurable via env or file)
        self.external_note = None
        try:
            self.external_note = os.getenv("EXTERNAL_NOTE")
            note_file = os.getenv("EXTERNAL_NOTE_FILE")
            if (not self.external_note) and note_file:
                try:
                    with open(note_file, "r", encoding="utf-8", errors="ignore") as f:
                        self.external_note = f.read().strip()
                except Exception:
                    pass
            if not self.external_note:
                # Sensible default short hint
                self.external_note = "외인/사고(external) 관련 질의는 KCD 외인 코드 범주에 해당합니다. 상세 기록(사고 유형·장소·기전 등)을 포함해 주시면 더 정확한 분류가 가능합니다."
        except Exception:
            self.external_note = None

    def _build_graph_if_possible(self) -> None:
        try:
            from langgraph.graph import StateGraph, END  # type: ignore

            sg = StateGraph(State)

            def expand_node(state: State) -> State:
                q = normalize_text(state.get("query", ""))
                # Single-call expand + classify to reduce LLM calls
                out = self.llm.expand_and_classify(q, max_expansions=3)
                return {
                    "expanded_queries": out.get("expanded_queries", [q]),
                    "classifications": out.get("classifications", {q: []}),
                    "external_matches": [s for s in (out.get("external_matches") or []) if isinstance(s, str) and s.strip()],
                }

            def classify_node(state: State) -> State:
                # Already classified in expand_node; pass through
                if state.get("classifications"):
                    return {}
                expanded = state.get("expanded_queries", [])
                classifications: Dict[str, List[str]] = {}
                for eq in expanded:
                    classifications[eq] = self.llm.classify_need(eq)
                return {"classifications": classifications}

            def retrieve_node(state: State) -> State:
                top_k = int(state.get("top_k", 5))
                expanded = state.get("expanded_queries", [])
                classifications = state.get("classifications", {})
                ext_matches = state.get("external_matches", []) or []
                results: Dict[str, List[Dict[str, Any]]] = {
                    "pathogen": [],
                    "resistance": [],
                    "external": [],
                }
                # Batch embed queries once
                q_embs = self.retriever.embed_queries(expanded)
                # Build per-category query lists
                by_cat: Dict[str, List[str]] = {"pathogen": [], "resistance": [], "external": []}
                has_external = False
                used_external_override = False
                external_queries: List[str] = []
                for eq in expanded:
                    for c in classifications.get(eq, []):
                        if c in by_cat:
                            by_cat[c].append(eq)
                            if c == "external":
                                has_external = True
                # Override external queries with external_matches when using Chroma
                try:
                    if has_external and ext_matches and getattr(self.retriever, "_use_chroma")() is True:
                        by_cat["external"] = [s for s in ext_matches if isinstance(s, str) and s.strip()]
                        used_external_override = True
                        external_queries = list(by_cat["external"])  # same as ext_matches filtered
                    else:
                        external_queries = list(by_cat.get("external", []))
                except Exception:
                    external_queries = list(by_cat.get("external", []))
                for c, qlist in by_cat.items():
                    if not qlist:
                        continue
                    kw_priority = bool(c == "external" and used_external_override)
                    hits = self.retriever.search_batch(c, qlist, q_embs, top_k=top_k, keyword_priority=kw_priority)
                    for text, score, bestq in hits:
                        results[c].append({"text": text, "score": score, "query": bestq})
                # Ensure per external query top-1 inclusion when using overrides
                try:
                    if used_external_override and external_queries:
                        ensured: Dict[str, Dict[str, Any]] = {}
                        for qx in external_queries:
                            top1 = self.retriever.search("external", qx, top_k=1, keyword_priority=True)
                            if top1:
                                t, sc = top1[0]
                                ensured[qx] = {"text": t, "score": sc, "query": qx}
                        have_by_query = {item.get("query"): True for item in results["external"]}
                        for qx, item in ensured.items():
                            if not have_by_query.get(qx):
                                results["external"].append(item)
                except Exception:
                    pass
                return {"results": results}

            def aggregate_node(state: State) -> State:
                top_k = int(state.get("top_k", 5))
                results = state.get("results", {})
                aggregated: Dict[str, List[Dict[str, Any]]] = {"pathogen": [], "resistance": [], "external": []}
                for c, arr in (results or {}).items():
                    seen: Dict[str, Dict[str, Any]] = {}
                    for item in arr:
                        t = item["text"]
                        if t not in seen or item["score"] > seen[t]["score"]:
                            seen[t] = item
                    limit = top_k
                    try:
                        if c == "external":
                            ext_qs = [s for s in (state.get("external_matches") or []) if isinstance(s, str) and s.strip()]
                            if ext_qs:
                                limit = max(int(top_k), len(ext_qs))
                    except Exception:
                        pass
                    aggregated[c] = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:limit]
                return {"results": aggregated}

            sg.add_node("expand", expand_node)
            sg.add_node("classify", classify_node)
            sg.add_node("retrieve", retrieve_node)
            sg.add_node("aggregate", aggregate_node)

            sg.set_entry_point("expand")
            sg.add_edge("expand", "classify")
            sg.add_edge("classify", "retrieve")
            sg.add_edge("retrieve", "aggregate")
            sg.add_edge("aggregate", END)

            self._graph = sg.compile()
            self._langgraph_available = True
            self.logger.info("LangGraph available: compiled state graph")
        except Exception as e:
            self._graph = None
            self._langgraph_available = False
            self.logger.warning("LangGraph not available, using fallback. error=%s", e)

    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        state: State = {"query": query, "top_k": int(top_k)}
        # Result cache check (normalized query + top_k)
        q_norm = normalize_text(query)
        cache_key = (q_norm, int(top_k))
        if cache_key in self._result_cache:
            res = self._result_cache.pop(cache_key)
            self._result_cache[cache_key] = res
            return res
        if self._langgraph_available and self._graph is not None:
            self.logger.debug("GraphRunner.run using LangGraph")
            out: State = self._graph.invoke(state)  # type: ignore[assignment]
            expanded = out.get("expanded_queries", [])
            classifications = out.get("classifications", {})
            results = out.get("results", {})
            out = {
                "query": query,
                "expanded_queries": expanded,
                "classifications": classifications,
                "results": results,
                "engine": "langgraph",
            }
            # Attach external_matches if present from state
            try:
                if (state.get("external_matches") or []):
                    out["external_matches"] = state.get("external_matches")
            except Exception:
                pass
            # Attach external note if any classification includes 'external'
            try:
                has_external = any("external" in (classifications.get(eq, []) or []) for eq in (expanded or []))
                if has_external and self.external_note:
                    out["external_note"] = self.external_note
            except Exception:
                pass
            # Cache and evict LRU if needed
            self._result_cache[cache_key] = out
            while len(self._result_cache) > max(0, self.cache_size):
                self._result_cache.popitem(last=False)
            return out
        # Fallback to direct logic
        self.logger.debug("GraphRunner.run using fallback pipeline")
        q = q_norm
        # Heuristic fast path: avoid LLM when not required and no abbreviation present
        use_llm = getattr(self.llm, "require_llm", False)
        if not has_abbreviation(q) and not use_llm:
            expanded = [q]
            classifications: Dict[str, List[str]] = {q: rule_based_classify(q)}
        else:
            if has_abbreviation(q):
                expanded = self.llm.expand_query(q, max_expansions=3)
            else:
                expanded = [q]
            classifications = {}
            for eq in expanded:
                classifications[eq] = self.llm.classify_need(eq)
        results: Dict[str, List[Dict[str, Any]]] = {"pathogen": [], "resistance": [], "external": []}
        for eq in expanded:
            cats = self.llm.classify_need(eq)
            classifications[eq] = cats
            for c in cats:
                for text, score in self.retriever.search(c, eq, top_k=top_k):
                    results[c].append({"text": text, "score": score, "query": eq})
        # aggregate
        aggregated: Dict[str, List[Dict[str, Any]]] = {"pathogen": [], "resistance": [], "external": []}
        for c, arr in results.items():
            seen: Dict[str, Dict[str, Any]] = {}
            for item in arr:
                t = item["text"]
                if t not in seen or item["score"] > seen[t]["score"]:
                    seen[t] = item
            aggregated[c] = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        out = {
            "query": q,
            "expanded_queries": expanded,
            "classifications": classifications,
            "results": aggregated,
            "engine": "fallback",
        }
        # Attach external note if any classification includes 'external'
        try:
            has_external = any("external" in (classifications.get(eq, []) or []) for eq in (expanded or []))
            if has_external and self.external_note:
                out["external_note"] = self.external_note
        except Exception:
            pass
        try:
            sizes = {k: len(v or []) for k, v in (aggregated or {}).items()}
            self.logger.debug("Fallback results sizes=%s", sizes)
        except Exception:
            pass
        # Cache and evict LRU if needed
        self._result_cache[cache_key] = out
        while len(self._result_cache) > max(0, self.cache_size):
            self._result_cache.popitem(last=False)
        return out

 
