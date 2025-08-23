from __future__ import annotations

from typing import Dict, List, Any

from .llm import LLMClient
from .retriever import Retriever
from .utils import has_abbreviation, normalize_text


class Pipeline:
    def __init__(self, data_dir: str | None = None):
        import os
        if not data_dir:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        model = os.getenv("MODEL", "gpt-5-mini")
        require_llm = bool(int(os.getenv("MCPO_REQUIRE_LLM", "0")))
        require_embeddings = bool(int(os.getenv("MCPO_REQUIRE_EMBEDDINGS", "0")))
        warm_embeddings = bool(int(os.getenv("MCPO_WARM_EMBEDDINGS", "0")))
        self.llm = LLMClient(model=model, require_llm=require_llm)
        self.retriever = Retriever(data_dir=data_dir, require_embeddings=require_embeddings)
        if warm_embeddings:
            try:
                self.retriever.warm_embeddings()
            except Exception:
                pass

    def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        q = normalize_text(query)

        # Combined expand + classify in one LLM call when possible
        try:
            combined = self.llm.expand_and_classify(q, max_expansions=3)
            expanded = combined.get("expanded_queries", [q])
            classifications: Dict[str, List[str]] = combined.get("classifications", {q: []})
            external_matches: List[str] = [s for s in (combined.get("external_matches") or []) if isinstance(s, str) and s.strip()]
        except Exception:
            # fallback (should not happen when MCPO_REQUIRE_LLM=1)
            if has_abbreviation(q):
                expanded = self.llm.expand_query(q, max_expansions=3)
            else:
                expanded = [q]
            classifications = {}
            for eq in expanded:
                classifications[eq] = self.llm.classify_need(eq)

        results: Dict[str, List[Dict[str, Any]]] = {
            "pathogen": [],
            "resistance": [],
            "external": [],
        }
        # Batch retrieval per category across queries
        q_embs = self.retriever.embed_queries(expanded)
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
        # If external category is present and LLM provided external_matches, use them for external queries
        try:
            if has_external:
                ext_matches = external_matches if ('external_matches' in locals()) else []
                # Use Chroma-driven override only when using ChromaDB
                if ext_matches and getattr(self.retriever, "_use_chroma")() is True:
                    by_cat["external"] = ext_matches
                    used_external_override = True
                    external_queries = list(ext_matches)
                else:
                    external_queries = list(by_cat.get("external", []))
            else:
                external_queries = []
        except Exception:
            # Graceful: ignore override if anything goes wrong
            external_queries = list(by_cat.get("external", []))
        for c, qlist in by_cat.items():
            if not qlist:
                continue
            kw_priority = bool(c == "external" and used_external_override)
            hits = self.retriever.search_batch(c, qlist, q_embs, top_k=top_k, keyword_priority=kw_priority)
            for text, score, bestq in hits:
                results[c].append({"text": text, "score": score, "query": bestq})

        # Ensure: when using external_matches with embeddings, include top-1 per query
        try:
            if used_external_override and external_queries:
                ensured: Dict[str, Dict[str, Any]] = {}
                for qx in external_queries:
                    top1 = self.retriever.search("external", qx, top_k=1, keyword_priority=True)
                    if top1:
                        t, sc = top1[0]
                        ensured[qx] = {"text": t, "score": sc, "query": qx}
                # Add any missing per-query top1
                have_by_query = {item.get("query"): True for item in results["external"]}
                for qx, item in ensured.items():
                    if not have_by_query.get(qx):
                        results["external"].append(item)
        except Exception:
            pass

        # Deduplicate per-category while keeping highest score
        for c, arr in results.items():
            seen = {}
            for item in arr:
                t = item["text"]
                if t not in seen or item["score"] > seen[t]["score"]:
                    seen[t] = item
            # sort by score desc
            limit = top_k
            if c == "external" and used_external_override and external_queries:
                # Guarantee at least one per query (may exceed top_k if needed)
                try:
                    limit = max(int(top_k), len(external_queries))
                except Exception:
                    limit = top_k
            results[c] = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:limit]

        out = {
            "query": q,
            "expanded_queries": expanded,
            "classifications": classifications,
            "results": results,
        }
        # Attach external_matches for transparency if present
        try:
            if 'external_matches' in locals() and external_matches:
                out["external_matches"] = external_matches
        except Exception:
            pass
        return out
