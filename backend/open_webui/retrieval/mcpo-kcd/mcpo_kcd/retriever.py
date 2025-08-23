from __future__ import annotations

import os
import math
from typing import Dict, List, Tuple, Optional
import logging
from collections import OrderedDict
from .utils import tokenize_ko


class Retriever:
    """Vector-store backed retriever (ChromaDB or Qdrant) with graceful keyword fallback.

    - Indexes the three KCD knowledge text files into the selected vector store (by category) with chunking + overlap.
    - If vector store/embeddings are unavailable and not required, falls back to BM25/keyword scoring.
    """

    def __init__(self, data_dir: str = "/home/ubuntu/open-webui-mcp/mcpo-kcd/data", model_name: str = "dragonkue/BGE-m3-ko", require_embeddings: Optional[bool] = None, context_overlap: Optional[int] = None):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        import os as _os
        self.model_name = _os.getenv("EMBEDDING_MODEL", model_name)
        self.require_embeddings = bool(int(_os.getenv("MCPO_REQUIRE_EMBEDDINGS", "0"))) if require_embeddings is None else require_embeddings
        try:
            # Only used in fallback mode
            self.context_overlap = int(_os.getenv("MCPO_CONTEXT_OVERLAP", "1")) if context_overlap is None else int(context_overlap)
        except Exception:
            self.context_overlap = 1

        # Vector store selection
        self.vector_store = (os.getenv("VECTOR_STORE", "chroma") or "chroma").lower()
        # Backwards-compat: USE_CHROMA still controls enabling vector store
        self.use_chroma = bool(int(os.getenv("USE_CHROMA", "1")))
        self.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", os.path.join(self.data_dir, "chroma"))
        # Qdrant config
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        try:
            self.chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        except Exception:
            self.chunk_size = 800
        try:
            self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
        except Exception:
            self.chunk_overlap = 150
        self.chroma_rebuild = bool(int(os.getenv("CHROMA_REBUILD", "0")))
        try:
            self.embedding_cache_size = int(os.getenv("EMBEDDING_CACHE_SIZE", "256"))
        except Exception:
            self.embedding_cache_size = 256
        # Hybrid search controls
        self.enable_hybrid = bool(int(os.getenv("ENABLE_HYBRID", "1")))
        try:
            self.hybrid_alpha = float(os.getenv("HYBRID_ALPHA", "0.6"))  # weight for vector side
        except Exception:
            self.hybrid_alpha = 0.6
        # Fusion strategy: 'linear' (default) or 'rrf'
        self.hybrid_fusion = (os.getenv("HYBRID_FUSION", "linear") or "linear").lower()
        try:
            self.rrf_k = int(os.getenv("RRF_K", "60"))
        except Exception:
            self.rrf_k = 60
        # When keyword-priority is requested (e.g., for external_matches), bias fusion toward BM25 by lowering alpha
        try:
            self.keyword_priority_alpha = float(os.getenv("KEYWORD_PRIORITY_ALPHA", "0.25"))
        except Exception:
            self.keyword_priority_alpha = 0.25

        # Fallback corpus (keyword)
        self.corpora: Dict[str, List[str]] = {"pathogen": [], "resistance": [], "external": []}

        # Embedding-based fast path via vector store
        self._chroma_client = None  # lazy (actual client for Chroma)
        self._vs_client = None      # unified client (Chroma or Qdrant adapter)
        self._chroma_collections: Dict[str, Optional[object]] = {"pathogen": None, "resistance": None, "external": None}
        self._embedding_fn = None
        self._embed_cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._manual_doc_embed = False  # when True, precompute doc embeddings and pass to Chroma
        # BM25 indices over chunks (for hybrid / fallback)
        self._bm25_index: Dict[str, Optional[object]] = {"pathogen": None, "resistance": None, "external": None}
        self._bm25_docs: Dict[str, List[str]] = {"pathogen": [], "resistance": [], "external": []}
        # Lazy build toggle for BM25 indices (reduces startup time)
        self.bm25_lazy = bool(int(os.getenv("BM25_LAZY", "1")))

        # Build vector store if possible; else load fallback corpus
        init_ok = False
        if self.use_chroma:
            if self.vector_store == "qdrant":
                init_ok = self._init_qdrant()
            else:
                init_ok = self._init_chroma()
        if init_ok:
            try:
                self._build_or_load_collections()
                # Build BM25 corpora to enable hybrid search
                if not self.bm25_lazy:
                    self._build_bm25_corpora()
                if self.vector_store == "qdrant":
                    self.logger.info(
                        "Retriever init with Qdrant (url=%s, model=%s, chunk=%d, overlap=%d)",
                        self.qdrant_url,
                        self.model_name,
                        self.chunk_size,
                        self.chunk_overlap,
                    )
                else:
                    self.logger.info(
                        "Retriever init with Chroma (dir=%s, model=%s, chunk=%d, overlap=%d)",
                        self.chroma_persist_dir,
                        self.model_name,
                        self.chunk_size,
                        self.chunk_overlap,
                    )
            except Exception as e:
                self.logger.warning("Vector-store init failed; attempting BM25 or keyword fallback. error=%s", e)
                try:
                    self._build_bm25_corpora()
                except Exception:
                    self._load_corpora()
        else:
            # No Chroma: try BM25 from chunks; else keyword lines
            try:
                if not self.bm25_lazy:
                    self._build_bm25_corpora()
            except Exception as e:
                self.logger.warning("BM25 init failed; using keyword fallback. error=%s", e)
                self._load_corpora()
            self.logger.info("Retriever init without vector store (BM25 or keyword)")

    # -----------------------
    # Vector store setup: ChromaDB
    # -----------------------
    def _init_chroma(self) -> bool:
        try:
            import chromadb  # type: ignore
            from chromadb.utils import embedding_functions  # type: ignore
            # Select embedding device (GPU/CPU)
            device_env = os.getenv("EMBEDDING_DEVICE", "auto") or "auto"
            device = None
            if device_env == "auto":
                try:
                    import torch  # type: ignore
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            else:
                device = device_env  # e.g., "cuda", "cuda:0", or "cpu"

            # Construct embedding function; pass device if supported
            try:
                self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.model_name,
                    device=device,
                )
            except TypeError:
                # Older Chroma may not support device kwarg
                self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.model_name
                )
            self.logger.info("Embedding function initialized (model=%s, device=%s)", self.model_name, device)
            # Persistent client
            self._chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)
            # Use Chroma client as the unified client
            self._vs_client = self._chroma_client
            return True
        except Exception as e:
            self._chroma_client = None
            self._vs_client = None
            self._embedding_fn = None
            if self.require_embeddings:
                raise RuntimeError(f"Chroma/embeddings required but unavailable: {e}")
            self.logger.warning("Chroma not available; will use keyword fallback. error=%s", e)
            return False

    # Vector store setup: Qdrant
    def _init_qdrant(self) -> bool:
        try:
            # Embedding function (same as Chroma path)
            device_env = os.getenv("EMBEDDING_DEVICE", "auto") or "auto"
            device = None
            if device_env == "auto":
                try:
                    import torch  # type: ignore
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            else:
                device = device_env
            try:
                from chromadb.utils import embedding_functions  # type: ignore
                self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.model_name,
                    device=device,
                )
            except Exception:
                # Fallback to sentence-transformers directly if chroma not present
                from sentence_transformers import SentenceTransformer  # type: ignore
                model = SentenceTransformer(self.model_name, device=device)

                def _embed(texts: List[str]):
                    return model.encode(texts, normalize_embeddings=True).tolist()

                self._embedding_fn = _embed  # type: ignore
            self.logger.info("Embedding function initialized (model=%s, device=%s)", self.model_name, device)

            # Qdrant client + adapter
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.models import Distance, VectorParams  # type: ignore

            client = QdrantClient(url=self.qdrant_url, api_key=(self.qdrant_api_key or None))

            class _QdrantCollection:
                def __init__(self, client: QdrantClient, name: str, embedding_fn):
                    self.client = client
                    self.name = name
                    self.embedding_fn = embedding_fn

                def count(self) -> int:
                    try:
                        res = self.client.count(self.name, exact=True)
                        return int(getattr(res, "count", 0) or 0)
                    except Exception:
                        return 0

                def add(self, ids: List[str], documents: List[str], metadatas: List[dict], embeddings: Optional[List[List[float]]] = None):
                    from qdrant_client.models import PointStruct  # type: ignore
                    # Qdrant requires vectors; embed if not provided
                    vecs = embeddings
                    if vecs is None and self.embedding_fn is not None:
                        vecs = self.embedding_fn(documents)
                    if vecs is None:
                        raise RuntimeError("Qdrant add requires embeddings or an embedding function")
                    points = []
                    for i, pid in enumerate(ids):
                        payload = {"document": documents[i]}
                        if metadatas and i < len(metadatas) and isinstance(metadatas[i], dict):
                            payload.update(metadatas[i])
                        points.append(PointStruct(id=pid, vector=vecs[i], payload=payload))
                    self.client.upsert(collection_name=self.name, points=points)

                def query(self, query_texts: Optional[List[str]] = None, query_embeddings: Optional[List[List[float]]] = None, n_results: int = 3, include: Optional[List[str]] = None):
                    if query_embeddings is None:
                        if query_texts is None:
                            raise RuntimeError("Qdrant query requires query_embeddings or query_texts")
                        if self.embedding_fn is None:
                            raise RuntimeError("Qdrant query cannot embed query_texts without embedding function")
                        query_embeddings = self.embedding_fn(query_texts)
                    docs_all: List[List[str]] = []
                    dists_all: List[List[float]] = []
                    for qv in query_embeddings or []:
                        hits = self.client.search(collection_name=self.name, query_vector=qv, limit=max(1, int(n_results)), with_payload=True)
                        docs: List[str] = []
                        dists: List[float] = []
                        for h in hits:
                            # Qdrant score is similarity (higher better) for cosine; convert to distance to match Chroma contract
                            sim = float(getattr(h, "score", 0.0) or 0.0)
                            dist = max(0.0, min(1.0, 1.0 - sim))
                            payload = getattr(h, "payload", {}) or {}
                            docs.append(str(payload.get("document", "")))
                            dists.append(dist)
                        docs_all.append(docs)
                        dists_all.append(dists)
                    return {"documents": docs_all, "distances": dists_all}

            class _QdrantClientAdapter:
                def __init__(self, client: QdrantClient):
                    self.client = client

                def delete_collection(self, name: str):
                    try:
                        self.client.delete_collection(name)
                    except Exception:
                        pass

                def get_or_create_collection(self, name: str, metadata: Optional[dict] = None, embedding_function=None):
                    # Determine vector dimension robustly (env override > probe)
                    dim: Optional[int] = None
                    try:
                        import os as _os  # local import
                        dim_env = _os.getenv("QDRANT_VECTOR_SIZE") or _os.getenv("EMBEDDING_DIM")
                        if dim_env:
                            dim = int(dim_env)
                    except Exception:
                        dim = None
                    if embedding_function is not None:
                        try:
                            test = embedding_function(["dim probe", "dim probe 2"])  # two inputs to favor 2D output
                            try:
                                import numpy as np  # type: ignore
                            except Exception:  # pragma: no cover
                                np = None  # type: ignore
                            if test is not None:
                                # numpy array
                                if 'np' in locals() and np is not None and hasattr(test, 'shape'):
                                    shape = getattr(test, 'shape', None)
                                    if shape is not None and len(shape) == 2:
                                        dim = int(shape[1])
                                    elif shape is not None and len(shape) == 1:
                                        dim = int(shape[0])
                                # list-like
                                if dim is None and isinstance(test, list) and test:
                                    first = test[0]
                                    # numpy vector
                                    if hasattr(first, 'shape'):
                                        try:
                                            shape1 = getattr(first, 'shape', None)
                                            if shape1 is not None and len(shape1) == 1:
                                                dim = int(shape1[0])
                                        except Exception:
                                            pass
                                    # python list/tuple vector
                                    if dim is None and isinstance(first, (list, tuple)) and len(first) > 0:
                                        dim = len(first)
                                    # degenerate 1D case
                                    if dim is None and isinstance(first, float):
                                        dim = len(test)
                        except Exception:
                            dim = None
                    try:
                        # If exists, this will succeed
                        self.client.get_collection(name)
                    except Exception:
                        if dim is None:
                            raise RuntimeError("Cannot determine embedding dimension for Qdrant collection creation")
                        self.client.create_collection(name, vectors_config=VectorParams(size=int(dim), distance=Distance.COSINE))
                    return _QdrantCollection(self.client, name, embedding_function)

            # Expose adapter as unified client
            self._vs_client = _QdrantClientAdapter(client)
            # In Qdrant path we always embed documents manually
            self._manual_doc_embed = True
            return True
        except Exception as e:
            self._vs_client = None
            self._embedding_fn = None
            if self.require_embeddings:
                raise RuntimeError(f"Qdrant/embeddings required but unavailable: {e}")
            self.logger.warning("Qdrant not available; will use keyword fallback. error=%s", e)
            return False

    @staticmethod
    def _read_file(path: str) -> str:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        if not text:
            return []
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        n = len(text)
        if n <= chunk_size:
            return [text.strip()]
        step = max(1, chunk_size - max(0, chunk_overlap))
        chunks: List[str] = []
        start = 0
        while start < n:
            end = min(n, start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            start = end - chunk_overlap
            if start < 0:
                start = 0
        return chunks

    @staticmethod
    def _chunk_text_external(text: str) -> List[str]:
        """Special chunking for external category: split by sections starting with "[제목]".

        A new chunk begins whenever a line starts with "[제목]". Content from that line
        up to the line before the next "[제목]" (or EOF) forms one chunk.
        Leading content before the first "[제목]" is ignored.
        """
        if not text:
            return []
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        chunks: List[str] = []
        buf: List[str] = []
        import re
        is_title = re.compile(r"^\s*\[제목\]")
        for line in lines:
            if is_title.match(line):
                if buf:
                    chunk = "\n".join(buf).strip()
                    if chunk:
                        chunks.append(chunk)
                    buf = []
                buf.append(line)
            else:
                if buf:
                    buf.append(line)
                else:
                    # Ignore preamble before first title
                    continue
        if buf:
            chunk = "\n".join(buf).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _build_or_load_collections(self) -> None:
        if not self._vs_client or not self._embedding_fn:
            raise RuntimeError("Vector-store client not initialized")

        files = {
            "pathogen": os.path.join(self.data_dir, "kcd_kb_pathogen.txt"),
            "resistance": os.path.join(self.data_dir, "kcd_kb_resistance.txt"),
            "external": os.path.join(self.data_dir, "kcd_kb_external.txt"),
        }

        for cat, path in files.items():
            name = f"kcd_{cat}"
            # Recreate collection if rebuild requested
            if self.chroma_rebuild:
                try:
                    self._vs_client.delete_collection(name)  # type: ignore[attr-defined]
                except Exception:
                    pass

            # Create/get collection; try with embedding_function first, then fallback without
            coll = None
            try:
                coll = self._vs_client.get_or_create_collection(  # type: ignore[attr-defined]
                    name=name,
                    metadata={"source": os.path.basename(path)},
                    embedding_function=self._embedding_fn,
                )
            except Exception as e:
                self.logger.warning("Collection init without embedding_function due to error: %s", e)
                coll = self._vs_client.get_or_create_collection(  # type: ignore[attr-defined]
                    name=name,
                    metadata={"source": os.path.basename(path)},
                )
                self._manual_doc_embed = True

            # If collection is empty, (re)build from file
            count = 0
            try:
                count = coll.count()  # type: ignore[attr-defined]
            except Exception:
                count = 0

            if count == 0:
                text = self._read_file(path)
                if cat == "external":
                    chunks = self._chunk_text_external(text)
                else:
                    chunks = self._chunk_text(text, self.chunk_size, self.chunk_overlap)
                if not chunks:
                    self.logger.warning("No chunks built for %s (file missing or empty)", cat)
                # IDs: Qdrant requires integer or UUID; use integers for Qdrant
                if getattr(self, 'vector_store', 'chroma') == 'qdrant':
                    ids = list(range(len(chunks)))
                else:
                    ids = [f"{cat}:{i}" for i in range(len(chunks))]
                metadatas = [{"category": cat, "file": os.path.basename(path), "chunk_index": i} for i in range(len(chunks))]
                # Add in manageable batches
                B = 200
                for i in range(0, len(chunks), B):
                    if self._manual_doc_embed and self._embedding_fn is not None:
                        try:
                            vecs = self._embedding_fn(chunks[i:i+B])
                        except Exception as e:
                            self.logger.warning("Manual embedding failed; adding without embeddings. error=%s", e)
                            vecs = None
                        if vecs is not None:
                            coll.add(ids=ids[i:i+B], embeddings=vecs, documents=chunks[i:i+B], metadatas=metadatas[i:i+B])  # type: ignore[attr-defined]
                        else:
                            coll.add(ids=ids[i:i+B], documents=chunks[i:i+B], metadatas=metadatas[i:i+B])  # type: ignore[attr-defined]
                    else:
                        coll.add(ids=ids[i:i+B], documents=chunks[i:i+B], metadatas=metadatas[i:i+B])  # type: ignore[attr-defined]
                self.logger.info("Built collection '%s' with %d chunks", name, len(chunks))
            else:
                self.logger.info("Using existing collection '%s' (count=%d)", name, count)

            self._chroma_collections[cat] = coll

    def _build_bm25_corpora(self) -> None:
        """Build BM25 indices on chunked documents per category."""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except Exception as e:
            # Disable hybrid if BM25 not available
            self.enable_hybrid = False
            raise RuntimeError(f"rank-bm25 is required for BM25 indexing: {e}")

        files = {
            "pathogen": os.path.join(self.data_dir, "kcd_kb_pathogen.txt"),
            "resistance": os.path.join(self.data_dir, "kcd_kb_resistance.txt"),
            "external": os.path.join(self.data_dir, "kcd_kb_external.txt"),
        }
        built: Dict[str, int] = {}
        for cat, path in files.items():
            text = self._read_file(path)
            if cat == "external":
                chunks = self._chunk_text_external(text)
            else:
                chunks = self._chunk_text(text, self.chunk_size, self.chunk_overlap)
            self._bm25_docs[cat] = chunks
            tokenized = [tokenize_ko(c) for c in chunks]
            if tokenized:
                self._bm25_index[cat] = BM25Okapi(tokenized)
            else:
                self._bm25_index[cat] = None
            built[cat] = len(chunks)
        self.logger.info("BM25 corpora built: %s", built)

    # -----------------------
    # Fusion helpers
    # -----------------------
    def _fuse_linear(self, vec_best: Dict[str, float], bm_best: Dict[str, float], alpha: Optional[float] = None) -> List[Tuple[str, float]]:
        if alpha is None:
            alpha = self.hybrid_alpha
        texts = set(vec_best) | set(bm_best)
        out = []
        for t in texts:
            s = alpha * vec_best.get(t, 0.0) + (1.0 - alpha) * bm_best.get(t, 0.0)
            out.append((t, float(s)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _fuse_rrf(self, vec_rank: Dict[str, int], bm_rank: Dict[str, int], alpha: Optional[float] = None) -> List[Tuple[str, float]]:
        # Reciprocal Rank Fusion with weighting
        k = max(1, int(self.rrf_k))
        if alpha is None:
            alpha = self.hybrid_alpha
        texts = set(vec_rank) | set(bm_rank)
        out = []
        for t in texts:
            s = 0.0
            if t in vec_rank:
                s += alpha * (1.0 / (k + vec_rank[t]))
            if t in bm_rank:
                s += (1.0 - alpha) * (1.0 / (k + bm_rank[t]))
            out.append((t, float(s)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _ensure_bm25(self) -> None:
        """Build BM25 indices on first use when lazy mode is enabled."""
        if not self.bm25_lazy:
            return
        if all(self._bm25_index.get(c) is not None for c in ("pathogen", "resistance", "external")):
            return
        try:
            self._build_bm25_corpora()
        except Exception as e:
            self.logger.warning("BM25 lazy build failed: %s", e)
            if not any(self.corpora.values()):
                self._load_corpora()

    # -----------------------
    # Query embedding helpers (with LRU cache)
    # -----------------------
    def _embed_queries_cached(self, queries: List[str]) -> Optional[List[List[float]]]:
        if not self._embedding_fn:
            return None
        # Gather cache hits and misses
        out: List[Optional[List[float]]] = [None] * len(queries)
        misses: List[str] = []
        miss_idx: List[int] = []
        for i, q in enumerate(queries):
            if q in self._embed_cache:
                # move to end (recently used)
                vec = self._embed_cache.pop(q)
                self._embed_cache[q] = vec
                out[i] = vec
            else:
                misses.append(q)
                miss_idx.append(i)
        if misses:
            try:
                vecs = self._embedding_fn(misses)
            except Exception as e:
                self.logger.warning("Embedding failed; cannot precompute query vectors. error=%s", e)
                return None
            for j, v in enumerate(vecs or []):
                i = miss_idx[j]
                out[i] = v
                # push to cache
                self._embed_cache[misses[j]] = v
                # evict LRU if needed
                while len(self._embed_cache) > max(0, self.embedding_cache_size):
                    self._embed_cache.popitem(last=False)
        # Validate
        final: List[List[float]] = []
        for v in out:
            if v is None:
                return None
            final.append(v)
        return final

    # -----------------------
    # Fallback corpus/keyword scoring
    # -----------------------
    def _load_corpora(self) -> None:
        files = {
            "pathogen": os.path.join(self.data_dir, "kcd_kb_pathogen.txt"),
            "resistance": os.path.join(self.data_dir, "kcd_kb_resistance.txt"),
            "external": os.path.join(self.data_dir, "kcd_kb_external.txt"),
        }
        for key, path in files.items():
            items: List[str] = []
            if os.path.exists(path):
                text = self._read_file(path)
                if key == "external":
                    # Use section-based chunks for external in keyword fallback too
                    items = self._chunk_text_external(text)
                else:
                    # Line-based items for non-external
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            s = line.strip()
                            if s:
                                items.append(s)
            self.corpora[key] = items
        try:
            sizes = {k: len(v) for k, v in self.corpora.items()}
            self.logger.info("Loaded corpora sizes=%s", sizes)
        except Exception:
            pass

    @staticmethod
    def _make_snippet(items: List[str], idx: int, overlap: int) -> str:
        start = max(0, idx - overlap)
        end = min(len(items), idx + overlap + 1)
        return "\n".join(items[start:end])

    @staticmethod
    def _keyword_score(query: str, text: str) -> float:
        # Very simple token overlap score
        import re

        def toks(s: str) -> List[str]:
            s = s.lower()
            s = re.sub(r"[^0-9a-z가-힣 ]+", " ", s)
            return [t for t in s.split() if len(t) > 1]

        q = set(toks(query))
        t = toks(text)
        if not q or not t:
            return 0.0
        overlap = sum(1 for tok in t if tok in q)
        return overlap / math.sqrt(len(t))

    # -----------------------
    # Public API
    # -----------------------
    def warm_embeddings(self) -> None:
        """No-op when using a persisted vector store. Left for API compatibility."""
        return

    def _use_chroma(self) -> bool:
        return (
            self._vs_client is not None
            and self.use_chroma
            and all(self._chroma_collections.get(c) is not None for c in ("pathogen", "resistance", "external"))
        )

    def search(self, cat: str, query: str, top_k: int = 3, keyword_priority: bool = False) -> List[Tuple[str, float]]:
        cat = {"pathogen", "resistance", "external"}.intersection({cat})
        if not cat:
            return []
        cat = next(iter(cat))

        # Hybrid (vector + BM25) when available
        if self._use_chroma() and self.enable_hybrid:
            self._ensure_bm25()
            if self._bm25_index.get(cat) is not None:
                vec_candidates: List[Tuple[str, float]] = []
                bm25_candidates: List[Tuple[str, float]] = []
                # Vector side
                coll = self._chroma_collections[cat]
                try:
                    q_emb = self._embed_queries_cached([query])
                    if q_emb is not None:
                        res = coll.query(query_embeddings=q_emb, n_results=max(1, int(top_k)*3), include=["documents", "distances"])  # type: ignore[attr-defined]
                    else:
                        res = coll.query(query_texts=[query], n_results=max(1, int(top_k)*3), include=["documents", "distances"])  # type: ignore[attr-defined]
                    docs = (res.get("documents") or [[]])[0]
                    dists = (res.get("distances") or [[]])[0]
                    for i, doc in enumerate(docs or []):
                        if doc is None:
                            continue
                        dist = float(dists[i]) if dists and i < len(dists) else 1.0
                        sim = max(0.0, min(1.0, 1.0 - dist))
                        vec_candidates.append((doc, sim))
                except Exception as e:
                    self.logger.warning("Hybrid: vector query failed; skipping vector side. error=%s", e)
                # BM25 side
                try:
                    idx = self._bm25_index.get(cat)
                    docs_bm = self._bm25_docs.get(cat) or []
                    if idx is not None and docs_bm:
                        scores = idx.get_scores(tokenize_ko(query))  # type: ignore[attr-defined]
                        pairs = list(enumerate(scores))
                        pairs.sort(key=lambda x: x[1], reverse=True)
                        pairs = pairs[:max(1, int(top_k)*3)]
                        vals = [p[1] for p in pairs]
                        if vals:
                            mn, mx = min(vals), max(vals)
                            for i, sc in pairs:
                                norm = 0.0 if mx == mn else (sc - mn) / (mx - mn)
                                bm25_candidates.append((docs_bm[i], norm))
                except Exception as e:
                    self.logger.warning("Hybrid: BM25 scoring failed; skipping BM25 side. error=%s", e)
                # Combine scores (linear or RRF)
                vec_best: Dict[str, float] = {}
                for t, s in vec_candidates:
                    if s > vec_best.get(t, 0.0):
                        vec_best[t] = s
                bm_best: Dict[str, float] = {}
                for t, s in bm25_candidates:
                    if s > bm_best.get(t, 0.0):
                        bm_best[t] = s
                eff_alpha = self.keyword_priority_alpha if keyword_priority else None
                if self.hybrid_fusion == "rrf":
                    # Build ranks (1-based)
                    vec_sorted = sorted(vec_best.items(), key=lambda x: x[1], reverse=True)
                    bm_sorted = sorted(bm_best.items(), key=lambda x: x[1], reverse=True)
                    vec_rank = {t: i + 1 for i, (t, _) in enumerate(vec_sorted)}
                    bm_rank = {t: i + 1 for i, (t, _) in enumerate(bm_sorted)}
                    fused = self._fuse_rrf(vec_rank, bm_rank, alpha=eff_alpha)
                    # Normalize to [0,1] where best possible = 1/(k+1)
                    scale = float(self.rrf_k + 1)
                    fused = [(t, min(1.0, s * scale)) for t, s in fused]
                else:
                    fused = self._fuse_linear(vec_best, bm_best, alpha=eff_alpha)
                return fused[:top_k]

        if self._use_chroma():
            coll = self._chroma_collections[cat]
            try:
                q_emb = self._embed_queries_cached([query])
                if q_emb is not None:
                    res = coll.query(query_embeddings=q_emb, n_results=max(1, int(top_k)), include=["documents", "distances"])  # type: ignore[attr-defined]
                else:
                    res = coll.query(query_texts=[query], n_results=max(1, int(top_k)), include=["documents", "distances"])  # type: ignore[attr-defined]
                docs = (res.get("documents") or [[]])[0]
                scores = (res.get("distances") or [[]])[0] or []
                # Distances are smaller=closer; convert to similarity
                out: List[Tuple[str, float]] = []
                for i, doc in enumerate(docs):
                    if doc is None:
                        continue
                    dist = float(scores[i]) if i < len(scores) else 1.0
                    sim = max(0.0, min(1.0, 1.0 - dist))
                    out.append((doc, sim))
                return out
            except Exception as e:
                self.logger.warning("Vector query failed; falling back to keyword. error=%s", e)

        # Prefer BM25 when available
        self._ensure_bm25()
        idx = self._bm25_index.get(cat)
        docs_bm = self._bm25_docs.get(cat) or []
        if idx is not None and docs_bm:
            try:
                scores = idx.get_scores(tokenize_ko(query))  # type: ignore[attr-defined]
                pairs = list(enumerate(scores))
                pairs.sort(key=lambda x: x[1], reverse=True)
                pairs = [p for p in pairs if p[1] > 0][:top_k]
                vals = [p[1] for p in pairs]
                mn, mx = (min(vals), max(vals)) if vals else (0.0, 0.0)
                out_bm = []
                for i, sc in pairs:
                    norm = 0.0 if mx == mn else (sc - mn) / (mx - mn)
                    out_bm.append((docs_bm[i], float(norm)))
                return out_bm
            except Exception:
                pass

        # Fallback: keyword scoring over line corpus
        items = self.corpora.get(cat) or []
        if not items:
            return []
        scores = [self._keyword_score(query, t) for t in items]
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out_kw = [(self._make_snippet(items, i, self.context_overlap), float(scores[i])) for i in idxs if scores[i] > 0]
        return out_kw

    # Batch search helpers
    def embed_queries(self, queries: List[str]) -> Optional[List[List[float]]]:
        # When using Chroma, pre-embed once and reuse across categories
        if self._use_chroma():
            return self._embed_queries_cached(queries)
        return None

    def search_batch(self, cat: str, queries: List[str], q_embs: Optional[List[List[float]]], top_k: int = 5, keyword_priority: bool = False):
        cat = {"pathogen", "resistance", "external"}.intersection({cat})
        if not cat:
            return []
        cat = next(iter(cat))

        # Hybrid batch mode
        if self._use_chroma() and self.enable_hybrid:
            self._ensure_bm25()
            if self._bm25_index.get(cat) is not None:
                coll = self._chroma_collections[cat]
                # Collect per-modality best scores and their source queries
                vec_best: Dict[str, Tuple[float, str]] = {}
                bm_best: Dict[str, Tuple[float, str]] = {}
                try:
                    if q_embs is None or (isinstance(q_embs, list) and len(q_embs) != len(queries)):
                        q_embs = self._embed_queries_cached(queries)
                    if q_embs is not None:
                        res = coll.query(query_embeddings=q_embs, n_results=max(1, int(top_k)*3), include=["documents", "distances"])  # type: ignore[attr-defined]
                    else:
                        res = coll.query(query_texts=queries, n_results=max(1, int(top_k)*3), include=["documents", "distances"])  # type: ignore[attr-defined]
                    all_docs = res.get("documents") or []
                    all_dists = res.get("distances") or []
                    for qi, (docs, dists) in enumerate(zip(all_docs, all_dists)):
                        for i, doc in enumerate(docs or []):
                            dist = float(dists[i]) if dists and i < len(dists) else 1.0
                            sim = max(0.0, min(1.0, 1.0 - dist))
                            prev = vec_best.get(doc)
                            if prev is None or sim > prev[0]:
                                vec_best[doc] = (sim, queries[qi])
                except Exception as e:
                    self.logger.warning("Hybrid: vector batch failed; skipping vector side. error=%s", e)
                # BM25 side per query
                idx = self._bm25_index.get(cat)
                docs_bm = self._bm25_docs.get(cat) or []
                if idx is not None and docs_bm:
                    for q in queries:
                        try:
                            scores = idx.get_scores(tokenize_ko(q))  # type: ignore[attr-defined]
                            pairs = list(enumerate(scores))
                            pairs.sort(key=lambda x: x[1], reverse=True)
                            pairs = pairs[:max(1, int(top_k)*3)]
                            vals = [p[1] for p in pairs]
                            if vals:
                                mn, mx = min(vals), max(vals)
                                for i, sc in pairs:
                                    norm = 0.0 if mx == mn else (sc - mn) / (mx - mn)
                                    t = docs_bm[i]
                                    prev = bm_best.get(t)
                                    if prev is None or norm > prev[0]:
                                        bm_best[t] = (norm, q)
                        except Exception:
                            continue
                # Fuse modalities
                eff_alpha = self.keyword_priority_alpha if keyword_priority else None
                if self.hybrid_fusion == "rrf":
                    # Build ranks (1-based)
                    vec_sorted = sorted(((t, s_q[0]) for t, s_q in vec_best.items()), key=lambda x: x[1], reverse=True)
                    bm_sorted = sorted(((t, s_q[0]) for t, s_q in bm_best.items()), key=lambda x: x[1], reverse=True)
                    vec_rank = {t: i + 1 for i, (t, _) in enumerate(vec_sorted)}
                    bm_rank = {t: i + 1 for i, (t, _) in enumerate(bm_sorted)}
                    fused = self._fuse_rrf(vec_rank, bm_rank, alpha=eff_alpha)
                    # Normalize to [0,1] where best possible = 1/(k+1)
                    scale = float(self.rrf_k + 1)
                    fused = [(t, min(1.0, s * scale)) for t, s in fused]
                    # Pick a representative query per text from the better-ranked modality
                    out = []
                    for t, sc in fused[:top_k]:
                        vq = vec_best.get(t)
                        bq = bm_best.get(t)
                        q = vq[1] if (vq and (not bq or vec_rank.get(t, 1e9) <= bm_rank.get(t, 1e9))) else (bq[1] if bq else (vq[1] if vq else queries[0]))
                        out.append((t, float(sc), q))
                    return out
                else:
                    # Linear fusion
                    vec_only = {t: s_q[0] for t, s_q in vec_best.items()}
                    bm_only = {t: s_q[0] for t, s_q in bm_best.items()}
                    fused = self._fuse_linear(vec_only, bm_only, alpha=eff_alpha)
                    out = []
                    for t, sc in fused[:top_k]:
                        vq = vec_best.get(t)
                        bq = bm_best.get(t)
                        # Choose the query from the stronger contributing modality
                        use_alpha = eff_alpha if eff_alpha is not None else self.hybrid_alpha
                        v_part = use_alpha * (vq[0] if vq else 0.0)
                        b_part = (1.0 - use_alpha) * (bq[0] if bq else 0.0)
                        q = (vq[1] if v_part >= b_part else (bq[1] if bq else (vq[1] if vq else queries[0])))
                        out.append((t, float(sc), q))
                    return out

        if self._use_chroma():
            coll = self._chroma_collections[cat]
            try:
                if q_embs is None or (isinstance(q_embs, list) and len(q_embs) != len(queries)):
                    q_embs = self._embed_queries_cached(queries)
                if q_embs is not None:
                    res = coll.query(query_embeddings=q_embs, n_results=max(1, int(top_k)), include=["documents", "distances"])  # type: ignore[attr-defined]
                else:
                    res = coll.query(query_texts=queries, n_results=max(1, int(top_k)), include=["documents", "distances"])  # type: ignore[attr-defined]
                all_docs = res.get("documents") or []
                all_dists = res.get("distances") or []
                out = []
                for qi, (docs, dists) in enumerate(zip(all_docs, all_dists)):
                    for i, doc in enumerate(docs or []):
                        dist = float(dists[i]) if dists and i < len(dists) else 1.0
                        sim = max(0.0, min(1.0, 1.0 - dist))
                        out.append((doc, sim, queries[qi]))
                # Deduplicate by text keeping best score
                best: Dict[str, Tuple[float, str]] = {}
                for text, score, q in out:
                    if text not in best or score > best[text][0]:
                        best[text] = (score, q)
                ranked = sorted(((t, sc, q) for t, (sc, q) in best.items()), key=lambda x: x[1], reverse=True)[:top_k]
                return ranked
            except Exception as e:
                self.logger.warning("Vector batch query failed; falling back to keyword. error=%s", e)

        # BM25 batch fallback
        self._ensure_bm25()
        idx = self._bm25_index.get(cat)
        docs_bm = self._bm25_docs.get(cat) or []
        if idx is not None and docs_bm:
            best_text: Dict[str, Tuple[float, str]] = {}
            for q in queries:
                try:
                    scores = idx.get_scores(tokenize_ko(q))  # type: ignore[attr-defined]
                    pairs = list(enumerate(scores))
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    pairs = [p for p in pairs if p[1] > 0][:top_k]
                    vals = [p[1] for p in pairs]
                    mn, mx = (min(vals), max(vals)) if vals else (0.0, 0.0)
                    for i, sc in pairs:
                        norm = 0.0 if mx == mn else (sc - mn) / (mx - mn)
                        t = docs_bm[i]
                        if t not in best_text or norm > best_text[t][0]:
                            best_text[t] = (float(norm), q)
                except Exception:
                    continue
            ranked = sorted(best_text.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
            return [(t, sc, q) for t, (sc, q) in ranked]

        # Fallback keyword scoring: take max across queries
        items = self.corpora.get(cat) or []
        if not items:
            return []
        best_scores = [-1.0] * len(items)
        best_q_idx = [-1] * len(items)
        for qi, q in enumerate(queries):
            scores = [self._keyword_score(q, t) for t in items]
            for i, sc in enumerate(scores):
                if sc > best_scores[i]:
                    best_scores[i] = sc
                    best_q_idx[i] = qi
        idxs = sorted(range(len(best_scores)), key=lambda i: best_scores[i], reverse=True)[:top_k]
        out = []
        for i in idxs:
            if best_scores[i] <= 0:
                continue
            out.append((self._make_snippet(items, i, self.context_overlap), float(best_scores[i]), queries[best_q_idx[i]] if 0 <= best_q_idx[i] < len(queries) else queries[0]))
        return out
