import hashlib
import random
from typing import TypedDict, Optional, List, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langgraph.graph import StateGraph, END # END is important for defining termination points

# Open WebUI specific imports
from open_webui.retrieval.utils import VectorSearchRetriever, RerankCompressor # Assuming these exist
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.config import (
    RAG_EMBEDDING_QUERY_PREFIX,
    RAG_EMBEDDING_CONTENT_PREFIX,
    TOP_K as DEFAULT_TOP_K,
    TOP_K_RERANKER as DEFAULT_TOP_K_RERANKER,
    RELEVANCE_THRESHOLD as DEFAULT_RELEVANCE_THRESHOLD,
    HYBRID_BM25_WEIGHT as DEFAULT_HYBRID_BM25_WEIGHT,
)
# For logging within LangGraph state if needed, or use standard logging
import logging
log = logging.getLogger(__name__)


class RAGState(TypedDict):
    """
    Represents the state of a RAG (Retrieval Augmented Generation) pipeline.
    """
    query: str
    collection_name: Optional[str]
    collection_names: Optional[List[str]]

    k: int
    k_reranker: int
    r_score_threshold: float
    hybrid_bm25_weight: float

    embedding_function: Optional[Callable]
    reranking_function: Optional[Callable]

    user: Optional[Any]
    request_context: Optional[Any]

    raw_documents_for_bm25: Optional[List[Document]]

    bm25_retriever: Optional[BM25Retriever]
    vector_retriever: Optional[VectorSearchRetriever]
    ensemble_retriever: Optional[EnsembleRetriever]

    retrieved_documents: List[Document]
    reranked_documents: List[Document]

    error_message: Optional[str]
    log_messages: List[str]


def validate_inputs_node(state: RAGState) -> RAGState:
    logs = state.get("log_messages", [])
    logs.append("Node: validate_inputs_node - Validating inputs.")

    query = state.get("query")
    if not query or not isinstance(query, str) or not query.strip():
        logs.append("Error: Query is missing or invalid.")
        state["error_message"] = "Query is missing or invalid."
        state["log_messages"] = logs
        return state

    collection_name = state.get("collection_name")
    collection_names = state.get("collection_names")
    if not (collection_name or collection_names):
        logs.append("Error: At least one of collection_name or collection_names must be provided.")
        state["error_message"] = "Collection name(s) must be provided."
        state["log_messages"] = logs
        return state

    k = state.get("k", DEFAULT_TOP_K)
    k_reranker = state.get("k_reranker", DEFAULT_TOP_K_RERANKER)
    r_score_threshold = state.get("r_score_threshold", DEFAULT_RELEVANCE_THRESHOLD)
    hybrid_bm25_weight = state.get("hybrid_bm25_weight", DEFAULT_HYBRID_BM25_WEIGHT)

    logs.append(f"Params: k={k}, k_reranker={k_reranker}, r_score_threshold={r_score_threshold}, hybrid_bm25_weight={hybrid_bm25_weight}")

    if not (0.0 <= hybrid_bm25_weight <= 1.0):
        logs.append(f"Error: hybrid_bm25_weight ({hybrid_bm25_weight}) must be between 0.0 and 1.0.")
        state["error_message"] = "hybrid_bm25_weight must be between 0.0 and 1.0."
        state["log_messages"] = logs
        return state

    if hybrid_bm25_weight < 1.0 and not state.get("embedding_function"):
        logs.append("Error: embedding_function is required for vector search (hybrid_bm25_weight < 1.0).")
        state["error_message"] = "embedding_function is required for vector search."
        state["log_messages"] = logs
        return state

    if not state.get("reranking_function") and not state.get("embedding_function"):
        logs.append("Warning: Reranking will use cosine similarity if embedding_function is available, otherwise it might be skipped or use a default if RerankCompressor allows. Provide reranking_function for model-based reranking.")

    current_raw_docs = state.get("raw_documents_for_bm25")

    initialized_state: RAGState = {
        "query": query,
        "collection_name": collection_name,
        "collection_names": collection_names,
        "k": k,
        "k_reranker": k_reranker,
        "r_score_threshold": r_score_threshold,
        "hybrid_bm25_weight": hybrid_bm25_weight,
        "embedding_function": state.get("embedding_function"),
        "reranking_function": state.get("reranking_function"),
        "user": state.get("user"),
        "request_context": state.get("request_context"),
        "raw_documents_for_bm25": current_raw_docs if current_raw_docs is not None else None,
        "bm25_retriever": None,
        "vector_retriever": None,
        "ensemble_retriever": None,
        "retrieved_documents": [],
        "reranked_documents": [],
        "error_message": None,
        "log_messages": logs,
    }

    logs.append("Input validation successful.")
    return initialized_state


def fetch_documents_for_bm25_node(state: RAGState) -> RAGState:
    logs = state["log_messages"]
    logs.append("Node: fetch_documents_for_bm25_node - Fetching documents for BM25.")

    bm25_collection_target = state.get("collection_name")
    if not bm25_collection_target and state.get("collection_names"): # Fallback for query_collection context
        bm25_collection_target = state["collection_names"][0]
        logs.append(f"Warning: Using the first collection '{bm25_collection_target}' from collection_names for BM25 documents.")

    if not bm25_collection_target:
        logs.append("Error: No collection specified for fetching BM25 documents.")
        state["error_message"] = "No collection specified for BM25 documents."
        return state

    try:
        collection_data = VECTOR_DB_CLIENT.get(collection_name=bm25_collection_target)
        if collection_data and collection_data.documents and len(collection_data.documents[0]) > 0:
            docs_content = collection_data.documents[0]
            docs_meta = collection_data.metadatas[0] if collection_data.metadatas and len(collection_data.metadatas[0]) == len(docs_content) else [{}] * len(docs_content)
            raw_docs = [Document(page_content=content, metadata=meta) for content, meta in zip(docs_content, docs_meta)]
            state["raw_documents_for_bm25"] = raw_docs
            logs.append(f"Fetched {len(raw_docs)} documents from '{bm25_collection_target}' for BM25.")
        else:
            logs.append(f"No documents found in '{bm25_collection_target}' for BM25. BM25 retriever will be empty.")
            state["raw_documents_for_bm25"] = []

    except Exception as e:
        error_msg = f"Error fetching documents for BM25 from '{bm25_collection_target}': {str(e)}"
        logs.append(error_msg)
        state["error_message"] = error_msg
    return state


def setup_bm25_retriever_node(state: RAGState) -> RAGState:
    logs = state["log_messages"]
    logs.append("Node: setup_bm25_retriever_node - Setting up BM25 retriever.")

    raw_documents = state.get("raw_documents_for_bm25", [])
    if not raw_documents:
        logs.append("No documents available for BM25 retriever setup. BM25 will not be used.")
        state["bm25_retriever"] = None
        return state

    try:
        texts = [doc.page_content for doc in raw_documents]
        metadatas = [doc.metadata for doc in raw_documents]

        bm25_retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas)
        bm25_retriever.k = state["k"]
        state["bm25_retriever"] = bm25_retriever
        logs.append(f"BM25 retriever initialized with {len(texts)} documents and k={bm25_retriever.k}.")
    except Exception as e:
        error_msg = f"Error initializing BM25 retriever: {str(e)}"
        logs.append(error_msg)
        state["error_message"] = error_msg
    return state


def setup_vector_retriever_node(state: RAGState) -> RAGState:
    logs = state["log_messages"]
    logs.append("Node: setup_vector_retriever_node - Setting up vector retriever.")

    # For query_doc_with_langgraph, collection_name is primary.
    # For query_collection_with_langgraph, collection_names will be used, but this node
    # is called per collection by the calling function. So, collection_name should be set in state.
    target_collection = state.get("collection_name")
    if not target_collection: # Should be set by the caller for this node
        logs.append("Error: 'collection_name' must be set in state for vector retriever setup.")
        state["error_message"] = "'collection_name' not set for vector retriever."
        return state

    embedding_function = state["embedding_function"]
    k = state["k"]

    try:
        vector_retriever = VectorSearchRetriever(
            collection_name=target_collection, # Should be a single string here
            embedding_function=embedding_function,
            top_k=k,
        )
        state["vector_retriever"] = vector_retriever
        logs.append(f"Vector retriever initialized for '{target_collection}' with k={k}.")
    except Exception as e:
        error_msg = f"Error initializing vector retriever for '{target_collection}': {str(e)}"
        logs.append(error_msg)
        state["error_message"] = error_msg
    return state


def setup_ensemble_retriever_node(state: RAGState) -> RAGState:
    logs = state["log_messages"]
    logs.append("Node: setup_ensemble_retriever_node - Setting up ensemble retriever.")

    bm25_retriever = state.get("bm25_retriever")
    vector_retriever = state.get("vector_retriever")
    hybrid_bm25_weight = state["hybrid_bm25_weight"]

    active_retrievers = []
    active_weights = []

    if bm25_retriever and hybrid_bm25_weight > 0:
        active_retrievers.append(bm25_retriever)
        active_weights.append(hybrid_bm25_weight)
        logs.append(f"BM25 retriever will be used in ensemble with weight {hybrid_bm25_weight}.")

    if vector_retriever and hybrid_bm25_weight < 1.0:
        active_retrievers.append(vector_retriever)
        active_weights.append(1.0 - hybrid_bm25_weight)
        logs.append(f"Vector retriever will be used in ensemble with weight {1.0 - hybrid_bm25_weight}.")

    if not active_retrievers:
        logs.append("Error: No retrievers are active for the ensemble based on weights and availability.")
        state["error_message"] = "Cannot form ensemble with no active retrievers."
        return state

    if len(active_retrievers) == 1:
        logs.append(f"Only one retriever type ({active_retrievers[0].__class__.__name__}) active. Ensemble will use this with weight 1.0.")
        active_weights = [1.0]

    try:
        ensemble_retriever = EnsembleRetriever(retrievers=active_retrievers, weights=active_weights)
        state["ensemble_retriever"] = ensemble_retriever
        logs.append(f"Ensemble retriever initialized with {len(active_retrievers)} retriever(s). Weights: {active_weights}")
    except Exception as e:
        error_msg = f"Error initializing ensemble retriever: {str(e)}"
        logs.append(error_msg)
        state["error_message"] = error_msg
    return state


def retrieve_documents_node(state: RAGState) -> RAGState:
    logs = state["log_messages"]
    logs.append("Node: retrieve_documents_node - Retrieving documents.")

    query = state["query"]
    retriever_to_use = state.get("ensemble_retriever")

    if not retriever_to_use:
        if state["hybrid_bm25_weight"] == 1.0 and state.get("bm25_retriever"):
            retriever_to_use = state["bm25_retriever"]
            logs.append("Using BM25 retriever directly.")
        elif state["hybrid_bm25_weight"] == 0.0 and state.get("vector_retriever"):
            retriever_to_use = state["vector_retriever"]
            logs.append("Using Vector retriever directly.")
        else:
            logs.append("Error: No appropriate retriever found (ensemble or direct).")
            state["error_message"] = "No retriever configured for document retrieval."
            return state

    try:
        docs = retriever_to_use.get_relevant_documents(query)
        state["retrieved_documents"] = docs
        logs.append(f"Retrieved {len(docs)} documents using {retriever_to_use.__class__.__name__}.")
    except Exception as e:
        error_msg = f"Error retrieving documents: {str(e)}"
        logs.append(error_msg)
        state["error_message"] = error_msg
    return state


def rerank_documents_node(state: RAGState) -> RAGState:
    logs = state["log_messages"]
    logs.append("Node: rerank_documents_node - Reranking documents.")

    retrieved_docs = state.get("retrieved_documents", [])
    if not retrieved_docs:
        logs.append("No documents to rerank. Skipping.")
        state["reranked_documents"] = []
        return state

    embedding_function = state.get("embedding_function")
    reranker_model_instance = state.get("reranking_function")
    k_final = state["k"]
    k_reranker_compressor = state["k_reranker"]
    r_score_threshold = state["r_score_threshold"]
    query = state["query"]

    if not reranker_model_instance and not embedding_function:
        logs.append("Warning: No reranking_function (model) or embedding_function (for cosine sim) provided. Reranking will be skipped, passing through retrieved documents.")
        state["reranked_documents"] = retrieved_docs
        return state

    try:
        compressor = RerankCompressor(
            embedding_function=embedding_function,
            top_n=k_reranker_compressor,
            reranking_function=reranker_model_instance,
            r_score=r_score_threshold,
        )

        reranked_compressed_docs = compressor.compress_documents(documents=retrieved_docs, query=query)

        if k_final < len(reranked_compressed_docs):
            final_docs = reranked_compressed_docs[:k_final]
            logs.append(f"Reranked documents further trimmed from {len(reranked_compressed_docs)} to final {k_final}.")
        else:
            final_docs = reranked_compressed_docs

        state["reranked_documents"] = final_docs
        logs.append(f"Reranked and compressed documents. Final count: {len(final_docs)}.")

    except Exception as e:
        error_msg = f"Error during document reranking/compression: {str(e)}"
        logs.append(error_msg)
        state["error_message"] = error_msg
    return state


def error_handler_node(state: RAGState) -> RAGState:
    logs = state["log_messages"]
    error = state.get("error_message", "An unspecified error occurred in RAG pipeline.")
    logs.append(f"Pipeline error: {error}. Terminating graph for this run.")
    return state

# Conditional Edge Functions
def route_after_validation(state: RAGState) -> str:
    logs = state["log_messages"]
    if state.get("error_message"):
        logs.append("Routing: Validation failed, proceeding to error handler.")
        return "error_handler_node"

    logs.append(f"Routing after validation. BM25 weight: {state['hybrid_bm25_weight']}.")
    if state['hybrid_bm25_weight'] > 0:
        if state.get("raw_documents_for_bm25") is None:
            logs.append("Routing: BM25 needed, documents not pre-loaded. Fetching BM25 documents.")
            return "fetch_bm25_documents"
        logs.append("Routing: BM25 needed, documents pre-loaded. Setting up BM25 retriever.")
        return "setup_bm25_retriever"

    logs.append("Routing: BM25 not needed (weight is 0). Proceeding to vector retriever setup.")
    return "setup_vector_retriever"


def route_after_bm25_fetch(state: RAGState) -> str:
    logs = state["log_messages"]
    if state.get("error_message"):
        logs.append("Routing: Error during BM25 document fetch. Proceeding to error handler.")
        return "error_handler_node"
    logs.append("Routing: BM25 documents fetched. Setting up BM25 retriever.")
    return "setup_bm25_retriever"


def route_after_bm25_setup(state: RAGState) -> str:
    logs = state["log_messages"]
    if state.get("error_message") and state['hybrid_bm25_weight'] == 1.0:
        logs.append("Routing: Error during BM25 setup (BM25 only path). Proceeding to error handler.")
        return "error_handler_node"
    # If error but not BM25 only, we might still proceed with vector

    if state['hybrid_bm25_weight'] < 1.0:
        logs.append("Routing: BM25 setup step done. Proceeding to vector retriever setup.")
        return "setup_vector_retriever"

    logs.append("Routing: BM25 setup step done (BM25 only path). Deciding final retriever.")
    return "decide_final_retriever_logic_node_name"


def route_after_vector_setup(state: RAGState) -> str:
    logs = state["log_messages"]
    if state.get("error_message") and (state['hybrid_bm25_weight'] == 0.0 or not state.get("bm25_retriever")):
        logs.append("Routing: Error during vector retriever setup (Vector only or BM25 also failed/unavailable). Proceeding to error handler.")
        return "error_handler_node"

    logs.append("Routing: Vector retriever setup step done. Deciding final retriever path.")
    return "decide_final_retriever_logic_node_name"


def decide_final_retriever_logic_node_conditional_edge(state: RAGState) -> str:
    logs = state["log_messages"]
    if state.get("error_message") and not (state.get("bm25_retriever") or state.get("vector_retriever")):
        logs.append("Routing: Error before final retriever decision, and no retrievers available. Proceeding to error handler.")
        return "error_handler_node"

    bm25_available = state.get("bm25_retriever") is not None
    vector_available = state.get("vector_retriever") is not None
    weight = state["hybrid_bm25_weight"]

    logs.append(f"Deciding final retriever: BM25 available={bm25_available}, Vector available={vector_available}, Weight={weight}")

    if weight > 0 and weight < 1: # Hybrid mode
        if bm25_available and vector_available:
            logs.append("Routing decision: Both BM25 and Vector available for hybrid. Setting up ensemble.")
            return "setup_ensemble_retriever"
        elif vector_available:
            logs.append("Routing decision: Hybrid mode, BM25 failed/unavailable. Fallback to Vector only.")
            state["hybrid_bm25_weight"] = 0.0
            return "retrieve_documents"
        elif bm25_available:
            logs.append("Routing decision: Hybrid mode, Vector failed/unavailable. Fallback to BM25 only.")
            state["hybrid_bm25_weight"] = 1.0
            return "retrieve_documents"
        else:
            logs.append("Error: Hybrid mode selected, but neither BM25 nor Vector retriever is available.")
            state["error_message"] = "Hybrid search failed: no retrievers available."
            return "error_handler_node"
    elif weight == 1.0: # BM25 only mode
        if bm25_available:
            logs.append("Routing decision: BM25 only mode selected and retriever available.")
            return "retrieve_documents"
        else: # BM25 setup must have failed
            logs.append("Error: BM25 only mode selected, but BM25 retriever not available.")
            state["error_message"] = "BM25 retriever not available for BM25-only search."
            return "error_handler_node"
    elif weight == 0.0: # Vector only mode
        if vector_available:
            logs.append("Routing decision: Vector only mode selected and retriever available.")
            return "retrieve_documents"
        else: # Vector setup must have failed
            logs.append("Error: Vector only mode selected, but Vector retriever not available.")
            state["error_message"] = "Vector retriever not available for vector-only search."
            return "error_handler_node"
    else:
        logs.append(f"Critical Error: Invalid hybrid_bm25_weight: {weight} post-validation.")
        state["error_message"] = f"Invalid hybrid_bm25_weight: {weight}"
        return "error_handler_node"

def route_after_retrieval(state: RAGState) -> str:
    logs = state["log_messages"]
    if state.get("error_message"):
        logs.append("Routing: Error during retrieval. Proceeding to error handler.")
        return "error_handler_node"
    if not state.get("retrieved_documents"):
        logs.append("Routing: No documents retrieved. Skipping reranking and ending.")
        return END
    logs.append("Routing: Documents retrieved. Proceeding to reranking.")
    return "rerank_documents"


# Compile the graph once when the module is loaded
RAG_APP_INSTANCE = None

def get_compiled_rag_graph() -> StateGraph:
    global RAG_APP_INSTANCE
    if RAG_APP_INSTANCE is None:
        RAG_APP_INSTANCE = create_rag_graph()
    return RAG_APP_INSTANCE

def create_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)
    graph.add_node("validate_inputs", validate_inputs_node)
    graph.add_node("fetch_bm25_documents", fetch_documents_for_bm25_node)
    graph.add_node("setup_bm25_retriever", setup_bm25_retriever_node)
    graph.add_node("setup_vector_retriever", setup_vector_retriever_node)
    graph.add_node("decide_final_retriever_logic_node_name", lambda state: state)
    graph.add_node("setup_ensemble_retriever", setup_ensemble_retriever_node)
    graph.add_node("retrieve_documents", retrieve_documents_node)
    graph.add_node("rerank_documents", rerank_documents_node)
    graph.add_node("error_handler_node", error_handler_node)

    graph.set_entry_point("validate_inputs")
    graph.add_conditional_edges("validate_inputs", route_after_validation)
    graph.add_conditional_edges("fetch_bm25_documents", route_after_bm25_fetch)
    graph.add_conditional_edges("setup_bm25_retriever", route_after_bm25_setup)
    graph.add_conditional_edges("setup_vector_retriever", route_after_vector_setup)
    graph.add_conditional_edges("decide_final_retriever_logic_node_name", decide_final_retriever_logic_node_conditional_edge)
    graph.add_edge("setup_ensemble_retriever", "retrieve_documents")
    graph.add_conditional_edges("retrieve_documents", route_after_retrieval)
    graph.add_edge("rerank_documents", END)
    graph.add_edge("error_handler_node", END)

    return graph.compile()

def query_doc_with_langgraph(
    collection_name: str, # Single collection name for this function
    query: str,
    embedding_function: Callable,
    k: int,
    reranking_function: Optional[Callable],
    k_reranker: int,
    r_score_threshold: float,
    hybrid_bm25_weight: float,
    user: Optional[Any] = None,
    request_context: Optional[Any] = None,
    raw_documents_for_bm25_override: Optional[List[Document]] = None,
) -> dict:
    initial_state: RAGState = {
        "query": query,
        "collection_name": collection_name,
        "collection_names": [collection_name] if collection_name else None, # For consistency if vector retriever uses list
        "k": k,
        "k_reranker": k_reranker,
        "r_score_threshold": r_score_threshold,
        "hybrid_bm25_weight": hybrid_bm25_weight,
        "embedding_function": embedding_function,
        "reranking_function": reranking_function,
        "user": user,
        "request_context": request_context,
        "raw_documents_for_bm25": raw_documents_for_bm25_override,
        "bm25_retriever": None, "vector_retriever": None, "ensemble_retriever": None, # Initialized as None
        "retrieved_documents": [], "reranked_documents": [], # Initialized as empty
        "error_message": None, "log_messages": [],
    }

    rag_app = get_compiled_rag_graph()
    final_state = rag_app.invoke(initial_state)

    if final_state.get("error_message"):
        full_log = "\n".join(final_state.get("log_messages", ["No log messages."]))
        raise Exception(f"RAG Pipeline failed for query '{query}' on '{collection_name}': {final_state['error_message']}\nLogs:\n{full_log}")

    output_documents = final_state.get("reranked_documents", [])
    if not output_documents and not final_state.get("error_message"): # If reranking was skipped or produced no results, use retrieved
        output_documents = final_state.get("retrieved_documents", [])

    distances = [doc.metadata.get("score", 0.0) if hasattr(doc, 'metadata') and doc.metadata else 0.0 for doc in output_documents]
    documents_content = [doc.page_content for doc in output_documents]
    metadatas = [doc.metadata if hasattr(doc, 'metadata') and doc.metadata else {} for doc in output_documents]

    return {
        "distances": [distances], "documents": [documents_content], "metadatas": [metadatas],
        "logs": final_state.get("log_messages", [])
    }

def _merge_and_sort_query_results(query_results: List[dict], k: int) -> dict:
    """
    Merges and sorts results from multiple query_doc_with_langgraph calls.
    (Adapted from utils.py)
    """
    combined_docs_map = {} # Using dict to handle potential duplicates by content hash

    for result_dict in query_results:
        if not result_dict or not result_dict.get("documents") or not result_dict["documents"][0]:
            continue # Skip empty or malformed results

        docs_content = result_dict["documents"][0]
        scores = result_dict["distances"][0] if result_dict.get("distances") and result_dict["distances"][0] else [0.0] * len(docs_content)
        metas = result_dict["metadatas"][0] if result_dict.get("metadatas") and result_dict["metadatas"][0] else [{}] * len(docs_content)

        for i, content in enumerate(docs_content):
            doc_hash = hashlib.sha256(content.encode()).hexdigest()
            score = scores[i]
            meta = metas[i]

            if doc_hash not in combined_docs_map or score > combined_docs_map[doc_hash]["score"]:
                combined_docs_map[doc_hash] = {"score": score, "content": content, "metadata": meta}

    # Sort by score descending
    sorted_combined_results = sorted(combined_docs_map.values(), key=lambda x: x["score"], reverse=True)

    # Take top k
    top_k_results = sorted_combined_results[:k]

    final_scores = [item["score"] for item in top_k_results]
    final_contents = [item["content"] for item in top_k_results]
    final_metadatas = [item["metadata"] for item in top_k_results]

    return {
        "distances": [final_scores],
        "documents": [final_contents],
        "metadatas": [final_metadatas]
    }

def query_collection_with_langgraph(
    collection_names: List[str],
    queries: List[str], # Using only queries[0] for now
    embedding_function: Callable,
    k: int,
    reranking_function: Optional[Callable],
    k_reranker: int,
    r_score_threshold: float,
    hybrid_bm25_weight: float,
    user: Optional[Any] = None,
    request_context: Optional[Any] = None,
) -> dict:
    if not queries:
        raise ValueError("Queries list cannot be empty.")

    main_query = queries[0] # Assuming we use the first query for all collections for now

    all_results = []
    collection_logs = {}

    # Use ThreadPoolExecutor to run query_doc_with_langgraph for each collection in parallel
    with ThreadPoolExecutor(max_workers=min(len(collection_names), 5)) as executor: # Limit workers
        future_to_collection = {
            executor.submit(
                query_doc_with_langgraph,
                collection_name=c_name, # Pass single collection name here
                query=main_query,
                embedding_function=embedding_function,
                k=k, # k is applied per collection, then merged and re-limited
                reranking_function=reranking_function,
                k_reranker=k_reranker,
                r_score_threshold=r_score_threshold,
                hybrid_bm25_weight=hybrid_bm25_weight,
                user=user,
                request_context=request_context,
                # raw_documents_for_bm25_override: For collection-specific BM25 docs, this would need to be a map
            ): c_name
            for c_name in collection_names
        }

        for future in as_completed(future_to_collection):
            collection_name_done = future_to_collection[future]
            try:
                result = future.result()
                all_results.append(result)
                collection_logs[collection_name_done] = result.get("logs", ["No logs available for this collection."])
            except Exception as e:
                log.error(f"Error processing collection {collection_name_done} with LangGraph: {str(e)}")
                collection_logs[collection_name_done] = [f"Error: {str(e)}"]
                # Decide if one failure means all fail, or just skip this collection's results
                # For now, skipping failed ones and attempting to merge successful ones

    if not all_results:
        # This could happen if all individual queries fail
        return {"distances": [[]], "documents": [[]], "metadatas": [[]], "logs_by_collection": collection_logs}

    merged_results = _merge_and_sort_query_results(all_results, k) # Final top-k from all collections
    merged_results["logs_by_collection"] = collection_logs # Add individual logs
    return merged_results


if __name__ == "__main__":
    # Setup basic logging for the __main__ test runs
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def dummy_embedding_function(texts: List[str], prefix: Optional[str] = None) -> List[List[float]]:
        # log.info(f"Embedding (dummy) for {len(texts)} texts, prefix: {prefix}")
        return [[random.random() for _ in range(10)] for _ in texts]

    class DummyReranker:
        def predict(self, RerankerInput: List[tuple[str, str]]):
            # log.info(f"Reranking (dummy) for {len(RerankerInput)} pairs")
            return [random.random() for _ in RerankerInput]

    sample_docs_coll1 = [
        Document(page_content="The cat sat on the mat.", metadata={"source": "coll1-doc1", "id": "c1_d1"}),
        Document(page_content="A black cat wandered.", metadata={"source": "coll1-doc2", "id": "c1_d2"}),
    ]
    sample_docs_coll2 = [
        Document(page_content="The dog chased the cat.", metadata={"source": "coll2-doc1", "id": "c2_d1"}),
        Document(page_content="A brown dog barked loudly.", metadata={"source": "coll2-doc2", "id": "c2_d2"}),
    ]

    class MockVectorDBClientForCollectionQuery:
        def get(self, collection_name: str):
            # log.info(f"MockVectorDBClient.get called for {collection_name}")
            docs_to_return = []
            if collection_name == "collection1":
                docs_to_return = sample_docs_coll1
            elif collection_name == "collection2":
                docs_to_return = sample_docs_coll2
            elif collection_name == "empty_collection_for_bm25":
                 return type('GetResult', (), {'documents': [[]], 'metadatas': [[]], 'ids': [[]]})()


            return type('GetResult', (), {
                'documents': [[d.page_content for d in docs_to_return]],
                'metadatas': [[d.metadata for d in docs_to_return]],
                'ids': [[d.metadata.get("id", f"id_{i}") for i, d in enumerate(docs_to_return)]]
            })()

        def search(self, collection_name: str, vectors: List[List[float]], limit: int, **kwargs):
            # log.info(f"MockVectorDBClient.search for {collection_name}, limit {limit}")
            if collection_name == "collection1":
                return type('SearchResult', (), {
                    'ids': [["c1_vec_id1", "c1_vec_id2"]],
                    'metadatas': [[{"source":"mock_c1_vdoc1"}, {"source":"mock_c1_vdoc2"}]],
                    'documents': [["Vector cat from collection1.", "Another mat from collection1."]],
                    'scores': [[0.89, 0.78]]
                })()
            elif collection_name == "collection2":
                 return type('SearchResult', (), {
                    'ids': [["c2_vec_id1"]],
                    'metadatas': [[{"source":"mock_c2_vdoc1"}]],
                    'documents': [["Vector dog from collection2."]],
                    'scores': [[0.92]]
                })()
            return type('SearchResult', (), {'ids': [[]], 'metadatas': [[]], 'documents': [[]], 'scores': [[]]})()


    original_vector_db_client = VECTOR_DB_CLIENT

    print("\n\n--- Testing query_doc_with_langgraph ---")
    setattr(open_webui.retrieval.vector.factory, "VECTOR_DB_CLIENT", MockVectorDBClientForCollectionQuery()) # Use the multi-collection mock

    doc_test_cases = [
        ("Hybrid (BM25 fetch)", {"collection_name": "collection1", "query": "cat on mat", "hybrid_bm25_weight": 0.5, "raw_documents_for_bm25_override": None, "embedding_function": dummy_embedding_function, "reranking_function": DummyReranker()}),
        ("Vector Only", {"collection_name": "collection2", "query": "dog chase", "hybrid_bm25_weight": 0.0, "embedding_function": dummy_embedding_function, "reranking_function": DummyReranker()}),
    ]
    for name, params in doc_test_cases:
        print(f"\n--- Running Test Case (query_doc): {name} ---")
        try:
            result = query_doc_with_langgraph(k=2, k_reranker=2, r_score_threshold=0.01, **params)
            log.info(f"Result for '{name}': Documents count: {len(result['documents'][0])}")
            # for log_entry in result.get("logs", []): print(f"  LOG: {log_entry}")
        except Exception as e:
            log.error(f"Error in test case '{name}': {str(e)}")


    print("\n\n--- Testing query_collection_with_langgraph ---")
    setattr(open_webui.retrieval.vector.factory, "VECTOR_DB_CLIENT", MockVectorDBClientForCollectionQuery())

    collection_test_params = {
        "collection_names": ["collection1", "collection2", "empty_collection_for_bm25"], # empty one to test robustness
        "queries": ["cat or dog"],
        "embedding_function": dummy_embedding_function,
        "k": 3, # Final k after merge
        "reranking_function": DummyReranker(),
        "k_reranker": 3, # k_reranker per individual graph call
        "r_score_threshold": 0.01,
        "hybrid_bm25_weight": 0.5,
    }
    print(f"\n--- Running Test Case (query_collection) ---")
    try:
        collection_result = query_collection_with_langgraph(**collection_test_params)
        log.info(f"Result for query_collection: Documents count: {len(collection_result['documents'][0])}")
        log.info(f"  Content: {collection_result['documents'][0]}")
        log.info(f"  Scores: {collection_result['distances'][0]}")
        # log.info(f"  Logs by Collection: {collection_result.get('logs_by_collection')}")
    except Exception as e:
        log.error(f"Error in query_collection test case: {str(e)}")

    setattr(open_webui.retrieval.vector.factory, "VECTOR_DB_CLIENT", original_vector_db_client)
