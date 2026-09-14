"""
THE RETRIEVER NODE
====================
Reads query_type and relevant_dates (set by the router) and searches our
Chroma vector store for the most relevant chunks.

QUERY EXPANSION:
Instead of embedding the user's question exactly as typed, we first ask
the LLM to reword it 3 different ways using different terminology. We then
embed all 4 versions (original + 3 rewrites) and search with each one,
merging and deduplicating results.

WHY THIS HELPS: semantic search finds chunks whose MEANING is similar to
the query -- but "similar" still has limits. If a user asks "did the Fed's
stance shift" and the document says "the Committee's assessment evolved,"
those are close in meaning but not identical, and a single embedding
might not surface the best chunk. Four different phrasings of the same
question, each embedded separately, catch matches a single phrasing
might miss.

We accept this only on comparison/contradiction questions
below, where the benefit of catching cross-document matches is highest.

DEDUPLICATION NOTE:
We normalise whitespace before comparing chunks so that minor formatting
differences (e.g. different line endings or extra spaces) between copies
of the same passage retrieved via different query phrasings don't produce
duplicates in the final result set.
"""
import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag_agent.utils import extract_text
from rag_agent.state import RAGState

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_TOP_K = {
    "factual": 4,
    "comparison": 8,
    "contradiction": 8,
}

# Query expansion is only worth its cost for questions that genuinely
# benefit from catching multiple phrasings across documents. A simple
# factual lookup ("what was the rate in March?") rarely needs it.
_EXPAND_FOR_TYPES = {"comparison", "contradiction"}


def _get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def _expand_query(query: str) -> list[str]:
    """Ask the LLM for 3 differently-worded versions of the question.
    Returns [original, rewrite1, rewrite2, rewrite3]. Falls back to just
    the original if the LLM call fails for any reason -- expansion is an
    enhancement, not a hard dependency."""
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)
    prompt = (
        "Rewrite the following question about Federal Reserve monetary "
        "policy in 3 different ways, keeping the same meaning but using "
        "different terminology. Return ONLY the 3 rewrites, one per line, "
        "no numbering, no extra text.\n\n"
        f"Question: {query}"
    )
    try:
        response = llm.invoke([("human", prompt)])
        rewrites = [
            line.strip()
            for line in extract_text(response.content).strip().split("\n")
            if line.strip()
        ]
        rewrites = rewrites[:3]
    except Exception:
        rewrites = []
    return [query] + rewrites


def _normalise(text: str) -> str:
    """Collapse whitespace so near-identical chunks from different query
    variants aren't treated as distinct. This catches cases where the same
    passage is returned with slightly different whitespace (e.g. trailing
    newlines or double spaces) depending on which query phrasing retrieved
    it -- without normalisation these slip through the exact-match
    deduplication and bloat the context window with repeated text."""
    return re.sub(r"\s+", " ", text).strip()


def _deduplicate(chunks: list[dict]) -> list[dict]:
    """Merge chunks found by multiple query variants, keeping one copy
    per unique (normalised_text, source) pair."""
    seen = {}
    for chunk in chunks:
        key = (_normalise(chunk["text"]), chunk["source"])
        if key not in seen:
            seen[key] = chunk
    return list(seen.values())


def retriever_node(state: RAGState) -> dict:
    query = state["query"]
    query_type = state.get("query_type") or "factual"
    relevant_dates = state.get("relevant_dates") or []
    retry_count = state.get("retry_count") or 0

    top_k = _TOP_K.get(query_type, 4)
    if retry_count > 0:
        top_k += 4  # widen search on retry

    vectorstore = _get_vectorstore()

    search_kwargs = {"k": top_k}
    if relevant_dates:
        source_filenames = [f"{date}_fomc_statement.txt" for date in relevant_dates]
        if len(source_filenames) == 1:
            search_kwargs["filter"] = {"source": source_filenames[0]}
        else:
            search_kwargs["filter"] = {"source": {"$in": source_filenames}}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # Decide whether to expand the query based on question type and
    # whether we're retrying (retries always expand, since the plain
    # search already failed to satisfy the verifier once).
    should_expand = query_type in _EXPAND_FOR_TYPES or retry_count > 0
    queries_to_run = _expand_query(query) if should_expand else [query]

    all_chunks: list[dict] = []
    for q in queries_to_run:
        docs = retriever.invoke(q)
        for d in docs:
            all_chunks.append(
                {
                    "text": d.page_content,
                    "source": d.metadata.get("source", "unknown"),
                }
            )

    deduped = _deduplicate(all_chunks)

    return {"retrieved_chunks": deduped}
