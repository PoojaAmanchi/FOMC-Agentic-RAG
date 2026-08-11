"""
THE SYNTHESIZER NODE
======================
Takes retrieved_chunks and writes an answer, citing which numbered
passage backed each claim.
"""
import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_agent.utils import extract_text
from rag_agent.state import RAGState

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_BASE_INSTRUCTION = """You are a monetary policy analyst. Answer the question using
ONLY the numbered passages below. Cite the passage number(s) for every claim
you make, like this: [1]. If the passages don't contain enough information
to answer, say so explicitly instead of guessing.
"""

_CONTRADICTION_ADDENDUM = """
This question asks about a possible CHANGE or SHIFT over time. Explicitly
compare the exact wording used across the different passages/dates. Point
out specific phrases that were added, removed, or reworded -- not just a
general summary of each document.
"""

_COMPARISON_ADDENDUM = """
This question asks you to compare specific meetings directly. Structure
your answer to address each meeting being compared, and explicitly state
what is similar or different between them.
"""


def _build_system_prompt(query_type: str) -> str:
    if query_type == "contradiction":
        return _BASE_INSTRUCTION + _CONTRADICTION_ADDENDUM
    if query_type == "comparison":
        return _BASE_INSTRUCTION + _COMPARISON_ADDENDUM
    return _BASE_INSTRUCTION


def _format_numbered_passages(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")
    return "\n\n".join(parts)


def _extract_cited_sources(chunks: list[dict], answer: str) -> list[str]:
    cited_indices = {int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer)}
    sources = []
    for idx in sorted(cited_indices):
        if 1 <= idx <= len(chunks):
            sources.append(chunks[idx - 1]["source"])
    return sorted(set(sources))


def synthesizer_node(state: RAGState) -> dict:
    query = state["query"]
    query_type = state.get("query_type") or "factual"
    chunks = state.get("retrieved_chunks") or []

    if not chunks:
        return {
            "draft_answer": "I could not find relevant information to answer this question.",
            "cited_sources": [],
        }

    numbered_passages = _format_numbered_passages(chunks)
    system_prompt = _build_system_prompt(query_type)

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    messages = [
        ("system", f"{system_prompt}\n\nPassages:\n{numbered_passages}"),
        ("human", query),
    ]
    response = llm.invoke(messages)
    answer = extract_text(response.content)

    cited_sources = _extract_cited_sources(chunks, answer)

    return {
        "draft_answer": answer,
        "cited_sources": cited_sources,
    }
