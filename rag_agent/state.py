"""
THE SHARED STATE — read this before any node file.
=====================================================
Every node in our LangGraph agent (router, retriever, synthesizer, verifier)
reads from and writes to ONE shared object of this shape. Think of it as a
folder being passed hand-to-hand between 5 coworkers on an assembly line --
each one reads what they need, does their job, and adds their own notes
before passing it to the next person.

LangGraph enforces this shape using a TypedDict: a plain Python dict that
has a fixed, named set of allowed keys. This isn't just Python style --
it's what lets LangGraph know how to merge partial updates from each node
without you writing that merging logic yourself.

WHY IS THIS DESIGN NECESSARY?
Without a shared, structured state, each node would have to somehow pass
its own custom output directly to the next specific node in code -- turning
our clean 5-node graph into a tangle of hardcoded handoffs. With a shared
state, every node just reads "whatever fields I need" and writes "the
fields I'm responsible for" -- so we can freely rewire, reorder, or add
retry loops between nodes without rewriting node internals.
"""
from typing import Any, Literal, Optional, TypedDict


class RAGState(TypedDict):
    """The one shared object passed through every node in the graph."""

    # ── Input ──────────────────────────────────────────────────────────
    # The raw question, exactly as the user typed it. Never modified.
    query: str

    # Set by the router. Determines how the retriever searches and which
    # prompt template the synthesizer uses.
    query_type: Optional[Literal[
        "factual",       # "What was the rate decision in March 2026?"
        "comparison",    # "Compare January and June statements on inflation"
        "contradiction", # "Did the Fed's tone on inflation shift over time?"
    ]]

    # ── Retrieval ──────────────────────────────────────────────────────
    # The router may extract which specific meeting dates are relevant
    # (e.g. a question mentioning "January and June" should narrow search
    # to those two documents specifically).
    relevant_dates: list[str]

    # The actual text chunks the retriever found, most relevant first.
    # Each chunk is a dict: {"text": ..., "source": ..., "score": ...}
    retrieved_chunks: list[dict[str, Any]]

    # ── Generation ─────────────────────────────────────────────────────
    # The synthesizer's answer BEFORE verification. We keep this separate
    # from "final_answer" because the verifier might reject it and trigger
    # a retry -- at that point draft_answer gets overwritten, but we
    # haven't committed to it as the final output yet.
    draft_answer: Optional[str]

    # Which source filenames actually got cited in the draft answer.
    cited_sources: list[str]

    # ── Verification (the "agentic" part) ───────────────────────────────
    # Did the verifier judge the draft answer as fully backed by the
    # retrieved chunks?
    is_grounded: Optional[bool]

    # The verifier's self-reported confidence, 0.0 to 1.0. We only accept
    # an answer if is_grounded=True AND confidence is above a threshold --
    # requiring both catches cases where the verifier is unsure even if it
    # technically didn't find an unsupported claim.
    confidence_score: Optional[float]

    # How many times we've already looped back to the retriever for this
    # question. This is what prevents an infinite retry loop -- the graph's
    # conditional edge checks this against a max (we'll use 2) before
    # deciding to retry again or give up honestly.
    retry_count: int

    # ── Output ─────────────────────────────────────────────────────────
    # The answer we actually return to the user. Only set once the
    # verifier accepts the draft, OR once retries are exhausted (in which
    # case this includes an honest low-confidence disclaimer).
    final_answer: Optional[str]
