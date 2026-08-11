"""
STREAMING -- Server-Sent Events (SSE) endpoint.
==================================================
The plain /query endpoint (routes.py) makes the user wait silently while
the ENTIRE agentic pipeline runs -- router, retriever, synthesizer,
verifier, and possibly a retry loop -- which can take several seconds of
total silence. That's a poor experience for something with this many
visible steps.

WHAT IS SSE, IN PLAIN TERMS?
Server-Sent Events is a simple one-way streaming protocol: the server
keeps the HTTP connection open and pushes small text messages to the
browser as they become available, instead of sending one big response at
the end. It's simpler than WebSockets (no two-way communication needed
here) and works over plain HTTP.

HOW WE STREAM PROGRESS FOR A LANGGRAPH AGENT:
LangGraph's compiled graph supports `.stream()` instead of `.invoke()` --
instead of running the whole graph and returning only the final state, it
yields the state AFTER EACH NODE completes. We use that to push a small
status message every time a node finishes, so the frontend can show
"Routing question..." -> "Searching documents..." -> "Writing answer..."
-> "Verifying answer..." live, instead of one long silent wait.
"""
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_agent.graph import app as compiled_graph

router = APIRouter()

# Human-readable labels for each node name, shown to the user as progress.
_NODE_LABELS = {
    "router": "Classifying your question...",
    "retriever": "Searching FOMC documents...",
    "synthesizer": "Drafting an answer...",
    "verifier": "Checking the answer is grounded...",
}


class StreamQueryRequest(BaseModel):
    question: str


def _sse_event(data: dict) -> str:
    """Format one Server-Sent Event: 'data: <json>\\n\\n' is the required
    wire format -- browsers' EventSource API parses exactly this shape."""
    return f"data: {json.dumps(data)}\n\n"


def _stream_graph_events(question: str):
    """Generator that yields SSE-formatted progress events as the LangGraph
    agent runs, ending with the final answer."""
    initial_state = {
        "query": question,
        "query_type": None,
        "relevant_dates": [],
        "retrieved_chunks": [],
        "draft_answer": None,
        "cited_sources": [],
        "is_grounded": None,
        "confidence_score": None,
        "retry_count": 0,
        "final_answer": None,
    }

    final_state = {}
    # .stream() yields a dict like {"router": {...partial state...}} after
    # EACH node finishes -- this is what lets us report progress live
    # instead of waiting for the whole graph to complete.
    for step in compiled_graph.stream(initial_state):
        for node_name, node_output in step.items():
            label = _NODE_LABELS.get(node_name, f"Running {node_name}...")
            yield _sse_event({"type": "progress", "node": node_name, "message": label})
            final_state.update(node_output)

    yield _sse_event({
        "type": "final",
        "answer": final_state.get("final_answer") or "",
        "is_grounded": bool(final_state.get("is_grounded")),
        "confidence": final_state.get("confidence_score") or 0.0,
        "retries_used": final_state.get("retry_count") or 0,
        "sources": final_state.get("cited_sources") or [],
    })


@router.post("/stream/query")
def stream_query(req: StreamQueryRequest):
    return StreamingResponse(
        _stream_graph_events(req.question),
        media_type="text/event-stream",
    )
