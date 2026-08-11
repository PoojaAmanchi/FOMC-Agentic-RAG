"""
ROUTES -- the actual endpoint definitions.
=============================================
Kept separate from api/main.py (which just creates the FastAPI app and
wires this router in) so endpoint logic doesn't get tangled with app
startup/config concerns -- a common separation in real FastAPI projects
once you have more than a couple of endpoints.
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag_agent.graph import run_query

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fomc_statements"


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    is_grounded: bool
    confidence: float
    retries_used: int
    sources: list[str]


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = run_query(req.question)

    return QueryResponse(
        question=req.question,
        answer=result.get("final_answer") or "",
        is_grounded=bool(result.get("is_grounded")),
        confidence=result.get("confidence_score") or 0.0,
        retries_used=result.get("retry_count") or 0,
        sources=result.get("cited_sources") or [],
    )


@router.get("/documents")
def list_documents():
    """List the FOMC documents currently ingested -- useful for a frontend
    to show the user what the system actually knows about."""
    if not DATA_DIR.exists():
        return {"documents": []}
    files = sorted(p.name for p in DATA_DIR.glob("*.txt"))
    return {"documents": files}
