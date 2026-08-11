"""
THE VERIFIER NODE -- the "agentic" heart of this system
===========================================================
After the synthesizer writes a draft answer, this node double-checks it
against the actual retrieved passages BEFORE we commit to showing it to
the user.

KEY FIX (out-of-scope detection):
The original verifier would sometimes mark an answer as grounded even
when the synthesizer answered a question the documents don't actually
cover -- because the LLM conflated "plausible answer" with "grounded in
text". The prompt now explicitly instructs the verifier to treat the
synthesizer's own "I don't have enough information" responses as a
special case: those should be marked grounded=True at high confidence,
since refusing to answer is the CORRECT response for out-of-scope
questions. This stops the retry loop from re-running on unanswerable
questions and prevents hallucinated answers from slipping through as
"grounded".
"""
import json
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_agent.utils import extract_text
from rag_agent.state import RAGState

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_CONFIDENCE_THRESHOLD = 0.7
_MAX_RETRIES = 2

VERIFICATION_PROMPT = """You are a rigorous fact-checker. Below are numbered
source passages and a draft answer that claims to be based on them.

SPECIAL CASE -- out-of-scope answers:
If the draft answer explicitly says it cannot find information in the
provided passages (e.g. "the documents do not contain", "I could not find",
"this is not addressed in the passages"), treat this as CORRECT and return:
  {{"is_grounded": true, "confidence": 0.95, "unsupported_claims": []}}
Refusing to answer when the context doesn't support an answer is the right
behaviour, not a failure to ground.

For all other answers:
Check EVERY factual claim in the draft answer against the passages. A claim
is only "grounded" if it is directly supported by the text -- not
plausible-sounding, not "probably true," but actually stated in the passages.

Respond ONLY with valid JSON in exactly this shape:
{{"is_grounded": true or false, "confidence": 0.0 to 1.0, "unsupported_claims": ["...", "..."]}}

Passages:
{numbered_passages}

Draft answer:
{draft_answer}
"""


def _format_numbered_passages(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")
    return "\n\n".join(parts)


def verifier_node(state: RAGState) -> dict:
    draft_answer = state.get("draft_answer") or ""
    chunks = state.get("retrieved_chunks") or []
    retry_count = state.get("retry_count") or 0

    if not draft_answer:
        return {
            "is_grounded": False,
            "confidence_score": 0.0,
            "final_answer": "Unable to generate an answer.",
            "retry_count": retry_count,
        }

    numbered_passages = _format_numbered_passages(chunks)
    prompt = VERIFICATION_PROMPT.format(
        numbered_passages=numbered_passages,
        draft_answer=draft_answer,
    )

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    response = llm.invoke([("human", prompt)])
    raw_text = extract_text(response.content).strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        verdict = json.loads(raw_text)
        is_grounded = verdict.get("is_grounded", False)
        confidence = float(verdict.get("confidence", 0.0))
        unsupported = verdict.get("unsupported_claims", [])
    except (json.JSONDecodeError, ValueError):
        is_grounded = False
        confidence = 0.0
        unsupported = []

    if is_grounded and confidence >= _CONFIDENCE_THRESHOLD:
        return {
            "is_grounded": True,
            "confidence_score": confidence,
            "final_answer": draft_answer,
            "retry_count": retry_count,
        }

    if retry_count < _MAX_RETRIES:
        return {
            "is_grounded": False,
            "confidence_score": confidence,
            "retry_count": retry_count + 1,
        }

    disclaimer = (
        f"\n\n⚠️ Low-confidence answer (confidence: {confidence:.0%}). "
        "Some claims may not be fully supported by the source documents."
    )
    if unsupported:
        disclaimer += "\n\nPotentially unsupported claims:\n"
        for claim in unsupported:
            disclaimer += f"- {claim}\n"

    return {
        "is_grounded": False,
        "confidence_score": confidence,
        "final_answer": draft_answer + disclaimer,
        "retry_count": retry_count,
    }
