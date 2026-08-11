"""
THE ROUTER NODE
================
First stop for every question. Its ONLY job: read the raw question and
decide two things BEFORE any searching happens:

  1. query_type   -- factual / comparison / contradiction. This determines
                      which prompt template the synthesizer uses later, and
                      (in a more advanced version) how broadly the retriever
                      searches.
  2. relevant_dates -- which of our 3 known FOMC meetings this question is
                      actually about, if any. If the question says "in
                      March," we want that mapped to "2026-03-18" so the
                      retriever can narrow its search instead of searching
                      everything blindly.

We ask Claude to respond in STRICT JSON (nothing else) so we can parse it
reliably with json.loads() -- no fragile regex-scraping of free-form text.
"""
import json
import os
from rag_agent.utils import extract_text
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_agent.prompts.router_prompt import KNOWN_MEETING_DATES, ROUTER_PROMPT
from rag_agent.state import RAGState

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _get_llm():
    # temperature=0 because classification should be as consistent as
    # possible -- we don't want the router randomly flip-flopping on the
    # same question asked twice.
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)


def router_node(state: RAGState) -> dict:
    """
    LangGraph convention: every node is a plain function that takes the
    current state and returns a dict of ONLY the fields it wants to
    update. LangGraph merges this dict into the full state automatically
    -- the router never needs to touch fields it doesn't care about.
    """
    query = state["query"]

    prompt = ROUTER_PROMPT.format(
        known_dates=", ".join(KNOWN_MEETING_DATES),
        query=query,
    )

    llm = _get_llm()
    response = llm.invoke([("human", prompt)])
    raw_text = extract_text(response.content).strip()

    # Defensive parsing: LLMs occasionally wrap JSON in markdown code
    # fences even when told not to. Strip those before parsing.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
        query_type = parsed.get("query_type", "factual")
        relevant_dates = parsed.get("relevant_dates", [])
    except json.JSONDecodeError:
        # If the LLM ever fails to return valid JSON, fail SAFE rather than
        # crash the whole pipeline -- default to the broadest search
        # strategy so the user still gets an answer, just a less targeted
        # one.
        query_type = "factual"
        relevant_dates = []

    return {
        "query_type": query_type,
        "relevant_dates": relevant_dates,
    }
