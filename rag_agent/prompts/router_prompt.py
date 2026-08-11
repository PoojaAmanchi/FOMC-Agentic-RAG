"""
Prompt template for the router node.

WHY IS THIS IN ITS OWN FILE?
Keeping prompts separate from the Python logic that calls them means you
can tweak wording, add examples, or test a different phrasing WITHOUT
touching the node's control-flow code. In a real project, prompts get
iterated on far more often than the surrounding logic -- separating them
keeps that iteration low-risk.
"""

# The 3 real FOMC meeting dates in our document collection. The router
# needs to know these explicitly so it can correctly extract dates
# mentioned in a user's question (e.g. "January" should map to
# "2026-01-28", not just be left as a vague word).
KNOWN_MEETING_DATES = ["2026-01-28", "2026-03-18", "2026-06-17"]

ROUTER_PROMPT = """You are a query classifier for a system that answers questions
about Federal Reserve FOMC statements. We have statements from exactly these
3 meetings: {known_dates}.

Classify the following question into exactly one of these types:
- "factual": asks about a single fact from one meeting (e.g. a rate decision,
  a specific number, a specific statement).
- "comparison": asks to compare two or more specific meetings directly.
- "contradiction": asks whether language, tone, or policy stance changed or
  shifted over time, without necessarily naming specific meetings.

Also extract which of the 3 known meeting dates (if any) are directly
relevant to answering this question. If the question doesn't mention or
imply specific dates (e.g. "how has tone evolved" with no dates named),
return an empty list -- this signals the retriever should search everything.

Respond ONLY with valid JSON in exactly this shape, no other text:
{{"query_type": "factual" | "comparison" | "contradiction", "relevant_dates": ["YYYY-MM-DD", ...]}}

Question: {query}
"""
