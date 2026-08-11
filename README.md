# FOMC Agentic RAG — Self-Correcting LangGraph Pipeline

Project 2 in the 6-project agentic AI portfolio path. A LangGraph state
machine that classifies questions, retrieves adaptively (with query
expansion), generates cited answers, and **checks its own work and
retries if it isn't grounded** — served via a real FastAPI backend with
streaming progress, plus a minimal frontend and a full RAGAS/ablation
evaluation suite.

Structured to match a real production-style agentic RAG repo: separate
`api/`, `rag_agent/`, `eval/`, and `frontend/` packages, not one flat
script.

---

## What makes this "agentic" (not just RAG)

1. **Router** classifies the question (factual / comparison /
   contradiction) before any searching happens
2. **Retriever** narrows or widens search based on that classification,
   and uses **query expansion** (rewording the question 3 ways via LLM)
   for comparison/contradiction questions or any retry
3. **Synthesizer** writes a cited answer using only retrieved passages
4. **Verifier** checks the draft against the actual retrieved text —
   if it isn't grounded, it loops back to the retriever (up to 2 retries)
   before honestly flagging low confidence rather than pretending certainty.
   Out-of-scope questions ("documents don't contain this") are correctly
   treated as grounded refusals, not retrieval failures.

## Architecture

```
        query
          |
          v
      +--------+
      | Router |  classifies: factual / comparison / contradiction
      +---+----+
          v
     +-----------+
     | Retriever |  Chroma search + query expansion for hard questions
     +-----+-----+
           v
     +-------------+
     | Synthesizer |  writes answer, cites passage numbers
     +------+------+
            v
      +----------+      not grounded, retries left
      | Verifier | --------------------------------+
      +----+-----+                                   |
           | grounded, or retries exhausted           |
           v                                          |
      final_answer                              back to Retriever
```

## Project structure

```
fomc-agentic-rag/
├── data/fomc_statements/       # 3 real FOMC documents
├── rag_agent/
│   ├── state.py                 # shared state passed between all nodes
│   ├── graph.py                  # wires nodes together + the retry loop
│   ├── nodes/
│   │   ├── router.py             # classifies question, extracts dates
│   │   ├── retriever.py           # Chroma search + query expansion
│   │   ├── synthesizer.py          # writes cited answer
│   │   └── verifier.py             # grounding check + retry decision
│   └── prompts/
│       └── router_prompt.py       # router's classification instructions
├── api/
│   ├── main.py                    # FastAPI app creation, CORS, router wiring
│   ├── routes.py                  # /query, /health, /documents endpoints
│   └── streaming.py                # /stream/query -- SSE live progress
├── frontend/
│   ├── index.html                 # minimal UI: question box + live progress
│   ├── app.js                      # connects to /stream/query, renders results
│   ├── styles.css
│   └── package.json                # `npm start` to serve locally
├── eval/
│   ├── data/
│   │   ├── eval_set.json           # externalized test questions + ground truth
│   │   └── results/                 # timestamped eval run outputs (JSON)
│   ├── run.py                       # runs eval_set.json through the full graph
│   └── ablation.py                   # sweeps chunk_size x top_k, measures impact
├── ingest.py                      # builds the Chroma vector store
├── requirements.txt
├── .env.example                   # copy to .env and add your key
└── .gitignore
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# add your GOOGLE_API_KEY -- get one FREE, no credit card, at
# https://aistudio.google.com/apikey
```

**Why Gemini instead of a paid API:** Google's Gemini API free tier is
genuinely free with no expiry (unlike Anthropic/OpenAI's one-time trial
credits) -- roughly 1,500 requests/day on Gemini Flash, which is more
than enough to build, test, and break this project without spending
anything. The whole system is built against LangChain's standard chat
model interface, so swapping to a different provider (Claude, GPT, Grok)
later is a two-line change per file, not a rewrite -- see any node file's
top few lines for exactly what changes.

## Run it

```bash
# 1. Build the vector store (once)
python ingest.py

# 2. Test the full agentic loop directly, see every step in the terminal
python -m rag_agent.graph

# 3. Start the API
uvicorn api.main:app --reload

# 4. In a separate terminal, serve the frontend
cd frontend && npm start
# open http://localhost:3000
```

With the frontend running, type a question and watch the progress
messages appear live ("Classifying your question..." then "Searching FOMC
documents..." then "Drafting an answer..." then "Checking the answer is
grounded...") before the final answer renders with its confidence score,
grounded/not-grounded badge, retry count, and cited sources.

**Plain (non-streaming) API usage:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Did the Feds language about inflation change between January and June 2026?"}'
```

**Streaming usage directly via curl:**
```bash
curl -N -X POST http://localhost:8000/stream/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Did the January minutes reveal disagreement?"}'
```

## Evaluate it

```bash
python -m eval.run
```

Loads `eval/data/eval_set.json`, runs every question through the full
agentic graph, scores with RAGAS (faithfulness, answer relevancy, context
precision), and saves a **timestamped** JSON to `eval/data/results/` —
run it again after any change and compare files to see if you actually
improved things, instead of just assuming you did.

**RAGAS version note:** this project uses ragas 0.4, which introduced a
new API. Metrics are now imported from `ragas.metrics.collections` and
instantiated with the LLM directly. The eval uses Google's
OpenAI-compatible endpoint so ragas's `llm_factory` (which expects an
OpenAI-style client) works with Gemini without any provider plugin.

## Run the ablation study

```bash
python -m eval.ablation
```

Systematically rebuilds the vector store at 3 different chunk sizes (200,
400, 800 characters) and tests 2 different `top_k` values (4, 8) for
each — 6 combinations total — measuring RAGAS scores for every one. This
answers, with actual evidence, "what chunk size and retrieval depth
actually work best for this data" instead of leaving defaults unexamined.

**Note:** this makes many LLM calls (one full eval pass per combination)
and will take several minutes and real API cost. Reduce `CHUNK_SIZES` or
`TOP_KS` in `eval/ablation.py` if you want a faster/cheaper run.

## Things to actually try

1. **Watch query expansion happen.** Ask a comparison/contradiction
   question and add a `print(queries_to_run)` in `retriever.py` temporarily
   -- see the 3 LLM-generated rewrites alongside your original question.
2. **Watch the retry loop trigger** in the terminal or the frontend's
   live progress -- ask something deliberately hard and see `retriever`
   run twice.
3. **Run the ablation study and read the summary table.** Which
   chunk_size actually won on faithfulness? Was it the default (400) or
   something else? This is real evidence for your interview answer about
   why you chose the settings you did.
4. **Compare `eval/data/results/` files over time** after making any
   change (e.g. lowering the verifier's confidence threshold) -- this is
   exactly the workflow a production ML team uses to catch regressions.

## Interview prep

**Q: Walk me through what happens when I ask a comparison question.**
Router classifies it as "comparison" and extracts any named meeting
dates. Retriever expands the query into 4 phrasings (query expansion
kicks in for comparison/contradiction types), searches with all 4,
narrows by date if dates were extracted, and deduplicates results.
Synthesizer writes an answer using a comparison-specific prompt that
explicitly asks it to address each meeting being compared. Verifier
checks every claim against the retrieved passages before accepting the
answer.

**Q: Why does query expansion only run for comparison/contradiction
questions, not every question?**
Cost/latency tradeoff -- a simple factual lookup rarely benefits enough to
justify 4x the embedding calls and a wider, noisier result set. We reserve
the expensive strategy for question types where catching multiple
phrasings across different documents actually matters most.

**Q: How does the verifier handle questions the documents don't cover?**
The verifier prompt includes an explicit out-of-scope rule: if the
synthesizer's answer says it can't find the information in the passages,
that's marked grounded=True at high confidence. This stops the retry loop
from re-running pointlessly on unanswerable questions and prevents the
fallback disclaimer from appearing for correct "I don't know" answers.

**Q: What did your ablation study actually show?**
Answer this with YOUR real numbers once you run it. Something like: "At
chunk_size=400, faithfulness was highest, but context precision dropped
slightly at chunk_size=800 because larger chunks introduced more
irrelevant text alongside the relevant sentence."

**Q: Why separate api/routes.py from api/streaming.py instead of one
file?**
Both handle HTTP concerns but solve different problems -- routes.py is
simple request/response, streaming.py manages a persistent connection and
a generator function. Keeping them separate means changes to one don't
risk breaking the other, and it's immediately clear where to look for
each kind of endpoint.

**Q: Why SSE instead of WebSockets for the streaming endpoint?**
We only need one-way communication (server pushes progress to the
browser); we never need the browser to send data back mid-stream. SSE is
simpler to implement and works over plain HTTP, so it's the right tool
for this specific need -- reaching for WebSockets here would be
unnecessary complexity.

## What's next

Project 3 takes the "structured, relational data" angle further with a
Neo4j-backed knowledge graph -- modeling entities (people, meetings,
policy positions) and their relationships explicitly, instead of relying
purely on semantic similarity search.
