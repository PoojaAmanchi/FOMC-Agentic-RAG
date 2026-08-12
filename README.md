# FOMC Agentic RAG

A self-correcting, agentic RAG system for answering questions about Federal Reserve (FOMC) statements. Instead of a single retrieve-then-generate pass, the system routes each question, retrieves adaptively based on question type, drafts an answer, and then runs an LLM-based **verifier** that checks the draft against the retrieved passages and triggers a retry if it isn't grounded.

Built with **LangGraph** for orchestration and served via **FastAPI**, with a live progress stream (SSE) so the frontend can show each pipeline step as it happens.

## How it works

The core of the project is a 4-node LangGraph pipeline with a conditional retry loop:

```
router → retriever → synthesizer → verifier ────┬─→ END (grounded answer)
                          ▲                     │
                          └────── retry  ───────┘ (not grounded, retries < max)
```

1. **Router** — classifies the question as `factual`, `comparison`, or `contradiction`, and extracts which known FOMC meeting date(s) it refers to, so retrieval can be narrowed instead of searching blindly.
2. **Retriever** — searches a Chroma vector store. `top_k` and whether to use it depend on question type (`comparison`/`contradiction` questions retrieve more, and use LLM-based **query expansion** — rewording the question 3 different ways and merging/deduplicating results — to catch cross-document matches a single phrasing might miss). On retry, the search widens further.
3. **Synthesizer** — writes an answer using *only* the retrieved passages, citing a numbered passage for every claim (`[1]`, `[2]`, ...). Its instructions adapt to question type (e.g. explicitly comparing wording across dates for `contradiction` questions).
4. **Verifier** — the agentic core. An LLM fact-checker re-reads the draft against the source passages and returns `is_grounded` + a `confidence` score. If either check fails and retries remain (max 2), the graph loops back to the retriever with a wider search; if retries are exhausted, the answer is still returned but with an honest low-confidence disclaimer and a list of any unsupported claims. Explicit "the documents don't cover this" refusals are treated as correctly grounded, so the loop doesn't keep retrying unanswerable questions.

All state is passed between nodes as a single `RAGState` `TypedDict` (see `rag_agent/state.py`), so nodes can be reordered or extended without rewiring each handoff by hand.

## Project structure

```
rag_agent/
  state.py            # shared state schema passed between all nodes
  graph.py             # wires the 4 nodes + retry edge into a LangGraph
  nodes/
    router.py          # query classification + date extraction
    retriever.py        # vector search, query expansion, dedup
    synthesizer.py       # cited draft answer generation
    verifier.py          # groundedness check + retry/disclaimer logic
  prompts/              # prompt templates (incl. known meeting dates)
  utils.py               # shared helpers (LLM response normalization)

api/
  main.py              # FastAPI app setup (CORS, router registration)
  routes.py             # POST /query, GET /documents, GET /health
  streaming.py           # POST /stream/query (SSE progress events)

frontend/               # static HTML/CSS/JS UI (no build step)

data/fomc_statements/    # source FOMC statement .txt files (Jan/Mar/Jun 2026)
ingest.py                # one-time script: chunk + embed documents into Chroma

eval/
  run.py                # scores the pipeline against eval/data/eval_set.json with RAGAS
  ablation.py             # sweeps chunk_size/top_k to find the best-performing settings
  data/eval_set.json       # question + ground_truth pairs for evaluation
```

The 3 known meeting dates (`2026-01-28`, `2026-03-18`, `2026-06-17`) are hardcoded in `rag_agent/prompts/router_prompt.py` so the router can map words like "January" to an exact source filename. The sample FOMC statements in `data/fomc_statements/` are short, synthetic example documents (dated 2026) built for this demo, not a live feed of real Fed releases — swap in real statements and update `KNOWN_MEETING_DATES` to point this at actual data.

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment**

Copy `.env_example` to `.env` and add a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) (no credit card required):

```bash
cp .env_example .env
```

```env
GOOGLE_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash
CHROMA_PERSIST_DIR=./chroma_store
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**3. Build the vector store** (one-time, run before anything else)

```bash
python ingest.py
```

This chunks and embeds the FOMC statements in `data/fomc_statements/` into a local Chroma store.

Every module defaults to `gemini-2.5-flash` if `GEMINI_MODEL` isn't set. Note that `gemini-2.5-flash` is an older generation than the current `gemini-3.x` line (e.g. `gemini-3.5-flash`) — if you switch to a Gemini 3.x model, check its docs first: Google no longer recommends overriding `temperature`/`top_p`/`top_k` on 3.x models, and the router/verifier nodes here rely on `temperature=0` for deterministic JSON output, so behavior may differ from what this project was built and tested against.

## Running it

**Command line** (runs one example query):

```bash
python -m rag_agent.graph
```

**API server:**

```bash
uvicorn api.main:app --reload
```

```bash
# Single request/response
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Did the Fed'"'"'s language about inflation change between January and June 2026?"}'

# Streamed progress (SSE)
curl -N -X POST http://localhost:8000/stream/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Did the January minutes reveal disagreement?"}'
```

Other endpoints: `GET /health`, `GET /documents` (lists ingested FOMC statement files).

**Frontend** (plain HTML/CSS/JS, no build step):

```bash
cd frontend
npm start   # serves on http://localhost:3000
```

The frontend calls `/stream/query` and renders each progress event (`"Classifying your question..."` → `"Searching FOMC documents..."` → `"Drafting an answer..."` → `"Checking the answer is grounded..."`) as it arrives, then shows the final answer with grounded/confidence/retry badges and cited sources. It talks to the API at a hardcoded `http://localhost:8000` (see `frontend/app.js`) — change `API_BASE` there if you run the API elsewhere. It reads the stream manually via `fetch` + `ReadableStream` rather than the browser's `EventSource` API, since `EventSource` only supports GET and this endpoint needs a POST body.

## Evaluation

The eval set (`eval/data/eval_set.json`) has 4 question/ground-truth pairs, including one deliberately out-of-scope question ("What was the Fed's stance on quantitative easing bond purchases in 2026?") to test that the pipeline correctly refuses rather than hallucinates.

**`python -m eval.run`** runs every question through the *full* agentic graph (router → retriever → synthesizer → verifier, including retries) and scores the results with [RAGAS](https://github.com/explodinggradients/ragas) (faithfulness, answer relevancy, context precision). RAGAS is driven by Gemini through Google's OpenAI-compatible endpoint, so the `openai` package is required even though no OpenAI account is used. To stay under the Gemini free tier's rate limit, it sleeps 60 seconds between questions. Each run is saved as a timestamped JSON file under `eval/data/results/` (a few sample runs are already checked in) so changes to the pipeline can be compared over time.

```bash
python -m eval.run
```

**`python -m eval.ablation`** sweeps `chunk_size` (200/400/800) × `top_k` (4/8), rebuilding a separate vector store (`chroma_store_ablation/`) for each chunk size. To keep the sweep's cost manageable, it deliberately uses a *simplified* retrieve-then-generate pass rather than the full agentic graph — no router, no verifier, no retries — since the goal is isolating retrieval quality itself, not the whole pipeline.

```bash
python -m eval.ablation
```

⚠️ Both scripts make real LLM calls and can take several minutes; `eval.ablation` in particular rebuilds the vector store multiple times and runs the whole eval set for every chunk/top_k combination.

## Requirements

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/apikey) API key (Gemini free tier is sufficient)
- Node.js (only if you want to serve the static frontend via `npm start`)
