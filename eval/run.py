"""
EVAL RUNNER
=============
Loads eval/data/eval_set.json, runs every question through the full
agentic graph, scores with RAGAS, and saves a TIMESTAMPED result file to
eval/data/results/. Keeping every run's results (instead of overwriting
one file) lets you track whether changes you make over time actually
improve or hurt quality -- e.g. compare today's run against last week's
after you tweak chunk_size or the confidence threshold.

Run with:
    python -m eval.run

RAGAS version note:
This project uses ragas>=0.4. The evaluate() API changed significantly
from 0.1/0.2:
  - Dataset format: EvaluationDataset(SingleTurnSample(...)) instead of
    HuggingFace Dataset with 'question'/'answer'/'contexts'/'ground_truth'
  - Column names: user_input, response, retrieved_contexts, reference
  - LLM: ragas 0.4 uses instructor-based structured output; pass the llm
    via llm_factory() with an OpenAI-compatible client (Google supports this).
  - Metrics: imported from ragas.metrics.collections (not ragas.metrics)
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.evaluation import evaluate
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    Faithfulness,
)

from rag_agent.graph import run_query

load_dotenv()

EVAL_SET_PATH = Path(__file__).resolve().parent / "data" / "eval_set.json"
RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Seconds to wait between questions to stay under the free tier
# rate limit (5 req/min). Each question uses ~3 LLM calls with
# query expansion disabled, so 60s gives the window time to reset.
SLEEP_BETWEEN_QUESTIONS = 60


def _make_ragas_llm():
    try:
        import openai
        client = openai.OpenAI(
            api_key=GOOGLE_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return llm_factory(model=GEMINI_MODEL, client=client, provider="openai")
    except Exception as e:
        raise RuntimeError(
            f"Could not create ragas LLM. Make sure GOOGLE_API_KEY is set "
            f"and the openai package is installed. Original error: {e}"
        )


def _make_ragas_embeddings():
    import openai
    from ragas.embeddings import embedding_factory

    client = openai.OpenAI(
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    return embedding_factory(
        "openai",
        model="text-embedding-004",
        client=client,
        interface="modern",
    )


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH) as f:
        return json.load(f)


def build_dataset(eval_set: list[dict]) -> tuple[EvaluationDataset, list[dict]]:
    samples = []
    raw_runs = []

    for i, item in enumerate(eval_set):
        result = run_query(item["question"])
        final_answer = result.get("final_answer") or ""
        retrieved_chunks = result.get("retrieved_chunks") or []
        context_texts = [c["text"] for c in retrieved_chunks]

        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=final_answer,
                retrieved_contexts=context_texts,
                reference=item["ground_truth"],
            )
        )

        raw_runs.append(
            {
                "question": item["question"],
                "is_grounded": result.get("is_grounded"),
                "confidence": result.get("confidence_score"),
                "retries_used": result.get("retry_count"),
                "sources_cited": result.get("cited_sources"),
            }
        )
        print(
            f"  done: {item['question'][:60]}... "
            f"(grounded={result.get('is_grounded')}, retries={result.get('retry_count')})"
        )

        # Wait between questions to stay under the free tier rate limit.
        # Skip the sleep after the last question.
        if i < len(eval_set) - 1:
            print(f"  waiting {SLEEP_BETWEEN_QUESTIONS}s before next question...")
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    return EvaluationDataset(samples=samples), raw_runs


def main():
    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} questions from {EVAL_SET_PATH}")
    print("Running through the full agentic graph...")

    dataset, raw_runs = build_dataset(eval_set)

    print("\nSetting up RAGAS evaluator (ragas 0.4 API)...")
    ragas_llm = _make_ragas_llm()
    ragas_embeddings = _make_ragas_embeddings()

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecisionWithReference(llm=ragas_llm),
    ]

    print("Running RAGAS evaluation...")
    results = evaluate(dataset=dataset, metrics=metrics)

    print("\n=== RAGAS Evaluation Results ===")
    print(results)

    df = results.to_pandas()

    combined = []
    for i, row in df.iterrows():
        combined.append(
            {
                **raw_runs[i],
                "faithfulness": _safe_float(row.get("faithfulness")),
                "answer_relevancy": _safe_float(row.get("answer_relevancy")),
                "context_precision": _safe_float(row.get("context_precision")),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nSaved timestamped results to {out_path}")
    print("\nRun this again after making changes to compare results over time.")


def _safe_float(value) -> float | None:
    try:
        f = float(value)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()