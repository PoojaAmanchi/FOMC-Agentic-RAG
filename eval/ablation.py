"""
ABLATION STUDY
================
Systematically sweeps combinations of chunk_size and top_k, rebuilding the
vector store for each chunk_size and re-running the full eval set, to
empirically find which settings actually produce the best RAGAS scores --
instead of guessing or leaving defaults unexamined.

WHY THIS MATTERS FOR INTERVIEWS:
Anyone can pick chunk_size=400 and top_k=4 because a tutorial used those
numbers. Being able to say "I swept chunk sizes from 200 to 800 and
measured the actual faithfulness/precision impact of each" is a
fundamentally different, stronger claim -- it shows you treat these as
real tunable parameters with measurable tradeoffs, not magic constants.

WHAT AN ABLATION STUDY ACTUALLY IS (in case this term is new):
"Ablation" means deliberately removing or changing ONE variable at a time
and measuring the effect on output quality, so you can isolate which
variable actually matters. Here we sweep two variables (chunk_size,
top_k) across a small grid and record scores for every combination.

Run with:
    python -m eval.ablation

WARNING: This rebuilds the vector store multiple times and runs the full
eval set through the agentic graph for every combination -- this makes
many LLM calls and will take several minutes and cost real API usage.

RAGAS version note: updated for ragas 0.4 API (EvaluationDataset,
SingleTurnSample, metric instantiation with LLM).
"""
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.evaluation import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    Faithfulness,
)

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fomc_statements"
EVAL_SET_PATH = Path(__file__).resolve().parent / "data" / "eval_set.json"
RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"
ABLATION_PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_store_ablation"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# The grid we're sweeping. Kept small deliberately -- this is a portfolio
# project, not a production hyperparameter search.
CHUNK_SIZES = [200, 400, 800]
TOP_KS = [4, 8]


def _make_ragas_llm():
    import openai
    client = openai.OpenAI(
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    return llm_factory(model=GEMINI_MODEL, client=client, provider="openai")


def _make_ragas_embeddings():
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH) as f:
        return json.load(f)


def build_vectorstore_for_chunk_size(chunk_size: int):
    """Rebuild the vector store from scratch with a specific chunk_size.
    We use a SEPARATE persist directory from the main app's vector store
    so ablation runs never corrupt your actual working index."""
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        loaded = TextLoader(str(path), encoding="utf-8").load()
        for d in loaded:
            d.metadata["source"] = path.name
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 6,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    persist_dir = str(ABLATION_PERSIST_DIR / f"chunk_{chunk_size}")
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_dir
    )
    return vectorstore


def run_eval_for_combo(
    vectorstore, top_k: int, eval_set: list[dict], ragas_llm, ragas_embeddings
) -> dict:
    """Run a simplified retrieve+generate pass (not the full agentic graph
    with router/verifier -- ablation studies isolate retrieval quality
    itself, so we skip the extra agentic machinery to keep each combo's
    cost and runtime manageable) and score with RAGAS."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    samples = []
    for item in eval_set:
        docs = retriever.invoke(item["question"])
        context_texts = [d.page_content for d in docs]
        numbered = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(context_texts))
        prompt = (
            f"Answer using ONLY this context, citing [N] for each claim:\n\n"
            f"{numbered}\n\nQuestion: {item['question']}"
        )
        response = llm.invoke([("human", prompt)])
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=response.content,
                retrieved_contexts=context_texts,
                reference=item["ground_truth"],
            )
        )

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecisionWithReference(llm=ragas_llm),
    ]
    results = evaluate(
        dataset=EvaluationDataset(samples=samples), metrics=metrics
    )
    df = results.to_pandas()

    def _safe(val):
        try:
            f = float(val)
            return None if f != f else round(f, 4)
        except (TypeError, ValueError):
            return None

    return {
        "faithfulness": _safe(df["faithfulness"].mean()),
        "answer_relevancy": _safe(df["answer_relevancy"].mean()),
        "context_precision": _safe(df["context_precision"].mean()),
    }


def main():
    eval_set = load_eval_set()
    ragas_llm = _make_ragas_llm()
    ragas_embeddings = _make_ragas_embeddings()
    all_results = []

    for chunk_size in CHUNK_SIZES:
        print(f"\n--- Building vector store: chunk_size={chunk_size} ---")
        vectorstore = build_vectorstore_for_chunk_size(chunk_size)

        for top_k in TOP_KS:
            print(f"  Running eval: chunk_size={chunk_size}, top_k={top_k}")
            scores = run_eval_for_combo(
                vectorstore, top_k, eval_set, ragas_llm, ragas_embeddings
            )
            all_results.append({"chunk_size": chunk_size, "top_k": top_k, **scores})
            print(f"    -> {scores}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"ablation_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Ablation sweep complete. Saved to {out_path} ===")
    print("\nSummary (sorted by faithfulness, best first):")
    sortable = [r for r in all_results if r["faithfulness"] is not None]
    for r in sorted(sortable, key=lambda x: x["faithfulness"], reverse=True):
        print(
            f"  chunk_size={r['chunk_size']:>4}  top_k={r['top_k']:>2}  "
            f"faithfulness={r['faithfulness']:.2f}  "
            f"relevancy={r['answer_relevancy']:.2f}  "
            f"precision={r['context_precision']:.2f}"
        )


if __name__ == "__main__":
    main()
