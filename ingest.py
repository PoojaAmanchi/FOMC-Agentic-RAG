"""
INGESTION -- run this ONCE before anything else, to build the vector store.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent / "data" / "fomc_statements"
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore():
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        loaded = TextLoader(str(path), encoding="utf-8").load()
        for d in loaded:
            d.metadata["source"] = path.name
        docs.extend(loaded)

    if not docs:
        raise RuntimeError(f"No .txt files found in {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR
    )
    vectorstore.persist()
    print(f"Ingested {len(docs)} docs -> {len(chunks)} chunks -> {PERSIST_DIR}")


if __name__ == "__main__":
    build_vectorstore()
