"""
build_sva_examples.py

Builds a ChromaDB collection 'sva_examples' from the NL2SVA annotated
benchmark data (FVEval/data_nl2sva/annotated_instructions_with_signals/).

Each document contains the NL spec as page_content and the reference SVA
assertion in metadata, enabling similarity-based retrieval at Phase 4 time.
"""

import argparse
import json
import os
import glob

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings


ANNOTATED_DIR = os.path.join(
    os.path.dirname(__file__),
    "FVEval", "data_nl2sva", "annotated_instructions_with_signals",
)
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "sva_examples"


def load_examples() -> list[Document]:
    """Parse all .jsonl + .sva pairs from the annotated NL2SVA dataset."""
    docs = []
    design_dirs = sorted(glob.glob(os.path.join(ANNOTATED_DIR, "*")))

    for design_dir in design_dirs:
        if not os.path.isdir(design_dir):
            continue
        design_name = os.path.basename(design_dir)

        jsonl_files = glob.glob(os.path.join(design_dir, "*.jsonl"))
        for jsonl_path in jsonl_files:
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    task_id = entry["task_id"]
                    prompt = entry["prompt"]

                    sva_path = os.path.join(design_dir, f"{task_id}.sva")
                    if not os.path.exists(sva_path):
                        continue

                    with open(sva_path) as sf:
                        ref_sva = sf.read().strip()

                    doc = Document(
                        page_content=f"Assertion specification: {prompt}",
                        metadata={
                            "reference_sva": ref_sva,
                            "design_name": design_name,
                            "task_id": task_id,
                        },
                    )
                    docs.append(doc)

    return docs


def build_collection(db_path: str = None):
    """Build (or rebuild) the sva_examples ChromaDB collection."""
    persist_dir = db_path or CHROMA_DB_PATH

    docs = load_examples()
    print(f"Loaded {len(docs)} (spec, SVA) pairs from NL2SVA annotated data")

    if not docs:
        print("No examples found. Check ANNOTATED_DIR path.")
        return

    ollama_url = os.getenv("OLLAMA_BASE_URL")
    ollama_model = os.getenv("OLLAMA_EMBEDDING_MODEL")
    if ollama_url and ollama_model:
        embeddings = OllamaEmbeddings(model=ollama_model, base_url=ollama_url)
        print(f"Using OllamaEmbeddings: {ollama_model} at {ollama_url}")
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Using HuggingFaceEmbeddings: all-MiniLM-L6-v2")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    existing = vector_store._collection.count()
    if existing > 0:
        print(f"Collection '{COLLECTION_NAME}' already has {existing} docs. Clearing...")
        vector_store._collection.delete(where={"task_id": {"$ne": ""}})

    vector_store.add_documents(docs)
    print(f"Stored {len(docs)} documents in '{COLLECTION_NAME}' collection")

    test_results = vector_store.similarity_search(
        "Check that the counter does not overflow", k=3
    )
    print(f"\nTest query: 'Check that the counter does not overflow'")
    for i, r in enumerate(test_results, 1):
        print(f"  [{i}] {r.metadata['design_name']}/{r.metadata['task_id']}")
        print(f"      Spec: {r.page_content[:80]}...")
        print(f"      SVA:  {r.metadata['reference_sva'][:80]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default=None,
                        help="ChromaDB persist directory (default: ./chroma_db)")
    args = parser.parse_args()
    build_collection(db_path=args.db_path)
