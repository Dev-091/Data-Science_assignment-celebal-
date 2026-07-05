from __future__ import annotations

from pathlib import Path

from rag_pipeline import build_answer, load_index, retrieve_chunks

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "artifacts" / "rag_index.pkl"


def ask_question(question: str) -> dict:
    bundle = load_index(INDEX_PATH)
    matches = retrieve_chunks(bundle, question, top_k=4)
    answer = build_answer(question, matches, bundle.vectorizer)
    return {
        "answer": answer,
        "sources": matches,
    }


if __name__ == "__main__":
    if not INDEX_PATH.exists():
        print("Index not found. Run 'python ingest.py' first.")
        raise SystemExit(1)

    while True:
        question = input("Ask a question (or type 'quit'): ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        result = ask_question(question)
        print("\nAnswer:")
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source.source_name} | {source.page_label} | {source.score:.3f}")
