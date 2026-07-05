from __future__ import annotations

from pathlib import Path

from rag_pipeline import build_index

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "rag_index.pkl"


def initialize_ingestion(source_dir: Path | None = None) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = source_dir or UPLOAD_DIR
    bundle = build_index(target_dir, INDEX_PATH)
    return bundle.stats


if __name__ == "__main__":
    stats = initialize_ingestion()
    print("Index built successfully")
    for key, value in stats.items():
        print(f"{key}: {value}")
