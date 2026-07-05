from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json"}
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkRecord:
    chunk_id: str
    source_path: str
    source_name: str
    page_label: str
    text: str


@dataclass
class SearchResult:
    score: float
    source_name: str
    page_label: str
    text: str


@dataclass
class IndexBundle:
    vectorizer: TfidfVectorizer
    matrix: Any
    chunks: list[ChunkRecord]
    stats: dict[str, Any]


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) > 2]


def keyword_overlap(question: str, text: str) -> float:
    q_tokens = tokenize(question)
    if not q_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    matches = sum(1 for token in q_tokens if token in text_tokens)
    return matches / len(set(q_tokens))


def split_sentences(text: str) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    parts = SENTENCE_SPLIT_PATTERN.split(clean)
    return [part.strip() for part in parts if part.strip()]


def read_pdf(path: Path) -> list[tuple[str, str]]:
    try:
        from pypdf import PdfReader
    except Exception:
        from PyPDF2 import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[str, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if normalize_text(text):
            pages.append((f"Page {index}", text))
    return pages


def read_text_file(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [("Full document", text)] if normalize_text(text) else []


def load_document_pages(path: Path) -> list[tuple[str, str]]:
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    return read_text_file(path)


def chunk_page(source_path: Path, page_label: str, text: str, sentences_per_chunk: int = 5, overlap: int = 1) -> list[ChunkRecord]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[ChunkRecord] = []
    step = max(1, sentences_per_chunk - overlap)
    for start in range(0, len(sentences), step):
        window = sentences[start : start + sentences_per_chunk]
        if not window:
            continue
        chunk_text = " ".join(window).strip()
        if len(chunk_text) < 40:
            continue
        chunk_id = f"{source_path.stem}-{page_label.replace(' ', '_')}-{start}"
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                source_path=str(source_path),
                source_name=source_path.name,
                page_label=page_label,
                text=chunk_text,
            )
        )
        if start + sentences_per_chunk >= len(sentences):
            break
    return chunks


def collect_supported_files(data_dir: Path) -> list[Path]:
    files = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def build_index(data_dir: Path, index_path: Path, sentences_per_chunk: int = 5, overlap: int = 1) -> IndexBundle:
    files = collect_supported_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No supported documents found in {data_dir}")

    chunks: list[ChunkRecord] = []
    for file_path in files:
        for page_label, text in load_document_pages(file_path):
            chunks.extend(chunk_page(file_path, page_label, text, sentences_per_chunk=sentences_per_chunk, overlap=overlap))

    if not chunks:
        raise ValueError("Documents were loaded, but no readable text chunks were created.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    stats = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "documents": len(files),
        "chunks": len(chunks),
        "sentences_per_chunk": sentences_per_chunk,
        "overlap": overlap,
        "files": [file.name for file in files],
    }

    bundle = IndexBundle(vectorizer=vectorizer, matrix=matrix, chunks=chunks, stats=stats)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return bundle


def load_index(index_path: Path) -> IndexBundle:
    with index_path.open("rb") as handle:
        return pickle.load(handle)


def retrieve_chunks(bundle: IndexBundle, question: str, top_k: int = 4) -> list[SearchResult]:
    query_vector = bundle.vectorizer.transform([question])
    dense_scores = linear_kernel(query_vector, bundle.matrix).ravel()
    scored: list[SearchResult] = []
    for chunk, dense_score in zip(bundle.chunks, dense_scores):
        lexical_score = keyword_overlap(question, chunk.text)
        final_score = (0.8 * float(dense_score)) + (0.2 * lexical_score)
        scored.append(
            SearchResult(
                score=final_score,
                source_name=chunk.source_name,
                page_label=chunk.page_label,
                text=chunk.text,
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def build_answer(question: str, matches: list[SearchResult], vectorizer: TfidfVectorizer) -> str:
    candidate_sentences: list[tuple[float, str]] = []
    for match in matches:
        for sentence in split_sentences(match.text):
            sentence_vector = vectorizer.transform([sentence])
            dense_score = float(linear_kernel(vectorizer.transform([question]), sentence_vector).ravel()[0])
            lexical_score = keyword_overlap(question, sentence)
            total_score = (0.75 * dense_score) + (0.25 * lexical_score)
            if total_score > 0:
                candidate_sentences.append((total_score + match.score, sentence))

    candidate_sentences.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, sentence in candidate_sentences:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) == 3:
            break

    if not selected:
        return "I could not find a relevant answer in the uploaded documents."
    return " ".join(selected)


def save_query_log(log_path: Path, question: str, answer: str, matches: list[SearchResult]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "sources": [asdict(match) for match in matches],
    }
    existing: list[dict[str, Any]] = []
    if log_path.exists():
        existing = json.loads(log_path.read_text(encoding="utf-8"))
    existing.append(payload)
    log_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
