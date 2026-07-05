from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from ingest import DATA_DIR, INDEX_PATH, UPLOAD_DIR, initialize_ingestion
from rag_pipeline import build_answer, load_index, retrieve_chunks, save_query_log

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
QUERY_LOG_PATH = LOGS_DIR / "query_log.json"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_uploaded_files(uploaded_files) -> int:
    ensure_dirs()
    saved = 0
    for file in uploaded_files:
        target = UPLOAD_DIR / file.name
        target.write_bytes(file.getbuffer())
        saved += 1
    return saved


def clear_uploaded_files() -> None:
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    if QUERY_LOG_PATH.exists():
        QUERY_LOG_PATH.unlink()


@st.cache_resource(show_spinner=False)
def get_bundle(index_mtime: float):
    return load_index(INDEX_PATH)


def reset_chat() -> None:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload your documents, process them once, and then ask questions. I will answer only from the uploaded content.",
            "sources": None,
        }
    ]


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1220;
            --panel: #111827;
            --panel-soft: #172033;
            --panel-strong: #1e293b;
            --border: rgba(148, 163, 184, 0.18);
            --text: #f3f4f6;
            --muted: #94a3b8;
            --accent-a: #2563eb;
            --accent-b: #0f766e;
        }
        .stApp {
            background: radial-gradient(circle at top, rgba(37, 99, 235, 0.14), transparent 26%), linear-gradient(180deg, #09111d 0%, #0f172a 100%);
            color: var(--text);
        }
        .block-container {
            max-width: 980px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, p, label, div {
            color: var(--text);
        }
        h1, h2, h3 {
            font-family: Georgia, "Times New Roman", serif;
        }
        .hero {
            background: linear-gradient(180deg, rgba(23, 32, 51, 0.96) 0%, rgba(17, 24, 39, 0.96) 100%);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            color: #ffffff;
        }
        .hero-copy {
            color: var(--muted);
            line-height: 1.65;
        }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.8rem;
            margin: 1rem 0 0.4rem 0;
        }
        .metric-card {
            background: rgba(17, 24, 39, 0.86);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.9rem;
        }
        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
        }
        .metric-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: #ffffff;
            margin-top: 0.2rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] * {
            color: var(--text);
        }
        div[data-testid="stFileUploader"] {
            background: rgba(17, 24, 39, 0.88);
            border: 1px dashed rgba(148, 163, 184, 0.36);
            border-radius: 16px;
            padding: 0.45rem;
        }
        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] span {
            color: var(--muted);
        }
        div[data-testid="stAlertContainer"] {
            border-radius: 16px;
        }
        div[data-testid="stChatMessage"] {
            background: rgba(17, 24, 39, 0.74);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.25rem 0.35rem;
        }
        div[data-testid="stExpander"] {
            background: rgba(17, 24, 39, 0.82);
            border: 1px solid var(--border);
            border-radius: 16px;
        }
        div[data-testid="stExpander"] * {
            color: var(--text);
        }
        div[data-testid="stChatInput"] textarea,
        div[data-testid="stTextInputRootElement"] input {
            background: rgba(17, 24, 39, 0.94) !important;
            color: var(--text) !important;
            border: 1px solid rgba(148, 163, 184, 0.24) !important;
            border-radius: 14px !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] > div {
            background: rgba(17, 24, 39, 0.94) !important;
            color: var(--text) !important;
        }
        div.stButton > button {
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--accent-a), var(--accent-b));
            color: white;
            font-weight: 600;
        }
        div.stButton > button:hover {
            filter: brightness(1.08);
        }
        .stCaption, .stMarkdown, .stText {
            color: var(--muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Week 7 RAG", page_icon="📄", layout="wide")
    ensure_dirs()
    apply_styles()

    if "messages" not in st.session_state:
        reset_chat()

    with st.sidebar:
        st.header("Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "md", "csv", "json"],
            accept_multiple_files=True,
        )

        if st.button("Process documents", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload at least one file first.")
            else:
                saved = save_uploaded_files(uploaded_files)
                with st.spinner("Building your document index..."):
                    stats = initialize_ingestion()
                get_bundle.clear()
                reset_chat()
                st.session_state["stats"] = stats
                st.success(f"Processed {saved} file(s). You can ask questions now.")
                st.rerun()

        if st.button("Reset uploaded data", use_container_width=True):
            clear_uploaded_files()
            get_bundle.clear()
            reset_chat()
            st.session_state.pop("stats", None)
            st.success("Uploaded files and index cleared.")
            st.rerun()

        st.caption("Supported formats: PDF, TXT, MD, CSV, JSON")

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Simple RAG Question Answering</div>
            <div class="hero-copy">Upload your own documents, process them once, and ask natural questions. Answers are built only from the text found in your uploaded files.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = st.session_state.get("stats")
    bundle = None
    if INDEX_PATH.exists():
        bundle = get_bundle(INDEX_PATH.stat().st_mtime)
        if stats is None:
            stats = bundle.stats
            st.session_state["stats"] = stats

    if stats:
        st.markdown(
            f"""
            <div class="metric-strip">
                <div class="metric-card"><div class="metric-label">Documents</div><div class="metric-value">{stats['documents']}</div></div>
                <div class="metric-card"><div class="metric-label">Chunks</div><div class="metric-value">{stats['chunks']}</div></div>
                <div class="metric-card"><div class="metric-label">Index Built</div><div class="metric-value">{stats['built_at']}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Upload documents from the sidebar and click Process documents to start.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                with st.expander("Sources used"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['source_name']} | {source['page_label']} | score {source['score']:.3f}**")
                        st.write(source["text"])

    user_query = st.chat_input("Ask a question from the uploaded documents", disabled=bundle is None)
    if user_query and bundle is not None:
        st.session_state.messages.append({"role": "user", "content": user_query, "sources": None})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching the documents..."):
                matches = retrieve_chunks(bundle, user_query, top_k=4)
                answer = build_answer(user_query, matches, bundle.vectorizer)
                save_query_log(QUERY_LOG_PATH, user_query, answer, matches)
            st.write(answer)
            source_payload = [
                {
                    "source_name": source.source_name,
                    "page_label": source.page_label,
                    "score": source.score,
                    "text": source.text,
                }
                for source in matches
            ]
            if source_payload:
                with st.expander("Sources used"):
                    for source in source_payload:
                        st.markdown(f"**{source['source_name']} | {source['page_label']} | score {source['score']:.3f}**")
                        st.write(source["text"])
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": source_payload})


if __name__ == "__main__":
    main()
