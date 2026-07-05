# Week 7 Simple RAG

This version is rebuilt from scratch for a simpler user experience and more reliable answers.

## Files

- `app.py` - Streamlit frontend
- `ingest.py` - document ingestion and index build entrypoint
- `query.py` - question answering entrypoint
- `rag_pipeline.py` - chunking, indexing, retrieval, and answer construction
- `data/uploads/` - documents uploaded through the app
- `artifacts/rag_index.pkl` - saved local index
- `logs/` - optional logs folder

## How it works

1. Upload PDF, TXT, MD, CSV, or JSON files from the sidebar.
2. Click `Process documents`.
3. Ask questions in the chat box.
4. The app retrieves the most relevant chunks and builds an answer directly from those sentences.

## Run

```bash
python -m streamlit run week7_Dev_sharma/app.py
```

## Notes

- Only files uploaded through the app are indexed.
- This app is optimized for fast local retrieval.
- Answers are extractive and grounded in your uploaded text.
- No external model server is required.
