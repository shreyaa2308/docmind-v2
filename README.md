# DocMind v2

DocMind v2 is a PDF question-answering service: it extracts prose, tables, and embedded images; captions images with GPT-4o mini; indexes chunks in local-embedding Qdrant; combines BM25 and dense retrieval; and answers only from reranked context.

## Files

- `src/config.py` loads every setting from environment variables and `.env`.
- `src/ingestion.py` extracts PyMuPDF text/images, pdfplumber tables as Markdown, captions images, then chunks at 800 characters with 100 overlap.
- `src/vectorstore.py` creates and writes the BGE-large (1024-dimension cosine) Qdrant collection.
- `src/retriever.py` persists chunks for BM25 and creates the weighted hybrid retriever plus FlashRank top-five reranking.
- `src/chain.py` defines the LCEL retrieval/prompt/model/string pipeline and source responses.
- `src/api.py` provides the FastAPI API and serves the no-build chat UI.
- `eval/run_eval.py` evaluates the golden dataset and enforces RAGAS quality gates.
- `frontend/index.html` is the self-contained chat widget.

## Configure and run

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, then run:

```bash
docker compose up --build
```

Open `http://localhost:8000`. Ingest a PDF through the API (the page currently assumes ingestion is performed separately):

```bash
curl -F "file=@/absolute/path/to/document.pdf" http://localhost:8000/ingest
```

Then submit `POST /query` with `{"question":"..."}`. `GET /health` checks service liveness.

## Evaluation

Replace the example record in `eval/golden_dataset.json` with verified questions and ground truths, then run:

```bash
python -m eval.run_eval
```

It returns status 1 if any metric fails its `.env` threshold. RAGAS and image captioning both call OpenAI and can incur API charges.
