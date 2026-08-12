# DocMind v2

A production-style document Q&A system: LangChain hybrid retrieval (keyword + semantic search) with reranking, multimodal ingestion (text, tables, and images), RAGAS evaluation, and a live deployed API + chat UI.

Rebuilt from an earlier traditional-Python version of DocMind, using LangChain, hybrid search, and a full evaluation and deployment pipeline.

## 🚀 Live Demo

**API:** https://docmind-v2.onrender.com
**Chat UI:** open `frontend/index.html` locally, paste the API URL above into the "API URL" field, and start asking questions

> Hosted on Render's free tier. If it's been idle, the first request may take 30-60 seconds to wake up — subsequent requests are fast.

Try it:
```bash
curl -X POST https://docmind-v2.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What BLEU score did the Transformer achieve on WMT 2014 English-to-German?"}'
```

## Architecture

```
PDF upload (runtime, via /ingest) ──┐
                                     ▼
                  ┌─────────────────────────────────┐
                  │  Ingestion (src/ingestion.py)     │
                  │  • Text  → PyMuPDF, chunked        │
                  │  • Tables → pdfplumber → Markdown   │
                  │  • Images → PyMuPDF → GPT-4o-mini    │
                  │             captions (searchable)     │
                  └─────────────────────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                                ▼
            BM25Retriever (keyword)         Qdrant (dense/semantic)
                     │                                │
                     └──────────── EnsembleRetriever ─┘
                              (0.4 sparse / 0.6 dense)
                                     │
                          FlashRank reranker (top 5)
                                     │
                      LCEL chain → GPT-4o-mini → cited answer
                                     │
                          RAGAS eval (faithfulness,
                          answer relevancy, context
                          precision/recall)
```

## Key features

- **Hybrid retrieval** — combines BM25 keyword search and dense vector search via LangChain's `EnsembleRetriever`, then reranks the combined results with FlashRank (a local cross-encoder — no extra API cost).
- **Multimodal ingestion** — tables are parsed and serialized to Markdown so row/column structure survives; images are captioned with GPT-4o-mini vision and indexed as searchable text.
- **Runtime document upload** — PDFs can be added after deployment via `POST /ingest`, not just at initial setup. The retriever rebuilds automatically so new documents are searchable immediately, without a server restart.
- **Grounded, cited answers** — the LLM is instructed to answer only from retrieved context and cite `[source, page]` for every claim.
- **Evaluation** — RAGAS scoring (faithfulness, answer relevancy, context precision, context recall) against a golden Q&A dataset, with pass/fail thresholds suitable for a CI gate.
- **Deployed** — Docker container running on Render, auto-redeploys on every push to `main`/`master`.

## Repo structure

```
docmind-v2/
├── src/
│   ├── config.py          # env-driven settings
│   ├── ingestion.py         # text + table + image extraction
│   ├── vectorstore.py         # Qdrant client + local embeddings
│   ├── retriever.py             # hybrid (BM25 + dense) + rerank
│   ├── chain.py                    # LCEL RAG chain
│   └── api.py                         # FastAPI app (/query, /ingest, /health)
├── eval/
│   ├── golden_dataset.json    # Q&A pairs with verified ground truth
│   └── run_eval.py               # RAGAS scoring + threshold gate
├── scripts/
│   ├── ingest.py    # CLI ingestion
│   └── query.py       # CLI query testing
├── frontend/
│   └── index.html    # standalone chat UI (no build step)
├── Dockerfile
├── docker-compose.yml   # local dev: API + Qdrant together
├── requirements.txt
└── .env.example
```

## Tech stack

| Layer | Tool |
|---|---|
| Orchestration | LangChain (LCEL) |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | BAAI/bge-large-en-v1.5 (local, via HuggingFace) |
| Vector store | Qdrant Cloud |
| Keyword search | BM25 (rank_bm25) |
| Reranking | FlashRank |
| PDF parsing | PyMuPDF (text/images), pdfplumber (tables) |
| API | FastAPI |
| Evaluation | RAGAS |
| Deployment | Docker → Render |

## Running locally

```bash
git clone https://github.com/shreyaa2308/docmind-v2.git
cd docmind-v2
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

Start Qdrant locally (or point `.env` at Qdrant Cloud):
```bash
docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant:v1.12.1
```

Ingest a document and query it:
```bash
mkdir -p data/raw   # add a PDF here
python scripts/ingest.py --input data/raw
python scripts/query.py "your question here"
```

Run the API:
```bash
uvicorn src.api:app --reload --port 8000
```

Open `frontend/index.html` in a browser, point it at `http://127.0.0.1:8000`, and chat.

## Running evaluation

```bash
python -m eval.run_eval --output eval/results/baseline.json
```

Scores faithfulness, answer relevancy, context precision, and context recall against `eval/golden_dataset.json`, and exits non-zero if any metric falls below its threshold — usable as a CI gate on retrieval/prompt changes.

## Deployment

Deployed on [Render](https://render.com) directly from this GitHub repo using the included `Dockerfile`:

1. Push to GitHub
2. Render → New Web Service → connect repo → Docker auto-detected → Free instance
3. Set environment variables (`OPENAI_API_KEY`, `OPENAI_MODEL`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`)
4. Deploy — auto-redeploys on every future push

Vector storage runs on [Qdrant Cloud](https://cloud.qdrant.io) (free tier) so it's reachable from the deployed service, not just localhost.

## What this project demonstrates

- Hybrid retrieval architecture (sparse + dense + reranking), not just naive vector similarity search
- Multimodal document understanding — tables and images, not just plain text
- A working evaluation harness with numeric, threshold-gated quality metrics
- A real deployed service with runtime document ingestion, not a static demo
