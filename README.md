
To fix it **so it looks good everywhere (VS Code + GitHub)** do these two things:

---

## 1) Fix VS Code preview (fastest)

### Option A (Best): Change preview theme

* Open `README.md`
* Press **Cmd+Shift+P**
* Type: **Markdown: Open Preview**
* Then open **Settings** → search: **Markdown Preview Theme**
* Set it to: **Light** (or “GitHub Light” if available)

### Option B: Use GitHub-style preview

Install extension:

* **Markdown Preview GitHub Styling**

 Enron Email Analyzer (RAG + Neo4j)

# Project Overview
This project analyzes the Enron email dataset using two main approaches:

1. **RAG (Retrieval-Augmented Generation):** Ask questions and retrieve relevant emails (semantic search).
2. **Graph Analysis (Neo4j):** Explore communication networks and identify risky senders.

## Folder Structure
```text
data/        - Raw and final exports (e.g., top2000.csv/parquet)
processed/   - Cleaned + scored datasets
reports/     - Evaluation outputs (e.g., eval_rag_results.csv)
src/         - Python scripts (pipeline + Streamlit app)
notebooks/   - Optional notebooks

## Main Scripts

* `src/01_clean_emails.py` — Cleans email data
* `src/02_score_risk.py` — Adds `risk_score` and `comm_score`
* `src/03_export_top2000.py` — Exports top risky emails
* `src/04_make_rag_dataset.py` — Prepares RAG metadata
* `src/05_chunk_emails.py` — Splits text into chunks
* `src/06_build_vector_index.py` — Builds FAISS index
* `src/07_load_neo4j.py` — Loads data into Neo4j
* `src/08_rag_search.py` — Semantic search
* `src/09_router_agent.py` — Routes queries (Graph vs RAG)
* `src/10_app_streamlit.py` — Streamlit UI
* `src/11_eval_rag.py` — Evaluation

## How to Run

### 1) Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run src/10_app_streamlit.py
```

## Evaluation Output

* `reports/eval_rag_results.csv` contains:

  * `query`
  * `latency_sec`
  * top retrieved message IDs and subjects


