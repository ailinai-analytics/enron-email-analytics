# Enron Email Analyzer (RAG + Neo4j)

## Project Overview
This project analyzes the Enron email dataset using two main approaches:

- **RAG (Retrieval-Augmented Generation)** for semantic email search  
- **Graph Analysis (Neo4j)** for communication network and risk analysis  

The system combines text analytics, vector search, and graph databases with a Streamlit UI.

---

## Folder Structure

data/
raw/ Raw Enron email data
processed/ Cleaned and scored datasets
final/ Final exports (top2000.csv / parquet)
rag/index/ FAISS index and metadata

reports/
eval_rag_results.csv

src/
01_clean_emails.py
02_score_risk.py
03_export_top2000.py
04_make_rag_dataset.py
05_chunk_emails.py
06_build_vector_index.py
07_load_neo4j.py
08_rag_search.py
09_router_agent.py
10_app_streamlit.py
11_eval_rag.py

notebooks/

---

## Main Scripts

- `01_clean_emails.py` — Cleans raw email data  
- `02_score_risk.py` — Adds `risk_score` and `comm_score`  
- `03_export_top2000.py` — Selects top risky emails  
- `04_make_rag_dataset.py` — Builds RAG metadata  
- `05_chunk_emails.py` — Splits emails into chunks  
- `06_build_vector_index.py` — Builds FAISS index  
- `07_load_neo4j.py` — Loads data into Neo4j  
- `08_rag_search.py` — Semantic search  
- `09_router_agent.py` — Routes queries (Graph vs RAG)  
- `10_app_streamlit.py` — Streamlit application  
- `11_eval_rag.py` — Evaluation script  

---

## How to Run

### Step 1: Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
reports/eval_rag_results.csv
