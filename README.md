Enron Email Analyzer (RAG + Neo4j)
Project Overview
This project analyzes the Enron email dataset using two main approaches:

1. RAG (Retrieval-Augmented Generation): Allows users to ask questions and retrieve relevant emails.
2. Graph Analysis (Neo4j): Shows email communication networks and risky senders.

Folder Structure
data/: Raw and processed data
processed/: Cleaned and scored datasets
reports/: Evaluation outputs
src/: Python source code
notebooks/: Optional notebooks
lib/: Helper files

Main Scripts
01_clean_emails.py – Cleans email data
02_score_risk.py – Adds risk_score and comm_score
03_export_top2000.py – Exports top risky emails
04_make_rag_dataset.py – Prepares RAG metadata
05_chunk_emails.py – Splits text into chunks
06_build_vector_index.py – Builds FAISS index
07_load_neo4j.py – Loads data into Neo4j
08_rag_search.py – Semantic search
09_router_agent.py – Routes queries
10_app_streamlit.py – Streamlit UI
11_eval_rag.py – Evaluation

How to Run
Create virtual environment:
python -m venv .venv
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Run app:
streamlit run src/10_app_streamlit.py

Evaluation Output
reports/eval_rag_results.csv contains query, latency, and top results.


