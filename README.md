# Enron Email Analyzer (RAG + Neo4j)

## Project Overview
This project analyzes the Enron email dataset using Retrieval-Augmented Generation (RAG) and Graph Analysis (Neo4j) to identify risky emails, explore communication networks, and enable semantic search over email content.

The system supports risk scoring, graph-based analysis, semantic search using FAISS, and an interactive Streamlit dashboard.

## Key Features
- Risk scoring based on sensitive keywords and communication patterns
- Graph analysis using Neo4j to visualize sender–receiver networks
- RAG pipeline for semantic search and question answering
- Streamlit UI for interactive exploration

## Folder Structure
data/        - Raw data and final exports (e.g., top2000.csv/parquet)  
processed/   - Cleaned and scored datasets  
reports/     - Evaluation outputs (e.g., eval_rag_results.csv)  
src/         - Python scripts (pipeline + Streamlit app)  
notebooks/   - Optional notebooks 

## Main Scripts
- src/01_clean_emails.py — Cleans raw email data  
- src/02_score_risk.py — Adds risk_score and comm_score  
- src/03_export_top2000.py — Exports top risky emails  
- src/04_make_rag_dataset.py — Prepares RAG metadata  
- src/05_chunk_emails.py — Splits email text into chunks  
- src/06_build_vector_index.py — Builds FAISS vector index  
- src/07_load_neo4j.py — Loads email graph into Neo4j  
- src/08_rag_search.py — Semantic search logic  
- src/09_router_agent.py — Routes queries (RAG vs Graph)  
- src/10_app_streamlit.py — Streamlit application  
- src/11_eval_rag.py — Evaluation and latency analysis  

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/AilinShamsaie/enron-email-analytics.git  
cd enron-email-analytics

### 2. Create virtual environment
python -m venv .venv  
source .venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Configure environment variables
Create a .env file in the project root:
NEO4J_URI=bolt://<your-uri>  
NEO4J_USER=neo4j  
NEO4J_PASSWORD=<your-password>

## Run the Pipeline (Optional)
python src/01_clean_emails.py  
python src/02_score_risk.py  
python src/03_export_top2000.py  
python src/04_make_rag_dataset.py  
python src/05_chunk_emails.py  
python src/06_build_vector_index.py  
python src/07_load_neo4j.py  

## Run the Application
streamlit run src/10_app_streamlit.py

## Evaluation Output
Evaluation results are saved in:
reports/eval_rag_results.csv  
This file contains query text, latency, and top retrieved results.

## Tech Stack
- Python
- SentenceTransformers + FAISS
- Neo4j
- Streamlit
- Pandas
<iframe
<iframe
width="800"
height="450"
src="https://www.youtube.com/embed/K5n16d47QB4?rel=0&modestbranding=1&end=39"
title="Demo Video"
frameborder="0"
allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen>
</iframe>
