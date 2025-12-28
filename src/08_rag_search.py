# src/08_rag_search.py
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

# Project root
ROOT = Path(__file__).resolve().parents[1]

IDX = ROOT / "data" / "rag" / "index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load model and index once
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(str(IDX / "faiss.index"))

# If you used parquet (recommended)
META_PARQUET = IDX / "meta.parquet"
META_CSV = IDX / "meta.csv"

if META_PARQUET.exists():
    meta = pd.read_parquet(META_PARQUET)
else:
    meta = pd.read_csv(META_CSV)


def rag_search(query: str, k: int = 5):
    """
    Semantic search over Enron email chunks
    """
    q_vec = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        row = meta.iloc[int(idx)].to_dict()
        row["score"] = float(score)
        results.append(row)

    return results


# Quick CLI test
if __name__ == "__main__":
    q = "SEC investigation document deletion"
    hits = rag_search(q, k=5)

    for i, h in enumerate(hits, 1):
        print(f"\n#{i} score={h['score']:.3f}")
        print("msg_id:", h["msg_id"])
        print("subject:", h.get("subject"))
