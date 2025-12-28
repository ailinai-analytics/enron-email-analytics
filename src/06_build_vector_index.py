# src/06_build_vector_index.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "data" / "rag" / "chunks.jsonl"
OUT  = ROOT / "data" / "rag" / "index"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 1) Load chunks
rows = []
with INP.open("r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)
assert "text" in df.columns, "chunks.jsonl must contain 'text'"

# 2) Embed
model = SentenceTransformer(MODEL)
emb = model.encode(
    df["text"].astype(str).tolist(),
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")

# 3) Build FAISS index (cosine sim via normalized vectors + IP)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

# 4) Save artifacts
faiss.write_index(index, str(OUT / "faiss.index"))

#  Keep text so RAG can show evidence
df.to_parquet(OUT / "meta.parquet", index=False)

np.save(OUT / "embeddings.npy", emb)

print(" Step 06 complete:", OUT)
print("Vectors:", len(df), "Dim:", emb.shape[1])
print("Saved:", OUT / "faiss.index")
print("Saved:", OUT / "meta.parquet")
