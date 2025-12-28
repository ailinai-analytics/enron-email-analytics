# src/rag_search.py
# FAISS + SentenceTransformers RAG search used by Streamlit app
from __future__ import annotations

from pathlib import Path
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "rag" / "index"

FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH_PARQUET = INDEX_DIR / "meta.parquet"
META_PATH_CSV = INDEX_DIR / "meta.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_index = None
_meta = None

def _load():
    global _model, _index, _meta

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    if _index is None:
        if not FAISS_PATH.exists():
            raise FileNotFoundError(f"Missing FAISS index: {FAISS_PATH}")
        _index = faiss.read_index(str(FAISS_PATH))

    if _meta is None:
        if META_PATH_PARQUET.exists():
            _meta = pd.read_parquet(META_PATH_PARQUET)
        elif META_PATH_CSV.exists():
            _meta = pd.read_csv(META_PATH_CSV)
        else:
            raise FileNotFoundError(f"Missing meta file: {META_PATH_PARQUET} or {META_PATH_CSV}")

        # required columns from your Step 05 chunks.jsonl
        required = {"chunk_id", "msg_id", "text"}
        missing = required - set(_meta.columns)
        if missing:
            raise ValueError(f"Meta file missing columns: {missing}. Rebuild index from Step 05/06.")

    return _model, _index, _meta


def rag_search(query: str, k: int = 8) -> list[dict]:
    """
    Returns list of dicts for Streamlit:
      chunk_id, msg_id, score, text, plus metadata columns.
    """
    query = (query or "").strip()
    if not query:
        return []

    model, index, meta = _load()

    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, k)

    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    out: list[dict] = []
    for score, i in zip(scores, idxs):
        if i < 0 or i >= len(meta):
            continue
        row = meta.iloc[i].to_dict()
        row["score"] = float(score)

        # nice UI helper columns (optional)
        text_val = str(row.get("text", ""))
        row["text_preview"] = (text_val[:400] + "…") if len(text_val) > 400 else text_val

        out.append(row)

    out.sort(key=lambda x: x.get("score", 0), reverse=True)
    return out
