# src/05_chunk_emails.py
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "data" / "rag" / "emails_rag.csv"
OUT  = ROOT / "data" / "rag" / "chunks.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 900
OVERLAP = 150

def norm(x):
    return " ".join(str(x).replace("\x00", " ").split())

def chunk_text(text: str):
    text = norm(text)
    out = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - OVERLAP)
    return out

df = pd.read_csv(INP, low_memory=False)

#  pick best text column (RAG needs REAL content)
# emails_rag.csv should already have "text"
TEXT_COL_CANDIDATES = ["text", "body_clean", "body", "content"]
text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
if not text_col:
    raise ValueError(f" No text column found in {INP}. Expected one of {TEXT_COL_CANDIDATES}.")

n = 0
skipped = 0

with OUT.open("w", encoding="utf-8") as f:
    for _, r in df.iterrows():
        msg_id = str(r.get("msg_id", "")).strip()
        if not msg_id:
            skipped += 1
            continue

        subject = norm(r.get("subject", ""))
        body_text = norm(r.get(text_col, ""))   # ✅ FIX: use text/body, NOT file path

        # Create the content to chunk
        full_text = f"Subject: {subject}\n\n{body_text}".strip()

        if len(full_text) < 80:
            skipped += 1
            continue

        chunks = chunk_text(full_text)
        for i, ch in enumerate(chunks):
            rec = {
                "chunk_id": f"{msg_id}__{i}",
                "msg_id": msg_id,
                "text": ch,

                # metadata (helps filtering + UI)
                "from": norm(r.get("from", "")),
                "to": norm(r.get("to", "")),
                "cc": norm(r.get("cc", "")),
                "date": norm(r.get("date", "")),
                "subject": subject,
                "risk_score": int(pd.to_numeric(r.get("risk_score", 0), errors="coerce") or 0),
                "comm_score": int(pd.to_numeric(r.get("comm_score", 0), errors="coerce") or 0),

                # keep file as metadata only (optional)
                "file": norm(r.get("file", "")),
                "source_text_col": text_col,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

print(" Step 05 complete:", OUT)
print(" Using text column:", text_col)
print(" Total chunks:", n)
print("Skipped rows:", skipped)
