# src/04_make_rag_dataset.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "data" / "outputs" / "top_2000_risky_emails_no_newsletters.csv"
OUT  = ROOT / "data" / "rag" / "emails_rag.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INP)

# Keep only what we need
keep = ["msg_id","date","from","to","cc","subject","file","risk_score","comm_score","recipient_count","is_newsletter"]
for c in keep:
    if c not in df.columns:
        df[c] = ""

df = df[keep].copy()
df.to_csv(OUT, index=False)

print(" Step 04 complete:", OUT, "rows=", len(df))# src/04_make_rag_dataset.py
#  Build RAG dataset WITH email text (so RAG answers match the question)

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#  Best input: data/final/top2000.csv (created by UPDATED Step 03)
# It contains: body, body_clean
INP  = ROOT / "data" / "final" / "top2000.csv"

OUT  = ROOT / "data" / "rag" / "emails_rag.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INP, low_memory=False)

# --- ensure columns exist ---
required = [
    "msg_id","date","from","to","cc","subject","file",
    "risk_score","comm_score","recipient_count","is_newsletter",
    "body","body_clean"
]
for c in required:
    if c not in df.columns:
        df[c] = ""

# --- choose best body column for RAG ---
df["body_for_rag"] = df["body_clean"].fillna("").astype(str).str.strip()
mask = df["body_for_rag"].eq("")
df.loc[mask, "body_for_rag"] = df.loc[mask, "body"].fillna("").astype(str).str.strip()

#  main text used for embeddings / retrieval
df["text"] = (
    "Subject: " + df["subject"].fillna("").astype(str).str.strip()
    + "\nFrom: " + df["from"].fillna("").astype(str).str.strip()
    + "\nTo: " + df["to"].fillna("").astype(str).str.strip()
    + "\nDate: " + df["date"].fillna("").astype(str).str.strip()
    + "\n\n" + df["body_for_rag"]
)

keep = [
    "msg_id","date","from","to","cc","subject","file",
    "risk_score","comm_score","recipient_count","is_newsletter",
    "text"
]

df = df[keep].copy()

# optional: remove empty rows
df = df[df["text"].fillna("").str.len() > 50].copy()

df.to_csv(OUT, index=False, encoding="utf-8")
print(" Step 04 complete:", OUT, "rows=", len(df))
print(" RAG will now work correctly because 'text' exists.")

