import csv
import time
from pathlib import Path

import pandas as pd
from rag_search import rag_search

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "eval_rag_results.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Put 15–30 queries you want to evaluate
QUERIES = [
    "SEC request document deletion",
    "document retention instructions",
    "audit exposure bankruptcy",
    "off book trading",
    "destroy documents",
    "lawsuit strategy",
    "whistleblower complaint",
    "energy trading losses",
    "credit risk exposure",
    "California power market manipulation",
]

def precision_at_k(hits_df: pd.DataFrame, relevant_ids: set, k: int = 5) -> float:
    top = hits_df.head(k)
    if top.empty:
        return 0.0
    got = set(top["msg_id"].astype(str).tolist())
    rel = len(got.intersection(relevant_ids))
    return rel / k

def main():
    rows = []
    for q in QUERIES:
        t0 = time.time()
        hits = rag_search(q, k=5)
        dt = time.time() - t0

        df = pd.DataFrame(hits) if isinstance(hits, list) else hits
        # You can manually fill relevant_ids later (human labeling)
        rows.append({
            "query": q,
            "latency_sec": round(dt, 3),
            "top_msg_ids": ",".join(df["msg_id"].astype(str).head(5).tolist()) if "msg_id" in df.columns else "",
            "top_subjects": " | ".join(df["subject"].astype(str).head(5).tolist()) if "subject" in df.columns else "",
        })

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"✅ saved: {OUT}")

if __name__ == "__main__":
    main()
