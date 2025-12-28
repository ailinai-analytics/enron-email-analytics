import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "data" / "processed" / "emails_clean.csv"
OUT  = ROOT / "data" / "processed" / "emails_scored.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

risk_terms = [
    "fraud","bribe","kickback","destroy","delete","lawsuit",
    "off book","off-book","investigation","urgent",
    "loss","risk","exposure","audit","sec"
]

#
term_patterns = []
for t in risk_terms:
    if " " in t:
        term_patterns.append(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b")
    else:
        term_patterns.append(r"\b" + re.escape(t) + r"\b")

pattern = "|".join(term_patterns)
rx = re.compile(pattern, flags=re.IGNORECASE)


chunksize = 20000
total = 0


if OUT.exists():
    OUT.unlink()

for i, df in enumerate(pd.read_csv(INP, chunksize=chunksize, low_memory=False), start=1):

   # Text used for scoring: subject + body_clean (if available), otherwise body

    body_col = "body_clean" if "body_clean" in df.columns else "body"
    text = (
        df["subject"].fillna("").astype(str)
        + " "
        + df[body_col].fillna("").astype(str)
    )

    
    df["risk_score"] = text.str.count(rx)

    # communication score
    df["from"] = df["from"].fillna("").astype(str)
    df["to"]   = df["to"].fillna("").astype(str)

    df["from_count"] = df.groupby("from")["from"].transform("size")
    df["to_count"] = df["to"].str.split(";").map(lambda x: len([p for p in x if p.strip()]))

    df["comm_score"] = df["from_count"] + df["to_count"]

    df.to_csv(OUT, index=False, mode="a", header=(i == 1))
    total += len(df)
    print(f" chunk {i} processed | total={total:,}")

print(f"\nDONE  Saved: {OUT}")
