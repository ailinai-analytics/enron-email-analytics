# src/07_load_neo4j.py
import os
import re
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

#  best input for Streamlit + Neo4j (has body)
INP = ROOT / "data" / "final" / "top2000.csv"

EMAIL_RX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD  = os.getenv("NEO4J_PASSWORD")

# Optional: set WIPE_GRAPH=1 in .env if you want a clean reload
WIPE_GRAPH = os.getenv("WIPE_GRAPH", "0").strip() in ("1", "true", "TRUE", "yes", "YES")


def safe_str(x) -> str:
    """Convert to string safely, turning NaN/None into ''."""
    if x is None:
        return ""
    try:
        # pandas NaN handling
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def extract_list(s):
    """Extract all valid emails (lowercase), deduplicated."""
    text = safe_str(s)
    found = [m.lower().strip() for m in EMAIL_RX.findall(text)]
    # dedupe while preserving order
    seen = set()
    out = []
    for e in found:
        if e and e not in seen:
            seen.add(e)
            out.append(e)
    return out


def first_email(s):
    """Pick the first valid email found (for From: sometimes includes Name <email>)."""
    lst = extract_list(s)
    return lst[0] if lst else ""


def to_int(x, default=0) -> int:
    """Safe int conversion (NaN/None -> default)."""
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return int(float(x))
    except Exception:
        return default


CYPHER = """
UNWIND $rows AS r

// Email node
MERGE (e:Email {msg_id: r.msg_id})
SET e.date = r.date,
    e.subject = r.subject,
    e.risk_score = r.risk_score,
    e.comm_score = r.comm_score,
    e.recipient_count = r.recipient_count,
    e.is_newsletter = r.is_newsletter,
    e.file = r.file,
    e.body = r.body,
    e.body_clean = r.body_clean

// Sender (skip if missing)
FOREACH (_ IN CASE WHEN r.from_email <> "" THEN [1] ELSE [] END |
    MERGE (s:Person {email: r.from_email})
    MERGE (s)-[:SENT]->(e)
)

// TO recipients
WITH e, r
UNWIND coalesce(r.to_list, []) AS t
WITH e, r, trim(toLower(t)) AS t2
WHERE t2 <> ""
MERGE (p:Person {email: t2})
MERGE (e)-[:TO]->(p)
MERGE (p)-[:RECEIVED]->(e)

// CC recipients
WITH e, r
UNWIND coalesce(r.cc_list, []) AS c
WITH e, r, trim(toLower(c)) AS c2
WHERE c2 <> ""
MERGE (p:Person {email: c2})
MERGE (e)-[:CC]->(p)
MERGE (p)-[:RECEIVED]->(e)
"""


def write_batch(tx, batch_rows):
    tx.run(CYPHER, rows=batch_rows)


def main():
    if not (URI and USER and PWD):
        raise RuntimeError("Missing .env values: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD")

    if not INP.exists():
        raise FileNotFoundError(f"Input file not found: {INP}")

    df = pd.read_csv(INP, low_memory=False)
    if df.empty:
        raise RuntimeError(f"CSV loaded but empty: {INP}")

    driver = GraphDatabase.driver(URI, auth=(USER, PWD))

    with driver.session() as s:
        # constraints (Neo4j 5 syntax)
        s.run("CREATE CONSTRAINT person_email IF NOT EXISTS FOR (p:Person) REQUIRE p.email IS UNIQUE")
        s.run("CREATE CONSTRAINT email_id IF NOT EXISTS FOR (e:Email) REQUIRE e.msg_id IS UNIQUE")

        if WIPE_GRAPH:
            s.run("MATCH (n) DETACH DELETE n")
            print("🧹 Graph wiped (WIPE_GRAPH=1)")

    rows = []
    for r in df.to_dict("records"):
        msg_id = safe_str(r.get("msg_id")).strip()
        if not msg_id:
            continue

        from_email = first_email(r.get("from"))

        to_list = extract_list(r.get("to"))
        cc_list = extract_list(r.get("cc"))

        rows.append({
            "msg_id": msg_id,
            "date": safe_str(r.get("date")),
            "subject": safe_str(r.get("subject")),
            "risk_score": to_int(r.get("risk_score"), 0),
            "comm_score": to_int(r.get("comm_score"), 0),
            "recipient_count": to_int(r.get("recipient_count"), 0),
            "is_newsletter": to_int(r.get("is_newsletter"), 0),
            "file": safe_str(r.get("file")),

            "from_email": from_email,
            "to_list": to_list,
            "cc_list": cc_list,

            "body": safe_str(r.get("body")),
            "body_clean": safe_str(r.get("body_clean")),
        })

    if not rows:
        raise RuntimeError("No valid rows found (msg_id missing everywhere). Check your CSV columns.")

    BATCH = 200
    total = len(rows)

    with driver.session() as s:
        for i in range(0, total, BATCH):
            batch = rows[i:i + BATCH]

            # Neo4j v5+
            if hasattr(s, "execute_write"):
                s.execute_write(write_batch, batch)
            else:
                # fallback for older neo4j driver versions
                s.write_transaction(write_batch, batch)

            print(f" loaded {min(i + BATCH, total)}/{total}")

    driver.close()
    print("DONE  Neo4j loaded with :SENT, :TO, :CC, :RECEIVED relationships")


if __name__ == "__main__":
    main()
