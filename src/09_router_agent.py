# src/09_router_agent.py
# CLI Router Agent (RAG + Neo4j + optional Email Body lookup)
# Works with:
# - RAG questions (semantic search)
# - Graph questions (top senders, connections, highest risk)
# - Show full body by msg_id (uses data/final/top2000.csv)

import os
import re
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

from rag_search import rag_search

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

EMAIL_RE = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I)

GRAPH_HINTS = [
    "who", "whom", "connected", "connection", "connections",
    "relationship", "network", "graph", "neighbors", "neighbours",
    "path", "centrality", "pagerank", "degree", "community",
    "top sender", "top senders", "top receivers", "hub",
]

ROOT = Path(__file__).resolve().parents[1]
FINAL_TOP2000 = ROOT / "data" / "final" / "top2000.csv"


# -------------------------
# Optional: body lookup
# -------------------------
def load_top2000_df() -> pd.DataFrame | None:
    if not FINAL_TOP2000.exists():
        return None
    df = pd.read_csv(FINAL_TOP2000, low_memory=False)
    # normalize
    for c in ["msg_id", "subject", "from", "to", "cc", "date", "body", "body_clean", "text"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def pick_body_col(df: pd.DataFrame) -> str | None:
    for c in ["body", "body_clean", "text"]:
        if c in df.columns:
            return c
    return None

def show_body_by_msg_id(msg_id: str):
    df = load_top2000_df()
    if df is None:
        print(" Body lookup not available. Missing:", FINAL_TOP2000)
        return
    if "msg_id" not in df.columns:
        print(" top2000.csv has no msg_id column.")
        return

    body_col = pick_body_col(df)
    if not body_col:
        print(" top2000.csv has no body/body_clean/text column.")
        return

    rows = df[df["msg_id"].astype(str) == str(msg_id)]
    if rows.empty:
        print(" msg_id not found in top2000.csv")
        return

    r = rows.iloc[0].to_dict()
    print("\n" + "=" * 70)
    print("MSG_ID:", r.get("msg_id"))
    print("DATE:", r.get("date"))
    print("FROM:", r.get("from"))
    print("TO:", r.get("to"))
    print("SUBJECT:", r.get("subject"))
    print("-" * 70)
    print(str(r.get(body_col, ""))[:6000])  # print up to 6000 chars
    print("=" * 70 + "\n")


# -------------------------
# Routing
# -------------------------
def route_question(q: str) -> str:
    ql = (q or "").lower().strip()
    if not ql:
        return "rag"

    # If the user includes an email address, it's usually a graph query:
    if EMAIL_RE.search(ql):
        return "graph"

    if any(h in ql for h in GRAPH_HINTS):
        return "graph"

    # msg_id body request
    if ql.startswith("body ") or ql.startswith("show "):
        return "body"

    return "rag"


# -------------------------
# Neo4j helper
# -------------------------
def get_driver():
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        return None
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# -------------------------
# Graph intents
# -------------------------
def graph_connections_for_email(email: str, limit: int = 20, min_risk: int = 0):
    cypher = """
    MATCH (s:Person)-[:SENT]->(e:Email)<-[:TO|:CC]-(r:Person)
    WHERE coalesce(e.risk_score,0) >= $min_risk
      AND (toLower(s.email)=toLower($email) OR toLower(r.email)=toLower($email))
    WITH s, r, count(e) AS emails, max(coalesce(e.risk_score,0)) AS max_risk
    RETURN
      CASE WHEN toLower(s.email)=toLower($email) THEN r.email ELSE s.email END AS connected_person,
      CASE WHEN toLower(s.email)=toLower($email) THEN "OUTGOING" ELSE "INCOMING" END AS direction,
      emails, max_risk
    ORDER BY max_risk DESC, emails DESC
    LIMIT $limit
    """
    return cypher, {"email": email, "limit": limit, "min_risk": min_risk}

def graph_top_risky_senders(limit: int = 20):
    cypher = """
    MATCH (p:Person)-[:SENT]->(e:Email)
    RETURN p.email AS sender, count(*) AS sent,
           avg(coalesce(e.risk_score,0)) AS avg_risk,
           max(coalesce(e.risk_score,0)) AS max_risk
    ORDER BY max_risk DESC, sent DESC
    LIMIT $limit
    """
    return cypher, {"limit": limit}

def graph_highest_risk_emails(limit: int = 20):
    cypher = """
    MATCH (e:Email)
    RETURN e.msg_id AS msg_id,
           coalesce(e.risk_score,0) AS risk_score,
           e.subject AS subject,
           e.date AS date
    ORDER BY risk_score DESC
    LIMIT $limit
    """
    return cypher, {"limit": limit}


def run_graph(question: str, limit: int = 20, min_risk: int = 0):
    driver = get_driver()
    if driver is None:
        return {"error": "Neo4j env vars missing. Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env"}

    ql = question.lower()

    # If question contains an email address -> connections
    m = EMAIL_RE.search(question)
    if m:
        email = m.group(0)
        cypher, params = graph_connections_for_email(email, limit=limit, min_risk=min_risk)

    # Top risky senders intent
    elif ("top" in ql and "sender" in ql) or ("top senders" in ql):
        cypher, params = graph_top_risky_senders(limit=limit)

    # Highest risk emails intent
    else:
        cypher, params = graph_highest_risk_emails(limit=limit)

    with driver.session() as s:
        rows = [r.data() for r in s.run(cypher, **params)]
    driver.close()
    return {"cypher": cypher, "rows": rows}


# -------------------------
# RAG
# -------------------------
def run_rag(question: str, k: int = 5):
    hits = rag_search(question, k=k)
    out = []
    for h in hits:
        out.append({
            "score": round(float(h.get("score", 0.0)), 3),
            "msg_id": h.get("msg_id"),
            "subject": h.get("subject"),
            "from": h.get("from"),
            "date": h.get("date"),
            "chunk_id": h.get("chunk_id"),
        })
    return out


# -------------------------
# Main CLI loop
# -------------------------
def main():
    print("\n✅ Router Agent ready (RAG + Neo4j + Body lookup)\n")
    print("Examples:")
    print("  - SEC investigation document deletion")
    print("  - Who is connected to kenneth.lay@enron.com?")
    print("  - top risky senders")
    print("  - body <msg_id>   (show full email body from data/final/top2000.csv)")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Q> ").strip()
        if not q or q.lower() == "exit":
            break

        route = route_question(q)
        print(f"\n[ROUTE] {route.upper()}")

        if route == "body":
            # Accept "body <msg_id>" or "show <msg_id>"
            parts = q.split()
            if len(parts) >= 2:
                show_body_by_msg_id(parts[1])
            else:
                print(" Usage: body <msg_id>")
            print("-" * 60)
            continue

        if route == "rag":
            results = run_rag(q, k=5)
            for i, r in enumerate(results, 1):
                print(f"\n#{i} score={r['score']}")
                print("msg_id:", r["msg_id"])
                print("subject:", r["subject"])
                print("from:", r["from"])
                print("date:", r["date"])
            print("\nTip: copy a msg_id and type: body <msg_id>")

        else:
            results = run_graph(q, limit=20, min_risk=0)
            if "error" in results:
                print("", results["error"])
            else:
                print("\n[Cypher]\n", results["cypher"])
                print("\n[Rows]")
                for r in results["rows"]:
                    print(r)

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
