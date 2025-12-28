# src/10_app_streamlit.py
# Enron Email Analyzer (RAG + Neo4j) — DESIGN UPGRADE 
#  RAG evidence + full email body viewer (data/final/top2000.csv or .parquet)
#  Neo4j graph analytics + PyVis visualization
#  Clean routing: Graph for "who/connected/email", otherwise RAG
#
# NOTE: This version fixes "dark again" by overriding Streamlit's real containers:
#   - div[data-testid="stAppViewContainer"]
#   - section.main
#   - header/footer + sidebar containers
# and also brightens inputs/expanders/selectboxes/markdown text.

import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network

from rag_search import rag_search  # must expose rag_search(query, k=...)

# -----------------------
# Load env
# -----------------------
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

EMAIL_RE = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I)

GRAPH_HINTS = [
    "who", "connected", "connection", "relationship", "network",
    "top sender", "top senders", "community", "pagerank",
    "path", "hub", "degree", "centrality", "neighbors", "neighbour"
]

# -----------------------
# Resolve project root robustly
# -----------------------
ROOT = Path(__file__).resolve().parents[1]

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Enron Email Analyzer", layout="wide")

# -----------------------
# CSS — Professional Bright Dark-Blue (NOT too dark)
# -----------------------
st.markdown(
    """
    <style>
      /* --- remove default header/decoration --- */
      header[data-testid="stHeader"] { height: 0px !important; }
      div[data-testid="stDecoration"] { height: 0px !important; }
      footer { visibility: hidden; }

      /* --- Main background (LIGHT) --- */
      div[data-testid="stAppViewContainer"]{
        background:
          radial-gradient(1100px 700px at 10% 10%, rgba(99,102,241,0.18), transparent 60%),
          radial-gradient(1000px 650px at 90% 10%, rgba(34,211,238,0.16), transparent 55%),
          radial-gradient(900px 650px at 50% 95%, rgba(244,114,182,0.14), transparent 60%),
          linear-gradient(180deg, #F6F8FF 0%, #EEF2FF 55%, #EAF0FF 100%) !important;
      }
      section.main, .stApp { background: transparent !important; }

      /* --- page width/padding --- */
      .block-container{
        max-width: 1400px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
      }

      /* --- Sidebar (LIGHT) --- */
      section[data-testid="stSidebar"]{
        background: rgba(255,255,255,0.85) !important;
        border-right: 1px solid rgba(15,23,42,0.10) !important;
        backdrop-filter: blur(10px);
      }

      /* --- Hero --- */
      .hero{
        padding: 1.05rem 1.15rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,0.14), rgba(244,114,182,0.10));
        border: 1px solid rgba(15,23,42,0.10);
        box-shadow: 0 14px 30px rgba(2,6,23,0.08);
        margin-bottom: 1rem;
      }
      .hero-title{
        font-size: 2.15rem;
        font-weight: 900;
        margin: 0;
        background: linear-gradient(90deg, #2563EB, #7C3AED, #DB2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .hero-subtitle{
        color: rgba(15,23,42,0.78);
        margin-top: 0.35rem;
        font-size: 1.02rem;
        font-weight: 600;
      }

      /* --- Form container --- */
      div[data-testid="stForm"]{
        background: rgba(255,255,255,0.92) !important;
        border: 1px solid rgba(15,23,42,0.10) !important;
        border-radius: 16px !important;
        padding: 0.75rem 0.9rem !important;
        box-shadow: 0 12px 26px rgba(2,6,23,0.08) !important;
      }

      /* --- Cards (Evidence) --- */
      .ev-card{
        padding: 0.85rem 0.95rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.10);
        box-shadow: 0 12px 24px rgba(2,6,23,0.08);
        margin-bottom: 0.7rem;
      }
      .ev-title{
        font-weight: 850;
        font-size: 1rem;
        color: rgba(15,23,42,0.95);
        margin: 0;
      }
      .ev-meta{
        color: rgba(15,23,42,0.72);
        font-size: 0.86rem;
        margin-top: 0.25rem;
        font-weight: 600;
      }
      .badge{
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 900;
        background: rgba(37,99,235,0.10);
        border: 1px solid rgba(37,99,235,0.18);
        color: rgba(15,23,42,0.85);
        white-space: nowrap;
      }

      /* --- Tabs (LIGHT pills) --- */
      .stTabs [data-baseweb="tab"]{
        border-radius: 999px;
        padding: 7px 14px;
        font-weight: 850;
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(15,23,42,0.10);
      }

      /* --- Pill --- */
      .pill{
        display:inline-flex; align-items:center; gap:0.45rem;
        padding:0.35rem 0.85rem;
        border-radius:999px;
        font-size:0.88rem;
        font-weight:900;
        border: 1px solid rgba(15,23,42,0.10);
        background: rgba(255,255,255,0.85);
        box-shadow: 0 10px 22px rgba(2,6,23,0.07);
        margin: 0.35rem 0 0.9rem 0;
      }

      /* --- Make text areas / selects clearly readable --- */
      textarea { background: #FFFFFF !important; color: #0F172A !important; }
      div[data-baseweb="select"] > div,
      div[data-baseweb="input"] > div { background: #FFFFFF !important; }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------
# RAG export loader (for full email body) — FIXED PATHS
# -----------------------
@st.cache_data(show_spinner=False)
def load_export_df() -> pd.DataFrame:
    pq_path = ROOT / "data" / "final" / "top2000.parquet"
    csv_path = ROOT / "data" / "final" / "top2000.csv"

    if pq_path.exists():
        df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        raise FileNotFoundError(f"Missing export file: {pq_path} OR {csv_path}")

    for col in ["msg_id", "subject", "from", "to", "cc", "date", "body", "body_clean", "text", "content"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def pick_body_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["body_clean", "body", "text", "content"]:
        if col in df.columns:
            return col
    return None


def get_email_row_by_msg_id(df: pd.DataFrame, msg_id: str) -> Optional[pd.Series]:
    if "msg_id" not in df.columns:
        return None
    m = str(msg_id)
    rows = df[df["msg_id"].astype(str) == m]
    if rows.empty:
        return None
    return rows.iloc[0]


# -----------------------
# Routing
# -----------------------
def route(q: str) -> str:
    ql = (q or "").lower().strip()
    if not ql:
        return "rag"
    if EMAIL_RE.search(ql):
        return "graph"
    if any(h in ql for h in GRAPH_HINTS):
        return "graph"
    return "rag"


# -----------------------
# Neo4j driver (cached)
# -----------------------
@st.cache_resource
def neo4j_driver_cached():
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        raise RuntimeError("Missing Neo4j env vars: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# -----------------------
# Neo4j queries (cached)
# -----------------------
@st.cache_data(show_spinner=False, ttl=300)
def graph_top_senders(limit: int = 20):
    driver = neo4j_driver_cached()
    cypher = """
    MATCH (p:Person)-[:SENT]->(e:Email)
    RETURN p.email AS sender,
           count(*) AS sent,
           max(coalesce(e.risk_score,0)) AS max_risk
    ORDER BY max_risk DESC, sent DESC
    LIMIT $limit
    """
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, limit=limit)]


@st.cache_data(show_spinner=False, ttl=300)
def fetch_network_edges_global(limit: int = 250, min_risk: int = 0):
    driver = neo4j_driver_cached()
    cypher = """
    MATCH (s:Person)-[:SENT]->(e:Email)<-[:RECEIVED]-(r:Person)
    WHERE coalesce(e.risk_score,0) >= $min_risk
    RETURN s.email AS sender,
           r.email AS receiver,
           count(e) AS emails,
           max(coalesce(e.risk_score,0)) AS max_risk
    ORDER BY max_risk DESC, emails DESC
    LIMIT $limit
    """
    with driver.session() as sess:
        return [rec.data() for rec in sess.run(cypher, min_risk=min_risk, limit=limit)]


@st.cache_data(show_spinner=False, ttl=300)
def fetch_neighbors_for_person(email: str, limit: int = 250, min_risk: int = 0):
    driver = neo4j_driver_cached()
    cypher = """
    MATCH (s:Person)-[:SENT]->(e:Email)<-[:RECEIVED]-(r:Person)
    WHERE coalesce(e.risk_score,0) >= $min_risk
      AND (toLower(s.email)=toLower($email) OR toLower(r.email)=toLower($email))
    RETURN s.email AS sender,
           r.email AS receiver,
           count(e) AS emails,
           max(coalesce(e.risk_score,0)) AS max_risk
    ORDER BY max_risk DESC, emails DESC
    LIMIT $limit
    """
    with driver.session() as sess:
        return [rec.data() for rec in sess.run(cypher, email=email, min_risk=min_risk, limit=limit)]


@st.cache_data(show_spinner=False, ttl=300)
def neighbors_table(email: str, limit: int = 50, min_risk: int = 0):
    driver = neo4j_driver_cached()
    cypher = """
    MATCH (s:Person)-[:SENT]->(e:Email)<-[:RECEIVED]-(r:Person)
    WHERE coalesce(e.risk_score,0) >= $min_risk
      AND (toLower(s.email)=toLower($email) OR toLower(r.email)=toLower($email))
    WITH s, r, count(e) AS emails, max(coalesce(e.risk_score,0)) AS max_risk
    RETURN
      CASE WHEN toLower(s.email)=toLower($email) THEN r.email ELSE s.email END AS connected_person,
      CASE WHEN toLower(s.email)=toLower($email) THEN "OUTGOING" ELSE "INCOMING" END AS direction,
      emails,
      max_risk
    ORDER BY max_risk DESC, emails DESC
    LIMIT $limit
    """
    with driver.session() as sess:
        return [rec.data() for rec in sess.run(cypher, email=email, min_risk=min_risk, limit=limit)]


# -----------------------
# Graph viz (PyVis) — lightened + more professional
# -----------------------
def risk_to_color(risk: int) -> str:
    if risk >= 15:
        return "#EF4444"  # red
    if risk >= 8:
        return "#F59E0B"  # orange
    return "#22C55E"      # green


def render_network_pyvis(edges, height_px: int = 700, focus_email: Optional[str] = None):
    # Brighter background so the graph area never looks "too dark"
    net = Network(height=f"{height_px}px", width="100%", bgcolor="#16224B", font_color="white", directed=True)
    net.barnes_hut()

    seen_nodes = set()
    max_emails = max([int(e.get("emails") or 1) for e in edges], default=1)

    def node_color(email: str, base: str):
        if focus_email and email.lower() == focus_email.lower():
            return "#A78BFA"  # highlight
        return base

    for row in edges:
        s = row.get("sender")
        r = row.get("receiver")
        if not s or not r:
            continue

        emails = int(row.get("emails") or 1)
        max_risk = int(row.get("max_risk") or 0)

        if s not in seen_nodes:
            net.add_node(s, label=s, color=node_color(s, "#60A5FA"), size=18, title=s)
            seen_nodes.add(s)

        if r not in seen_nodes:
            net.add_node(r, label=r, color=node_color(r, "#F472B6"), size=18, title=r)
            seen_nodes.add(r)

        width = 1 + int(8 * (emails / max_emails)) if max_emails else 2
        net.add_edge(
            s, r,
            color=risk_to_color(max_risk),
            width=width,
            title=f"emails={emails}<br>max_risk={max_risk}"
        )

    net.set_options("""
    var options = {
      "nodes": {"shape":"dot", "font":{"size":14, "color":"#FFFFFF"}},
      "edges": {"arrows":{"to":{"enabled":true}}, "smooth":{"type":"dynamic"}},
      "physics": {"enabled": true, "stabilization":{"iterations":160}}
    }
    """)

    html = net.generate_html()
    components.html(html, height=height_px + 40, scrolling=True)


# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.markdown('<div class="side-title">Controls</div>', unsafe_allow_html=True)

    st.markdown("### 🔎 RAG settings")
    k = st.slider("Top-K evidence", 3, 15, 8)

    st.divider()

    st.markdown("### 🧠 Graph settings")
    min_risk = st.slider("Min risk_score", 0, 50, 0)
    edge_limit = st.slider("Max edges", 50, 600, 250, step=50)
    sender_limit = st.slider("Top senders rows", 5, 50, 20, step=5)

    st.divider()

    st.markdown("### ✨ Examples")
    if st.button("RAG: SEC request document deletion"):
        st.session_state["example_q"] = "SEC request document deletion"
    if st.button("RAG: audit exposure bankruptcy"):
        st.session_state["example_q"] = "audit exposure bankruptcy document preservation"
    if st.button("GRAPH: Who is connected to kenneth.lay@enron.com?"):
        st.session_state["example_q"] = "Who is connected to kenneth.lay@enron.com?"
    if st.button("GRAPH: Top risky senders"):
        st.session_state["example_q"] = "Who are the top risky senders?"

    st.divider()
    st.markdown("### ✅ Routing tips")
    st.markdown("- **Graph**: who / connected / network / degree / email")
    st.markdown("- **RAG**: everything else")


# -----------------------
# Hero header
# -----------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-title">📧 Enron Email Analyzer</div>
      <div class="hero-subtitle">
        RAG (semantic search) + Neo4j (network graph). Ask a question → Run → explore evidence & connections.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Input
# -----------------------
default_q = st.session_state.get("example_q", "SEC request document deletion")

with st.form("query_form", clear_on_submit=False):
    q = st.text_input("Ask a question", value=default_q)
    run = st.form_submit_button("Run", type="primary")

# -----------------------
# Session state
# -----------------------
if "route" not in st.session_state:
    st.session_state.route = None
if "rag_hits" not in st.session_state:
    st.session_state.rag_hits = None
if "graph" not in st.session_state:
    st.session_state.graph = None

# -----------------------
# Run
# -----------------------
if run:
    chosen = route(q)
    st.session_state.route = chosen

    if chosen == "rag":
        with st.spinner("Searching RAG index..."):
            hits = rag_search(q, k=k)
        st.session_state.rag_hits = pd.DataFrame(hits) if isinstance(hits, list) else hits
        st.session_state.graph = None

    else:
        m = EMAIL_RE.search(q or "")
        focus_email = m.group(0) if m else None

        with st.spinner("Querying Neo4j..."):
            top_rows = pd.DataFrame(graph_top_senders(limit=sender_limit))

            if focus_email:
                conn = pd.DataFrame(neighbors_table(focus_email, limit=50, min_risk=min_risk))
                edges = fetch_neighbors_for_person(focus_email, limit=edge_limit, min_risk=min_risk)
            else:
                conn = None
                edges = fetch_network_edges_global(limit=edge_limit, min_risk=min_risk)

        st.session_state.graph = {
            "focus_email": focus_email,
            "top_rows": top_rows,
            "conn_rows": conn,
            "edges": edges,
        }
        st.session_state.rag_hits = None

# -----------------------
# Results
# -----------------------
if st.session_state.route:
    route_name = st.session_state.route.upper()
    pill_class = "pill-rag" if st.session_state.route == "rag" else "pill-graph"
    icon = "🔎" if st.session_state.route == "rag" else "🧠"
    st.markdown(
        f'<div class="pill {pill_class}">{icon} Route selected: {route_name}</div>',
        unsafe_allow_html=True
    )

    if st.session_state.route == "rag":
        tab_rag, tab_graph = st.tabs(["🔎 RAG Evidence", "🧠 Graph Analytics"])
    else:
        tab_graph, tab_rag = st.tabs(["🧠 Graph Analytics", "🔎 RAG Evidence"])

    # -----------------------
    # RAG TAB
    # -----------------------
    with tab_rag:
        if st.session_state.rag_hits is None:
            st.info("This question routed to GRAPH. Open Graph Analytics tab.")
        else:
            st.subheader("Top evidence (RAG) — matches your question")
            hits_df = st.session_state.rag_hits.copy()

            st.markdown("### Evidence cards (easy to read)")
            show_n = min(10, len(hits_df))
            for _, row in hits_df.head(show_n).iterrows():
                subj = str(row.get("subject", "(no subject)"))
                score = float(row.get("score", 0) or 0)
                msg_id = str(row.get("msg_id", ""))
                sender = str(row.get("from", ""))
                date = str(row.get("date", ""))

                st.markdown(
                    f"""
                    <div class="ev-card">
                      <div class="ev-top">
                        <div>
                          <div class="ev-title">{subj}</div>
                          <div class="ev-meta">From: {sender} • Date: {date} • msg_id: {msg_id}</div>
                        </div>
                        <div class="badge">score {score:.3f}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with st.expander("See raw table (optional)", expanded=False):
                st.dataframe(hits_df, use_container_width=True)

            st.markdown("### 📧 View full email body")
            try:
                export_df = load_export_df()
                body_col = pick_body_column(export_df)

                if "msg_id" in hits_df.columns and body_col:
                    msg_ids = hits_df["msg_id"].astype(str).unique().tolist()
                    selected_msg = st.selectbox("Select an email (msg_id)", msg_ids, index=0)

                    row = get_email_row_by_msg_id(export_df, selected_msg)

                    if row is None:
                        st.warning("Body not found for this msg_id in data/final/top2000.(parquet|csv).")
                    else:
                        meta_cols = [c for c in ["date", "from", "to", "cc", "subject", "risk_score", "comm_score"] if c in row.index]
                        with st.expander("Email details", expanded=True):
                            for c in meta_cols:
                                st.write(f"**{c}:** {row[c]}")

                        st.text_area("Email body", value=str(row[body_col]), height=420)

                        txt = (
                            f"Subject: {row.get('subject','')}\n"
                            f"From: {row.get('from','')}\n"
                            f"To: {row.get('to','')}\n"
                            f"Cc: {row.get('cc','')}\n"
                            f"Date: {row.get('date','')}\n\n"
                            f"{row[body_col]}"
                        )
                        st.download_button(
                            "⬇️ Download this email (.txt)",
                            data=txt.encode("utf-8"),
                            file_name=f"{selected_msg}.txt",
                            mime="text/plain",
                        )
                else:
                    if "msg_id" not in hits_df.columns:
                        st.info("RAG results do not include msg_id, so body viewer cannot link emails.")
                    elif not body_col:
                        st.info("Export file found, but no body/text column exists (expected: body_clean/body/text).")
            except Exception as e:
                st.info(f"Body viewer not available: {e}")

    # -----------------------
    # GRAPH TAB
    # -----------------------
    with tab_graph:
        gd = st.session_state.graph
        if gd is None:
            st.info("This question routed to RAG. Open RAG Evidence tab.")
        else:
            st.subheader("Graph results (Neo4j)")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Edges shown", len(gd["edges"]))
            c2.metric("Min risk", min_risk)
            c3.metric("Max edges", edge_limit)
            c4.metric("Focused person", gd["focus_email"] or "None")

            st.write("")
            left, right = st.columns([0.95, 1.05], gap="small")

            with left:
                if gd["focus_email"]:
                    st.markdown(f"### Connections for **{gd['focus_email']}**")
                    if gd["conn_rows"] is None or gd["conn_rows"].empty:
                        st.warning("No connections found. Try min_risk=0 and increase Max edges.")
                    else:
                        st.dataframe(gd["conn_rows"], use_container_width=True)

                    with st.expander("Global Top risky senders (optional)", expanded=False):
                        st.dataframe(gd["top_rows"], use_container_width=True)
                else:
                    st.markdown("### Top risky senders")
                    st.dataframe(gd["top_rows"], use_container_width=True)

            with right:
                st.markdown("### Network visualization")
                if not gd["edges"]:
                    st.warning("No edges found. Try min_risk=0 or increase Max edges.")
                else:
                    st.caption("Edge color: green=low, orange=medium, red=high risk. Thickness ~ email count.")
                    render_network_pyvis(gd["edges"], height_px=650, focus_email=gd["focus_email"])

else:
    st.info("Type a question and click **Run**. Results will appear here.")
