# src/03_export_top2000.py
import re
import warnings
import pandas as pd
from pathlib import Path

# -----------------------
# Config
# -----------------------
MAX_PER_SUBJECT = 5
TOPK = 2000
CHUNKSIZE = 200000

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "data" / "processed" / "emails_scored.csv"

# your existing outputs (keep them)
OUT_DIR = ROOT / "data" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL  = OUT_DIR / "top_2000_risky_emails.csv"
OUT_NEWS = OUT_DIR / "top_2000_newsletters.csv"
OUT_REAL = OUT_DIR / "top_2000_risky_emails_no_newsletters.csv"
OUT_INT  = OUT_DIR / "top_2000_risky_emails_internal_only.csv"

# ✅ NEW: Streamlit body viewer expects this file:
FINAL_DIR = ROOT / "data" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
OUT_FINAL = FINAL_DIR / "top2000.csv"   # <- Streamlit loads data/final/top2000.csv

# ✅ IMPORTANT: include body columns
USECOLS = [
    "msg_id", "date", "from", "to", "cc",
    "risk_score", "comm_score", "subject", "file",
    "body", "body_clean"   # ✅ add these
]

EXTRA_COLS = ["recipient_count", "is_newsletter"]
SAVE_COLS = USECOLS + EXTRA_COLS

# -----------------------
# Newsletter / broadcast detection
# -----------------------
SUBJECT_LISTY_PATTERNS = [
    r"^\s*\[[^\]]+\]\s*",
    r"\bnews(?:desk|clips?)\b",
    r"\bclip(?:s|ping)?\b",
    r"\bdispatch\b",
    r"\bdigest\b",
    r"\bbulletin\b",
    r"\bnewsletter\b",
    r"\bupdate\b",
    r"\balert\b",
    r"\breport\b",
    r"\bheadlines?\b",
    r"\bdaily\b",
    r"\bweekly\b",
    r"\bmonth(?:ly)?\b",
    r"^\s*issue\s*\d+\b",
    r"\bissue\s*\d+\b",
    r"\benron mentions?\b",
    r"\bmentions\s*\(part\b",
    r"\bmajor papers?\b",
    r"^\s*articles\s*$",
    r"\benergy issues?\b",
    r"\bca energy\b",
    r"\barticles?\b",
]
SUBJECT_LISTY_RX = re.compile("|".join(SUBJECT_LISTY_PATTERNS), flags=re.IGNORECASE)

MARKET_HEADLINE_PATTERNS = [
    r"\bnatural gas\b",
    r"\bfutures?\b",
    r"\bshort-?covering\b",
    r"\bsell-?off\b",
    r"\bquiet trading\b",
    r"\brally\b",
    r"\bdip\b",
    r"\bpower market\b",
    r"\bmarket oversight\b",
    r"\bferc\b",
]
MARKET_HEADLINE_RX = re.compile("|".join(MARKET_HEADLINE_PATTERNS), flags=re.IGNORECASE)

def mark_news_subject(subject: pd.Series) -> pd.Series:
    s = subject.fillna("").astype(str).str.strip()
    return (
        s.str.contains(SUBJECT_LISTY_RX, na=False, regex=True)
        | s.str.contains(MARKET_HEADLINE_RX, na=False, regex=True)
    )

LIST_HINTS_RX = re.compile(
    r"(?:all@|list@|newsletter@|announce@|alerts?@|reports?@|no-?reply@|noreply@)",
    re.IGNORECASE,
)

def mark_broadcast_like(from_s: pd.Series, to_s: pd.Series, cc_s: pd.Series) -> pd.Series:
    blob = (
        from_s.fillna("").astype(str)
        + " "
        + to_s.fillna("").astype(str)
        + " "
        + cc_s.fillna("").astype(str)
    )
    return blob.str.contains(LIST_HINTS_RX, na=False, regex=True)

AUTO_SENDER_RX = re.compile(
    r"(?:no-?reply|noreply|do-?not-?reply|mailer-daemon|postmaster|"
    r"alerts?|dispatch|daily|weekly|update|bulletin|digest|"
    r"subscriptions?|newsletter|broadcast|listserv|mailing\s*list|news(?:desk)?)",
    re.IGNORECASE,
)

def mark_auto_sender(from_s: pd.Series) -> pd.Series:
    f = from_s.fillna("").astype(str)
    return f.str.contains(AUTO_SENDER_RX, na=False, regex=True)

# -----------------------
# Email parsing + internal detection
# -----------------------
EMAIL_RX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)

def extract_emails(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.findall(EMAIL_RX)

def recipient_count(to_s: pd.Series, cc_s: pd.Series) -> pd.Series:
    return extract_emails(to_s).map(len) + extract_emails(cc_s).map(len)

def mark_internal_only(from_s: pd.Series, to_s: pd.Series, cc_s: pd.Series) -> pd.Series:
    from_s = from_s.fillna("").astype(str)
    from_is_enron = from_s.str.contains(r"@enron\.com\b", case=False, na=False, regex=True)

    to_list = extract_emails(to_s)
    cc_list = extract_emails(cc_s)
    all_rcpt = (to_list + cc_list)

    def all_enron(lst):
        if not lst:
            return False
        return all(e.lower().endswith("@enron.com") for e in lst)

    recipients_all_enron = all_rcpt.map(all_enron)
    return from_is_enron & recipients_all_enron

def normalize_subject(subj: pd.Series) -> pd.Series:
    s = subj.fillna("").astype(str).str.lower()
    s = s.str.replace(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", " <date> ", regex=True)
    s = s.str.replace(r"\d+", " <num> ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

# -----------------------
# Date parsing (no warnings)
# -----------------------
def safe_parse_date(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    s = s.str.replace(r"\s*\([A-Z]{2,5}\)\s*$", "", regex=True)  # remove "(PDT)"
    s = s.str.replace(r"^(\w{3},)\s(\d)\s", r"\1 0\2 ", regex=True)  # zero pad day

    dt = pd.to_datetime(
        s,
        format="%a, %d %b %Y %H:%M:%S %z",
        errors="coerce",
        utc=True,
    )

    mask = dt.isna() & s.ne("")
    if mask.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dt.loc[mask] = pd.to_datetime(s.loc[mask], errors="coerce", utc=True)

    dt = dt.mask(dt < pd.Timestamp("1990-01-01", tz="UTC"))
    return dt

# -----------------------
# Sorting / caps
# -----------------------
SORT_COLS = ["risk_score", "comm_score", "date_dt"]

def cap_per_subject(df: pd.DataFrame, max_per_subject: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    tmp = df.copy()
    tmp["subj_norm"] = normalize_subject(tmp["subject"])
    tmp = tmp.sort_values(SORT_COLS, ascending=[False, False, False], kind="mergesort")
    tmp = tmp.groupby("subj_norm", group_keys=False).head(max_per_subject)
    tmp = tmp.drop(columns=["subj_norm"])
    return tmp

def keep_best_topk(best: pd.DataFrame, incoming: pd.DataFrame, k: int, max_per_subject: int | None = None) -> pd.DataFrame:
    if incoming is None or incoming.empty:
        return best

    combined = incoming.copy() if best is None or best.empty else pd.concat([best, incoming], ignore_index=True)

    combined["risk_score"] = pd.to_numeric(combined["risk_score"], errors="coerce").fillna(0).astype(int)
    combined["comm_score"] = pd.to_numeric(combined["comm_score"], errors="coerce").fillna(0).astype(int)

    if "date_dt" not in combined.columns:
        combined["date_dt"] = safe_parse_date(combined["date"])
    combined["date_dt"] = combined["date_dt"].fillna(pd.Timestamp.min.tz_localize("UTC"))

    combined = combined.drop_duplicates(subset=["msg_id"], keep="first")
    combined = combined.sort_values(SORT_COLS, ascending=[False, False, False], kind="mergesort")

    if max_per_subject is not None:
        combined = cap_per_subject(combined, max_per_subject)

    return combined.head(k)

# -----------------------
# Main streaming
# -----------------------
best_all  = pd.DataFrame(columns=SAVE_COLS + ["date_dt"])
best_news = pd.DataFrame(columns=SAVE_COLS + ["date_dt"])
best_real = pd.DataFrame(columns=SAVE_COLS + ["date_dt"])
best_int  = pd.DataFrame(columns=SAVE_COLS + ["date_dt"])

total_read = 0
total_kept = 0

BAD_FROM_RX = re.compile(r"(?:announcements|no-?reply|mailer-daemon|postmaster)", re.IGNORECASE)

for i, raw in enumerate(pd.read_csv(INP, chunksize=CHUNKSIZE, low_memory=False), start=1):
    total_read += len(raw)

    # ensure required cols exist
    for c in USECOLS:
        if c not in raw.columns:
            raw[c] = ""

    chunk = raw.loc[:, USECOLS].copy()

    chunk["date_dt"] = safe_parse_date(chunk["date"])
    chunk["valid_date"] = chunk["date_dt"].notna()

    chunk["risk_score"] = pd.to_numeric(chunk["risk_score"], errors="coerce").fillna(0).astype(int)
    chunk["comm_score"] = pd.to_numeric(chunk["comm_score"], errors="coerce").fillna(0).astype(int)

    chunk = chunk.loc[chunk["risk_score"] > 0].copy()
    total_kept += len(chunk)

    if chunk.empty:
        print(f" chunk {i} | read={total_read:,} | kept={total_kept:,} | (no risky)")
        continue

    is_news_subj   = mark_news_subject(chunk["subject"])
    is_broadcast   = mark_broadcast_like(chunk["from"], chunk["to"], chunk["cc"])
    is_auto_sender = mark_auto_sender(chunk["from"])
    is_news = (is_news_subj | is_broadcast | is_auto_sender)

    chunk["recipient_count"] = recipient_count(chunk["to"], chunk["cc"])
    chunk["is_newsletter"] = is_news.astype(int)

    chunk["date_dt"] = chunk["date_dt"].fillna(pd.Timestamp.min.tz_localize("UTC"))

    chunk_news = chunk.loc[is_news].copy()

    # REAL candidates
    rcpt_cnt = chunk["recipient_count"]
    subj_norm_all = normalize_subject(chunk["subject"])
    freq = subj_norm_all.map(subj_norm_all.value_counts())
    not_repetitive = freq <= 10

    chunk_real = chunk.loc[(~is_news) & (rcpt_cnt <= 3) & not_repetitive & (chunk["valid_date"])].copy()

    real_from_enron = chunk_real["from"].fillna("").astype(str).str.contains(r"@enron\.com\b", case=False, na=False, regex=True)
    chunk_real = chunk_real.loc[real_from_enron].copy()

    bad_from = chunk_real["from"].fillna("").astype(str).str.contains(BAD_FROM_RX, na=False)
    chunk_real = chunk_real.loc[~bad_from].copy()

    internal_mask = mark_internal_only(chunk_real["from"], chunk_real["to"], chunk_real["cc"])
    chunk_int = chunk_real.loc[internal_mask].copy()

    chunk_all_valid = chunk.loc[chunk["valid_date"]].copy()
    chunk_news_valid = chunk_news.loc[chunk_news["valid_date"]].copy()

    best_all  = keep_best_topk(best_all,  chunk_all_valid,  TOPK)
    best_news = keep_best_topk(best_news, chunk_news_valid, TOPK)
    best_real = keep_best_topk(best_real, chunk_real, TOPK, max_per_subject=MAX_PER_SUBJECT)
    best_int  = keep_best_topk(best_int,  chunk_int,  TOPK, max_per_subject=MAX_PER_SUBJECT)

    print(
        f"chunk {i} | read={total_read:,} | kept={total_kept:,} | "
        f"all={len(best_all):,} real={len(best_real):,} news={len(best_news):,} internal={len(best_int):,}"
    )

# -----------------------
# Save CSVs
# -----------------------
best_all[SAVE_COLS].to_csv(OUT_ALL, index=False)
best_news[SAVE_COLS].to_csv(OUT_NEWS, index=False)
best_real[SAVE_COLS].to_csv(OUT_REAL, index=False)
best_int[SAVE_COLS].to_csv(OUT_INT, index=False)

#  NEW: write Streamlit viewer file (pick REAL, or ALL if you prefer)
# I recommend REAL because it’s more “human” emails than newsletters.
best_real[SAVE_COLS].to_csv(OUT_FINAL, index=False)

print("\nDONE ")
print(f"Saved ALL      : {OUT_ALL}  rows={len(best_all)}")
print(f"Saved NEWS     : {OUT_NEWS} rows={len(best_news)}")
print(f"Saved REAL     : {OUT_REAL} rows={len(best_real)}")
print(f"Saved INTERNAL : {OUT_INT}  rows={len(best_int)}")
print(f"Saved FINAL    : {OUT_FINAL} rows={len(best_real)}  <-- Streamlit body viewer uses this")
