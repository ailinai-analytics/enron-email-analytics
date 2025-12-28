import pandas as pd
import re
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw" / "emails.csv"
OUT  = ROOT / "data" / "processed" / "emails_clean.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def get_header_value(raw: str, header: str) -> str:
    if not raw:
        return ""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(rf"(?im)^{re.escape(header)}\s*:\s*(.*)$")
    m = pattern.search(raw)
    if not m:
        return ""
    value = (m.group(1) or "").strip()

    # folded headers (lines starting with space/tab continue the previous header)
    start = m.end()
    cont_lines = []
    for line in raw[start:].split("\n"):
        if line.startswith((" ", "\t")):
            cont_lines.append(line.strip())
        else:
            break
    if cont_lines:
        value = value + " " + " ".join(cont_lines)
    return value.strip()

def split_headers_body(raw: str):
    """
    More robust than only '\n\n'.
    Handles cases where body separator is:
      - '\n\n'
      - '\n \n' (spaces)
      - '\n\t\n'
    """
    if not raw:
        return "", ""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").strip("\n")

    m = re.search(r"\n[ \t]*\n", raw)  # blank line (maybe with spaces/tabs) separates headers and body
    if not m:
        return raw, ""  # no clear separator; treat everything as headers
    split_at = m.start()
    headers = raw[:split_at]
    body = raw[m.end():]
    return headers, body

def normalize_list(s: str) -> str:
    if not s:
        return ""
    s = s.replace(",", ";")
    s = re.sub(r"\s*;\s*", ";", s.strip())
    parts = [p.strip() for p in s.split(";") if p.strip()]
    seen, out = set(), []
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return ";".join(out)

def make_msg_id(file_val: str, date: str, from_: str, subject: str) -> str:
    base = f"{file_val}|{date}|{from_}|{subject}"
    return hashlib.md5(base.encode("utf-8", errors="ignore")).hexdigest()

def clean_email_body(text: str) -> str:
    """
    Light cleaning for better RAG:
    - remove common forwarded/reply separators
    - remove common email footers/disclaimers (basic heuristics)
    - collapse excessive whitespace
    """
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # cut off long reply chains (keep the newest part)
    cut_markers = [
        r"^-{2,}\s*Original Message\s*-{2,}$",
        r"^From:\s.*$",
        r"^Sent:\s.*$",
        r"^To:\s.*$",
        r"^Cc:\s.*$",
        r"^Subject:\s.*$",
        r"^---+\s*Forwarded by.*$",
    ]
    for pat in cut_markers:
        m = re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            t = t[:m.start()].strip()
            break

    # remove common Enron/enterprise disclaimer blocks (heuristic)
    disclaimer_markers = [
        r"^This e-mail.*confidential.*$",
        r"^The information contained in this email.*$",
        r"^NOTICE:\s.*$",
    ]
    for pat in disclaimer_markers:
        m = re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            t = t[:m.start()].strip()
            break

    # collapse whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# ---------------- chunk processing ----------------
if OUT.exists():
    OUT.unlink()

chunksize = 2000
total_in = 0
total_out = 0

reader = pd.read_csv(RAW, chunksize=chunksize)

for chunk_i, df in enumerate(reader, start=1):
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"file", "message"}.issubset(df.columns):
        raise ValueError(f"Expected columns ['file','message'], found: {list(df.columns)}")

    out_rows = []
    for row in df.itertuples(index=False):
        file_val = str(getattr(row, "file", "") or "")
        raw_msg  = str(getattr(row, "message", "") or "")

        headers, body = split_headers_body(raw_msg)

        date    = get_header_value(headers, "Date")
        from_   = get_header_value(headers, "From")
        to      = get_header_value(headers, "To")
        cc      = get_header_value(headers, "Cc")
        subject = get_header_value(headers, "Subject")

        # If Subject header missing, sometimes it appears in body/header weirdly.
        # Keep it as-is (empty is OK).

        msg_id = make_msg_id(file_val, date or "", from_ or "", subject or "")

        subject = (subject or "").strip()
        body_raw = (body or "").strip()
        body_clean = clean_email_body(body_raw)

        # drop totally empty emails
        if not subject and not body_clean and not body_raw:
            continue

        out_rows.append({
            "msg_id": msg_id,
            "date": (date or "").strip(),
            "from": (from_ or "").strip(),
            "to": normalize_list(to),
            "cc": normalize_list(cc),
            "subject": subject,
            "body": body_raw,            # keep original
            "body_clean": body_clean,    # better for embeddings/RAG
            "file": file_val
        })

    out_df = pd.DataFrame(out_rows)

    out_df.to_csv(
        OUT,
        index=False,
        mode="a",
        header=(chunk_i == 1),
        encoding="utf-8",
        lineterminator="\n"
    )

    total_in += len(df)
    total_out += len(out_df)
    print(f"chunk {chunk_i} | read {total_in:,} rows | wrote {total_out:,} cleaned rows")

print(f"\nDONE Saved: {OUT}")
print(f"Total input rows: {total_in:,}")
print(f"Total cleaned rows: {total_out:,}")
print("Tip: in Neo4j/RAG, use body_clean for embedding and keep body for full evidence.")
