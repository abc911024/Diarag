# -*- coding: utf-8 -*-
"""
M0 Time Range Inference Experiment
- Model: gpt-oss-20b (default via Ollama OpenAI-compatible/chat API)
- Task: classify temporal type E/R/A/L, then infer [start_year, end_year]
- Supports Type A event-anchor lookup from corpus_with_time.csv

Expected files:
1. bridge_rewrite_dataset_v3_typeA_event_anchors.csv or bridge_rewrite_dataset_v3_corrected.csv
2. corpus_with_time.csv

Run examples:
python m0_gptoss_experiment.py \
  --dataset bridge_rewrite_dataset_v3_typeA_event_anchors.csv \
  --corpus corpus_with_time.csv \
  --model gpt-oss:20b \
  --backend ollama \
  --out m0_results.csv

If your local server is OpenAI-compatible:
python m0_gptoss_experiment.py \
  --dataset bridge_rewrite_dataset_v3_typeA_event_anchors.csv \
  --corpus corpus_with_time.csv \
  --backend openai_compatible \
  --base-url http://localhost:11434/v1 \
  --api-key ollama \
  --model gpt-oss:20b \
  --out m0_results.csv
"""

import argparse
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# -----------------------------
# Utility functions
# -----------------------------

def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def extract_json(text: str) -> Dict[str, Any]:
    """Extract first JSON object from LLM output."""
    if text is None:
        return {}
    text = text.strip()
    # Remove markdown fences
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find first {...}
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            # Common fixes
            candidate = candidate.replace("\n", " ")
            candidate = re.sub(r",\s*\}", "}", candidate)
            candidate = re.sub(r",\s*\]", "]", candidate)
            try:
                return json.loads(candidate)
            except Exception:
                return {}
    return {}


def normalize_type(t: str) -> str:
    t = safe_str(t).strip().upper()
    if "EXPLICIT" in t or t in {"E", "TYPE E"}:
        return "Type E"
    if "RELATIVE" in t or t in {"R", "TYPE R"}:
        return "Type R"
    if "EVENT" in t or "ANCHOR" in t or t in {"A", "TYPE A"}:
        return "Type A"
    if "LATENT" in t or t in {"L", "TYPE L"}:
        return "Type L"
    # fallback if model outputs original PPT naming accidentally
    if t == "TYPE B":
        return "Type R"
    if t == "TYPE C":
        return "Type A"
    return "Type L"


def parse_int_year(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x)
    m = re.search(r"(19\d{2}|20\d{2})", s)
    if m:
        return int(m.group(1))
    try:
        v = int(float(s))
        if 1900 <= v <= 2100:
            return v
    except Exception:
        pass
    return None


def parse_gold_range(row: pd.Series) -> Tuple[Optional[int], Optional[int]]:
    """Flexible gold range parser."""
    possible_start_cols = ["gold_start", "gold_start_year", "start_year", "t_start", "gold_t_start"]
    possible_end_cols = ["gold_end", "gold_end_year", "end_year", "t_end", "gold_t_end"]

    gs, ge = None, None
    for c in possible_start_cols:
        if c in row.index:
            gs = parse_int_year(row[c])
            if gs is not None:
                break
    for c in possible_end_cols:
        if c in row.index:
            ge = parse_int_year(row[c])
            if ge is not None:
                break

    if gs is not None and ge is not None:
        return gs, ge

    # Try common combined fields
    for c in ["gold_range", "range", "time_range", "original_range"]:
        if c in row.index:
            s = safe_str(row[c])
            years = [int(y) for y in re.findall(r"19\d{2}|20\d{2}", s)]
            if len(years) >= 2:
                return min(years[0], years[1]), max(years[0], years[1])
            if len(years) == 1:
                return years[0], years[0]

    # Try original question / explicit question
    for c in ["original_question", "question", "explicit_query"]:
        if c in row.index:
            s = safe_str(row[c])
            years = [int(y) for y in re.findall(r"19\d{2}|20\d{2}", s)]
            if len(years) >= 2:
                return min(years[0], years[1]), max(years[0], years[1])
            if len(years) == 1:
                y = years[0]
                if re.search(r"before", s, flags=re.I):
                    return None, y - 1
                if re.search(r"after", s, flags=re.I):
                    return y + 1, None
                return y, y
    return None, None


def infer_query_col(df: pd.DataFrame) -> str:
    candidates = [
        "type_a_query", "rewritten_query", "implicit_query", "query_rewrite",
        "question_rewrite", "query", "question", "original_question"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find query column. Available columns: {list(df.columns)}")


def infer_ticker_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["ticker", "symbol", "stock", "company"]:
        if c in df.columns:
            return c
    return None


def infer_corpus_year_col(corpus: pd.DataFrame) -> str:
    for c in ["publish_year", "year", "date_year", "pub_year", "timestamp", "date", "published_at"]:
        if c in corpus.columns:
            return c
    raise ValueError(f"Cannot find year/date column in corpus. Available columns: {list(corpus.columns)}")


def infer_corpus_text_cols(corpus: pd.DataFrame) -> List[str]:
    cols = []
    for c in ["title", "headline", "text", "content", "body", "summary"]:
        if c in corpus.columns:
            cols.append(c)
    if not cols:
        # fallback: use all object columns except ticker/date columns
        cols = [c for c in corpus.columns if corpus[c].dtype == "object"]
    return cols


def year_from_corpus_value(x: Any) -> Optional[int]:
    return parse_int_year(x)


def compute_iou(pred_s: Optional[int], pred_e: Optional[int], gold_s: Optional[int], gold_e: Optional[int]) -> Optional[float]:
    if None in [pred_s, pred_e, gold_s, gold_e]:
        return None
    if pred_s > pred_e:
        pred_s, pred_e = pred_e, pred_s
    if gold_s > gold_e:
        gold_s, gold_e = gold_e, gold_s
    inter = max(0, min(pred_e, gold_e) - max(pred_s, gold_s) + 1)
    union = max(pred_e, gold_e) - min(pred_s, gold_s) + 1
    if union <= 0:
        return None
    return inter / union


def exact_match(pred_s: Optional[int], pred_e: Optional[int], gold_s: Optional[int], gold_e: Optional[int]) -> Optional[int]:
    if None in [pred_s, pred_e, gold_s, gold_e]:
        return None
    return int(pred_s == gold_s and pred_e == gold_e)


def acceptable_acc(pred_s: Optional[int], pred_e: Optional[int], gold_s: Optional[int], gold_e: Optional[int], tolerance: int = 1) -> Optional[int]:
    if None in [pred_s, pred_e, gold_s, gold_e]:
        return None
    return int(abs(pred_s - gold_s) <= tolerance and abs(pred_e - gold_e) <= tolerance)


def boundary_mae(pred_s: Optional[int], pred_e: Optional[int], gold_s: Optional[int], gold_e: Optional[int]) -> Optional[float]:
    if None in [pred_s, pred_e, gold_s, gold_e]:
        return None
    return (abs(pred_s - gold_s) + abs(pred_e - gold_e)) / 2


# -----------------------------
# LLM client
# -----------------------------

class LLMClient:
    def __init__(self, backend: str, model: str, base_url: str, api_key: str, temperature: float = 0.0, timeout: int = 120):
        self.backend = backend
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                if self.backend == "ollama":
                    # Native Ollama chat endpoint
                    url = f"{self.base_url}/api/chat" if not self.base_url.endswith("/api") else f"{self.base_url}/chat"
                    payload = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": self.temperature},
                    }
                    r = requests.post(url, json=payload, timeout=self.timeout)
                    r.raise_for_status()
                    data = r.json()
                    return data.get("message", {}).get("content", "")
                else:
                    # OpenAI-compatible /v1/chat/completions
                    url = f"{self.base_url}/chat/completions"
                    headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    payload = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": self.temperature,
                    }
                    r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                    r.raise_for_status()
                    data = r.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 + attempt)
        return ""


# -----------------------------
# Prompts
# -----------------------------

CLASSIFY_SYSTEM = """You are Module 0 for temporal query understanding in diachronic question answering.
Classify the temporal intent of the query into exactly one of the following types:

Type E: Explicit. The query directly contains an explicit year, date, or time range.
Type R: Relative. The query contains a relative temporal expression, such as recent years, past few years, last decade.
Type A: Event-anchored. The query refers to an event that determines the time range, such as after the merger, since the crisis, before the rate hike, between event A and event B.
Type L: Latent. The query contains no explicit, relative, or event-based temporal expression. The time range must be inferred from topic semantics.

Priority rule:
- If explicit years appear, classify as Type E unless the query primarily asks around named events and the years only appear in metadata-like text.
- If named events define the boundary, classify as Type A.
- If relative expressions define the boundary, classify as Type R.
- If no temporal clue exists, classify as Type L.

Return only valid JSON."""

CLASSIFY_USER = """Query:
{query}

Return JSON in this schema:
{{
  "temporal_type": "Type E | Type R | Type A | Type L",
  "temporal_expression": "the key temporal expression or null",
  "anchor_event": "the event anchor if Type A, otherwise null",
  "reason": "brief reason"
}}"""

INFER_SYSTEM = """You are Module 0: Time Range Inference for diachronic question answering.
Your task is to infer the most appropriate analysis time range [start_year, end_year].
Return only valid JSON.

Rules by type:
1. Type E: extract explicit year/range from the query.
2. Type R: resolve relative expressions using corpus_max_year as the reference year, not the real current year.
   - recent years / past few years = last 3 years
   - past several years = last 5 years
   - past decade / last decade = last 10 years
3. Type A: use the event evidence snippets to identify the anchor event year. Then infer the range:
   - before EVENT => [corpus_min_year, event_year - 1]
   - after/since EVENT => [event_year + 1, corpus_max_year] or [event_year, corpus_max_year] if the wording is since
   - between EVENT1 and EVENT2 => [event1_year, event2_year]
4. Type L: infer a reasonable analysis range from the query topic. For general trend/evolution questions, prefer the full corpus range.

Constraints:
- start_year and end_year must be integers within [corpus_min_year, corpus_max_year].
- If uncertain, output confidence = low but still provide the best range.
- Do not include explanations outside JSON."""

INFER_USER = """Corpus time span:
corpus_min_year = {corpus_min_year}
corpus_max_year = {corpus_max_year}

Ticker: {ticker}
Query:
{query}

Temporal classification:
{classification_json}

Event evidence snippets from corpus, if any:
{event_snippets}

Return JSON in this schema:
{{
  "start_year": 2012,
  "end_year": 2022,
  "temporal_type": "Type E | Type R | Type A | Type L",
  "anchor_event": "event text or null",
  "anchor_event_year": 2020,
  "direction": "before | after | since | between | none",
  "confidence": "high | medium | low",
  "reason": "brief reason"
}}"""


# -----------------------------
# Corpus retrieval for event anchors
# -----------------------------

def prepare_corpus(corpus: pd.DataFrame) -> pd.DataFrame:
    year_col = infer_corpus_year_col(corpus)
    text_cols = infer_corpus_text_cols(corpus)
    c = corpus.copy()
    c["__year"] = c[year_col].apply(year_from_corpus_value)
    c["__doc_text"] = c[text_cols].fillna("").astype(str).agg(" | ".join, axis=1)
    return c.dropna(subset=["__year"])


def retrieve_event_snippets(
    corpus: pd.DataFrame,
    query: str,
    ticker: str = "",
    anchor_event: str = "",
    top_k: int = 6,
) -> str:
    if corpus.empty:
        return "No corpus snippets available."

    df = corpus.copy()
    # Ticker filter when possible
    ticker_col = infer_ticker_col(df)
    if ticker and ticker_col:
        sub = df[df[ticker_col].astype(str).str.upper() == ticker.upper()]
        if len(sub) >= 3:
            df = sub

    search_text = (anchor_event or "") + " " + query

    if SKLEARN_OK and len(df) > 0:
        docs = df["__doc_text"].fillna("").astype(str).tolist()
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=20000)
            X = vec.fit_transform(docs + [search_text])
            sims = cosine_similarity(X[-1], X[:-1]).flatten()
            order = sims.argsort()[::-1][:top_k]
            chosen = df.iloc[order].copy()
            chosen["__score"] = sims[order]
        except Exception:
            chosen = df.head(top_k).copy()
            chosen["__score"] = 0.0
    else:
        # Simple keyword fallback
        keywords = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", search_text) if len(w) > 3]
        def score_doc(s: str) -> int:
            sl = s.lower()
            return sum(1 for w in keywords if w in sl)
        df["__score"] = df["__doc_text"].apply(score_doc)
        chosen = df.sort_values("__score", ascending=False).head(top_k)

    lines = []
    for i, (_, r) in enumerate(chosen.iterrows(), 1):
        text = safe_str(r["__doc_text"])
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 700:
            text = text[:700] + "..."
        lines.append(f"[{i}] year={int(r['__year'])}, score={float(r.get('__score', 0)):.4f}, text={text}")
    return "\n".join(lines) if lines else "No corpus snippets available."


# -----------------------------
# Main experiment
# -----------------------------

def clamp_year(y: Optional[int], min_y: int, max_y: int) -> Optional[int]:
    if y is None:
        return None
    return max(min_y, min(max_y, int(y)))


def run_one(row: pd.Series, query_col: str, ticker_col: Optional[str], corpus: pd.DataFrame, llm: LLMClient, corpus_min: int, corpus_max: int) -> Dict[str, Any]:
    query = safe_str(row[query_col])
    ticker = safe_str(row[ticker_col]) if ticker_col else ""

    # 1) Classification
    class_raw = llm.chat(CLASSIFY_SYSTEM, CLASSIFY_USER.format(query=query))
    class_json = extract_json(class_raw)
    temporal_type = normalize_type(class_json.get("temporal_type", ""))
    class_json["temporal_type"] = temporal_type

    # If dataset already marks Type A and current query uses before/after/between event wording, keep Type A.
    # This prevents the model from misclassifying event queries as latent.
    if re.search(r"\b(before|after|since|between)\b", query, flags=re.I) and not re.search(r"19\d{2}|20\d{2}", query):
        temporal_type = "Type A"
        class_json["temporal_type"] = "Type A"

    anchor_event = safe_str(class_json.get("anchor_event", ""))
    # Use existing metadata from the Type A rewritten dataset when available
    for c in ["event_anchor_text", "anchor_event", "event_source_title"]:
        if c in row.index and safe_str(row[c]):
            anchor_event = safe_str(row[c])
            break

    # 2) Retrieve event evidence for Type A or event-like queries
    event_snippets = ""
    if temporal_type == "Type A":
        event_snippets = retrieve_event_snippets(corpus, query=query, ticker=ticker, anchor_event=anchor_event, top_k=6)
    else:
        event_snippets = "No event-anchor retrieval needed."

    # 3) Inference
    infer_raw = llm.chat(
        INFER_SYSTEM,
        INFER_USER.format(
            corpus_min_year=corpus_min,
            corpus_max_year=corpus_max,
            ticker=ticker,
            query=query,
            classification_json=json.dumps(class_json, ensure_ascii=False),
            event_snippets=event_snippets,
        ),
    )
    infer_json = extract_json(infer_raw)

    pred_s = parse_int_year(infer_json.get("start_year"))
    pred_e = parse_int_year(infer_json.get("end_year"))
    pred_s = clamp_year(pred_s, corpus_min, corpus_max)
    pred_e = clamp_year(pred_e, corpus_min, corpus_max)
    if pred_s is not None and pred_e is not None and pred_s > pred_e:
        pred_s, pred_e = pred_e, pred_s

    gold_s, gold_e = parse_gold_range(row)
    # Fill open-ended gold from before/after original question if needed
    if gold_s is None and gold_e is not None:
        gold_s = corpus_min
    if gold_e is None and gold_s is not None:
        gold_e = corpus_max

    result = {
        "query": query,
        "ticker": ticker,
        "pred_type": temporal_type,
        "pred_start_year": pred_s,
        "pred_end_year": pred_e,
        "gold_start_year": gold_s,
        "gold_end_year": gold_e,
        "exact_match": exact_match(pred_s, pred_e, gold_s, gold_e),
        "acceptable_acc_pm1": acceptable_acc(pred_s, pred_e, gold_s, gold_e, tolerance=1),
        "temporal_iou": compute_iou(pred_s, pred_e, gold_s, gold_e),
        "boundary_mae": boundary_mae(pred_s, pred_e, gold_s, gold_e),
        "confidence": infer_json.get("confidence"),
        "anchor_event": infer_json.get("anchor_event") or anchor_event,
        "anchor_event_year": infer_json.get("anchor_event_year"),
        "direction": infer_json.get("direction"),
        "reason": infer_json.get("reason"),
        "classification_raw": class_raw,
        "inference_raw": infer_raw,
        "event_snippets": event_snippets,
    }
    return result


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in [("ALL", results)] + list(results.groupby("pred_type")):
        rows.append({
            "group": name,
            "n": len(g),
            "exact_match": g["exact_match"].dropna().mean() if "exact_match" in g else None,
            "acceptable_acc_pm1": g["acceptable_acc_pm1"].dropna().mean() if "acceptable_acc_pm1" in g else None,
            "temporal_iou": g["temporal_iou"].dropna().mean() if "temporal_iou" in g else None,
            "boundary_mae": g["boundary_mae"].dropna().mean() if "boundary_mae" in g else None,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to rewrite dataset CSV")
    parser.add_argument("--corpus", required=True, help="Path to corpus_with_time CSV")
    parser.add_argument("--out", default="m0_results.csv")
    parser.add_argument("--summary-out", default="m0_summary.csv")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openai_compatible"])
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL or OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "ollama"))
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--limit", type=int, default=None, help="Run first N rows only for testing")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between rows")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    corpus_raw = pd.read_csv(args.corpus)
    corpus = prepare_corpus(corpus_raw)

    query_col = infer_query_col(df)
    ticker_col = infer_ticker_col(df)

    corpus_min = int(corpus["__year"].min())
    corpus_max = int(corpus["__year"].max())

    print(f"Dataset rows: {len(df)}")
    print(f"Query column: {query_col}")
    print(f"Ticker column: {ticker_col}")
    print(f"Corpus time span: {corpus_min}-{corpus_max}")
    print(f"Model: {args.model} / backend={args.backend}")

    if args.limit:
        df = df.head(args.limit).copy()

    llm = LLMClient(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0.0,
    )

    results = []
    for idx, row in df.iterrows():
        print(f"\n[{len(results)+1}/{len(df)}] row_index={idx}")
        try:
            res = run_one(row, query_col, ticker_col, corpus, llm, corpus_min, corpus_max)
        except Exception as e:
            res = {
                "query": safe_str(row[query_col]),
                "ticker": safe_str(row[ticker_col]) if ticker_col else "",
                "error": repr(e),
            }
            print("ERROR:", repr(e))
        results.append(res)

        # Save incrementally to avoid losing progress
        pd.DataFrame(results).to_csv(args.out, index=False, encoding="utf-8-sig")
        print("pred:", res.get("pred_type"), res.get("pred_start_year"), res.get("pred_end_year"),
              "gold:", res.get("gold_start_year"), res.get("gold_end_year"),
              "iou:", res.get("temporal_iou"))
        if args.sleep > 0:
            time.sleep(args.sleep)

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    summary = summarize(res_df)
    summary.to_csv(args.summary_out, index=False, encoding="utf-8-sig")

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {args.out}")
    print(f"Saved summary to: {args.summary_out}")


if __name__ == "__main__":
    main()
