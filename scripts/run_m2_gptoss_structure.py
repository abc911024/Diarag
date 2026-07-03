# -*- coding: utf-8 -*-
"""
M2: Temporal Evidence Structuring with gpt-oss-20b

Input:
    m1_qwen_results.csv

Output:
    m2_structured_evidence.csv
    m2_summary.csv

Purpose:
    Convert M1 top-k retrieved evidence into a year-by-year temporal evidence structure.

M2 does NOT retrieve new evidence.
M2 only structures the evidence retrieved by M1.

Example:
    python run_m2_gptoss_structure.py ^
      --m1-results m1_qwen_results.csv ^
      --backend ollama ^
      --model gpt-oss:20b ^
      --out m2_structured_evidence.csv ^
      --summary-out m2_summary.csv

Test 5 rows:
    python run_m2_gptoss_structure.py ^
      --m1-results m1_qwen_results.csv ^
      --backend ollama ^
      --model gpt-oss:20b ^
      --out m2_structured_evidence_test.csv ^
      --summary-out m2_summary_test.csv ^
      --limit 5
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


# ============================================================
# Basic utilities
# ============================================================

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def parse_year(x: Any) -> Optional[int]:
    if x is None:
        return None

    try:
        y = int(float(x))
        if 1900 <= y <= 2100:
            return y
    except Exception:
        pass

    m = re.search(r"(19|20)\d{2}", str(x))
    if m:
        return int(m.group(0))

    return None


def ensure_year_order(start_year: Optional[int], end_year: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    if start_year is None or end_year is None:
        return start_year, end_year
    if start_year > end_year:
        return end_year, start_year
    return start_year, end_year


def year_list(start_year: Optional[int], end_year: Optional[int]) -> List[int]:
    start_year, end_year = ensure_year_order(start_year, end_year)
    if start_year is None or end_year is None:
        return []
    return list(range(start_year, end_year + 1))


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse JSON from model output.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    if not text:
        return None

    raw = text.strip()

    # Remove markdown fences
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # First direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Try to extract first JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


def clip_text(text: str, max_chars: int) -> str:
    text = normalize_text(text).replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def safe_mean(values: List[float]) -> float:
    values = [v for v in values if v is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


# ============================================================
# LLM clients
# ============================================================

def call_ollama(
    prompt: str,
    model: str,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    timeout: int = 300,
) -> str:
    """
    Call Ollama chat API.
    """
    url = base_url.rstrip("/") + "/api/chat"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise temporal evidence structuring module. "
                    "You must return valid JSON only. Do not include markdown."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    return data.get("message", {}).get("content", "")


def call_openai_compatible(
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    timeout: int = 300,
) -> str:
    """
    Call OpenAI-compatible chat completion endpoint.
    """
    url = base_url.rstrip("/") + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise temporal evidence structuring module. "
                    "You must return valid JSON only. Do not include markdown."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]


def call_llm(prompt: str, args) -> str:
    if args.backend == "ollama":
        return call_ollama(
            prompt=prompt,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            timeout=args.timeout,
        )

    if args.backend == "openai_compatible":
        if not args.api_key:
            raise ValueError("--api-key is required for openai_compatible backend.")
        return call_openai_compatible(
            prompt=prompt,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            timeout=args.timeout,
        )

    raise ValueError(f"Unsupported backend: {args.backend}")


# ============================================================
# Evidence extraction from M1 output
# ============================================================

def extract_topk_evidence_from_row(row: pd.Series, max_top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Extract top1~topK evidence from M1 result row.

    Expected columns:
        top1_year
        top1_title
        top1_text_preview
        top1_score
        top1_doc_id
    """
    evidence = []

    for rank in range(1, max_top_k + 1):
        year_col = f"top{rank}_year"
        title_col = f"top{rank}_title"
        text_col = f"top{rank}_text_preview"
        score_col = f"top{rank}_score"
        doc_id_col = f"top{rank}_doc_id"

        if year_col not in row.index:
            continue

        year = parse_year(row.get(year_col))
        title = normalize_text(row.get(title_col, ""))
        text = normalize_text(row.get(text_col, ""))
        score = row.get(score_col, "")
        doc_id = normalize_text(row.get(doc_id_col, ""))

        if year is None and not title and not text:
            continue

        if not title and not text:
            continue

        evidence.append({
            "rank": rank,
            "doc_id": doc_id,
            "year": year,
            "title": title,
            "text": text,
            "score": score,
        })

    return evidence


def group_evidence_by_year(
    evidence: List[Dict[str, Any]],
    start_year: Optional[int],
    end_year: Optional[int],
    max_evidence_per_year: int = 3,
    max_chars_per_evidence: int = 700,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group top-kevidence by year.
    Also create empty list for years with no evidence in predicted range.
    """
    years = year_list(start_year, end_year)
    grouped: Dict[str, List[Dict[str, Any]]] = {str(y): [] for y in years}

    for ev in evidence:
        y = ev.get("year")
        if y is None:
            continue
        y_str = str(int(y))
        if y_str not in grouped:
            grouped[y_str] = []

        if len(grouped[y_str]) < max_evidence_per_year:
            grouped[y_str].append({
                "rank": ev.get("rank"),
                "doc_id": ev.get("doc_id", ""),
                "title": clip_text(ev.get("title", ""), 250),
                "text": clip_text(ev.get("text", ""), max_chars_per_evidence),
                "score": ev.get("score", "")
            })

    return grouped


def build_compact_evidence_text(grouped: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Convert grouped evidence dict into compact readable text for prompt.
    """
    parts = []

    for year in sorted(grouped.keys()):
        evs = grouped[year]
        parts.append(f"\nYear {year}:")

        if not evs:
            parts.append("- No retrieved evidence.")
            continue

        for ev in evs:
            parts.append(
                f"- Rank {ev.get('rank')}, Title: {ev.get('title')}\n"
                f"  Text: {ev.get('text')}"
            )

    return "\n".join(parts)


# ============================================================
# M2 prompt
# ============================================================

def build_m2_prompt(
    query_text: str,
    target_subject: str,
    start_year: Optional[int],
    end_year: Optional[int],
    evidence_by_year_text: str,
) -> str:
    return f"""
You are Module 2: Temporal Evidence Structuring for diachronic question answering.

Your task is to organize retrieved evidence into a year-by-year temporal evidence structure.

Important rules:
1. Use ONLY the provided retrieved evidence.
2. Do NOT invent facts, prices, years, or events.
3. Do NOT answer the final multiple-choice question.
4. Your job is evidence structuring, not final reasoning.
5. For each year in the predicted time range, summarize only evidence relevant to the query.
6. If a year has no retrieved evidence, write "No sufficient evidence."
7. If the evidence is related to the subject but does not directly support stock price trend judgment, use signal "not_directly_related".
8. Assign exactly one signal for each year from:
   positive, negative, stable, mixed, volatile, not_directly_related, no_evidence

Definitions of signals:
- positive: evidence suggests favorable movement, growth, support, improved market condition, or positive trend.
- negative: evidence suggests decline, pressure, loss, unfavorable market condition, or negative trend.
- stable: evidence suggests little change or steady condition.
- mixed: evidence contains both positive and negative signals.
- volatile: evidence emphasizes fluctuation, uncertainty, or instability.
- not_directly_related: evidence mentions the subject/topic but does not clearly support a trend judgment.
- no_evidence: no retrieved evidence is available for that year.

Query:
{query_text}

Target subject / ticker:
{target_subject}

Predicted time range:
{start_year} to {end_year}

Retrieved evidence grouped by year:
{evidence_by_year_text}

Return ONLY valid JSON in the following format:

{{
  "yearly_evidence": {
    "2015": {{
      "summary": "...",
      "signal": "positive | negative | stable | mixed | volatile | not_directly_related | no_evidence",
      "supporting_titles": ["..."],
      "evidence_count": 1
    }}
  }},
  "overall_observation": "...",
  "evidence_sufficiency": "high | medium | low",
  "coverage_warning": "..."
}}
""".strip()


# ============================================================
# Fallback M2 without LLM
# ============================================================

def build_fallback_structure(
    grouped: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    If LLM call or JSON parsing fails, create rule-based fallback.
    """
    yearly = {}

    for year in sorted(grouped.keys()):
        evs = grouped[year]
        if not evs:
            yearly[year] = {
                "summary": "No sufficient evidence.",
                "signal": "no_evidence",
                "supporting_titles": [],
                "evidence_count": 0
            }
        else:
            titles = [ev.get("title", "") for ev in evs if ev.get("title", "")]
            summary = "Retrieved evidence is available for this year, but no model-generated summary was produced."
            yearly[year] = {
                "summary": summary,
                "signal": "not_directly_related",
                "supporting_titles": titles[:3],
                "evidence_count": len(evs)
            }

    return {
        "yearly_evidence": yearly,
        "overall_observation": "Fallback structure generated because the LLM output was unavailable or invalid.",
        "evidence_sufficiency": "low",
        "coverage_warning": "LLM structuring failed; summaries should be manually checked."
    }


# ============================================================
# Metrics and summary for M2
# ============================================================

def compute_m2_stats(structure: Dict[str, Any]) -> Dict[str, Any]:
    yearly = structure.get("yearly_evidence", {})
    if not isinstance(yearly, dict):
        return {
            "num_years_in_structure": 0,
            "num_years_with_evidence": 0,
            "num_years_without_evidence": 0,
            "evidence_year_coverage": 0.0,
            "signal_distribution": "{}",
        }

    num_years = len(yearly)
    with_evidence = 0
    without_evidence = 0
    signal_counts: Dict[str, int] = {}

    for _, item in yearly.items():
        if not isinstance(item, dict):
            continue

        count = item.get("evidence_count", 0)
        signal = normalize_text(item.get("signal", "unknown"))

        try:
            count = int(count)
        except Exception:
            count = 0

        if count > 0 and signal != "no_evidence":
            with_evidence += 1
        else:
            without_evidence += 1

        signal_counts[signal] = signal_counts.get(signal, 0) + 1

    coverage = with_evidence / num_years if num_years else 0.0

    return {
        "num_years_in_structure": num_years,
        "num_years_with_evidence": with_evidence,
        "num_years_without_evidence": without_evidence,
        "evidence_year_coverage": coverage,
        "signal_distribution": json.dumps(signal_counts, ensure_ascii=False),
    }


def yearly_json_to_text(structure: Dict[str, Any]) -> str:
    yearly = structure.get("yearly_evidence", {})
    if not isinstance(yearly, dict):
        return ""

    lines = []
    for year in sorted(yearly.keys()):
        item = yearly[year]
        if not isinstance(item, dict):
            continue
        summary = normalize_text(item.get("summary", ""))
        signal = normalize_text(item.get("signal", ""))
        count = item.get("evidence_count", 0)
        lines.append(f"{year}: [{signal}, evidence_count={count}] {summary}")

    return "\n".join(lines)


# ============================================================
# Run one row
# ============================================================

def run_m2_one(row: pd.Series, args) -> Dict[str, Any]:
    query_id = normalize_text(row.get("query_id", ""))
    query_text = normalize_text(row.get("query_text", ""))
    target_subject = normalize_text(row.get("target_subject", ""))

    pred_start = parse_year(row.get("predicted_start_year"))
    pred_end = parse_year(row.get("predicted_end_year"))
    pred_start, pred_end = ensure_year_order(pred_start, pred_end)

    gold_start = parse_year(row.get("gold_start_year"))
    gold_end = parse_year(row.get("gold_end_year"))
    gold_start, gold_end = ensure_year_order(gold_start, gold_end)

    evidence = extract_topk_evidence_from_row(row, max_top_k=args.top_k)

    grouped = group_evidence_by_year(
        evidence=evidence,
        start_year=pred_start,
        end_year=pred_end,
        max_evidence_per_year=args.max_evidence_per_year,
        max_chars_per_evidence=args.max_chars_per_evidence,
    )

    evidence_by_year_text = build_compact_evidence_text(grouped)

    prompt = build_m2_prompt(
        query_text=query_text,
        target_subject=target_subject,
        start_year=pred_start,
        end_year=pred_end,
        evidence_by_year_text=evidence_by_year_text,
    )

    raw_output = ""
    parsed = None
    status = "ok"

    if args.no_llm:
        parsed = build_fallback_structure(grouped)
        raw_output = ""
        status = "no_llm_fallback"
    else:
        try:
            raw_output = call_llm(prompt, args)
            parsed = safe_json_loads(raw_output)

            if parsed is None:
                status = "json_parse_failed"
                parsed = build_fallback_structure(grouped)

        except Exception as e:
            status = f"llm_error: {type(e).__name__}: {str(e)}"
            parsed = build_fallback_structure(grouped)

    stats = compute_m2_stats(parsed)

    out = {
        "query_id": query_id,
        "query_text": query_text,
        "target_subject": target_subject,

        "predicted_start_year": pred_start,
        "predicted_end_year": pred_end,
        "gold_start_year": gold_start,
        "gold_end_year": gold_end,

        "method": row.get("method", ""),
        "rewrite_type": row.get("rewrite_type", ""),
        "query_type_name": row.get("query_type_name", ""),
        "query_type_normalized": row.get("query_type_normalized", ""),

        "m1_retrieved_years": row.get("retrieved_years", ""),
        "m1_temporal_coverage_at_k_pred_range": row.get("temporal_coverage_at_k_pred_range", ""),
        "m1_temporal_coverage_at_k_gold_range": row.get("temporal_coverage_at_k_gold_range", ""),
        "m1_temporal_precision_at_k_pred_range": row.get("temporal_precision_at_k_pred_range", ""),
        "m1_temporal_precision_at_k_gold_range": row.get("temporal_precision_at_k_gold_range", ""),

        "m2_status": status,
        "yearly_evidence_json": json.dumps(parsed.get("yearly_evidence", {}), ensure_ascii=False),
        "yearly_summary_text": yearly_json_to_text(parsed),
        "overall_observation": normalize_text(parsed.get("overall_observation", "")),
        "evidence_sufficiency": normalize_text(parsed.get("evidence_sufficiency", "")),
        "coverage_warning": normalize_text(parsed.get("coverage_warning", "")),

        "raw_m2_output": raw_output,
    }

    out.update(stats)

    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--m1-results", required=True, help="Path to m1_qwen_results.csv")
    parser.add_argument("--out", default="m2_structured_evidence.csv")
    parser.add_argument("--summary-out", default="m2_summary.csv")

    parser.add_argument(
        "--backend",
        choices=["ollama", "openai_compatible"],
        default="ollama"
    )
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="gpt-oss:20b")

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300)

    parser.add_argument(
        "--max-evidence-per-year",
        type=int,
        default=3,
        help="Maximum evidence items included per year in M2 prompt."
    )
    parser.add_argument(
        "--max-chars-per-evidence",
        type=int,
        default=700,
        help="Maximum characters per evidence item included in M2 prompt."
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use rule-based fallback only. Useful for debugging."
    )

    args = parser.parse_args()

    if not os.path.exists(args.m1_results):
        raise FileNotFoundError(f"M1 results file not found: {args.m1_results}")

    print("=" * 90)
    print("M2 Temporal Evidence Structuring with gpt-oss-20b")
    print("=" * 90)

    print(f"\n[1] Loading M1 results: {args.m1_results}")
    df = pd.read_csv(args.m1_results)

    required_cols = [
        "query_id",
        "query_text",
        "predicted_start_year",
        "predicted_end_year",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"M1 results missing required column: {col}")

    if args.limit:
        df = df.head(args.limit).copy()
        print(f"Using first {len(df)} rows for testing.")
    else:
        print(f"Using all rows: {len(df)}")

    print("\n[2] Running M2 structuring...")
    rows = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        out = run_m2_one(row, args)
        rows.append(out)

        print(
            f"[{i}/{len(df)}] "
            f"qid={out.get('query_id')} | "
            f"range=[{out.get('predicted_start_year')},{out.get('predicted_end_year')}] | "
            f"m2_status={out.get('m2_status')} | "
            f"years_with_evidence={out.get('num_years_with_evidence')}/"
            f"{out.get('num_years_in_structure')} | "
            f"sufficiency={out.get('evidence_sufficiency')}"
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    results = pd.DataFrame(rows)

    print("\n[3] Saving M2 results...")
    results.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"M2 results saved to: {args.out}")

    print("\n[4] Building summary...")

    summary_rows = []

    def add_summary(name: str, sub: pd.DataFrame):
        if len(sub) == 0:
            return

        status_counts = sub["m2_status"].value_counts().to_dict()
        suff_counts = sub["evidence_sufficiency"].value_counts().to_dict()

        summary_rows.append({
            "group": name,
            "num_queries": len(sub),
            "avg_num_years_in_structure": sub["num_years_in_structure"].mean(),
            "avg_num_years_with_evidence": sub["num_years_with_evidence"].mean(),
            "avg_num_years_without_evidence": sub["num_years_without_evidence"].mean(),
            "avg_evidence_year_coverage": sub["evidence_year_coverage"].mean(),
            "status_counts": json.dumps(status_counts, ensure_ascii=False),
            "evidence_sufficiency_counts": json.dumps(suff_counts, ensure_ascii=False),
        })

    add_summary("ALL", results)

    for col in ["query_type_normalized", "query_type_name", "rewrite_type", "method"]:
        if col in results.columns:
            for value, sub in results.groupby(col):
                value = normalize_text(value)
                if value:
                    add_summary(f"{col}={value}", sub)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.summary_out, index=False, encoding="utf-8-sig")
    print(f"M2 summary saved to: {args.summary_out}")

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
