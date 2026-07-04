# -*- coding: utf-8 -*-
"""
Naive RAG baseline: M3 final MCQA answer selection.

Input:
    1. m1_naive_qwen_results_with_source.csv
    2. dqabench_MCQA.json

Output:
    1. m3_naive_rag_answers.csv
    2. m3_naive_rag_summary.csv

Example:
    python run_m3_naive_rag_gptoss.py ^
      --m1-results m1_naive_qwen_results_with_source.csv ^
      --mcqa-json dqabench_MCQA.json ^
      --backend ollama ^
      --model gpt-oss:20b ^
      --out m3_naive_rag_answers.csv ^
      --summary-out m3_naive_rag_summary.csv ^
      --timeout 600
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests


# ============================================================
# Utilities
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


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    raw = text.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            return None

    return None


def extract_option_from_text(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r'"predicted_option"\s*:\s*"([ABCD])"',
        r"'predicted_option'\s*:\s*'([ABCD])'",
        r"\bOption\s*([ABCD])\b",
        r"\bAnswer\s*[:：]?\s*([ABCD])\b",
        r"\bpredicted_option\s*[:：]?\s*([ABCD])\b",
        r"^\s*([ABCD])\s*$",
    ]

    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1).upper()

    return ""


def list_to_string(x: Any) -> str:
    if isinstance(x, list):
        return ", ".join(str(i) for i in x)
    return normalize_text(x)


# ============================================================
# Load MCQA
# ============================================================

def load_mcqa_json(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"MCQA file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("MCQA JSON should be a list.")

    out = {}
    for item in data:
        qid = normalize_text(item.get("id", ""))
        if qid:
            out[qid] = item

    return out


def get_choices_text(choices: Dict[str, Any]) -> str:
    return "\n".join([
        f"A. {normalize_text(choices.get('A', ''))}",
        f"B. {normalize_text(choices.get('B', ''))}",
        f"C. {normalize_text(choices.get('C', ''))}",
        f"D. {normalize_text(choices.get('D', ''))}",
    ])


# ============================================================
# Evidence formatting
# ============================================================

def build_raw_evidence_text(row: pd.Series, top_k: int = 10, max_chars_per_evidence: int = 500) -> str:
    blocks = []

    for i in range(1, top_k + 1):
        year = normalize_text(row.get(f"top{i}_year", ""))
        score = normalize_text(row.get(f"top{i}_score", ""))
        title = normalize_text(row.get(f"top{i}_title", ""))
        text = normalize_text(row.get(f"top{i}_text_preview", ""))

        if not title and not text:
            continue

        if len(text) > max_chars_per_evidence:
            text = text[:max_chars_per_evidence] + "..."

        blocks.append(
            f"[Evidence {i}]\n"
            f"Year: {year}\n"
            f"Score: {score}\n"
            f"Title: {title}\n"
            f"Text: {text}"
        )

    if not blocks:
        return "No retrieved evidence is available."

    return "\n\n".join(blocks)


# ============================================================
# LLM call
# ============================================================

def call_ollama(prompt: str, model: str, base_url: str, temperature: float, timeout: int) -> str:
    url = base_url.rstrip("/") + "/api/chat"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise final answer selection module for "
                    "multiple-choice diachronic question answering. "
                    "Use only the provided evidence. Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "")


def call_openai_compatible(prompt: str, model: str, base_url: str, api_key: str, temperature: float, timeout: int) -> str:
    url = base_url.rstrip("/") + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise final answer selection module for "
                    "multiple-choice diachronic question answering. "
                    "Use only the provided evidence. Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
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
# Prompt
# ============================================================

def build_prompt(
    rewritten_query: str,
    target_subject: str,
    original_question: str,
    choices: Dict[str, Any],
    raw_evidence_text: str,
) -> str:
    choices_text = get_choices_text(choices)

    return f"""
You are answering a multiple-choice diachronic question using raw evidence retrieved by a Naive RAG baseline.

Important:
This is the Naive RAG baseline.
The evidence was retrieved from the full corpus without temporal filtering and without Module 2 evidence structuring.

Rules:
1. Use only the provided evidence.
2. Do not use outside knowledge.
3. Choose the best answer from A, B, C, or D.
4. If the evidence is incomplete or noisy, still choose the best-supported option.
5. Return ONLY valid JSON. Do not include markdown or extra text.

Rewritten question:
{rewritten_query}

Target subject / ticker:
{target_subject}

Original benchmark question:
{original_question}

Answer choices:
{choices_text}

Raw top-k evidence from Naive RAG:
{raw_evidence_text}

Return JSON in exactly this format:
{{
  "predicted_option": "A | B | C | D",
  "predicted_answer_text": "...",
  "overall_trend": "...",
  "supporting_years": [2015, 2016],
  "reason": "...",
  "evidence_sufficiency": "high | medium | low"
}}
""".strip()


# ============================================================
# Run one row
# ============================================================

def run_one(row: pd.Series, mcqa_map: Dict[str, Dict[str, Any]], args) -> Dict[str, Any]:
    query_id = normalize_text(row.get("query_id", ""))
    source_id = normalize_text(row.get("source_id", ""))
    query_text = normalize_text(row.get("query_text", ""))
    target_subject = normalize_text(row.get("target_subject", ""))

    if not source_id:
        return {
            "query_id": query_id,
            "source_id": source_id,
            "query_text": query_text,
            "gold_answer": "",
            "predicted_option": "",
            "is_correct": 0,
            "m3_status": "missing_source_id",
            "raw_m3_output": "",
        }

    if source_id not in mcqa_map:
        return {
            "query_id": query_id,
            "source_id": source_id,
            "query_text": query_text,
            "gold_answer": "",
            "predicted_option": "",
            "is_correct": 0,
            "m3_status": "source_id_not_found_in_mcqa",
            "raw_m3_output": "",
        }

    item = mcqa_map[source_id]
    original_question = normalize_text(item.get("question", ""))
    choices = item.get("choices", {})
    gold_answer = normalize_text(item.get("correct_answer_key", "")).upper()

    raw_evidence_text = build_raw_evidence_text(
        row=row,
        top_k=args.top_k,
        max_chars_per_evidence=args.max_chars_per_evidence,
    )

    prompt = build_prompt(
        rewritten_query=query_text,
        target_subject=target_subject,
        original_question=original_question,
        choices=choices,
        raw_evidence_text=raw_evidence_text,
    )

    raw_output = ""
    parsed = None
    status = "ok"

    try:
        raw_output = call_llm(prompt, args)
        parsed = safe_json_loads(raw_output)

        if parsed is None:
            status = "json_parse_failed"
            predicted_option = extract_option_from_text(raw_output)
            parsed = {
                "predicted_option": predicted_option,
                "predicted_answer_text": "",
                "overall_trend": "",
                "supporting_years": [],
                "reason": "Fallback option extraction was used because JSON parsing failed.",
                "evidence_sufficiency": "",
            }
        else:
            predicted_option = normalize_text(parsed.get("predicted_option", "")).upper()

            if predicted_option not in ["A", "B", "C", "D"]:
                fallback = extract_option_from_text(raw_output)
                if fallback:
                    predicted_option = fallback
                    status = "invalid_option_fallback_used"
                else:
                    predicted_option = ""
                    status = "invalid_option"

    except Exception as e:
        status = f"llm_error: {type(e).__name__}: {str(e)}"
        predicted_option = ""
        parsed = {
            "predicted_option": "",
            "predicted_answer_text": "",
            "overall_trend": "",
            "supporting_years": [],
            "reason": status,
            "evidence_sufficiency": "",
        }

    is_correct = int(predicted_option == gold_answer) if predicted_option and gold_answer else 0

    return {
        "query_id": query_id,
        "source_id": source_id,
        "query_text": query_text,
        "target_subject": target_subject,

        "rewrite_type": row.get("rewrite_type", ""),
        "query_type_name": row.get("query_type_name", ""),
        "query_type_normalized": row.get("query_type_normalized", ""),
        "method": "naive_rag_full_corpus",

        "gold_start_year": row.get("gold_start_year", ""),
        "gold_end_year": row.get("gold_end_year", ""),

        "original_question": original_question,
        "choice_A": normalize_text(choices.get("A", "")),
        "choice_B": normalize_text(choices.get("B", "")),
        "choice_C": normalize_text(choices.get("C", "")),
        "choice_D": normalize_text(choices.get("D", "")),

        "gold_answer": gold_answer,
        "predicted_option": predicted_option,
        "is_correct": is_correct,

        "predicted_answer_text": normalize_text(parsed.get("predicted_answer_text", "")),
        "overall_trend": normalize_text(parsed.get("overall_trend", "")),
        "supporting_years": list_to_string(parsed.get("supporting_years", [])),
        "reason": normalize_text(parsed.get("reason", "")),
        "m3_evidence_sufficiency": normalize_text(parsed.get("evidence_sufficiency", "")),

        "retrieved_years": row.get("retrieved_years", ""),
        "temporal_coverage_at_k_gold_range": row.get("temporal_coverage_at_k_gold_range", ""),
        "temporal_precision_at_k_gold_range": row.get("temporal_precision_at_k_gold_range", ""),

        "m3_status": status,
        "raw_m3_output": raw_output,
    }


# ============================================================
# Summary
# ============================================================

def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_group(name: str, sub: pd.DataFrame):
        if len(sub) == 0:
            return

        rows.append({
            "group": name,
            "num_queries": len(sub),
            "num_answered": int(sub["predicted_option"].isin(["A", "B", "C", "D"]).sum()),
            "num_correct": int(sub["is_correct"].sum()),
            "mcqa_accuracy": float(sub["is_correct"].mean()) if len(sub) else 0.0,
            "status_counts": json.dumps(sub["m3_status"].value_counts().to_dict(), ensure_ascii=False),
        })

    add_group("ALL", results)

    for col in ["query_type_normalized", "query_type_name", "rewrite_type", "m3_evidence_sufficiency", "method"]:
        if col in results.columns:
            for value, sub in results.groupby(col):
                value = normalize_text(value)
                if value:
                    add_group(f"{col}={value}", sub)

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--m1-results", required=True)
    parser.add_argument("--mcqa-json", required=True)

    parser.add_argument("--out", default="m3_naive_rag_answers.csv")
    parser.add_argument("--summary-out", default="m3_naive_rag_summary.csv")

    parser.add_argument("--backend", choices=["ollama", "openai_compatible"], default="ollama")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="gpt-oss:20b")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-chars-per-evidence", type=int, default=500)

    args = parser.parse_args()

    print("=" * 90)
    print("Naive RAG Baseline: M3 Final MCQA Answer Selection")
    print("=" * 90)

    if not os.path.exists(args.m1_results):
        raise FileNotFoundError(f"M1 results file not found: {args.m1_results}")

    if not os.path.exists(args.mcqa_json):
        raise FileNotFoundError(f"MCQA file not found: {args.mcqa_json}")

    print(f"\n[1] Loading Naive RAG retrieval results: {args.m1_results}")
    m1 = pd.read_csv(args.m1_results)

    required_cols = ["query_id", "source_id", "query_text"]
    for col in required_cols:
        if col not in m1.columns:
            raise ValueError(f"Missing required column in M1 results: {col}")

    if args.limit:
        m1 = m1.head(args.limit).copy()
        print(f"Using first {len(m1)} rows for testing.")
    else:
        print(f"Using all rows: {len(m1)}")

    print(f"\n[2] Loading MCQA dataset: {args.mcqa_json}")
    mcqa_map = load_mcqa_json(args.mcqa_json)
    print(f"MCQA questions loaded: {len(mcqa_map)}")

    print("\n[3] Running Naive RAG M3 answer selection...")
    outputs = []

    for i, (_, row) in enumerate(m1.iterrows(), start=1):
        out = run_one(row, mcqa_map, args)
        outputs.append(out)

        print(
            f"[{i}/{len(m1)}] "
            f"qid={out.get('query_id')} | "
            f"source={out.get('source_id')} | "
            f"pred={out.get('predicted_option')} | "
            f"gold={out.get('gold_answer')} | "
            f"correct={out.get('is_correct')} | "
            f"status={out.get('m3_status')}"
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    results = pd.DataFrame(outputs)

    print("\n[4] Saving answers...")
    results.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Saved to: {args.out}")

    print("\n[5] Building summary...")
    summary = build_summary(results)
    summary.to_csv(args.summary_out, index=False, encoding="utf-8-sig")
    print(f"Saved to: {args.summary_out}")

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()