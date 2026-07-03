# -*- coding: utf-8 -*-
"""
M3: Final MCQA Answer Selection with gpt-oss-20b

Input:
    1. m2_structured_evidence_with_source.csv
    2. dqabench_MCQA.json

Output:
    1. m3_answers.csv
    2. m3_summary.csv

Purpose:
    Use structured temporal evidence from M2 to select the best A/B/C/D answer.
    Evaluate final answer quality using MCQA Accuracy.

Example test 5 rows:
    python run_m3_gptoss_mcqa.py ^
      --m2-results m2_structured_evidence_with_source.csv ^
      --mcqa-json dqabench_MCQA.json ^
      --backend ollama ^
      --model gpt-oss:20b ^
      --out m3_answers_test.csv ^
      --summary-out m3_summary_test.csv ^
      --limit 5

Run all:
    python run_m3_gptoss_mcqa.py ^
      --m2-results m2_structured_evidence_with_source.csv ^
      --mcqa-json dqabench_MCQA.json ^
      --backend ollama ^
      --model gpt-oss:20b ^
      --out m3_answers.csv ^
      --summary-out m3_summary.csv
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, Optional, List

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


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from model output.
    Handles JSON wrapped in markdown fences.
    """
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
        candidate = raw[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


def extract_option_from_text(text: str) -> str:
    """
    Fallback parser if model does not return valid JSON.
    """
    if not text:
        return ""

    text = text.strip()

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

    # 最後保守抓第一個單出現的 A/B/C/D
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1).upper()

    return ""


def list_to_string(x: Any) -> str:
    if isinstance(x, list):
        return ", ".join(str(i) for i in x)
    return normalize_text(x)


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
    url = base_url.rstrip("/") + "/api/chat"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Module 3: a precise final answer selection module for "
                    "multiple-choice diachronic question answering. "
                    "You must answer using only the provided structured evidence. "
                    "Return valid JSON only."
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


def call_openai_compatible(
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    timeout: int = 300,
) -> str:
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
                    "You are Module 3: a precise final answer selection module for "
                    "multiple-choice diachronic question answering. "
                    "You must answer using only the provided structured evidence. "
                    "Return valid JSON only."
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
# Load MCQA dataset
# ============================================================

def load_mcqa_json(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"MCQA json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("MCQA JSON should be a list of question objects.")

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
# Prompt
# ============================================================

def build_m3_prompt(
    rewritten_query: str,
    target_subject: str,
    original_question: str,
    choices: Dict[str, Any],
    yearly_summary_text: str,
    overall_observation: str,
    evidence_sufficiency: str,
    coverage_warning: str,
) -> str:
    choices_text = get_choices_text(choices)

    return f"""
You are Module 3: Final Answer Selection for diachronic multiple-choice question answering.

Your task:
Choose the best answer from A, B, C, or D based ONLY on the structured temporal evidence provided below.

Important rules:
1. Use only the provided structured evidence.
2. Do not use outside knowledge.
3. Do not assume facts that are not supported by the evidence.
4. The rewritten question is the question to answer.
5. The answer choices are inherited from the original benchmark question.
6. If the evidence is incomplete, still choose the best-supported option, but mention the uncertainty in the reason.
7. Return ONLY valid JSON. Do not include markdown or extra text.

Rewritten question:
{rewritten_query}

Target subject / ticker:
{target_subject}

Original benchmark question:
{original_question}

Answer choices:
{choices_text}

Structured yearly evidence from Module 2:
{yearly_summary_text}

Overall observation from Module 2:
{overall_observation}

Evidence sufficiency from Module 2:
{evidence_sufficiency}

Coverage warning:
{coverage_warning}

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
# Run one M3 item
# ============================================================

def run_m3_one(row: pd.Series, mcqa_map: Dict[str, Dict[str, Any]], args) -> Dict[str, Any]:
    query_id = normalize_text(row.get("query_id", ""))
    source_id = normalize_text(row.get("source_id", ""))
    query_text = normalize_text(row.get("query_text", ""))
    target_subject = normalize_text(row.get("target_subject", ""))

    if not source_id:
        return {
            "query_id": query_id,
            "source_id": source_id,
            "query_text": query_text,
            "m3_status": "missing_source_id",
            "predicted_option": "",
            "gold_answer": "",
            "is_correct": 0,
            "raw_m3_output": "",
        }

    if source_id not in mcqa_map:
        return {
            "query_id": query_id,
            "source_id": source_id,
            "query_text": query_text,
            "m3_status": "source_id_not_found_in_mcqa",
            "predicted_option": "",
            "gold_answer": "",
            "is_correct": 0,
            "raw_m3_output": "",
        }

    item = mcqa_map[source_id]
    original_question = normalize_text(item.get("question", ""))
    choices = item.get("choices", {})
    gold_answer = normalize_text(item.get("correct_answer_key", "")).upper()

    choice_A = normalize_text(choices.get("A", ""))
    choice_B = normalize_text(choices.get("B", ""))
    choice_C = normalize_text(choices.get("C", ""))
    choice_D = normalize_text(choices.get("D", ""))

    yearly_summary_text = normalize_text(row.get("yearly_summary_text", ""))
    overall_observation = normalize_text(row.get("overall_observation", ""))
    evidence_sufficiency = normalize_text(row.get("evidence_sufficiency", ""))
    coverage_warning = normalize_text(row.get("coverage_warning", ""))

    prompt = build_m3_prompt(
        rewritten_query=query_text,
        target_subject=target_subject,
        original_question=original_question,
        choices=choices,
        yearly_summary_text=yearly_summary_text,
        overall_observation=overall_observation,
        evidence_sufficiency=evidence_sufficiency,
        coverage_warning=coverage_warning,
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
                "evidence_sufficiency": evidence_sufficiency,
            }
        else:
            predicted_option = normalize_text(parsed.get("predicted_option", "")).upper()
            if predicted_option not in ["A", "B", "C", "D"]:
                fallback = extract_option_from_text(raw_output)
                if fallback:
                    predicted_option = fallback
                    status = "invalid_option_fallback_used"
                else:
                    status = "invalid_option"
                    predicted_option = ""

    except Exception as e:
        status = f"llm_error: {type(e).__name__}: {str(e)}"
        predicted_option = ""
        parsed = {
            "predicted_option": "",
            "predicted_answer_text": "",
            "overall_trend": "",
            "supporting_years": [],
            "reason": status,
            "evidence_sufficiency": evidence_sufficiency,
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
        "method": row.get("method", ""),

        "predicted_start_year": row.get("predicted_start_year", ""),
        "predicted_end_year": row.get("predicted_end_year", ""),
        "gold_start_year": row.get("gold_start_year", ""),
        "gold_end_year": row.get("gold_end_year", ""),

        "original_question": original_question,
        "choice_A": choice_A,
        "choice_B": choice_B,
        "choice_C": choice_C,
        "choice_D": choice_D,

        "gold_answer": gold_answer,
        "predicted_option": predicted_option,
        "is_correct": is_correct,

        "predicted_answer_text": normalize_text(parsed.get("predicted_answer_text", "")),
        "overall_trend": normalize_text(parsed.get("overall_trend", "")),
        "supporting_years": list_to_string(parsed.get("supporting_years", [])),
        "reason": normalize_text(parsed.get("reason", "")),
        "m3_evidence_sufficiency": normalize_text(parsed.get("evidence_sufficiency", "")),

        "m2_status": row.get("m2_status", ""),
        "m2_evidence_sufficiency": row.get("evidence_sufficiency", ""),
        "m2_evidence_year_coverage": row.get("evidence_year_coverage", ""),
        "m2_num_years_with_evidence": row.get("num_years_with_evidence", ""),
        "m2_num_years_without_evidence": row.get("num_years_without_evidence", ""),

        "m3_status": status,
        "raw_m3_output": raw_output,
    }


# ============================================================
# Summary
# ============================================================

def safe_accuracy(sub: pd.DataFrame) -> float:
    if len(sub) == 0:
        return 0.0
    return float(sub["is_correct"].mean())


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
            "mcqa_accuracy": safe_accuracy(sub),
            "status_counts": json.dumps(sub["m3_status"].value_counts().to_dict(), ensure_ascii=False),
        })

    add_group("ALL", results)

    for col in ["query_type_normalized", "query_type_name", "rewrite_type", "m2_evidence_sufficiency", "method"]:
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

    parser.add_argument("--m2-results", required=True, help="Path to m2_structured_evidence_with_source.csv")
    parser.add_argument("--mcqa-json", required=True, help="Path to dqabench_MCQA.json")

    parser.add_argument("--out", default="m3_answers.csv")
    parser.add_argument("--summary-out", default="m3_summary.csv")

    parser.add_argument(
        "--backend",
        choices=["ollama", "openai_compatible"],
        default="ollama",
    )
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="gpt-oss:20b")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300)

    args = parser.parse_args()

    if not os.path.exists(args.m2_results):
        raise FileNotFoundError(f"M2 results file not found: {args.m2_results}")

    print("=" * 90)
    print("M3 Final MCQA Answer Selection with gpt-oss-20b")
    print("=" * 90)

    print(f"\n[1] Loading M2 results: {args.m2_results}")
    m2 = pd.read_csv(args.m2_results)

    required_cols = [
        "query_id",
        "source_id",
        "query_text",
        "yearly_summary_text",
        "overall_observation",
        "evidence_sufficiency",
    ]

    for col in required_cols:
        if col not in m2.columns:
            raise ValueError(f"M2 results missing required column: {col}")

    if args.limit:
        m2 = m2.head(args.limit).copy()
        print(f"Using first {len(m2)} rows for testing.")
    else:
        print(f"Using all rows: {len(m2)}")

    print(f"\n[2] Loading MCQA dataset: {args.mcqa_json}")
    mcqa_map = load_mcqa_json(args.mcqa_json)
    print(f"MCQA questions loaded: {len(mcqa_map)}")

    print("\n[3] Running M3 answer selection...")
    outputs = []

    for i, (_, row) in enumerate(m2.iterrows(), start=1):
        out = run_m3_one(row, mcqa_map, args)
        outputs.append(out)

        print(
            f"[{i}/{len(m2)}] "
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

    print("\n[4] Saving M3 results...")
    results.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"M3 answers saved to: {args.out}")

    print("\n[5] Building summary...")
    summary = build_summary(results)
    summary.to_csv(args.summary_out, index=False, encoding="utf-8-sig")
    print(f"M3 summary saved to: {args.summary_out}")

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()