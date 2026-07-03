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
Classify the temporal intent of the query into exactly one of the following