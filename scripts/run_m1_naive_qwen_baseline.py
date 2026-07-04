# -*- coding: utf-8 -*-
"""
Naive RAG baseline retrieval with Qwen embeddings.

Purpose:
    Retrieve top-k evidence from the full corpus without temporal filtering.
    This produces an M1-like result file that can be passed to
    run_m3_gptoss_wo_m2.py for final MCQA answer selection.

Input:
    1. bridge_v3_m0_runs.csv
    2. corpus_index_qwen.csv
    3. corpus_embeddings_qwen.npy

Output:
    1. m1_naive_qwen_results_with_source.csv
    2. m1_naive_qwen_summary.csv

Example:
    python run_m1_naive_qwen_baseline.py ^
      --queries bridge_v3_m0_runs.csv ^
      --corpus-index corpus_index_qwen.csv ^
      --corpus-embeddings corpus_embeddings_qwen.npy ^
      --model Qwen/Qwen3-Embedding-0.6B ^
      --out m1_naive_qwen_results_with_source.csv ^
      --summary-out m1_naive_qwen_summary.csv ^
      --top-k 10 ^
      --method FINAL_prior_guided_qualified_boundary
"""

import argparse
import json
import os
from typing import Any, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


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


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True, name: str = "") -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    if required:
        raise ValueError(f"Cannot find required column for {name}. Tried: {candidates}. Existing columns: {df.columns.tolist()}")
    return ""


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def compute_temporal_metrics(row: pd.Series, retrieved_years: List[int]) -> dict:
    """
    For naive baseline, retrieval is not filtered by time.
    We still compute gold-range temporal coverage and precision
    for diagnostic comparison.
    """
    try:
        gold_start = int(row.get("gold_start_year"))
        gold_end = int(row.get("gold_end_year"))
    except Exception:
        return {
            "temporal_coverage_at_k_gold_range": np.nan,
            "temporal_precision_at_k_gold_range": np.nan,
        }

    if gold_end < gold_start:
        return {
            "temporal_coverage_at_k_gold_range": np.nan,
            "temporal_precision_at_k_gold_range": np.nan,
        }

    gold_years = set(range(gold_start, gold_end + 1))
    retrieved_year_set = set(y for y in retrieved_years if isinstance(y, int))

    if len(gold_years) == 0:
        coverage = np.nan
    else:
        coverage = len(gold_years.intersection(retrieved_year_set)) / len(gold_years)

    if len(retrieved_years) == 0:
        precision = np.nan
    else:
        precision = sum(1 for y in retrieved_years if y in gold_years) / len(retrieved_years)

    return {
        "temporal_coverage_at_k_gold_range": coverage,
        "temporal_precision_at_k_gold_range": precision,
    }


def safe_int_year(x: Any):
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


# ============================================================
# Main retrieval
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--queries", required=True, help="M0 / bridge query file, e.g. bridge_v3_m0_runs.csv")
    parser.add_argument("--corpus-index", required=True, help="Corpus metadata CSV, e.g. corpus_index_qwen.csv")
    parser.add_argument("--corpus-embeddings", required=True, help="Corpus embedding matrix, e.g. corpus_embeddings_qwen.npy")

    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--out", default="m1_naive_qwen_results_with_source.csv")
    parser.add_argument("--summary-out", default="m1_naive_qwen_summary.csv")

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--method", default="FINAL_prior_guided_qualified_boundary")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-query-chars", type=int, default=2000)
    parser.add_argument("--device", default=None, help="Optional: cuda or cpu")
    parser.add_argument("--normalize", action="store_true", help="Normalize query/corpus embeddings before cosine similarity")

    args = parser.parse_args()

    print("=" * 90)
    print("Naive RAG Baseline Retrieval with Qwen Embeddings")
    print("=" * 90)

    if not os.path.exists(args.queries):
        raise FileNotFoundError(args.queries)

    if not os.path.exists(args.corpus_index):
        raise FileNotFoundError(args.corpus_index)

    if not os.path.exists(args.corpus_embeddings):
        raise FileNotFoundError(args.corpus_embeddings)

    print(f"\n[1] Loading queries: {args.queries}")
    queries = pd.read_csv(args.queries)
    queries.columns = [str(c).strip() for c in queries.columns]

    if "method" in queries.columns and args.method:
        before = len(queries)
        filtered = queries[queries["method"].astype(str) == args.method].copy()
        if len(filtered) > 0:
            queries = filtered
            print(f"Filtered by method={args.method}: {len(queries)} / {before}")
        else:
            print("Warning: method filter returned 0 rows. Using all queries.")

    if args.limit:
        queries = queries.head(args.limit).copy()
        print(f"Using first {len(queries)} queries for testing.")
    else:
        print(f"Using all queries: {len(queries)}")

    required_query_cols = ["query_id", "query_text", "source_id"]
    for col in required_query_cols:
        if col not in queries.columns:
            raise ValueError(f"Queries missing required column: {col}. Existing columns: {queries.columns.tolist()}")

    print(f"\n[2] Loading corpus index: {args.corpus_index}")
    corpus = pd.read_csv(args.corpus_index)
    corpus.columns = [str(c).strip() for c in corpus.columns]
    print(f"Corpus rows: {len(corpus)}")

    print(f"\n[3] Loading corpus embeddings: {args.corpus_embeddings}")
    corpus_emb = np.load(args.corpus_embeddings)
    print(f"Corpus embeddings shape: {corpus_emb.shape}")

    if len(corpus) != corpus_emb.shape[0]:
        raise ValueError(f"Corpus rows ({len(corpus)}) != embedding rows ({corpus_emb.shape[0]})")

    # Flexible column detection
    doc_id_col = find_col(corpus, ["doc_id", "id", "chunk_id", "document_id"], required=False, name="doc_id")
    year_col = find_col(corpus, ["publish_year", "year", "event_year", "publication_year"], required=True, name="publish year")
    title_col = find_col(corpus, ["title", "headline"], required=False, name="title")
    text_col = find_col(corpus, ["text", "content", "chunk_text", "passage"], required=True, name="text")

    print("\nDetected corpus columns:")
    print(f"doc_id : {doc_id_col}")
    print(f"year   : {year_col}")
    print(f"title  : {title_col}")
    print(f"text   : {text_col}")

    if args.normalize:
        print("\nNormalizing corpus embeddings...")
        corpus_emb = l2_normalize(corpus_emb)

    print(f"\n[4] Loading embedding model: {args.model}")
    if args.device:
        model = SentenceTransformer(args.model, device=args.device)
    else:
        model = SentenceTransformer(args.model)

    print("\n[5] Running full-corpus retrieval...")
    outputs = []

    for idx, row in queries.iterrows():
        query_id = normalize_text(row.get("query_id"))
        query_text = normalize_text(row.get("query_text"))[:args.max_query_chars]

        query_emb = model.encode(
            [query_text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=args.normalize,
            show_progress_bar=False,
        )

        scores = np.dot(corpus_emb, query_emb[0])

        top_k = min(args.top_k, len(corpus))
        top_indices = np.argsort(-scores)[:top_k]

        out = {
            "query_id": query_id,
            "source_id": normalize_text(row.get("source_id")),
            "query_text": normalize_text(row.get("query_text")),
            "target_subject": row.get("target_subject", ""),
            "method": "naive_full_corpus_qwen",
            "rewrite_type": row.get("rewrite_type", ""),
            "query_type_name": row.get("query_type_name", ""),
            "query_type_normalized": row.get("query_type_normalized", ""),
            "predicted_start_year": row.get("predicted_start_year", ""),
            "predicted_end_year": row.get("predicted_end_year", ""),
            "gold_start_year": row.get("gold_start_year", ""),
            "gold_end_year": row.get("gold_end_year", ""),
        }

        retrieved_years = []

        for rank, ci in enumerate(top_indices, start=1):
            c = corpus.iloc[ci]
            year = safe_int_year(c.get(year_col))
            if year is not None:
                retrieved_years.append(year)

            doc_id = normalize_text(c.get(doc_id_col)) if doc_id_col else str(ci)
            title = normalize_text(c.get(title_col)) if title_col else ""
            text = normalize_text(c.get(text_col))

            out[f"top{rank}_doc_id"] = doc_id
            out[f"top{rank}_year"] = year if year is not None else ""
            out[f"top{rank}_score"] = float(scores[ci])
            out[f"top{rank}_title"] = title
            out[f"top{rank}_text_preview"] = text[:1000]

        out["retrieved_years"] = json.dumps(retrieved_years, ensure_ascii=False)

        metrics = compute_temporal_metrics(row, retrieved_years)
        out.update(metrics)

        outputs.append(out)

        print(
            f"[{len(outputs)}/{len(queries)}] "
            f"qid={query_id} | years={retrieved_years} | "
            f"cov_gold={out['temporal_coverage_at_k_gold_range']:.4f} | "
            f"prec_gold={out['temporal_precision_at_k_gold_range']:.4f}"
        )

    results = pd.DataFrame(outputs)

    print("\n[6] Saving retrieval results...")
    results.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Saved to: {args.out}")

    print("\n[7] Building summary...")
    rows = []

    def add_summary(group_name: str, sub: pd.DataFrame):
        rows.append({
            "group": group_name,
            "num_queries": len(sub),
            "avg_temporal_coverage_at_k_gold_range": sub["temporal_coverage_at_k_gold_range"].mean(),
            "avg_temporal_precision_at_k_gold_range": sub["temporal_precision_at_k_gold_range"].mean(),
        })

    add_summary("ALL", results)

    for col in ["query_type_normalized", "query_type_name", "rewrite_type"]:
        if col in results.columns:
            for value, sub in results.groupby(col):
                value = normalize_text(value)
                if value:
                    add_summary(f"{col}={value}", sub)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.summary_out, index=False, encoding="utf-8-sig")
    print(f"Saved to: {args.summary_out}")

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()