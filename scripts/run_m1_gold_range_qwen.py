# -*- coding: utf-8 -*-
"""
Gold-range w/o M2 retrieval experiment.

Purpose:
    Use gold_start_year / gold_end_year instead of M0 predicted range
    to perform temporal retrieval.

Pipeline:
    ADQAB-Implicit query
    -> gold time range
    -> Qwen embedding retrieval within gold range
    -> raw top-k evidence
    -> output M1-like CSV for M3 w/o M2

Input:
    1. bridge_v3_m0_runs.csv
    2. corpus_index_qwen.csv
    3. corpus_embeddings_qwen.npy

Output:
    1. m1_gold_range_qwen_results_with_source.csv
    2. m1_gold_range_qwen_summary.csv

Example:
    python run_m1_gold_range_qwen.py ^
      --queries bridge_v3_m0_runs.csv ^
      --corpus-index corpus_index_qwen.csv ^
      --corpus-embeddings corpus_embeddings_qwen.npy ^
      --model Qwen/Qwen3-Embedding-0.6B ^
      --out m1_gold_range_qwen_results_with_source.csv ^
      --summary-out m1_gold_range_qwen_summary.csv ^
      --top-k 10 ^
      --method FINAL_prior_guided_qualified_boundary ^
      --normalize ^
      --device cuda
"""

import argparse
import json
import os
from typing import Any, List, Optional

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


def safe_int(x: Any) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True, name: str = "") -> str:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(
            f"Cannot find required column for {name}. "
            f"Tried: {candidates}. Existing columns: {df.columns.tolist()}"
        )
    return ""


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def compute_temporal_metrics(gold_start: int, gold_end: int, retrieved_years: List[int]) -> dict:
    if gold_start is None or gold_end is None or gold_end < gold_start:
        return {
            "temporal_coverage_at_k_gold_range": np.nan,
            "temporal_precision_at_k_gold_range": np.nan,
        }

    gold_years = set(range(gold_start, gold_end + 1))
    retrieved_year_set = set(retrieved_years)

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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--queries", required=True, help="Query file, e.g. bridge_v3_m0_runs.csv")
    parser.add_argument("--corpus-index", required=True, help="Corpus metadata CSV, e.g. corpus_index_qwen.csv")
    parser.add_argument("--corpus-embeddings", required=True, help="Corpus embeddings, e.g. corpus_embeddings_qwen.npy")

    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--out", default="m1_gold_range_qwen_results_with_source.csv")
    parser.add_argument("--summary-out", default="m1_gold_range_qwen_summary.csv")

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--method", default="FINAL_prior_guided_qualified_boundary")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-query-chars", type=int, default=2000)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--device", default=None, help="Optional: cuda or cpu")

    parser.add_argument(
        "--per-year-quota",
        type=int,
        default=2,
        help="Coverage-aware reranking quota per year. Use 0 to disable.",
    )

    args = parser.parse_args()

    print("=" * 90)
    print("Gold-range w/o M2: Qwen Temporal Retrieval")
    print("=" * 90)

    if not os.path.exists(args.queries):
        raise FileNotFoundError(args.queries)
    if not os.path.exists(args.corpus_index):
        raise FileNotFoundError(args.corpus_index)
    if not os.path.exists(args.corpus_embeddings):
        raise FileNotFoundError(args.corpus_embeddings)

    # ------------------------------------------------------------
    # Load queries
    # ------------------------------------------------------------
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

    required_query_cols = [
        "query_id",
        "source_id",
        "query_text",
        "gold_start_year",
        "gold_end_year",
    ]

    for col in required_query_cols:
        if col not in queries.columns:
            raise ValueError(f"Queries missing required column: {col}. Existing columns: {queries.columns.tolist()}")

    # ------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------
    print(f"\n[2] Loading corpus index: {args.corpus_index}")
    corpus = pd.read_csv(args.corpus_index)
    corpus.columns = [str(c).strip() for c in corpus.columns]
    print(f"Corpus rows: {len(corpus)}")

    print(f"\n[3] Loading corpus embeddings: {args.corpus_embeddings}")
    corpus_emb = np.load(args.corpus_embeddings)
    print(f"Corpus embeddings shape: {corpus_emb.shape}")

    if len(corpus) != corpus_emb.shape[0]:
        raise ValueError(f"Corpus rows ({len(corpus)}) != embedding rows ({corpus_emb.shape[0]})")

    doc_id_col = find_col(corpus, ["doc_id", "id", "chunk_id", "document_id"], required=False, name="doc_id")
    year_col = find_col(corpus, ["publish_year", "year", "event_year", "publication_year"], required=True, name="year")
    title_col = find_col(corpus, ["title", "headline"], required=False, name="title")
    text_col = find_col(corpus, ["text", "content", "chunk_text", "passage"], required=True, name="text")

    print("\nDetected corpus columns:")
    print(f"doc_id : {doc_id_col}")
    print(f"year   : {year_col}")
    print(f"title  : {title_col}")
    print(f"text   : {text_col}")

    corpus_years = corpus[year_col].apply(safe_int)

    if args.normalize:
        print("\nNormalizing corpus embeddings...")
        corpus_emb = l2_normalize(corpus_emb)

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    print(f"\n[4] Loading embedding model: {args.model}")
    if args.device:
        model = SentenceTransformer(args.model, device=args.device)
    else:
        model = SentenceTransformer(args.model)

    # ------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------
    print("\n[5] Running gold-range temporal retrieval...")
    outputs = []

    for idx, row in queries.iterrows():
        query_id = normalize_text(row.get("query_id"))
        query_text = normalize_text(row.get("query_text"))[:args.max_query_chars]

        gold_start = safe_int(row.get("gold_start_year"))
        gold_end = safe_int(row.get("gold_end_year"))

        out = {
            "query_id": query_id,
            "source_id": normalize_text(row.get("source_id")),
            "query_text": normalize_text(row.get("query_text")),
            "target_subject": row.get("target_subject", ""),
            "method": "gold_range_temporal_retrieval_wo_m2",

            "rewrite_type": row.get("rewrite_type", ""),
            "query_type_name": row.get("query_type_name", ""),
            "query_type_normalized": row.get("query_type_normalized", ""),

            # For compatibility with M3 w/o M2 script
            "predicted_start_year": gold_start,
            "predicted_end_year": gold_end,

            "gold_start_year": gold_start,
            "gold_end_year": gold_end,
        }

        if gold_start is None or gold_end is None or gold_end < gold_start:
            print(f"[Warning] Invalid gold range for qid={query_id}: {gold_start}-{gold_end}")
            outputs.append(out)
            continue

        # Encode query
        query_emb = model.encode(
            [query_text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=args.normalize,
            show_progress_bar=False,
        )

        scores = np.dot(corpus_emb, query_emb[0])

        # Filter by gold range
        candidate_mask = (corpus_years >= gold_start) & (corpus_years <= gold_end)
        candidate_indices = np.where(candidate_mask.to_numpy())[0]

        if len(candidate_indices) == 0:
            print(f"[Warning] No corpus candidates in gold range for qid={query_id}: {gold_start}-{gold_end}")
            outputs.append(out)
            continue

        candidate_scores = scores[candidate_indices]

        # First get a larger pool for reranking
        pool_size = min(max(args.top_k * 5, args.top_k), len(candidate_indices))
        pool_order = np.argsort(-candidate_scores)[:pool_size]
        pool_indices = candidate_indices[pool_order]

        # Coverage-aware reranking
        selected = []
        year_counts = {}

        if args.per_year_quota and args.per_year_quota > 0:
            # First pass: limit per-year quota
            for ci in pool_indices:
                y = safe_int(corpus.iloc[ci].get(year_col))
                if y is None:
                    continue
                if year_counts.get(y, 0) < args.per_year_quota:
                    selected.append(ci)
                    year_counts[y] = year_counts.get(y, 0) + 1
                if len(selected) >= args.top_k:
                    break

            # Second pass: fill remaining by similarity
            if len(selected) < args.top_k:
                for ci in pool_indices:
                    if ci not in selected:
                        selected.append(ci)
                    if len(selected) >= args.top_k:
                        break
        else:
            selected = list(pool_indices[:args.top_k])

        retrieved_years = []

        for rank, ci in enumerate(selected, start=1):
            c = corpus.iloc[ci]
            year = safe_int(c.get(year_col))
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

        metrics = compute_temporal_metrics(gold_start, gold_end, retrieved_years)
        out.update(metrics)

        outputs.append(out)

        print(
            f"[{len(outputs)}/{len(queries)}] "
            f"qid={query_id} | gold=[{gold_start},{gold_end}] | "
            f"years={retrieved_years} | "
            f"cov_gold={out['temporal_coverage_at_k_gold_range']:.4f} | "
            f"prec_gold={out['temporal_precision_at_k_gold_range']:.4f}"
        )

    results = pd.DataFrame(outputs)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    print("\n[6] Saving retrieval results...")
    results.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Saved to: {args.out}")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
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