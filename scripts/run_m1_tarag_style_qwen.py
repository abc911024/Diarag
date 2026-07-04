# -*- coding: utf-8 -*-
"""
TA-RAG-style baseline retrieval with Qwen embeddings.

This is not a full reproduction of TA-RAG.
It approximates TA-RAG's key retrieval idea:
1. use a temporal range,
2. create hypothetical temporal queries anchored at different years,
3. average their embeddings,
4. retrieve evidence within the temporal range,
5. pass raw chronological evidence to M3.

Outputs an M1-like CSV that can be passed to run_m3_naive_rag_gptoss.py.

Example:
    python run_m1_tarag_style_qwen.py ^
      --queries bridge_v3_m0_runs.csv ^
      --corpus-index corpus_index_qwen.csv ^
      --corpus-embeddings corpus_embeddings_qwen.npy ^
      --model Qwen/Qwen3-Embedding-0.6B ^
      --range-source gold ^
      --out m1_tarag_style_gold_qwen_results_with_source.csv ^
      --summary-out m1_tarag_style_gold_qwen_summary.csv ^
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


def choose_anchor_years(start: int, end: int, num_anchors: int) -> List[int]:
    years = list(range(start, end + 1))
    if len(years) <= num_anchors:
        return years
    positions = np.linspace(0, len(years) - 1, num_anchors)
    return sorted(set(years[int(round(p))] for p in positions))


def compute_temporal_metrics(gold_start: int, gold_end: int, retrieved_years: List[int]) -> dict:
    if gold_start is None or gold_end is None or gold_end < gold_start:
        return {
            "temporal_coverage_at_k_gold_range": np.nan,
            "temporal_precision_at_k_gold_range": np.nan,
        }

    gold_years = set(range(gold_start, gold_end + 1))
    retrieved_year_set = set(retrieved_years)

    coverage = len(gold_years.intersection(retrieved_year_set)) / len(gold_years) if gold_years else np.nan
    precision = sum(1 for y in retrieved_years if y in gold_years) / len(retrieved_years) if retrieved_years else np.nan

    return {
        "temporal_coverage_at_k_gold_range": coverage,
        "temporal_precision_at_k_gold_range": precision,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--queries", required=True)
    parser.add_argument("--corpus-index", required=True)
    parser.add_argument("--corpus-embeddings", required=True)

    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--out", default="m1_tarag_style_qwen_results_with_source.csv")
    parser.add_argument("--summary-out", default="m1_tarag_style_qwen_summary.csv")

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--method", default="FINAL_prior_guided_qualified_boundary")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--range-source",
        choices=["gold", "predicted"],
        default="gold",
        help="gold = use gold_start_year/gold_end_year; predicted = use predicted_start_year/predicted_end_year",
    )
    parser.add_argument("--num-anchors", type=int, default=5)
    parser.add_argument("--max-query-chars", type=int, default=2000)
    parser.add_argument("--sort-output-by-year", action="store_true", default=True)

    args = parser.parse_args()

    print("=" * 90)
    print("TA-RAG-style Qwen Retrieval Baseline")
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

    required = ["query_id", "source_id", "query_text", "gold_start_year", "gold_end_year"]
    if args.range_source == "predicted":
        required += ["predicted_start_year", "predicted_end_year"]

    for col in required:
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

    print(f"\n[4] Loading embedding model: {args.model}")
    if args.device:
        model = SentenceTransformer(args.model, device=args.device)
    else:
        model = SentenceTransformer(args.model)

    print("\n[5] Running TA-RAG-style retrieval...")
    outputs = []

    for _, row in queries.iterrows():
        query_id = normalize_text(row.get("query_id"))
        query_text = normalize_text(row.get("query_text"))[:args.max_query_chars]

        gold_start = safe_int(row.get("gold_start_year"))
        gold_end = safe_int(row.get("gold_end_year"))

        if args.range_source == "gold":
            range_start = gold_start
            range_end = gold_end
        else:
            range_start = safe_int(row.get("predicted_start_year"))
            range_end = safe_int(row.get("predicted_end_year"))

        out = {
            "query_id": query_id,
            "source_id": normalize_text(row.get("source_id")),
            "query_text": normalize_text(row.get("query_text")),
            "target_subject": row.get("target_subject", ""),
            "method": f"tarag_style_{args.range_source}_range",

            "rewrite_type": row.get("rewrite_type", ""),
            "query_type_name": row.get("query_type_name", ""),
            "query_type_normalized": row.get("query_type_normalized", ""),

            "predicted_start_year": range_start,
            "predicted_end_year": range_end,
            "gold_start_year": gold_start,
            "gold_end_year": gold_end,
            "range_source": args.range_source,
        }

        if range_start is None or range_end is None or range_end < range_start:
            print(f"[Warning] Invalid range for qid={query_id}: {range_start}-{range_end}")
            outputs.append(out)
            continue

        anchor_years = choose_anchor_years(range_start, range_end, args.num_anchors)
        anchored_queries = [f"In {y}, {query_text}" for y in anchor_years]

        anchor_embs = model.encode(
            anchored_queries,
            batch_size=len(anchored_queries),
            convert_to_numpy=True,
            normalize_embeddings=args.normalize,
            show_progress_bar=False,
        )

        query_emb = anchor_embs.mean(axis=0, keepdims=True)

        if args.normalize:
            query_emb = l2_normalize(query_emb)

        scores = np.dot(corpus_emb, query_emb[0])

        candidate_mask = (corpus_years >= range_start) & (corpus_years <= range_end)
        candidate_indices = np.where(candidate_mask.to_numpy())[0]

        if len(candidate_indices) == 0:
            print(f"[Warning] No candidates for qid={query_id}: {range_start}-{range_end}")
            outputs.append(out)
            continue

        candidate_scores = scores[candidate_indices]
        top_k = min(args.top_k, len(candidate_indices))
        top_order = np.argsort(-candidate_scores)[:top_k]
        selected = list(candidate_indices[top_order])

        if args.sort_output_by_year:
            selected = sorted(
                selected,
                key=lambda ci: (
                    safe_int(corpus.iloc[ci].get(year_col)) if safe_int(corpus.iloc[ci].get(year_col)) is not None else 9999,
                    -scores[ci],
                ),
            )

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
        out["anchor_years"] = json.dumps(anchor_years, ensure_ascii=False)

        metrics = compute_temporal_metrics(gold_start, gold_end, retrieved_years)
        out.update(metrics)

        outputs.append(out)

        print(
            f"[{len(outputs)}/{len(queries)}] "
            f"qid={query_id} | range=[{range_start},{range_end}] | "
            f"anchors={anchor_years} | "
            f"years={retrieved_years} | "
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