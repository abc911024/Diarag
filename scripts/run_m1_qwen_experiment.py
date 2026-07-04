# -*- coding: utf-8 -*-
"""
M1 Qwen Embedding Temporal Retrieval Experiment

功能：
1. 讀取 M0 結果 bridge_v3_m0_runs.csv
2. 讀取 Qwen corpus embedding：
   - corpus_index_qwen.csv
   - corpus_embeddings_qwen.npy
3. 根據 M0 預測年份範圍 predicted_start_year / predicted_end_year 過濾 corpus
4. 使用 Qwen 對 query 做 embedding
5. 計算 query embedding 與候選 corpus embeddings 的 cosine similarity
6. 取 Top-k evidence
7. 計算：
   - Temporal Coverage@k
   - Temporal Precision@k
8. 輸出：
   - m1_qwen_results.csv
   - m1_qwen_summary.csv

使用範例：

python run_m1_qwen_experiment.py ^
  --m0-results bridge_v3_m0_runs.csv ^
  --corpus-index corpus_index_qwen.csv ^
  --corpus-embeddings corpus_embeddings_qwen.npy ^
  --model Qwen/Qwen3-Embedding-0.6B ^
  --out m1_qwen_results.csv ^
  --summary-out m1_qwen_summary.csv ^
  --top-k 10 ^
  --method-filter FINAL_prior_guided_qualified_boundary

先測 5 筆：

python run_m1_qwen_experiment.py ^
  --m0-results bridge_v3_m0_runs.csv ^
  --corpus-index corpus_index_qwen.csv ^
  --corpus-embeddings corpus_embeddings_qwen.npy ^
  --model Qwen/Qwen3-Embedding-0.6B ^
  --out m1_qwen_results_test.csv ^
  --summary-out m1_qwen_summary_test.csv ^
  --top-k 10 ^
  --method-filter FINAL_prior_guided_qualified_boundary ^
  --limit 5
"""

import argparse
import os
import re
import json
from typing import Any, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# Basic utilities
# ============================================================

def normalize_text(x: Any) -> str:
    """把任意值轉成乾淨字串。"""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def parse_year(x: Any) -> Optional[int]:
    """從 int / float / string 中解析年份。"""
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
    """確保 start_year <= end_year。"""
    if start_year is None or end_year is None:
        return start_year, end_year
    if start_year > end_year:
        return end_year, start_year
    return start_year, end_year


def year_list(start_year: Optional[int], end_year: Optional[int]) -> List[int]:
    """產生 inclusive year list。"""
    start_year, end_year = ensure_year_order(start_year, end_year)
    if start_year is None or end_year is None:
        return []
    return list(range(start_year, end_year + 1))


def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """row-wise normalize，用於 cosine similarity。"""
    mat = mat.astype("float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """normalize 單一向量。"""
    vec = vec.astype("float32")
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def years_to_string(years: List[int]) -> str:
    cleaned = []
    for y in years:
        try:
            cleaned.append(int(y))
        except Exception:
            pass
    return ",".join(str(y) for y in sorted(set(cleaned)))


# ============================================================
# Temporal metrics
# ============================================================

def temporal_coverage_at_k(
    retrieved_years: List[int],
    target_start_year: Optional[int],
    target_end_year: Optional[int],
) -> float:
    """
    Temporal Coverage@k:
    在 target range 中，有多少比例的年份至少被 top-k evidence 覆蓋。

    例如：
    target range = 2018-2022，共 5 年
    retrieved years = 2018, 2019, 2021
    coverage = 3 / 5 = 0.6
    """
    target_years = set(year_list(target_start_year, target_end_year))
    if not target_years:
        return 0.0

    retrieved_set = set()
    for y in retrieved_years:
        try:
            retrieved_set.add(int(y))
        except Exception:
            pass

    covered = target_years.intersection(retrieved_set)
    return len(covered) / len(target_years)


def temporal_precision_at_k(
    retrieved_years: List[int],
    target_start_year: Optional[int],
    target_end_year: Optional[int],
) -> float:
    """
    Temporal Precision@k:
    top-k retrieved chunks 中，有多少比例的 evidence 年份落在 target range。

    注意：
    如果 M1 已經先用 predicted range 做 time filter，
    temporal_precision_at_k_pred_range 通常會是 1.0。
    但 temporal_precision_at_k_gold_range 不一定是 1.0，
    因為 predicted range 可能與 gold range 不一致。
    """
    if not retrieved_years:
        return 0.0

    target_years = set(year_list(target_start_year, target_end_year))
    if not target_years:
        return 0.0

    in_range = 0
    total = 0

    for y in retrieved_years:
        try:
            y = int(y)
            total += 1
            if y in target_years:
                in_range += 1
        except Exception:
            continue

    if total == 0:
        return 0.0

    return in_range / total


def boundary_overlap_iou(
    pred_start: Optional[int],
    pred_end: Optional[int],
    gold_start: Optional[int],
    gold_end: Optional[int],
) -> float:
    """
    M0 range 與 gold range 的 IoU，放在 M1 結果中方便分析。
    """
    pred_start, pred_end = ensure_year_order(pred_start, pred_end)
    gold_start, gold_end = ensure_year_order(gold_start, gold_end)

    if pred_start is None or pred_end is None or gold_start is None or gold_end is None:
        return 0.0

    inter_start = max(pred_start, gold_start)
    inter_end = min(pred_end, gold_end)
    intersection = max(0, inter_end - inter_start + 1)

    union_start = min(pred_start, gold_start)
    union_end = max(pred_end, gold_end)
    union = max(1, union_end - union_start + 1)

    return intersection / union


# ============================================================
# Load M0 results
# ============================================================

def detect_m0_columns(m0: pd.DataFrame) -> Dict[str, str]:
    """
    自動偵測 M0 欄位名稱。
    你的 bridge_v3_m0_runs.csv 主要會用：
    - query_id
    - query_text
    - target_subject
    - predicted_start_year
    - predicted_end_year
    - gold_start_year
    - gold_end_year
    - method
    """

    def pick(candidates: List[str], required: bool = True) -> str:
        for c in candidates:
            if c in m0.columns:
                return c
        if required:
            raise ValueError(f"Cannot find required column from candidates: {candidates}")
        return ""

    cols = {
        "query_id": pick(["query_id", "id", "qid"], required=True),
        "query_text": pick(["query_text", "query", "rewritten_query", "question"], required=True),
        "target_subject": pick(["target_subject", "ticker", "symbol", "subject"], required=False),
        "pred_start": pick(["predicted_start_year", "pred_start_year", "pred_start", "start_year"], required=True),
        "pred_end": pick(["predicted_end_year", "pred_end_year", "pred_end", "end_year"], required=True),
        "gold_start": pick(["gold_start_year", "gold_start", "answer_start_year"], required=False),
        "gold_end": pick(["gold_end_year", "gold_end", "answer_end_year"], required=False),
        "method": pick(["method", "m0_method"], required=False),
    }

    return cols


def load_m0_results(args) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if not os.path.exists(args.m0_results):
        raise FileNotFoundError(f"M0 results not found: {args.m0_results}")

    m0 = pd.read_csv(args.m0_results)
    cols = detect_m0_columns(m0)

    print("Detected M0 columns:")
    for k, v in cols.items():
        print(f"  {k}: {v if v else '(none)'}")

    # method filter
    if args.method_filter:
        if not cols["method"]:
            raise ValueError("--method-filter was provided, but no method column was found in M0 results.")

        before = len(m0)
        m0 = m0[m0[cols["method"]].astype(str) == args.method_filter].copy()
        print(f"\nMethod filter = {args.method_filter}")
        print(f"M0 rows after filter: {len(m0)} / {before}")

    # 如果同一 query_id 有多筆，保留第一筆或依照 overlap_score 排序
    if args.dedupe_by_query:
        before = len(m0)

        if "overlap_score" in m0.columns:
            m0["_overlap_score_float"] = pd.to_numeric(m0["overlap_score"], errors="coerce").fillna(0.0)
            m0 = (
                m0.sort_values([cols["query_id"], "_overlap_score_float"], ascending=[True, False])
                  .groupby(cols["query_id"], as_index=False)
                  .head(1)
                  .drop(columns=["_overlap_score_float"])
                  .copy()
            )
        else:
            m0 = m0.drop_duplicates(subset=[cols["query_id"]], keep="first").copy()

        print(f"M0 rows after dedupe_by_query: {len(m0)} / {before}")

    if args.limit:
        m0 = m0.head(args.limit).copy()
        print(f"M0 rows after limit: {len(m0)}")

    return m0.reset_index(drop=True), cols


# ============================================================
# Load corpus index and embeddings
# ============================================================

def load_corpus_and_embeddings(args) -> Tuple[pd.DataFrame, np.ndarray]:
    if not os.path.exists(args.corpus_index):
        raise FileNotFoundError(f"Corpus index not found: {args.corpus_index}")

    if not os.path.exists(args.corpus_embeddings):
        raise FileNotFoundError(f"Corpus embeddings not found: {args.corpus_embeddings}")

    corpus = pd.read_csv(args.corpus_index)
    embeddings = np.load(args.corpus_embeddings)

    if len(corpus) != embeddings.shape[0]:
        raise ValueError(
            f"corpus_index rows ({len(corpus)}) != embeddings rows ({embeddings.shape[0]}). "
            "請確認 corpus_index_qwen.csv 和 corpus_embeddings_qwen.npy 是同一次 build 產生的。"
        )

    # 必要欄位
    if "publish_year" not in corpus.columns:
        raise ValueError("corpus_index missing required column: publish_year")

    if "doc_id" not in corpus.columns:
        corpus["doc_id"] = [f"doc_{i:06d}" for i in range(len(corpus))]

    if "title" not in corpus.columns:
        corpus["title"] = ""

    if "text" not in corpus.columns:
        if "doc_text" in corpus.columns:
            corpus["text"] = corpus["doc_text"]
        else:
            corpus["text"] = ""

    if "doc_text" not in corpus.columns:
        corpus["doc_text"] = (
            corpus["title"].fillna("").astype(str)
            + "\n"
            + corpus["text"].fillna("").astype(str)
        )

    if "ticker" not in corpus.columns:
        corpus["ticker"] = ""

    corpus["publish_year"] = corpus["publish_year"].apply(parse_year)
    valid_mask = corpus["publish_year"].notna()

    corpus = corpus[valid_mask].copy()
    embeddings = embeddings[valid_mask.values]

    corpus["publish_year"] = corpus["publish_year"].astype(int)

    # normalize corpus embeddings
    embeddings = normalize_matrix(embeddings)

    corpus = corpus.reset_index(drop=True)

    print(f"\nCorpus rows: {len(corpus)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Corpus year range: {corpus['publish_year'].min()} - {corpus['publish_year'].max()}")

    return corpus, embeddings


# ============================================================
# Retrieval functions
# ============================================================

def build_retrieval_query(
    row: pd.Series,
    cols: Dict[str, str],
    use_query_prefix: bool = False,
) -> str:
    query = normalize_text(row.get(cols["query_text"], ""))
    target = normalize_text(row.get(cols["target_subject"], "")) if cols["target_subject"] else ""

    # 把 ticker / subject 加到 query 前面，增強 retrieval focus
    if target and target.lower() not in query.lower():
        query = f"{target} {query}"

    if use_query_prefix:
        query = f"query: {query}"

    return query


def filter_candidates(
    corpus: pd.DataFrame,
    start_year: Optional[int],
    end_year: Optional[int],
    target_subject: str = "",
    use_ticker_filter: bool = False,
) -> np.ndarray:
    """
    根據 predicted_start_year / predicted_end_year 過濾 corpus。
    可選擇是否加 ticker filter。
    """
    start_year, end_year = ensure_year_order(start_year, end_year)

    if start_year is None or end_year is None:
        return np.array([], dtype=int)

    mask = (
        (corpus["publish_year"] >= start_year)
        & (corpus["publish_year"] <= end_year)
    )

    if use_ticker_filter and target_subject:
        target = re.escape(target_subject)

        candidate_text = (
            corpus["ticker"].fillna("").astype(str)
            + " "
            + corpus["title"].fillna("").astype(str)
            + " "
            + corpus["text"].fillna("").astype(str)
        )

        target_mask = candidate_text.str.contains(target, case=False, regex=True)
        mask = mask & target_mask

    return np.where(mask.values)[0]


def select_topk_by_score(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    top_k: int,
) -> List[int]:
    if len(candidate_indices) == 0:
        return []

    order = np.argsort(candidate_scores)[::-1]
    selected = candidate_indices[order[:top_k]]
    return selected.tolist()


def select_topk_coverage_aware(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    corpus: pd.DataFrame,
    top_k: int,
    per_year_quota: int = 2,
) -> List[int]:
    """
    Coverage-aware reranking:
    避免 top-k 全部集中在同一年。

    做法：
    1. 先依 score 排序
    2. 每一年最多先選 per_year_quota 筆
    3. 若不足 top-k，再用 score 補滿
    """
    if len(candidate_indices) == 0:
        return []

    order = np.argsort(candidate_scores)[::-1]
    sorted_indices = candidate_indices[order]

    selected = []
    selected_set = set()
    year_count: Dict[int, int] = {}

    # First pass: quota by year
    for idx in sorted_indices:
        y = int(corpus.iloc[idx]["publish_year"])
        if year_count.get(y, 0) < per_year_quota:
            selected.append(int(idx))
            selected_set.add(int(idx))
            year_count[y] = year_count.get(y, 0) + 1

        if len(selected) >= top_k:
            break

    # Second pass: fill by score
    if len(selected) < top_k:
        for idx in sorted_indices:
            idx = int(idx)
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
            if len(selected) >= top_k:
                break

    return selected[:top_k]


def retrieve_one_query(
    row: pd.Series,
    cols: Dict[str, str],
    corpus: pd.DataFrame,
    corpus_embeddings: np.ndarray,
    model: SentenceTransformer,
    args,
) -> Dict[str, Any]:
    query_id = normalize_text(row.get(cols["query_id"], ""))
    query_text = normalize_text(row.get(cols["query_text"], ""))
    target_subject = normalize_text(row.get(cols["target_subject"], "")) if cols["target_subject"] else ""

    pred_start = parse_year(row.get(cols["pred_start"], None))
    pred_end = parse_year(row.get(cols["pred_end"], None))
    pred_start, pred_end = ensure_year_order(pred_start, pred_end)

    gold_start = parse_year(row.get(cols["gold_start"], None)) if cols["gold_start"] else None
    gold_end = parse_year(row.get(cols["gold_end"], None)) if cols["gold_end"] else None
    gold_start, gold_end = ensure_year_order(gold_start, gold_end)

    method = normalize_text(row.get(cols["method"], "")) if cols["method"] else ""

    retrieval_query = build_retrieval_query(
        row=row,
        cols=cols,
        use_query_prefix=args.use_query_prefix,
    )

    # Step 1: time filter
    candidate_indices = filter_candidates(
        corpus=corpus,
        start_year=pred_start,
        end_year=pred_end,
        target_subject=target_subject,
        use_ticker_filter=args.use_ticker_filter,
    )

    base_out = {
        "query_id": query_id,
        "query_text": query_text,
        "retrieval_query": retrieval_query,
        "target_subject": target_subject,
        "method": method,

        "predicted_start_year": pred_start,
        "predicted_end_year": pred_end,
        "gold_start_year": gold_start,
        "gold_end_year": gold_end,

        "m0_range_iou_with_gold": boundary_overlap_iou(pred_start, pred_end, gold_start, gold_end),
        "m0_is_acceptable": row.get("is_acceptable", ""),
        "m0_overlap_score": row.get("overlap_score", ""),
        "m0_start_boundary_error": row.get("start_boundary_error", ""),
        "m0_end_boundary_error": row.get("end_boundary_error", ""),

        "num_candidates_after_time_filter": len(candidate_indices),
    }

    # 沒有候選 evidence
    if len(candidate_indices) == 0:
        base_out.update({
            "num_retrieved": 0,
            "retrieved_years": "",
            "temporal_coverage_at_k_pred_range": 0.0,
            "temporal_coverage_at_k_gold_range": 0.0,
            "temporal_precision_at_k_pred_range": 0.0,
            "temporal_precision_at_k_gold_range": 0.0,
            "retrieved_evidence_json": "[]",
        })
        return base_out

    # Step 2: query embedding
    query_embedding = model.encode(
        [retrieval_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    query_embedding = normalize_vector(query_embedding)

    # Step 3: similarity scores
    candidate_embeddings = corpus_embeddings[candidate_indices]
    candidate_scores = candidate_embeddings @ query_embedding

    # Step 4: select top-k
    if args.rerank == "coverage":
        selected_indices = select_topk_coverage_aware(
            candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            corpus=corpus,
            top_k=args.top_k,
            per_year_quota=args.per_year_quota,
        )
    else:
        selected_indices = select_topk_by_score(
            candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            top_k=args.top_k,
        )

    # score map
    score_map = {
        int(idx): float(score)
        for idx, score in zip(candidate_indices, candidate_scores)
    }

    retrieved = corpus.iloc[selected_indices].copy()
    retrieved["retrieval_score"] = [score_map[int(idx)] for idx in selected_indices]
    retrieved_years = retrieved["publish_year"].astype(int).tolist()

    # Metrics
    cov_pred = temporal_coverage_at_k(retrieved_years, pred_start, pred_end)
    cov_gold = temporal_coverage_at_k(retrieved_years, gold_start, gold_end)
    prec_pred = temporal_precision_at_k(retrieved_years, pred_start, pred_end)
    prec_gold = temporal_precision_at_k(retrieved_years, gold_start, gold_end)

    evidence_list = []
    for rank, (idx, ev) in enumerate(retrieved.iterrows(), start=1):
        doc_text = normalize_text(ev.get("doc_text", ev.get("text", ""))).replace("\n", " ")
        title = normalize_text(ev.get("title", ""))

        evidence_item = {
            "rank": rank,
            "doc_id": normalize_text(ev.get("doc_id", "")),
            "year": int(ev.get("publish_year")),
            "score": float(ev.get("retrieval_score")),
            "title": title,
            "text_preview": doc_text[:args.preview_chars],
        }
        evidence_list.append(evidence_item)

    base_out.update({
        "num_retrieved": len(retrieved),
        "retrieved_years": years_to_string(retrieved_years),

        "temporal_coverage_at_k_pred_range": cov_pred,
        "temporal_coverage_at_k_gold_range": cov_gold,
        "temporal_precision_at_k_pred_range": prec_pred,
        "temporal_precision_at_k_gold_range": prec_gold,

        "retrieved_evidence_json": json.dumps(evidence_list, ensure_ascii=False),
    })

    # Top-k evidence 展開成欄位，方便人工檢查
    for ev in evidence_list:
        rank = ev["rank"]
        base_out[f"top{rank}_doc_id"] = ev["doc_id"]
        base_out[f"top{rank}_year"] = ev["year"]
        base_out[f"top{rank}_score"] = ev["score"]
        base_out[f"top{rank}_title"] = ev["title"]
        base_out[f"top{rank}_text_preview"] = ev["text_preview"]

    return base_out


# ============================================================
# Summary
# ============================================================

def build_summary(results: pd.DataFrame, args) -> pd.DataFrame:
    rows = []

    def add_group(name: str, sub: pd.DataFrame):
        if len(sub) == 0:
            return

        rows.append({
            "group": name,
            "num_queries": len(sub),

            "avg_num_candidates_after_time_filter": safe_mean(sub["num_candidates_after_time_filter"]),
            "avg_num_retrieved": safe_mean(sub["num_retrieved"]),

            "avg_m0_range_iou_with_gold": safe_mean(sub["m0_range_iou_with_gold"]),

            "avg_temporal_coverage_at_k_pred_range": safe_mean(sub["temporal_coverage_at_k_pred_range"]),
            "avg_temporal_coverage_at_k_gold_range": safe_mean(sub["temporal_coverage_at_k_gold_range"]),

            "avg_temporal_precision_at_k_pred_range": safe_mean(sub["temporal_precision_at_k_pred_range"]),
            "avg_temporal_precision_at_k_gold_range": safe_mean(sub["temporal_precision_at_k_gold_range"]),
        })

    add_group("ALL", results)

    if "method" in results.columns:
        for method, sub in results.groupby("method"):
            if normalize_text(method):
                add_group(f"method={method}", sub)

    # 如果你的 M0 檔案有 type 欄位，這裡會自動分組
    for col in ["query_type_normalized", "query_type_name", "rewrite_type", "predicted_type", "gold_type"]:
        if col in results.columns:
            for value, sub in results.groupby(col):
                if normalize_text(value):
                    add_group(f"{col}={value}", sub)

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--m0-results", required=True, help="Path to bridge_v3_m0_runs.csv")
    parser.add_argument("--corpus-index", required=True, help="Path to corpus_index_qwen.csv")
    parser.add_argument("--corpus-embeddings", required=True, help="Path to corpus_embeddings_qwen.npy")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")

    parser.add_argument("--out", default="m1_qwen_results.csv")
    parser.add_argument("--summary-out", default="m1_qwen_summary.csv")

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--method-filter",
        default=None,
        help="只跑某一個 M0 method，例如 FINAL_prior_guided_qualified_boundary"
    )

    parser.add_argument(
        "--dedupe-by-query",
        action="store_true",
        help="若同一 query_id 有多筆，只保留一筆。若有 overlap_score，保留 overlap_score 最高者。"
    )

    parser.add_argument(
        "--rerank",
        choices=["none", "coverage"],
        default="coverage",
        help="是否使用 coverage-aware reranking。"
    )

    parser.add_argument(
        "--per-year-quota",
        type=int,
        default=2,
        help="coverage-aware reranking 時，每一年最多先選幾筆。"
    )

    parser.add_argument(
        "--use-ticker-filter",
        action="store_true",
        help="若啟用，只保留 title/text/ticker 中包含 target_subject 的候選文章。"
    )

    parser.add_argument(
        "--use-query-prefix",
        action="store_true",
        help="若你 corpus embedding 時用了 passage: prefix，這裡可以加 query: prefix。"
    )

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="每篇 evidence preview 最多保留幾個字元。"
    )

    args = parser.parse_args()

    print("=" * 90)
    print("M1 Qwen Embedding Temporal Retrieval Experiment")
    print("=" * 90)

    print("\n[1] Loading M0 results...")
    m0, cols = load_m0_results(args)
    print(f"M0 rows to run: {len(m0)}")

    print("\n[2] Loading corpus index and embeddings...")
    corpus, corpus_embeddings = load_corpus_and_embeddings(args)

    print("\n[3] Loading Qwen embedding model...")
    print(f"Model: {args.model}")

    try:
        model = SentenceTransformer(args.model, trust_remote_code=True)
    except TypeError:
        model = SentenceTransformer(args.model)

    print("\n[4] Running retrieval...")
    outputs = []

    for i, (_, row) in enumerate(m0.iterrows(), start=1):
        out = retrieve_one_query(
            row=row,
            cols=cols,
            corpus=corpus,
            corpus_embeddings=corpus_embeddings,
            model=model,
            args=args,
        )
        outputs.append(out)

        print(
            f"[{i}/{len(m0)}] "
            f"qid={out.get('query_id')} | "
            f"range=[{out.get('predicted_start_year')},{out.get('predicted_end_year')}] | "
            f"candidates={out.get('num_candidates_after_time_filter')} | "
            f"years={out.get('retrieved_years')} | "
            f"cov_gold={out.get('temporal_coverage_at_k_gold_range'):.3f} | "
            f"prec_gold={out.get('temporal_precision_at_k_gold_range'):.3f}"
        )

    results = pd.DataFrame(outputs)

    # 把 M0 原始欄位補回來，方便後面分析
    # 用 query_id 對齊，但避免欄位重複太多，這裡只補一些常見欄位
    optional_cols_to_copy = [
        "source_id",
        "bridge_id",
        "rewrite_type",
        "query_type_name",
        "query_type_normalized",
        "gold_type",
        "predicted_type",
        "temporal_expression",
        "anchor_event",
        "anchor_direction",
        "anchor_event_year",
    ]

    query_id_col = cols["query_id"]
    for c in optional_cols_to_copy:
        if c in m0.columns and c not in results.columns:
            mapping = dict(zip(m0[query_id_col].astype(str), m0[c]))
            results[c] = results["query_id"].astype(str).map(mapping)

    print("\n[5] Saving M1 results...")
    results.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"M1 results saved to: {args.out}")

    print("\n[6] Building summary...")
    summary = build_summary(results, args)
    summary.to_csv(args.summary_out, index=False, encoding="utf-8-sig")
    print(f"M1 summary saved to: {args.summary_out}")

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()