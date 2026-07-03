# -*- coding: utf-8 -*-
"""
Build corpus embeddings for M1 Temporal Retrieval.

Input:
    corpus_with_time.jsonl

Output:
    corpus_index.csv
    corpus_embeddings.npy

Example:
    python build_corpus_embeddings.py ^
      --corpus corpus_with_time.jsonl ^
      --model sentence-transformers/all-MiniLM-L6-v2 ^
      --out-index corpus_index.csv ^
      --out-embeddings corpus_embeddings.npy

If you want a stronger embedding model:
    --model BAAI/bge-small-en-v1.5
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def parse_year(x: Any) -> Optional[int]:
    if x is None:
        return None

    try:
        return int(float(x))
    except Exception:
        pass

    m = re.search(r"(19|20)\d{2}", str(x))
    if m:
        return int(m.group(0))

    return None


def load_jsonl(path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Warning] Skip invalid JSON at line {line_id + 1}")
                continue

            obj["_line_id"] = line_id
            rows.append(obj)

    if not rows:
        raise ValueError("No valid JSONL records loaded.")

    return pd.DataFrame(rows)


def pick_column(df: pd.DataFrame, candidates: List[str], default: str = "") -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return default


def build_doc_text(row: pd.Series, title_col: str, text_col: str) -> str:
    title = normalize_text(row.get(title_col, "")) if title_col else ""
    text = normalize_text(row.get(text_col, "")) if text_col else ""

    if title and text:
        return f"{title}\n{text}"
    if title:
        return title
    return text


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to corpus_with_time.jsonl"
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name"
    )
    parser.add_argument(
        "--out-index",
        default="corpus_index.csv",
        help="Output CSV storing metadata and document text"
    )
    parser.add_argument(
        "--out-embeddings",
        default="corpus_embeddings.npy",
        help="Output numpy file storing embeddings"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Maximum characters kept for embedding per document"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize embeddings for cosine similarity"
    )

    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        raise FileNotFoundError(f"Corpus file not found: {args.corpus}")

    print(f"Loading corpus: {args.corpus}")
    df = load_jsonl(args.corpus)

    print(f"Loaded records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # 自動偵測常見欄位名稱
    year_col = pick_column(df, ["publish_year", "year", "date_year", "pub_year"])
    title_col = pick_column(df, ["title", "headline", "article_title"])
    text_col = pick_column(df, ["text", "content", "body", "article_text", "chunk_text"])
    ticker_col = pick_column(df, ["ticker", "symbol", "stock", "company_ticker"])
    doc_id_col = pick_column(df, ["doc_id", "id", "article_id", "news_id"])

    if not year_col:
        raise ValueError(
            "Cannot find year column. Expected one of: "
            "publish_year, year, date_year, pub_year"
        )

    if not text_col and not title_col:
        raise ValueError(
            "Cannot find text/title column. Expected text/content/body/chunk_text or title/headline."
        )

    print("\nDetected columns:")
    print(f"year_col   = {year_col}")
    print(f"title_col  = {title_col if title_col else '(none)'}")
    print(f"text_col   = {text_col if text_col else '(none)'}")
    print(f"ticker_col = {ticker_col if ticker_col else '(none)'}")
    print(f"doc_id_col = {doc_id_col if doc_id_col else '(none)'}")

    # 建立標準 index dataframe
    index_df = pd.DataFrame()
    index_df["row_id"] = range(len(df))
    index_df["line_id"] = df["_line_id"]

    if doc_id_col:
        index_df["doc_id"] = df[doc_id_col].apply(normalize_text)
    else:
        index_df["doc_id"] = [f"doc_{i:06d}" for i in range(len(df))]

    if ticker_col:
        index_df["ticker"] = df[ticker_col].apply(normalize_text)
    else:
        index_df["ticker"] = ""

    index_df["publish_year"] = df[year_col].apply(parse_year)

    if title_col:
        index_df["title"] = df[title_col].apply(normalize_text)
    else:
        index_df["title"] = ""

    if text_col:
        index_df["text"] = df[text_col].apply(normalize_text)
    else:
        index_df["text"] = ""

    # 移除沒有年份或沒有文字的資料
    index_df["doc_text"] = df.apply(
        lambda row: build_doc_text(row, title_col, text_col),
        axis=1
    )

    index_df["doc_text"] = index_df["doc_text"].apply(
        lambda x: normalize_text(x)[: args.max_chars]
    )

    before = len(index_df)
    index_df = index_df.dropna(subset=["publish_year"]).copy()
    index_df = index_df[index_df["doc_text"].str.len() > 0].copy()
    index_df["publish_year"] = index_df["publish_year"].astype(int)

    after = len(index_df)
    print(f"\nValid records after cleaning: {after} / {before}")

    if after == 0:
        raise ValueError("No valid records after cleaning.")

    print("\nCorpus year range:")
    print(index_df["publish_year"].min(), "to", index_df["publish_year"].max())

    print(f"\nLoading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    texts = index_df["doc_text"].tolist()

    print("\nEncoding corpus embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize,
    )

    print(f"Embeddings shape: {embeddings.shape}")

    # 儲存
    index_df.to_csv(args.out_index, index=False, encoding="utf-8-sig")
    np.save(args.out_embeddings, embeddings)

    print("\nDone.")
    print(f"Saved index to: {args.out_index}")
    print(f"Saved embeddings to: {args.out_embeddings}")


if __name__ == "__main__":
    main()
