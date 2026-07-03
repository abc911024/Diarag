# -*- coding: utf-8 -*-
import pandas as pd

M1_PATH = "m1_qwen_results.csv"
M0_PATH = "bridge_v3_m0_runs.csv"
OUT_PATH = "m1_qwen_results_with_source.csv"

m1 = pd.read_csv(M1_PATH)
m0 = pd.read_csv(M0_PATH)

# 清掉欄位名稱前後空白
m1.columns = [str(c).strip() for c in m1.columns]
m0.columns = [str(c).strip() for c in m0.columns]

print("M1 columns:")
print(m1.columns.tolist())
print("\nM0 columns:")
print(m0.columns.tolist())

# 確認 query_id
if "query_id" not in m1.columns:
    raise ValueError("M1 missing query_id")

if "query_id" not in m0.columns:
    raise ValueError("M0 missing query_id")

# 找 source_id 欄位
source_candidates = ["source_id", "original_id", "adqab_id", "id"]
source_col = None

for c in source_candidates:
    if c in m0.columns:
        source_col = c
        break

if source_col is None:
    raise ValueError(
        "Cannot find source_id-like column in M0. "
        "Please check M0 columns printed above."
    )

print(f"\nUsing M0 source column: {source_col}")

# 如果有 method，就優先使用 FINAL_prior_guided_qualified_boundary
if "method" in m0.columns:
    before = len(m0)
    filtered = m0[m0["method"].astype(str) == "FINAL_prior_guided_qualified_boundary"].copy()
    if len(filtered) > 0:
        m0 = filtered
        print(f"Method filtered: {len(m0)} / {before}")
    else:
        print("Warning: method filter produced 0 rows, using all M0 rows instead.")

# 準備 merge mapping
cols = ["query_id", source_col]

if "bridge_id" in m0.columns:
    cols.append("bridge_id")

m0_map = m0[cols].drop_duplicates(subset=["query_id"]).copy()
m0_map = m0_map.rename(columns={source_col: "source_id_from_m0"})

# 如果 M1 原本有 source_id，先保留，但避免 merge 後衝突
if "source_id" in m1.columns:
    m1 = m1.rename(columns={"source_id": "source_id_from_m1"})

merged = m1.merge(m0_map, on="query_id", how="left")

# 決定最後 source_id
if "source_id_from_m1" in merged.columns:
    merged["source_id"] = merged["source_id_from_m1"].fillna(merged["source_id_from_m0"])
else:
    merged["source_id"] = merged["source_id_from_m0"]

missing = merged["source_id"].isna().sum()
print(f"\nTotal rows: {len(merged)}")
print(f"Missing source_id: {missing}")

if missing > 0:
    print("\nRows missing source_id:")
    print(merged.loc[merged["source_id"].isna(), ["query_id"]].head(20))

# 欄位順序
front = ["query_id", "source_id"]

if "bridge_id" in merged.columns:
    front.append("bridge_id")

# 移除暫存欄位
drop_cols = [c for c in ["source_id_from_m1", "source_id_from_m0"] if c in merged.columns]
merged = merged.drop(columns=drop_cols)

other_cols = [c for c in merged.columns if c not in front]
merged = merged[front + other_cols]

merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"\nSaved to: {OUT_PATH}")

print("\nPreview:")
print(merged[["query_id", "source_id"]].head())