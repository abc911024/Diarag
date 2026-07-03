# -*- coding: utf-8 -*-
import pandas as pd

M2_PATH = "m2_structured_evidence.csv"
M0_PATH = "bridge_v3_m0_runs.csv"
OUT_PATH = "m2_structured_evidence_with_source.csv"

# 讀取 M2 與 M0
m2 = pd.read_csv(M2_PATH)
m0 = pd.read_csv(M0_PATH)

# 只保留主要使用的 M0 method
if "method" in m0.columns:
    m0 = m0[m0["method"] == "FINAL_prior_guided_qualified_boundary"].copy()

# 檢查必要欄位
required_m2 = ["query_id"]
required_m0 = ["query_id", "source_id"]

for col in required_m2:
    if col not in m2.columns:
        raise ValueError(f"M2 missing column: {col}")

for col in required_m0:
    if col not in m0.columns:
        raise ValueError(f"M0 missing column: {col}")

# 避免重複欄位
cols_to_merge = ["query_id", "source_id"]

if "bridge_id" in m0.columns:
    cols_to_merge.append("bridge_id")

m0_map = m0[cols_to_merge].drop_duplicates(subset=["query_id"])

# merge 回 M2
merged = m2.merge(m0_map, on="query_id", how="left")

# 檢查有多少筆沒有對到 source_id
missing = merged["source_id"].isna().sum()
print(f"Total rows: {len(merged)}")
print(f"Missing source_id: {missing}")

if missing > 0:
    print("Warning: Some rows did not match source_id. Please check query_id consistency.")

# 調整欄位順序：把 source_id 放前面
front_cols = ["query_id", "source_id"]

if "bridge_id" in merged.columns:
    front_cols.append("bridge_id")

other_cols = [c for c in merged.columns if c not in front_cols]
merged = merged[front_cols + other_cols]

merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"Saved to: {OUT_PATH}")