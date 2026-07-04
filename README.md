# DiaRAG

面向時序推理的檢索增強生成(Structured Diachronic Reasoning for Retrieval-Augmented Generation)。

一般的 RAG 只看語意相似度做檢索,遇到**時序型問題(diachronic queries)**——也就是問「趨勢」「演變」「隨時間如何變化」的問題就會失效,因為答案必須綜合橫跨多個時間點的證據,而不是單一段落。DiaRAG 要補上既有時序 RAG 研究留下的兩個缺口:當查詢沒有明講時間範圍時,如何推斷出正確的時間範圍;以及如何真的跨時期做推理,而不是把這件事丟給生成模型自己處理。

本 repo 對應的最終書面報告已附在專案討論紀錄中,以下的模組說明、資料集與實驗數字皆整理自該報告。

## 四個模組(Pipeline)

論文把整個流程拆成四個模組:

| 模組 | 名稱 | 做什麼 | 對應程式碼 |
|---|---|---|---|
| **M0** | 時間範圍推斷 | 判斷查詢的時序類型(explicit 明確 / relative 相對 / event-anchored 事件錨定 / latent 隱性),並推斷出一個 `[start_year, end_year]` 目前實作的是:直接生成一組候選時間窗並評分挑選。 | `scripts/run_m0_experiment_gptoss.py` |
| **M1** | 時序檢索 | 用 Qwen3-Embedding-0.6B 對語料做稠密向量檢索,依 M0 推斷出的年份範圍篩選候選段落,並用 coverage-aware 重排序機制,限制同一年份被選中的段落數,以增加時間分佈的多樣性。 | `build_qwen_embeddings.py`(建索引)、`scripts/run_m1_qwen_experiment.py` |
| **M2** | 時序證據結構化 | 把 M1 檢索到的段落依發布年份分組,用 `gpt-oss-20b`(透過 Ollama)幫每一年生成摘要、時序訊號(positive/negative/stable/mixed/volatile/not_directly_related/no_evidence)與證據充足度判斷,組成逐年的時序證據結構。不會重新檢索。 | `scripts/run_m2_gptoss_structure.py` |
| **M3** | 結構化時序推理 | 把 M2 產出的逐年證據結構、原始四選一選項一起丟給 `gpt-oss-20b`,選出最終答案,並回傳支持年份、證據充足度與簡短理由。 | `scripts/run_m3_gptoss_mcqa.py` |

四個模組之間用 CSV/JSON 交接資料:`M0 結果 → M1 檢索結果 → M2 結構化證據 → M3 最終答案`。

另外還有對應最終報告 Baselines 一節與內部消融實驗(ablation)的腳本,都是「换一種 M1 檢索方式,再接一種 M3 答案選擇方式」的組合:
| 名稱 | 說明 | M1(檢索) | M3(答案選擇) |
|---|---|---|---|
| Naive RAG | 不做時序過濾,直接對全語料做語意檢索,做為「完全不管時間」的下限基準。 | `scripts/run_m1_naive_qwen_baseline.py` | `scripts/run_m3_naive_rag_gptoss.py`(不經過 M2 結構化) |
| TA-RAG 風格 | 用一組錨定在不同年份的「假設性時序查詢」平均向量做檢索,近似既有研究 TA-RAG 的核心做法(並非完整重現)。 | `scripts/run_m1_tarag_style_qwen.py` | `scripts/run_m3_naive_rag_gptoss.py`(不經過 M2 結構化) |
| Ours w/o M0 | 消融實驗:直接用 gold 時間範圍做檢索,跳過 M0 的時間範圍推斷,藉此單獨評估 M1-M3。 | `scripts/run_m1_gold_range_qwen.py` | `scripts/run_m3_gptoss_wo_m2.py`(或接 M2 走完整流程) |
| Ours w/o M2 | 消融實驗:M1 檢索到的原始段落直接丟給 M3,不經過 M2 的逐年結構化。 | `scripts/run_m1_qwen_experiment.py` | `scripts/run_m3_gptoss_wo_m2.py` |

## 專案結構

```
.
├── scripts/
│   ├── run_m0_experiment_gptoss.py    # M0:時間範圍推斷(LLM 版,經 Ollama 呼叫 gpt-oss-20b)
│   ├── run_m1_qwen_experiment.py      # M1:Qwen3-Embedding 稠密檢索 + coverage-aware 重排序
│   ├── run_m2_gptoss_structure.py     # M2:把 M1 證據結構化成逐年時序證據
│   ├── run_m3_gptoss_mcqa.py          # M3:依 M2 結構化證據選出最終 MCQA 答案
│   ├── run_m1_naive_qwen_baseline.py  # 對照組:Naive RAG 檢索(全語料、不做時序過濾)
│   ├── run_m1_tarag_style_qwen.py     # 對照組:TA-RAG 風格檢索(錨定年份假設查詢平均向量)
│   ├── run_m3_naive_rag_gptoss.py     # 對照組:Naive RAG / TA-RAG 風格的最終答案選擇(不經 M2)
│   ├── run_m1_gold_range_qwen.py      # 消融:Ours w/o M0(用 gold 範圍檢索,跳過 M0)
│   └── run_m3_gptoss_wo_m2.py         # 消融:Ours w/o M2(M1 原始段落直接進 M3,不經結構化)
├── m0-graph-lite/                     # M0 輔助特徵層(Postgres 時序結構)+ sql/
├── Dataest-Bridge/                    # ADQAB-Implicit 改寫流程(bridge_pipeline_v3.1.py)
├── exp_bridge_v3_m0/                  # 在 Bridge v3 改寫資料集上單獨評測 M0 的實驗腳本
├── exp_ect_qa/                        # ECT-QA 可行性實驗(9 步驟腳本)
├── postgres/init/                     # Postgres schema(qa、content、runtime、m0_graph)
├── docker-compose.yml                 # 本地開發用的 postgres+pgvector 與 adminer
├── build_qwen_embeddings.py           # 建立 M1 用的語料向量索引
├── requirements.txt                   # 完整環境凍結(conda freeze)

```

`scripts/` 資料夾原本有 26 個檔案(M0 的 Postgres 規則式實作、DB seed/export 工具、smoke test 等),整理後留下真正拿去產生最終報告數字、以及對照組/消融實驗用的 9 個檔案:M0/M1/M2/M3 主流程各一個,加上 Naive RAG、TA-RAG 風格、Ours w/o M0、Ours w/o M2 四組對照/消融實驗腳本。這些都是單檔可獨立執行(不依賴 Postgres,只需要 pandas/numpy/requests,M1 相關腳本額外需要 `sentence-transformers`),彼此靠 CSV 檔案交接,詳見下方「執行 pipeline」。其餘刪除的檔案(Postgres 規則式 M0、DB 匯入匯出工具、smoke test)仍保留在 git 歷史紀錄中,需要的話可以從 commit log 找回來。

`dataset/` 與 `outputs/` 不在這個 repo 裡(見下方「資料」一節)。

## 資料

原始語料與實驗輸出檔案刻意不進版本控制(`.gitignore`:`dataset/`、`outputs/`、`exp_ect_qa/processed/`、`*.jsonl`、`*.csv`)——光是原始語料就有 ~370MB,而且程式碼本身完全不需要這些檔案就能閱讀、審查。這同時也是當初「檔案太大推不上去」的根本修正:原本的 `.gitignore` 排除的是一個不存在的 `data/` 資料夾,而真正的大型資料夾 `dataset/` 從未被排除。

要在本機真的跑起來,需要從團隊 Google Drive(`實作/ds_final/`)把以下檔案複製回專案根目錄:

- `dataset/corpus_with_time.jsonl`、`dataset/corpus_with_time.csv`、`dataset/C_final.jsonl`、`dataset/dqabench_MCQA.json` —— M1/M3 使用的語料與 MCQA 題庫
- `Dataest-Bridge/dqabench_MCQA.json` —— 同一份題庫,改寫流程用
- `m0_final/bridge_rewrite_dataset_v3.csv` —— ADQAB-Implicit 的改寫結果(204 筆真實題目,`exp_bridge_v3_m0` 需要這份檔案才能跑)
- `exp_ect_qa/processed/*` —— ECT-QA 匯入的中間檔案(只有要重跑 `exp_ect_qa/scripts/` 時才需要)
- 其餘任何 `outputs/` 資料夾,如果想直接看過去實驗的產出而不是重新產生一次

這個 repo 裡沒有任何 script 能從零生成 `corpus_with_time.jsonl`,但它的源頭其實找得到:`C_final.jsonl`、`dqabench_MCQA.json`、`C_real_metadata.csv`、`C_synth_metadata.csv` 這幾個檔名都跟 TA-RAG 論文(本專案 baseline 之一,見上方比較表)公開釋出的 repo [`kwunhang/TA-RAG`](https://github.com/kwunhang/TA-RAG) 裡 `DQABench/` 資料夾完全一致。該 repo 的 `construct_corpus.ipynb` 說明了原始語料組成:23,737 篇來自 FNSPID 資料集的真實財經新聞(25 檔股票)+ 3,300 篇合成新聞,共 27,037 篇文件,`id` 對 `C_real_metadata.csv`/`C_synth_metadata.csv` 的 `global_id` 即可接上每篇文件的真實發布日期。也就是說,即使拿不到團隊的 Drive 備份,理論上也能從這個公開 repo 重建出 `corpus_with_time`(`doc_id` + `text` + `publish_year`),只是要自己額外做這個 join、且原始 repo 的 `C_final.jsonl` 是用 Git LFS 存的 121MB 檔案。實測過(見下方「現況」)這個 join 邏輯與 `build_qwen_embeddings.py` 的欄位偵測完全相容。

## 環境建置

```bash
cp .env.example .env        # 若不用預設帳密,填入實際的 Postgres 帳密
docker compose up -d        # 啟動 postgres-pgvector(5432)與 adminer(8080)
pip install -r requirements.txt
```

`docker-compose.yml` 掛載了 `postgres/init/`,第一次啟動時會自動建立 schema(`qa`、`content`、`runtime`、`m0_graph`)。

> 補充:若沒有 Docker,原生安裝的 PostgreSQL 16 + `postgresql-16-pgvector`(可透過 apt 直接安裝)也已驗證過整套 schema 可以無錯誤套用。

## 執行 pipeline

需要先有 Ollama(或其他 OpenAI-compatible 的本地推論服務)跑著 `gpt-oss-20b`,以及能下載 `Qwen/Qwen3-Embedding-0.6B` 的網路環境。以下每一步的輸出都是下一步的輸入。

**M0:時間範圍推斷**

```bash
python scripts/run_m0_experiment_gptoss.py \
  --dataset bridge_rewrite_dataset_v3.csv \
  --corpus corpus_with_time.csv \
  --model gpt-oss:20b \
  --backend ollama \
  --out bridge_v3_m0_runs.csv
```

若本地服務是 OpenAI-compatible API(例如非 Ollama 的本地推論伺服器),把 `--backend` 換成 `openai_compatible` 並補上 `--base-url`、`--api-key` 即可,其餘 M1/M2/M3 腳本也都支援同樣的兩種 backend。

**建立語料向量索引(M1 之前只需做一次)**

```bash
python build_qwen_embeddings.py \
  --corpus corpus_with_time.jsonl \
  --model Qwen/Qwen3-Embedding-0.6B \
  --out-index corpus_index_qwen.csv \
  --out-embeddings corpus_embeddings_qwen.npy
```

**M1:時序檢索**

```bash
python scripts/run_m1_qwen_experiment.py \
  --m0-results bridge_v3_m0_runs.csv \
  --corpus-index corpus_index_qwen.csv \
  --corpus-embeddings corpus_embeddings_qwen.npy \
  --model Qwen/Qwen3-Embedding-0.6B \
  --out m1_qwen_results.csv \
  --summary-out m1_qwen_summary.csv \
  --top-k 10 \
  --method-filter FINAL_prior_guided_qualified_boundary
```

**M2:時序證據結構化**

```bash
python scripts/run_m2_gptoss_structure.py \
  --m1-results m1_qwen_results.csv \
  --backend ollama --model gpt-oss:20b \
  --out m2_structured_evidence.csv \
  --summary-out m2_summary.csv
```

**M3:最終 MCQA 答案選擇**

```bash
python scripts/run_m3_gptoss_mcqa.py \
  --m2-results m2_structured_evidence.csv \
  --mcqa-json dqabench_MCQA.json \
  --backend ollama --model gpt-oss:20b \
  --out m3_answers.csv \
  --summary-out m3_summary.csv
```

每支 script 都支援 `--limit 5` 之類的參數,建議先跑 5 筆確認流程正常,再跑全部 204 筆(全部跑完可能要數小時,取決於本地 GPU 與模型速度)。

**對照組與消融實驗(baseline / ablation)**

```bash
# Naive RAG:全語料檢索,不做時序過濾
python scripts/run_m1_naive_qwen_baseline.py \
  --queries bridge_v3_m0_runs.csv \
  --corpus-index corpus_index_qwen.csv \
  --corpus-embeddings corpus_embeddings_qwen.npy \
  --out m1_naive_qwen_results_with_source.csv \
  --summary-out m1_naive_qwen_summary.csv

# TA-RAG 風格檢索(用 gold 或 predicted range 都可以,--range-source 切換)
python scripts/run_m1_tarag_style_qwen.py \
  --queries bridge_v3_m0_runs.csv \
  --corpus-index corpus_index_qwen.csv \
  --corpus-embeddings corpus_embeddings_qwen.npy \
  --range-source gold \
  --out m1_tarag_style_gold_qwen_results_with_source.csv \
  --summary-out m1_tarag_style_gold_qwen_summary.csv

# 上面兩種檢索結果都可以接這支做最終答案(不經過 M2 結構化)
python scripts/run_m3_naive_rag_gptoss.py \
  --m1-results m1_naive_qwen_results_with_source.csv \
  --mcqa-json dqabench_MCQA.json \
  --backend ollama --model gpt-oss:20b \
  --out m3_naive_rag_answers.csv \
  --summary-out m3_naive_rag_summary.csv

# Ours w/o M0:用 gold 範圍取代 M0 預測範圍做檢索
python scripts/run_m1_gold_range_qwen.py \
  --queries bridge_v3_m0_runs.csv \
  --corpus-index corpus_index_qwen.csv \
  --corpus-embeddings corpus_embeddings_qwen.npy \
  --out m1_gold_range_qwen_results_with_source.csv \
  --summary-out m1_gold_range_qwen_summary.csv

# Ours w/o M2:M1 原始段落直接進 M3,跳過逐年結構化
python scripts/run_m3_gptoss_wo_m2.py \
  --m1-results m1_qwen_results.csv \
  --mcqa-json dqabench_MCQA.json \
  --backend ollama --model gpt-oss:20b \
  --out m3_wo_m2_answers.csv \
  --summary-out m3_wo_m2_summary.csv
```

## 論文最終結果(節錄自最終書面報告)

以下數字全部來自 ADQAB-Implicit(204 筆改寫題目,event-anchored / latent / relative-time 各 68 筆),摘自最終報告的實驗結果章節,不是本 repo 自己重新量測的:

**M0 時間範圍推斷**(整體,兩種評分策略):

| 評分策略 | Acceptable Match Accuracy | Temporal IoU | Boundary MAE(年) |
|---|---|---|---|
| Stable temporal-prior scorer | 0.2157 | 0.5872 | 1.9534 |
| Prior-guided fusion scorer | 0.1814 | 0.5734 | 2.0098 |

依查詢類型拆分(stable temporal-prior scorer):event-anchored 準確率最高(0.35,IoU 0.72),latent 與 relative-time 都偏低(各 0.147,IoU 0.52)。最主要的邊界誤差型態是「過度延伸(over-extension)」,佔 176 例,其次是「延伸不足」103 例。

**M1 時序檢索**(Coverage@10 / Precision@10,對 gold 時間範圍評估):

| 類型 | 題數 | Coverage@10 | Precision@10 |
|---|---|---|---|
| 全部 | 204 | 0.6342 | 0.6534 |
| Event-anchored | 68 | 0.7638 | 0.8118 |
| Latent | 68 | 0.5493 | 0.5779 |
| Relative | 68 | 0.5895 | 0.5706 |

**M2 時序證據結構化**:整體平均每筆查詢的時序結構涵蓋 7.35 年,證據年份覆蓋率 0.8041;event-anchored 覆蓋率最高(0.9252),latent 最低(0.7224)。

**M3 最終答案選擇(MCQA Accuracy)**:整體 0.5147(105/204 正確);event-anchored 0.5735,latent 與 relative-time 皆為 0.4853。

四個模組的結果呈一致的趨勢:**event-anchored 查詢全程表現最好,latent 與 relative-time 查詢全程較弱**——時間線索越明確,後續檢索、結構化與答案選擇的品質就越好。

## 現況與已知限制

- 最終報告本身就明講:受限於時間與資源,目前的 M0 實作是簡化版的「時間範圍提案模組」(直接生成候選範圍並評分),不是完整設計中的四類分類器 + 逐類推斷 + 迭代驗證。完整版留待未來工作。
- 沒有向量索引(FAISS 等),M1 是對預先算好的向量做暴力法 cosine similarity。
- 已用真實資料驗證過 M0 輸出與 MCQA 題庫的 `source_id` 對接、以及 `corpus_with_time`(源自 TA-RAG 公開語料)與 `build_qwen_embeddings.py` 欄位偵測邏輯的相容性;尚未驗證的只剩需要連上 HuggingFace / Ollama 才能執行的實際向量化與 LLM 推理步驟。
