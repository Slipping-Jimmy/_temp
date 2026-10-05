# Multimodal Playground Load Test

這是一個可單獨複製到內網執行的端到端負載測試資料夾。它測量以下完整流程：

1. 建立 conversation。
2. 上傳一張圖片。
3. 發出 multimodal chat SSE 請求。
4. 讀取第一段模型文字及完整回覆。
5. 保存逐筆結果、Locust 統計，以及可選的 vLLM/GPU telemetry。

目前的可攜版本已包含 TONO single 與 mixed 資料，共 4,672 張圖片；來源、checksum、驗證方式、授權及引用資訊記錄於 [`data/DATASET_NOTICE.md`](data/DATASET_NOTICE.md)。TONO 授權為 CC BY-NC 4.0，使用前仍須確認本次內部用途符合非商業限制。

## 資料夾內容

```text
multimodal_load_test/
├── .env.example
├── README.md
├── requirements.txt
├── questions.json
├── images_manifest.csv
├── data/
│   ├── DATASET_NOTICE.md
│   └── tono/
│       ├── release/          # 4,569 single-violation images
│       └── release_mixed/    # 103 mixed-violation images
├── wheelhouse/
│   └── cp39-manylinux2014-x86_64/  # 完整離線安裝 wheels
├── build_manifest.py
├── locustfile.py
├── collect_metrics.py
├── run_test.sh
├── analyze_results.py
├── loadtest_lib.py
└── tests/
```

`.env`、圖片、token 與測試結果預設都不會被 Git 追蹤；但圖片實體已放在此可攜資料夾中，直接複製整個資料夾即可帶入內網。

## 1. 安裝

建議在獨立 load-generator 主機執行，不要與 H200/vLLM 共用 CPU、RAM 或網路資源。

若內網主機與準備時確認的環境相同（CPython 3.9、Linux x86-64、glibc 2.17 以上），直接使用隨附 wheelhouse，不需連接套件站，也不需編譯 gevent：

```bash
cd multimodal_load_test
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install \
  --no-index \
  --find-links ./wheelhouse/cp39-manylinux2014-x86_64 \
  "pip==24.0"
python3 -m pip install \
  --no-index \
  --find-links ./wheelhouse/cp39-manylinux2014-x86_64 \
  -r requirements.txt
```

可先用以下命令確認環境：

```bash
python3 --version
uname -m
ldd --version | head -n 1
```

如果不是 Python 3.9 x86-64，才使用可連套件站的安裝方式：

```bash
cd multimodal_load_test
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

支援 Python 3.10 以上。正式執行前可驗證工具：

```bash
python3 -m unittest discover -s tests -v
```

## 2. 資料集與 manifest

目前資料已放在：

```text
data/tono/
```

對應 manifest 已建立為：

```text
images_manifest.csv
```

因此使用隨附資料時不需再下載、解壓或建立 manifest。也可以改用任意其他資料集位置，之後在 `.env` 設定絕對路徑並重新建立 manifest。

重新建立 manifest：

```bash
python3 build_manifest.py ./data/tono --output ./images_manifest.csv
```

manifest 會保存相對路徑、類別、single/mixed、解析度、格式、大小、SHA-256 與圖片是否可解碼。壓測只會使用 `valid=1` 且實際存在的檔案。

若 manifest 不存在，`run_test.sh` 會在第一次執行時自動建立。

## 3. 設定

```bash
cp .env.example .env
chmod 600 .env
```

至少修改：

```dotenv
PLAYGROUND_BASE_URL=http://內網主機:port
PLAYGROUND_TOKEN=新的短效測試token
DATASET_DIR=./data/tono
```

不要沿用曾貼在聊天、郵件或 issue 中的 token，也不要把 `.env` 放進結果資料夾。

重要參數：

| 參數 | 用途 |
|---|---|
| `USERS_STEPS` | 依序執行的 Locust concurrency，例如 `"1 2 4 8 16 32"` |
| `SPAWN_RATE` | 每秒啟動的 users |
| `RUN_TIME` | 每個級別持續時間，例如 `5m` |
| `PROMPT_PROFILE` | `short` 或 `mixed` |
| `CONVERSATION_MODE` | `per_user`（預設，每個虛擬 user 重用 conversation）或 `per_request` |
| `FIXED_CONVERSATION_ID` | 可選；所有請求使用一個已存在的 conversation ID |
| `REQUEST_TIMEOUT_SECONDS` | 每個 HTTP/SSE request 的固定 timeout |
| `REQUIRE_DONE_MARKER` | 預設 `1`；SSE 必須收到 `[DONE]`、`[END]` 或 recognized finish event |
| `CAPTURE_RESPONSE` | 是否在逐筆 CSV 保存截短後的模型回覆 |
| `VLLM_METRICS_URL` | 可選的 vLLM `/metrics` URL |
| `COLLECT_LOCAL_GPU` | load test 若直接跑在 H200 主機才設為 `1` |

建議正式測試前確認 H200 環境沒有其他流量，並先記錄模型、vLLM、driver、CUDA 和容器版本。這些設定不需調整，但解讀結果時需要知道。

## 4. Smoke test

Smoke 模式會覆寫成 1 user、1 分鐘：

```bash
./run_test.sh ./.env --smoke
```

完成後檢查：

- conversation、attachment、SSE 都是成功狀態。
- `ttft_ms` 有值。
- `response_preview` 是真正模型內容，不是 status event。
- `completion_marker=1`，且回覆不包含 `[ERROR]`。

如果 SSE 並非標準 `data:` 格式，而是每行直接傳文字，可以在 `.env` 加上：

```dotenv
ALLOW_RAW_STREAM=1
```

## 5. 正式執行

```bash
./run_test.sh ./.env
```

每一級 users 都會啟動獨立 Locust process，避免前一級的 client 狀態污染下一級。預設沒有 think time，每個 user 完成一筆端到端 transaction 後立即送下一筆，適合尋找最大負載。

預設 `CONVERSATION_MODE=per_user`：每個 Locust user 第一次請求建立 conversation，後續重用它；每筆 transaction 仍產生唯一的 `chat_id`。這可避免 conversation 建立／儲存層掩蓋模型負載。若要刻意測完整 conversation 建立路徑，才改用 `per_request`。

`short` profile 固定使用短二元判斷，適合測容量上限；完成後建議再將 `.env` 改成：

```dotenv
PROMPT_PROFILE=mixed
```

用接近臨場不確定性的題目組合重跑一次。

不要一開始就直接壓到很高。先跑 `1 2 4 8 16 32`，查看結果後再以另一個 `RUN_ID` 細測邊界，例如 `24 28 32 36 40`。

## 6. Telemetry

若 load generator 能連到 vLLM Prometheus endpoint：

```dotenv
VLLM_METRICS_URL=http://vllm-host:8000/metrics
```

runner 會同步保存 running/waiting requests、KV cache、prompt/generation tokens、success、prefix cache 與 preemption 指標。

若測試就是在 H200 主機執行，可另外設定：

```dotenv
COLLECT_LOCAL_GPU=1
```

否則 GPU 指標應在 H200 主機另外保存；不要把 load generator 移到 H200 只為了收 `nvidia-smi`，那會改變被測環境。最簡單的遠端方式是在 H200 主機執行：

```bash
python3 collect_metrics.py \
  --output-dir /tmp/multimodal-server-monitor \
  --vllm-url http://127.0.0.1:8000/metrics \
  --gpu
```

這種獨立收集沒有自動 stage 名稱，應保留開始／結束時間，帶回後與 client UTC 時間對齊。若可從 load generator 直接讀 `/metrics`，優先使用 runner 的整合方式。

## 7. 結果內容

每次執行會產生：

```text
results/<run_id>/
├── run_config.txt             # 不含 token
├── images_manifest.csv        # 測試使用的資料快照
├── questions.json             # 題庫快照
├── monitor/
│   ├── vllm_metrics.csv
│   └── gpu_metrics.csv
├── stage_001_1u/
│   ├── transactions_<pid>.csv
│   ├── locust_stats.csv
│   ├── locust_stats_history.csv
│   ├── locust_failures.csv
│   ├── locust_report.html
│   └── locust.log
├── summary.csv
└── report.md
```

逐筆 transaction 包含：

- 圖片類別、解析度與大小。
- prompt ID/profile。
- conversation ID、唯一 chat ID、attachment ID，以及 API 回傳的 request/correlation trace ID。
- conversation、upload、SSE status 與 latency。
- SSE headers time、TTFT、完整 SSE time、完整 transaction time。
- content event 數、回覆字數、completion marker。
- 明確的 error type 與截短錯誤內容。
- 最多五個截短後的 SSE 原始事件，供 smoke test 診斷格式；可用 `CAPTURE_RESPONSE=0` 關閉。

`sse_event_count` 不是 token 數。真正 token throughput 只從 vLLM metrics 計算。

## 8. 將結果帶回本地分析

整個 `results/<run_id>/` 複製回本地，不需要圖片原檔，也不需要 token：

```bash
python3 analyze_results.py /path/to/results/<run_id>
```

它會重新產生：

- `summary.csv`：每級負載的吞吐、錯誤率與 p50/p95/p99。
- `report.md`：暫定通過級別、峰值吞吐、錯誤分布及 telemetry 摘要。

暫定通過門檻來自 `.env`：

```dotenv
MAX_ERROR_RATE_PERCENT=1
MAX_P95_TTFT_MS=3000
MAX_P95_TOTAL_MS=10000
```

改動門檻後可以在本地重新分析，不必重跑壓測。最終建議容量仍應在候選級別做至少 30 分鐘確認，正式營運點則保留約 20–30% headroom。

## 注意事項

- 預設每個虛擬 user 建立並重用一個 conversation，但每筆 transaction 仍上傳新 attachment 並使用唯一 chat ID。請確認 attachment 的 TTL 或清理方式。
- 大量重複使用同一張圖片可能觸發 multimodal cache。manifest 快照可用來檢查圖片分布。
- load generator CPU 建議低於 70%；若 Locust 出現 CPU warning，結果代表 client 上限，不是 H200 上限。
- `response_preview` 預設保存最多 500 字。若輸入不是合成資料或回覆可能含敏感資訊，將 `CAPTURE_RESPONSE=0`。
