# Multimodal load-test update 20261027-v10

修正測試端的 SSE 解析、完成判定、時間上限與統計。生成參數維持 v9：

```dotenv
LLM_ARGUMENT_JSON='{"max_tokens":1024,"temperature":1.0,"top_p":0.95,"presence_penalty":0.2,"enable_thinking":false}'
```

不傳 frequency_penalty、repetition_penalty、top_k。本包不變更後端 API 或模型設定。

## 修正

- 保留純數字、JSON 字串、陣列及非協定 JSON 物件；不再 strip 掉內容空白。
- SSE 以空行分隔完整事件，正確合併多行 data，辨識 event: completed / done / end / error。
- 原文完整保留；TTFT 與嚴格答案驗證才將 ↵／␣ 視為空白。純前綴與空白不算首 token。
- HTTP headers／body 都受絕對時間上限約束。使用有界佇列與每請求一個額外 daemon reader，
  不會因沒有新行而卡住測試呼叫端。取消會嘗試 shutdown socket，reader 最後負責關閉；
  特殊傳輸無法取得 socket 時，read timeout 負責最終回收 reader。
- 保留 SSE 事件分隔空行，新增非空行、data 行、heartbeat 行計數；可區分空回覆與解析問題。
- 明確的 finish_reason=length/max_tokens 列為 sse_output_length_limit，不算正常完成。
  不根據中文錯誤訊息自行推論 vLLM 的 finish_reason。
- 新增包含失敗請求的 all-request TTFT 統計，無首 token 者另計，不當成 0 ms。
- 回答明細也列出完全沒有 raw 行的請求，不再把它們從 HTML 遺漏。
- TTFT 從實際發起 HTTP 請求起算；request_dispatch_skew_ms 顯示該時點相對同步放行誤差。
- 原 max_sse_event_gap_ms 改為真正的 wire 行空窗，包含 heartbeat，不再誤當模型 idle。

主要串流修正適用 unthrottled_load.py 的 burst / sse-burst / open-loop。
共享 loadtest_lib.py 也更新，因此一併備份；Locust 主迴圈未改用新版 deadline/framing。
高並發會多出 reader threads，請留意 load generator CPU/thread 與同步誤差。
新舊版本的 TTFT 與答案格式率定義有修正，請重新量測，不把舊數字直接混成同一曲線。

## 安裝

將更新包放在內網 `_temp`，若只有 tar.gz 先解壓縮：

```bash
tar -xzf multimodal_load_test_update_20261027_v10.tar.gz
bash ./multimodal_load_test_update_20261027_v10/apply_update.sh \
  /home/aiadm/ModelUseData/ivan/multimodal_load_test
```

git pull 已有展開的目錄時，可直接執行 apply_update.sh，無需再解壓縮。
安裝自動備份受影響的檔案，保留 .env、data、results 與 venv。
既有 .env 確認：

```dotenv
FIXED_CONVERSATION_ID=
UNTHROTTLED_CONVERSATION_STRATEGY=per_request
REQUEST_TIMEOUT_SECONDS=120
SSE_MAX_TOTAL_SECONDS=180
CAPTURE_RESPONSE=1
REQUIRE_DONE_MARKER=1
```

## 先重測小批次

```bash
cd /home/aiadm/ModelUseData/ivan/multimodal_load_test
source .venv-loadtest/bin/activate
python3 unthrottled_load.py \
  --config ./.env --output-root ./results \
  sse-burst --concurrency "5 10 20" --rounds 3 \
  --between-rounds 20s --prepare-workers 8
```

開頭必須印出 Test script version: 20261027-v10。
初次驗證先不加 continue-after-failure；若失敗，帶回整個結果，包括 run_config.txt、
transactions.csv、sse_raw.jsonl、summary.csv、report.md、report.html、answers.html。
新 CSV 的 sse_data_line_count=0 表示沒有收到 data 行，應與 heartbeat／非空行計數一起看。
39 項本機測試涵蓋真實 loopback HTTP、headers/body 停滯、多行事件、完成判定、
UTF-8、數字片段、token 長度上限、上傳完整流程與全請求 TTFT；未連接內網模型驗證。
