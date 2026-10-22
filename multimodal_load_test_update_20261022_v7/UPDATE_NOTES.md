# Multimodal load-test update 20261022-v7

將 Gemma 4 31B 生成設定透過 `llm_argument` 傳給 playground API。
包含 v6 的 API 回覆明細工具；不含資料集或 token。

## 設定

`unthrottled_load.py` 的 burst / sse-burst / open-loop 均使用：

```dotenv
LLM_ARGUMENT_JSON='{"max_tokens":1024,"temperature":1.0,"top_p":0.95,"top_k":64,"repetition_penalty":1.05,"frequency_penalty":0.0,"presence_penalty":0.0,"enable_thinking":false}'
```

既有 `.env` 未填此欄位也使用上述預設。提供 JSON 時可僅覆寫部分欄位。
錯誤型別、不合理數值與未知欄位會於執行前拒絕，不會默默改值或移除參數重試。
max_tokens 必須是正整數，不沿用後端未明確定義的 0。

temperature=1.0、top_p=0.95、top_k=64 來自
[Google Gemma 4 官方建議](https://ai.google.dev/gemma/docs/core/model_card_4#best-practices)。
max_tokens=1024 是本次實驗的輸出上限；repetition_penalty=1.05 是實驗起點，非官方指定。
enable_thinking=false 需由後端正確套用到 Gemma 4 chat template，
參考 [官方 thinking 文件](https://ai.google.dev/gemma/docs/capabilities/thinking)。

請同事確認 API schema 接收 top_k、repetition_penalty，以及真正送往 vLLM 的值。
啟動訊息與 `run_config.txt` 的 `LLM_ARGUMENT_JSON_REQUESTED` 只證明 client 要求的值，
不證明後端已生效。此更新未更改 Locust 的 locustfile.py。

## 安裝

在內網 `_temp` pull 後，可直接使用已展開的目錄：

```bash
bash ./multimodal_load_test_update_20261022_v7/apply_update.sh \
  /home/aiadm/ModelUseData/ivan/multimodal_load_test
```

如果只轉移 tar.gz：

```bash
tar -xzf multimodal_load_test_update_20261022_v7.tar.gz
bash ./multimodal_load_test_update_20261022_v7/apply_update.sh \
  /home/aiadm/ModelUseData/ivan/multimodal_load_test
```

安裝會備份程式，不修改既有 .env、資料集與結果。
確認 .env 各項只設定一次：

```dotenv
FIXED_CONVERSATION_ID=
UNTHROTTLED_CONVERSATION_STRATEGY=per_request
REQUEST_TIMEOUT_SECONDS=120
SSE_MAX_TOTAL_SECONDS=180
CAPTURE_RESPONSE=1
REQUIRE_DONE_MARKER=1
```

## 重測

```bash
cd /home/aiadm/ModelUseData/ivan/multimodal_load_test
source .venv-loadtest/bin/activate
python3 unthrottled_load.py \
  --config ./.env --output-root ./results \
  sse-burst --concurrency "5 10 20" --rounds 3 \
  --between-rounds 20s --prepare-workers 8
```

首先確認設定欄位被後端接受，檢查完整完成率、嚴格答案格式率、重複輸出和原始 SSE。
每次自動建立新結果目錄，並保留原始 SSE 與 answers.html。
到達 max_tokens 而被截斷，不應視為正常短答案。

品質確認後重新從低級別測效能，避免與舊生成設定混成同一曲線：

```bash
python3 unthrottled_load.py \
  --config ./.env --output-root ./results \
  sse-burst --concurrency "50 100 150 200 250 300 350 400 450 500" \
  --rounds 3 --between-rounds 20s --prepare-workers 8 \
  --continue-after-failure
```

後者即使失敗仍會繼續到清單結束；如果出現準備階段失敗或服務不健康，
該級之後的資料不應當成健康狀態下的獨立容量點。
