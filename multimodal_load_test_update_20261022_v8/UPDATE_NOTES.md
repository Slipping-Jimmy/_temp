# Multimodal load-test update 20261022-v8

修正 v7 多送 API 不支援欄位的問題，僅傳送同事提供的六個 llm_argument 欄位。
將 frequency_penalty 設為 0.2，作為降低重複生成的實驗起點，不保證完全消除迴圈。
不再傳 top_k 或 repetition_penalty；若設定含這些未知欄位，啟動時明確拒絕。

## 安裝

在內網 `_temp` 更新後執行（若只有 tar.gz，先解壓縮）：

```bash
bash ./multimodal_load_test_update_20261022_v8/apply_update.sh \
  /home/aiadm/ModelUseData/ivan/multimodal_load_test
```

既有 .env 與結果保留，程式自動備份。請編輯現有 .env，將 LLM_ARGUMENT_JSON
整行替換為以下內容，並確保只留一行同名設定：

```dotenv
LLM_ARGUMENT_JSON='{"max_tokens":1024,"temperature":1.0,"top_p":0.95,"frequency_penalty":0.2,"presence_penalty":0.0,"enable_thinking":false}'
```

若沒有設定此變數也會使用這組預設；若舊 .env 明確設了 frequency_penalty=0.0，
則仍會覆蓋預設，所以應改為 0.2。不得沿用 v7 的 top_k/repetition_penalty。
每次啟動印出要求的參數，並保存到 run_config.txt。

## 重測

```bash
cd /home/aiadm/ModelUseData/ivan/multimodal_load_test
source .venv-loadtest/bin/activate
python3 unthrottled_load.py \
  --config ./.env --output-root ./results \
  sse-burst --concurrency "5 10 20" --rounds 3 \
  --between-rounds 20s --prepare-workers 8
```

先核對 API 完成率、嚴格答案格式、輸出長度與重複行為。
要比較參數效果，使用同批圖片與同級負載，將 frequency_penalty=0.0 作為對照。
確認仍可正常產出符合／不符合後，再重建容量曲線，不與舊參數結果混用。
維持 max_tokens=1024 與 SSE_MAX_TOTAL_SECONDS=180，對重複生成提供兩層上限。
完整 API 內容可查看 answers.html 或用 inspect_sse.py 查看既有 sse_raw.jsonl。
