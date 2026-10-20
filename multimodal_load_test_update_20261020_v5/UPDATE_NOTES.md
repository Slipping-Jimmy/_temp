# Multimodal load-test update 20261020-v5

這是給既有 `multimodal_load_test` 的增量修正包，不含圖片資料集，也不會覆蓋
`.env`、`data/`、`results/` 或 `.venv-loadtest/`。

## 為什麼需要 v5

前一輪 35-concurrency 的唯一失敗在約 2.1 秒就收到模型內容，之後持續輸出異常內容
約 50 分鐘，最後回傳「內容已達語言模型上限」。那不是 60 秒排隊 timeout；同一個
固定 conversation 在整場測試被重用 423 次，造成 context 污染。

## 修正內容

- `sse-burst` 預設禁止使用 `FIXED_CONVERSATION_ID`，避免把 context limit 誤認成容量上限。
- `shared` 模式每個 round 建立新的 conversation，該 round 內的 N 筆仍共用它。
- 60 秒 cutoff 不再預設為 TTFT timeout；同時記錄 TTFT、完整 SSE 時間與最大 SSE
  事件空窗，再自動分類為 `first_token`、`total_request`、`idle_stream` 或
  `explicit_timeout`。
- 只有實際失敗發生在約 55–75 秒或收到明確 timeout 才列為候選；120 秒 client read
  timeout、180 秒測試端安全中止與 context/token 上限錯誤不會冒充系統 60 秒 cutoff。
- `sse_record_save_error` 若很快發生，只算後端紀錄持久化失敗；若已收到首 token、
  接著 SSE 沉默約 60 秒才出現，則標成 `idle_stream` 候選，並同時保留儲存錯誤供追查。
- 已快速收到內容、最後才發生 context/error 的串流不再算 60 秒 cutoff。
- 修正成功 round 被錯寫成 `round_valid=0` 的問題。
- 新增 `SSE_MAX_TOTAL_SECONDS`，預設 180 秒；持續送垃圾 chunk 也不能無限掛住。
- 新增 `answer_valid`、`normalized_answer` 和每級有效答案率。short profile 必須嚴格輸出
  「符合」或「不符合」才算有效答案，但格式錯誤不會與 HTTP/SSE 失敗混在一起。
- HTML/Markdown 報告同步顯示有效答案數。
- 不需要新增 Python 套件。

## 安裝

```bash
tar -xzf multimodal_load_test_update_20261020_v5.tar.gz
cd multimodal_load_test_update_20261020_v5
chmod +x apply_update.sh
./apply_update.sh ~/ModelUseData/ivan/multimodal_load_test
```

安裝後編輯既有 `.env`：

```dotenv
FIXED_CONVERSATION_ID=
UNTHROTTLED_CONVERSATION_STRATEGY=shared
SSE_MAX_TOTAL_SECONDS=180
```

若不清空固定 ID，`sse-burst` 會直接拒絕啟動並說明原因。不要用
`--allow-fixed-conversation` 做容量測試；該選項只保留給刻意測 conversation context limit。

## 下一輪

前一輪正常的 35 筆請求仍只有約 3.66 秒 p95，沒有碰到 60 秒，因此從 35 往上測：

```bash
cd ~/ModelUseData/ivan/multimodal_load_test
source .venv-loadtest/bin/activate

python3 unthrottled_load.py \
  --config ./.env \
  --output-root ./results \
  sse-burst \
  --concurrency "35 40 45 50 55 60" \
  --rounds 3 \
  --between-rounds 20s \
  --prepare-workers 8
```

第一個出現 transport/SSE 失敗的級別仍會停止。若是 60 秒 cutoff，`summary.csv` 的
`cutoff_60s_candidates` 會大於 0，並可從 `timeout_basis_counts` 看判定依據；逐筆資料則看
`ttft_ms`、`sse_total_ms`、`max_sse_event_gap_ms` 與 `timeout_basis`。答案格式品質另外看
`answer_valid_rate_percent`。
