# Multimodal load-test update 20261020-v5

這是給既有 `multimodal_load_test` 的增量修正包，不含圖片資料集，也不會覆蓋
`.env`、`data/`、`results/` 或 `.venv-loadtest/`。

## 為什麼需要 v5

前一輪 35-concurrency 的唯一失敗在約 2.1 秒就收到模型內容，之後持續輸出異常內容
約 50 分鐘，最後回傳「內容已達語言模型上限」。那不是 60 秒排隊 timeout；同一個
固定 conversation 在整場測試被重用 423 次，造成 context 污染。

## 修正內容

- `sse-burst` 預設禁止使用 `FIXED_CONVERSATION_ID`，避免把 context limit 誤認成容量上限。
- `per_request` 改為 `sse-burst` 模型/GPU 容量測試的建議策略；每筆使用獨立
  conversation，但 conversation 與 upload 都在同步放行 SSE 前完成，不會計入容量曲線。
- `shared` 模式仍保留：每個 round 建立新的 conversation，該 round 內的 N 筆共用它，
  用來刻意測試同一 conversation 的併發寫入與持久化可靠性。
- 同事已確認系統限制為「request 發出後 60 秒內沒有第一個模型 token」。容量 cutoff
  改為只依首 token：TTFT 超過 60 秒，或沒有模型內容且在約 55–75 秒失敗的候選。
- 首 token 及時到達後的儲存錯誤／SSE 空窗保留為失敗，不列入此項 60 秒 timeout。
  TTFT 超過 60 秒，即使最後完成回答，也會列入超限候選。
- 新增 `first_token_within_60s`，與完整回答的 `success`／`completion_marker` 分開記錄。
- `CAPTURE_RESPONSE=1` 額外保存每級 `sse_raw.jsonl`，包括 SSE 事件名稱、原始資料行、
  完成標記、接收時間與 chat ID，用來核對是否有未辨識的 API 完成格式。
- 單獨的 `[answer]`／`[thought]` 前綴不當作模型首 token；目前仍等待已辨識完成訊號，
  不會收到首 token 就提前關閉串流。完整回答圖移除 60 秒水平線，避免混淆兩項指標。
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
UNTHROTTLED_CONVERSATION_STRATEGY=per_request
SSE_MAX_TOTAL_SECONDS=180
```

若不清空固定 ID，`sse-burst` 會直接拒絕啟動並說明原因。不要用
`--allow-fixed-conversation` 做容量測試；該選項只保留給刻意測 conversation context limit。

## 下一輪

先從 5 筆重新建立 `per_request` 基線，再每次增加 5 筆。先前 `shared` 測試在不同圖片、
不同 conversation 上都曾出現「已收到模型內容後，SSE 沉默約 59 秒，最後回報無法儲存
紀錄」。這兩筆已有及時首 token，因此不屬於本次已確認的首 token timeout。
先前推測的併發寫入競態仍待後端 log 驗證；也需要核對原始 SSE 的完成訊號。

```bash
cd ~/ModelUseData/ivan/multimodal_load_test
source .venv-loadtest/bin/activate

python3 unthrottled_load.py \
  --config ./.env \
  --output-root ./results \
  sse-burst \
  --concurrency "5 10 15 20 25 30 35 40 45 50" \
  --rounds 3 \
  --between-rounds 20s \
  --prepare-workers 8
```

第一個出現 transport/SSE 失敗的級別仍會停止。若是 60 秒 cutoff，`summary.csv` 的
`cutoff_60s_candidates` 會大於 0，並可從 `timeout_basis_counts` 看判定依據；逐筆資料則看
`ttft_ms`、`first_token_within_60s`、`sse_total_ms`、`max_sse_event_gap_ms` 與
`timeout_basis`（僅 `first_token`）。答案格式品質另外看
`answer_valid_rate_percent`。
