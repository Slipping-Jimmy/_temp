# Multimodal load-test update 20261022-v9

依最新需求改用 presence_penalty=0.2，完全不傳 frequency_penalty。
省略參數代表由後端採用預設，不代表 client 已確認後端設定為 0。
temperature=1.0、top_p=0.95、max_tokens=1024、enable_thinking=false 維持不變。
presence_penalty=0.2 是實驗起點，不保證避免重複生成。

## 安裝與設定

在內網 `_temp` 更新後執行：

```bash
bash ./multimodal_load_test_update_20261022_v9/apply_update.sh \
  /home/aiadm/ModelUseData/ivan/multimodal_load_test
```

若只轉移 tar.gz，先解壓縮。安裝會備份程式，保留既有 .env 與結果。
請將現有 .env 的 LLM_ARGUMENT_JSON 整行替換，勿留下重複設定：

```dotenv
LLM_ARGUMENT_JSON='{"max_tokens":1024,"temperature":1.0,"top_p":0.95,"presence_penalty":0.2,"enable_thinking":false}'
```

程式不再接受 frequency_penalty、top_k、repetition_penalty，舊設定殘留時明確報錯，
避免在使用者不知情的情況下測到不同設定。未提供 LLM_ARGUMENT_JSON 時也使用上述預設。
啟動時會印出實際送出的參數，run_config.txt 亦保存 client 要求的設定。

## 重測

```bash
cd /home/aiadm/ModelUseData/ivan/multimodal_load_test
source .venv-loadtest/bin/activate
python3 unthrottled_load.py \
  --config ./.env --output-root ./results \
  sse-burst --concurrency "5 10 20" --rounds 3 \
  --between-rounds 20s --prepare-workers 8
```

先比較重複輸出、格式合格率及完整完成率。需要對照時僅將 presence_penalty 改為 0.0，
保留同批圖片與負載條件。確認品質後再重新量測效能，勿與舊生成參數混成同一曲線。
