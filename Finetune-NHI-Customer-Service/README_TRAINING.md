# Gemma-3 Reflection SFT 訓練手冊

本文件記錄了在內網開發環境（VDI/離線環境）中進行 Gemma-3-27b 模型微調（SFT）的標準作業流程與技術細節。

## 🚀 快速啟動流程

### 0. 防止連線中斷
訓練時間較長，請先啟動 `tmux` 以避免 SSH 斷線導致訓練終止。
```bash
tmux new -s gemma_sft
```
*   **常用快捷鍵**: `Ctrl+b` 然後按 `d` (脫離)；`tmux a -t gemma_sft` (重新連回)。

### 1. 啟動容器環境 (H200 機器)
執行腳本以掛載模型目錄與當前工作目錄：
```bash
./run_container.sh
```

### 2. 激活 Python 環境
進入容器後，切換至專屬的虛擬環境：
```bash
source /opt/python-venvs/sft_gemma/bin/activate
```

### 3. 開始訓練
執行訓練腳本。**注意：** 每次訓練前請修改腳本內的 `output_dir` (checkpoint 名稱) 避免覆蓋舊檔案。
```bash
python sft_trainer_unsloth_gemma3.py
```

---

## 📊 監控訓練指標 (Metrics)

由於環境不連外網且 VDI 有特定埠口限制，請依照以下方式查看訓練曲線：

### A. 使用 TensorBoard (轉發路徑: H200 -> W119 -> VDI)
1.  **H200 (機器 A)**:
    在訓練目錄啟動 TensorBoard 並綁定所有 IP：
    ```bash
    tensorboard --logdir ./checkpoints/<YOUR_CHECKPOINT> --port 6006 --bind_all
    ```
2.  **W119 (機器 B)**:
    建立 SSH Tunnel 將 A 的 6006 轉發至本地：
    ```bash
    ssh -L 6006:localhost:6006 <user>@H200_IP
    ```
3.  **VDI 瀏覽器**:
    透過 B 機器轉發至 9999 port (請先確保 9999 port 上的舊服務已關閉)。
    *   在 VDI 瀏覽器輸入 `http://W119_IP:9999` 查看。

### B. 快速產出靜態圖表 (無需 TensorBoard)
若環境未安裝 TensorBoard，可直接讀取 `trainer_state.json` 畫圖：
```bash
python -c "import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; state = json.load(open('./checkpoints/<YOUR_CHECKPOINT>/trainer_state.json')); tl, ts, el, es = [], [], [], []; [ (tl.append(e['loss']), ts.append(e['step'])) for e in state['log_history'] if 'loss' in e ]; [ (el.append(e['eval_loss']), es.append(e['step'])) for e in state['log_history'] if 'eval_loss' in e ]; fig, ax = plt.subplots(figsize=(10,5)); ax.plot(ts, tl, label='Train Loss'); ax.plot(es, el, label='Eval Loss', marker='o'); ax.legend(); fig.savefig('curve.png')"
```

---

## 🛠️ 重要技術細節 (Know-how)

### 1. Reflection 標籤轉換
為了配合後端 `app.py` 的 `stop_token_ids` 煞車機制，腳本會自動將資料中的標籤進行替換：
*   `<reflection>` ➡ `<unused0>` (Gemma-3 內建特殊 Token)
*   `</reflection>` ➡ `<unused1>`
*   **優點**: 模型回答完正式內容後產出 `<unused0>`，後端會立即停止生成，隱藏思考過程並節省算力。

### 2. Unsloth 極速優化
目前的腳本已針對 Unsloth 框架優化：
*   **`lora_dropout = 0`**: 開啟 C++/CUDA 核心加速，速度提升一倍且節省 VRAM。
*   **`train_on_responses_only`**: 模型僅針對 Assistant 的回覆計算 Loss，不學習 User 的提問，大幅提升回答品質。

### 3. 注意事項
*   **GPU 記憶體**: Gemma-3-27b 體積極大，即便使用 4-bit 載入，建議仍使用 H200/H100 等高顯存機器。
*   **Checkpoint**: 若訓練中斷，可將 `model_path` 改向最新的 checkpoint 資料夾路徑以接續訓練。
