# Multimodal load-test report: 20261005T070541Z

## 結論

- 觀測到的最高成功吞吐量：**0.972 requests/s** （32 users，stage_006_32u）。
- 沒有負載級別通過目前的暫定 SLO，需先查看錯誤與 latency 明細。
- 暫定門檻：error ≤ 1%、p95 TTFT ≤ 3000 ms、p95 SSE total ≤ 10000 ms。
- 最終容量仍應搭配 vLLM queue、GPU 及至少 30 分鐘耐久測試確認。

## 各級負載

| Users | Requests | Success RPS | Error % | p50 TTFT ms | p95 TTFT ms | p95 SSE ms | p95 E2E ms | Avg output chars | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 15 | 0.049 | 0.00 | 159.2 | 809.4 | 60026.0 | 60099.9 | 22.9 | N |
| 2 | 30 | 0.098 | 0.00 | 139.8 | 174.1 | 60031.5 | 60111.2 | 24.7 | N |
| 4 | 66 | 0.202 | 6.06 | 147.3 | 241.4 | 60041.6 | 60154.2 | 23.5 | N |
| 8 | 133 | 0.396 | 5.26 | 147.6 | 5718.7 | 60049.6 | 60168.5 | 22.8 | N |
| 16 | 213 | 0.587 | 8.92 | 185.4 | 60067.5 | 60073.9 | 60282.0 | 32.5 | N |
| 32 | 368 | 0.972 | 12.77 | 298.9 | 60101.5 | 60104.3 | 60412.3 | 28.0 | N |

## 錯誤分布

- stage_003_4u: conversation_http_error=4
- stage_004_8u: conversation_http_error=7
- stage_005_16u: conversation_http_error=18, unknown=1
- stage_006_32u: conversation_http_error=46, unknown=1

## Server telemetry

未收集 vLLM/GPU telemetry；容量仍可由 client 結果計算，但無法完整定位瓶頸。

## 解讀注意事項

- SSE chunk 數不是 token 數；token throughput 只採用 vLLM metrics。
- 此報告的 RPS 使用整個 stage wall time，包含 ramp-up，數字較保守。
- TONO 主要是不合規樣本，本報告不將模型準確率視為容量判定條件。
- 若圖片被大量重複，multimodal cache 可能讓容量偏高，應一併查看 manifest 與 cache 指標。
