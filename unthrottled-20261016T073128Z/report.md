# Unthrottled load test: unthrottled-20261016T073128Z

## Stages

| Stage | Mode | Target | Scheduled | Admitted | Success | Error % | 60s candidates | Prep failures | Mean E2E ms | p95 E2E ms | Max E2E ms | Max release skew ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage_001_sse_burst_1c | sse_burst | 1 | 3 | 3 | 3 | 0.00 | 0 | 0 | 269.0 | 286.7 | 288.2 | 0.1 |
| stage_002_sse_burst_5c | sse_burst | 5 | 15 | 15 | 15 | 0.00 | 0 | 0 | 558.8 | 730.6 | 942.7 | 4.9 |
| stage_003_sse_burst_10c | sse_burst | 10 | 30 | 30 | 30 | 0.00 | 0 | 0 | 801.5 | 974.3 | 1168.5 | 8.4 |
| stage_004_sse_burst_15c | sse_burst | 15 | 45 | 45 | 45 | 0.00 | 0 | 0 | 1000.5 | 1443.4 | 1838.5 | 13.3 |
| stage_005_sse_burst_20c | sse_burst | 20 | 60 | 60 | 60 | 0.00 | 0 | 0 | 1497.4 | 2137.3 | 2157.1 | 22.4 |
| stage_006_sse_burst_25c | sse_burst | 25 | 75 | 75 | 75 | 0.00 | 0 | 0 | 1791.1 | 2103.5 | 2271.6 | 27.2 |
| stage_007_sse_burst_30c | sse_burst | 30 | 90 | 90 | 90 | 0.00 | 0 | 0 | 2095.8 | 2546.4 | 4100.8 | 27.8 |
| stage_008_sse_burst_35c | sse_burst | 35 | 105 | 105 | 104 | 0.95 | 1 | 0 | 2708.0 | 3660.1 | 3692.5 | 30.5 |
  - errors: `{"sse_error_event": 1}`
  - 60-second cutoff candidates: **1**

## 60-second cutoff boundary

The first cutoff candidate occurred at **stage_008_sse_burst_35c** (target=35, count=1).

## Interpretation

- `burst` sends the target number of jobs together, then waits for the round to finish.
- `sse_burst` uploads all attachments first and then releases all SSE/model requests together; its E2E values measure SSE release through complete model answer.
- `open_loop` schedules at its target requests/s even while earlier jobs are still running.
- `admission_limited > 0` means the client safety cap was reached; increase `--max-inflight` before treating that stage as a server capacity result.
- A `cutoff_60s_candidates` value is a request that failed at roughly 60 seconds or reported a timeout. Confirm its cause with the API/vLLM logs using the recorded trace IDs.
