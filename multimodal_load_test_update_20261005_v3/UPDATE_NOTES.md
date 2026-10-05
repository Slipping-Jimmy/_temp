# Multimodal load-test runtime update 20261005-v3

This incremental update adds the load model needed for a single vendor/account that can send many requests without waiting for earlier model responses.

## What changes

- Adds `unthrottled_load.py`; this is independent of Locust's closed-loop users.
- `burst` sends N end-to-end image requests together, then waits for that round to finish.
- `open-loop` sends at a fixed request rate even while prior requests remain in flight.
- All calls still use the one `PLAYGROUND_TOKEN`; every request gets a unique UUID `chat_id`.
- Default `UNTHROTTLED_CONVERSATION_STRATEGY=shared` reuses one conversation so conversation creation is not mistaken for model capacity.
- Produces per-request timestamps, in-flight count, error type, trace IDs, `summary.csv`, and `report.md` with 60-second cutoff candidates.
- No additional Python package is required: it uses `requests`, which is already required by the existing test package.

## Install

Extract this directory anywhere on the intranet machine, then run:

```bash
chmod +x apply_update.sh
./apply_update.sh ~/ModelUseData/ivan/multimodal_load_test
```

The installer backs up replaced runtime files into `_update_backup_<timestamp>/` and preserves `.env`, data, results, and `.venv-loadtest`. It only appends `UNTHROTTLED_CONVERSATION_STRATEGY=shared` if that setting does not already exist.

## First run: concurrent burst boundary

```bash
cd ~/ModelUseData/ivan/multimodal_load_test
source .venv-loadtest/bin/activate

python3 unthrottled_load.py \
  --config ./.env \
  --output-root ./results \
  burst \
  --concurrency "8 16 24 32" \
  --rounds 3 \
  --between-rounds 20s
```

Then inspect `results/<run_id>/summary.csv` and `report.md`. The first level with `cutoff_60s_candidates > 0`, an SSE failure, or a sharp tail-latency rise is the range to refine. It is a cutoff *candidate*, not proof that vLLM is the cause; correlate the exact request timestamps and trace IDs with the API / vLLM logs.

## Sustained fire-and-forget rate

```bash
python3 unthrottled_load.py \
  --config ./.env \
  --output-root ./results \
  open-loop \
  --arrival-rates "2 4 6 8" \
  --duration 5m \
  --max-inflight 128 \
  --conversation-pool-size 1
```

If `admission_limited` is nonzero, the load generator reached its own protective `--max-inflight` setting. Increase it before interpreting that stage as the server limit.
