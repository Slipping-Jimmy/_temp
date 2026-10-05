# Multimodal load-test runtime update 20261005-v2

This is an incremental update for an existing `multimodal_load_test` directory.

## Changes

- Treat raw SSE `[ERROR]` events as failures.
- Require `[DONE]`, `[END]`, or another recognized completion marker by default.
- Record requests interrupted when Locust stops a stage as failures.
- Reuse one conversation per Locust virtual user by default.
- Continue to generate a unique UUID `chat_id` for every transaction.
- Preserve lower-level details for HTTP status 0 failures.
- Add conversation/chat/attachment IDs and response trace IDs to transaction CSV.
- Keep the dataset, results, virtual environment, and real token out of this package.

## Install

Place the extracted update directory anywhere on the intranet machine, then run:

```bash
chmod +x apply_update.sh
./apply_update.sh ~/ModelUseData/ivan/multimodal_load_test
```

The installer creates `_update_backup_<timestamp>/` inside the target before replacing files.

It updates only these `.env` settings while preserving the token and all other values:

```dotenv
CONVERSATION_MODE=per_user
REQUIRE_DONE_MARKER=1
```

It adds an empty `FIXED_CONVERSATION_ID=` only when the key does not exist.

## First verification

```bash
cd ~/ModelUseData/ivan/multimodal_load_test
source .venv-loadtest/bin/activate
./run_test.sh ./.env --smoke
```

Do not start a 30-minute run until the smoke and short staircase runs distinguish model communication timeout, record-save error, and connection error correctly.

