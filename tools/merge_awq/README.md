# Merge LoRA Then AWQ Quantize

This folder contains scripts for exporting a Gemma 3 LoRA adapter into a full
merged model, then quantizing the merged model for vLLM serving.

The intended flow is:

```text
bf16 base + LoRA adapter
  -> merged bf16 model
  -> AWQ quantized full model
  -> vLLM serve without --enable-lora
```

Do not merge into an already-AWQ model. Start from the original bf16/base HF
checkpoint used for training.

## 1. Install

On the H200 machine:

```bash
pip install -U transformers peft accelerate safetensors datasets
pip install -U llmcompressor compressed-tensors
```

If Gemma 3 loading fails, align the `transformers` version with the version that
successfully loads your local `google--gemma-3-27b-it` checkpoint.

## 2. Merge LoRA Into Base

```bash
python tools/merge_awq/merge_lora_to_bf16.py \
  --base ./models/google--gemma-3-27b-it \
  --lora ./checkpoints/YOUR_LORA_CHECKPOINT \
  --out ./models/gemma3-27b-nhi-merged-bf16
```

Expected output is a standalone HF checkpoint under:

```text
./models/gemma3-27b-nhi-merged-bf16
```

Run a quick quality check on this merged bf16 model before quantizing. If the
merged bf16 model is poor, AWQ will not fix it.

## 3. Build Calibration Data

Use the same SFT CSV format used by the training scripts:

```bash
python tools/merge_awq/build_calibration_jsonl.py \
  --input training_data/20260518/gemma3_sft_general_260518.csv \
  --input training_data/20260603/gemma3_sft_key_qa_expansion.csv \
  --out ./calibration/gemma3_nhi_calibration.jsonl \
  --limit 512 \
  --seed 3407
```

The output JSONL has one `text` field per line and uses the Gemma 3 chat
template.

## 4. AWQ Quantize The Merged Model

```bash
python tools/merge_awq/quantize_awq_llmcompressor.py \
  --model ./models/gemma3-27b-nhi-merged-bf16 \
  --calibration ./calibration/gemma3_nhi_calibration.jsonl \
  --out ./models/gemma3-27b-nhi-merged-awq \
  --num-calibration-samples 256 \
  --max-seq-length 4096
```

If H200 memory allows, try `--num-calibration-samples 512`.

## 5. Serve With vLLM

The AWQ output is a full model. Do not mount the LoRA adapter again.

```bash
vllm serve ./models/gemma3-27b-nhi-merged-awq \
  --tensor-parallel-size 2 \
  --quantization awq \
  --max-model-len 4096
```

Depending on your vLLM version and the output format from `llmcompressor`, vLLM
may auto-detect the quantization config. If `--quantization awq` fails, try
omitting it.

