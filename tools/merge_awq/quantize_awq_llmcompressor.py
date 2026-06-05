#!/usr/bin/env python3
import argparse
import gc
import os

import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize a merged HF model with llm-compressor AWQ."
    )
    parser.add_argument("--model", required=True, help="Merged bf16 model path.")
    parser.add_argument("--calibration", required=True, help="Calibration JSONL with a text field.")
    parser.add_argument("--out", required=True, help="Output directory for quantized model.")
    parser.add_argument("--num-calibration-samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--scheme", default="W4A16_ASYM")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--ignore",
        action="append",
        default=["lm_head"],
        help="Module pattern to ignore. Can be passed multiple times.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading merged model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    dataset = load_dataset(
        "json",
        data_files={"calibration": args.calibration},
        split="calibration",
    )
    if args.num_calibration_samples > 0:
        dataset = dataset.select(range(min(args.num_calibration_samples, len(dataset))))

    recipe = [
        AWQModifier(),
        QuantizationModifier(
            targets=["Linear"],
            scheme=args.scheme,
            ignore=args.ignore,
        ),
    ]

    print(
        "Running AWQ calibration:",
        f"samples={len(dataset)}",
        f"max_seq_length={args.max_seq_length}",
        f"batch_size={args.batch_size}",
        f"scheme={args.scheme}",
    )
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(dataset),
        batch_size=args.batch_size,
        data_collator="truncation",
    )

    print(f"Saving quantized model to {args.out}...")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()

