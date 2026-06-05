#!/usr/bin/env python3
import argparse
import gc
import os

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a PEFT/LoRA adapter into the original bf16 base model."
    )
    parser.add_argument("--base", required=True, help="Path or HF id of the bf16 base model.")
    parser.add_argument("--lora", required=True, help="Path to the LoRA adapter checkpoint.")
    parser.add_argument("--out", required=True, help="Output directory for the merged model.")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Load on CPU. Slow, but useful when debugging loader issues.",
    )
    return parser.parse_args()


def load_base_model(args):
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.cpu:
        kwargs["device_map"] = {"": "cpu"}
    else:
        kwargs["device_map"] = "auto"

    try:
        print("Loading base with AutoModelForImageTextToText...")
        return AutoModelForImageTextToText.from_pretrained(args.base, **kwargs)
    except Exception as exc:
        print(f"AutoModelForImageTextToText failed: {exc}")
        print("Falling back to AutoModelForCausalLM...")
        return AutoModelForCausalLM.from_pretrained(args.base, **kwargs)


def save_tokenizer_or_processor(base_path, out_path, trust_remote_code=False):
    try:
        processor = AutoProcessor.from_pretrained(
            base_path,
            trust_remote_code=trust_remote_code,
        )
        processor.save_pretrained(out_path)
        print("Saved processor.")
        return
    except Exception as exc:
        print(f"AutoProcessor save failed: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(out_path)
    print("Saved tokenizer.")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    base = load_base_model(args)
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base,
        args.lora,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )

    print("Merging LoRA into base...")
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.torch_dtype = torch.bfloat16

    print(f"Saving merged model to {args.out}...")
    merged.save_pretrained(
        args.out,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    save_tokenizer_or_processor(args.base, args.out, args.trust_remote_code)

    del model
    del merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()

