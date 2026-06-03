import os
import json
import ast
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Must be imported before most training stack imports.
from unsloth import FastModel

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template, train_on_responses_only


# =========================
# Config
# =========================
RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL_PATH = "./models/google--gemma-3-27b-it"
GENERAL_DATA_PATH = "./training_data/20260518/gemma3_sft_general_260518.csv"
KEY_QA_EXPANSION_DATA_PATH = "./training_data/20260518/gemma3_sft_key_qa_expansion.csv"

OUTPUT_DIR = (
    f"./checkpoints/"
    f"wingeneai-gemma-3-27b-clean-sft-general-keyqa-v2-{RUN_NAME}"
)
TRAINING_LOG_PATH = os.path.join(OUTPUT_DIR, "training_metrics.jsonl")

MAX_SEQ_LENGTH = 4096
VAL_RATIO = 0.1
SHUFFLE_SEED = 3407

DEBUG_PRINT_TRAIN_SAMPLES = 1
DEBUG_PRINT_VAL_SAMPLES = 1
MAP_NUM_PROC = 4

# ===== ADJUSTED: 更激进的 key_qa 学习 =====
# 增加 batch size 使 key_qa 被处理更多次
PER_DEVICE_TRAIN_BATCH_SIZE = 2  # 从 1 增加到 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4   # 从 8 减少到 4（实际有效批大小相同）
LEARNING_RATE = 5e-6              # 从 8e-6 降低到 5e-6（更平稳的学习）
NUM_TRAIN_EPOCHS = 3              # 从 2 增加到 3（更多 key_qa 曝光）
WARMUP_RATIO = 0.15               # 从 0.1 增加到 0.15（更长的预热）

EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 10
LOGGING_STEPS = 5

# ===== ADJUSTED: 改进验证策略 =====
# 现在包含 key_qa 在验证中，可以监控其效果
# 注意：这改变了最佳模型的选择逻辑
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "eval_loss"
GREATER_IS_BETTER = False

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0

# ===== NEW: key_qa 重复采样配置 =====
# 通过重复 key_qa 数据增加其在训练中的权重
KEY_QA_REPEAT_FACTOR = 3  # key_qa 在训练中重复 3 次


# =========================
# Parsing / formatting
# =========================
def safe_parse(x):
    if x is None:
        return None

    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return None

    try:
        return json.loads(x)
    except Exception:
        pass

    try:
        return ast.literal_eval(x)
    except Exception:
        return None


def normalize_role(role):
    role = str(role).strip().lower()
    if role == "model":
        return "assistant"
    return role


def normalize_conversation(conv):
    """
    Clean data is expected as:
        system -> user -> assistant

    Gemma-3 chat templates are safest with user/assistant turns, so the system
    instruction is folded into the first user turn. No prompt rewriting, task
    filtering, apology filtering, or legacy data cleanup happens here.
    """
    if not isinstance(conv, list) or len(conv) == 0:
        return None

    normalized = []
    system_text = ""

    for msg in conv:
        if not isinstance(msg, dict):
            continue

        role = normalize_role(msg.get("role", ""))
        content = msg.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if content is None or str(content).strip() == "":
            continue

        content = str(content).strip()

        if role == "system":
            system_text = content if not system_text else system_text + "\n\n" + content
            continue

        if role == "user" and system_text:
            content = "[SYSTEM INSTRUCTION]\n" + system_text + "\n\n[USER]\n" + content
            system_text = ""

        if len(normalized) > 0 and normalized[-1]["role"] == role:
            normalized[-1]["content"] += "\n\n" + content
            continue

        normalized.append({"role": role, "content": content})

    if system_text:
        normalized.insert(0, {"role": "user", "content": "[SYSTEM INSTRUCTION]\n" + system_text})

    while normalized and normalized[0]["role"] != "user":
        normalized = normalized[1:]

    if len(normalized) < 2:
        return None
    if not any(msg["role"] == "assistant" for msg in normalized):
        return None

    return normalized


def process(batch, tokenizer):
    texts = []
    skipped_parse = 0
    skipped_normalize = 0
    skipped_template = 0

    for content in batch["content"]:
        conv = safe_parse(content)
        if conv is None:
            texts.append("")
            skipped_parse += 1
            continue

        conv = normalize_conversation(conv)
        if conv is None:
            texts.append("")
            skipped_normalize += 1
            continue

        try:
            text = tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            print(f"Template conversion error: {exc}")
            texts.append("")
            skipped_template += 1
            continue

        if text is None or str(text).strip() == "":
            texts.append("")
            skipped_template += 1
            continue

        texts.append(text)

    if skipped_parse or skipped_normalize or skipped_template:
        print(
            "Batch skipped:",
            f"parse={skipped_parse}",
            f"normalize={skipped_normalize}",
            f"template={skipped_template}",
        )

    return {"text": texts}


# =========================
# Dataset helpers
# =========================
def require_content(dataset, name):
    if "content" not in dataset.column_names:
        raise ValueError(f"{name} dataset must contain a content column")

    dataset = dataset.filter(
        lambda x: x["content"] is not None and str(x["content"]).strip() != ""
    )
    if len(dataset) == 0:
        raise ValueError(f"{name} dataset is empty after content filtering")
    return dataset


def load_clean_csv(path, name):
    dataset = load_dataset("csv", data_files={name: path})[name]
    return require_content(dataset, name)


def build_train_val_datasets():
    if not os.path.exists(GENERAL_DATA_PATH):
        raise FileNotFoundError(f"GENERAL_DATA_PATH does not exist: {GENERAL_DATA_PATH}")
    if not os.path.exists(KEY_QA_EXPANSION_DATA_PATH):
        raise FileNotFoundError(
            f"KEY_QA_EXPANSION_DATA_PATH does not exist: {KEY_QA_EXPANSION_DATA_PATH}"
        )

    general_dataset = load_clean_csv(GENERAL_DATA_PATH, "general")

    split = general_dataset.train_test_split(
        test_size=VAL_RATIO,
        seed=SHUFFLE_SEED,
        shuffle=True,
    )
    train_dataset = split["train"]
    val_dataset = split["test"]

    print("\n=========================")
    print("General split")
    print("=========================")
    print("general total:", len(general_dataset))
    print("general train:", len(train_dataset))
    print("general val:", len(val_dataset))
    print("=========================\n")

    key_qa_expansion_dataset = load_clean_csv(
        KEY_QA_EXPANSION_DATA_PATH,
        "key_qa_expansion",
    )
    
    # ===== ADJUSTED: key_qa 重复采样 =====
    # 将 key_qa 数据重复 KEY_QA_REPEAT_FACTOR 次，增加学习机会
    print(f"Original key_qa size: {len(key_qa_expansion_dataset)}")
    key_qa_repeated = concatenate_datasets(
        [key_qa_expansion_dataset] * KEY_QA_REPEAT_FACTOR
    )
    print(f"Repeated key_qa size: {len(key_qa_repeated)}")
    
    train_dataset = concatenate_datasets([train_dataset, key_qa_repeated]).shuffle(
        seed=SHUFFLE_SEED
    )

    # ===== ADJUSTED: 验证集现在包含一部分 key_qa =====
    # 将 key_qa 的 10% 加入验证集，用于监控记忆效果
    key_qa_split = key_qa_expansion_dataset.train_test_split(
        test_size=VAL_RATIO,
        seed=SHUFFLE_SEED,
        shuffle=True,
    )
    val_dataset = concatenate_datasets([val_dataset, key_qa_split["test"]])

    print("\n=========================")
    print("Final raw dataset sizes (WITH key_qa IN VALIDATION)")
    print("=========================")
    print("train total:", len(train_dataset))
    print("validation total:", len(val_dataset))
    print("  - general val:", len(split["test"]))
    print("  - key_qa val:", len(key_qa_split["test"]))
    print("key_qa in train (with repeat):", len(key_qa_repeated))
    print("repeat factor:", KEY_QA_REPEAT_FACTOR)
    print("=========================\n")

    return train_dataset, val_dataset


# =========================
# Debug helpers
# =========================
def debug_dataset_text(dataset, tokenizer, n=2, name="dataset"):
    print("\n=========================")
    print(f"Debug samples: {name}")
    print("=========================")

    for idx in range(min(n, len(dataset))):
        text = dataset[idx]["text"]
        token_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])

        print(f"\n--- {name} sample {idx} ---")
        print(text[:3000])
        print("text chars:", len(text))
        print("token length:", token_len)
        print("user marker count:", text.count("<start_of_turn>user\n"))
        print("model marker count:", text.count("<start_of_turn>model\n"))
        print("contains [SYSTEM INSTRUCTION]:", "[SYSTEM INSTRUCTION]" in text)
        print("contains [USER]:", "[USER]" in text)

    print("=========================\n")


def sanity_check_markers(dataset, sample_size=100):
    checked = min(sample_size, len(dataset))
    bad = 0

    for idx in range(checked):
        text = dataset[idx]["text"]
        user_count = text.count("<start_of_turn>user\n")
        model_count = text.count("<start_of_turn>model\n")

        if user_count == 0 or model_count == 0:
            bad += 1
            print(f"Marker issue sample={idx}, user={user_count}, model={model_count}")
            print(text[:1000])

    print(f"Marker sanity check: checked={checked}, bad={bad}")
    if bad > 0:
        raise RuntimeError("Some samples are missing Gemma user/model markers.")


def print_length_stats(dataset, tokenizer, name="train"):
    lengths = [
        len(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
        for row in dataset
    ]

    if not lengths:
        print(f"{name}: no data")
        return

    lengths_sorted = sorted(lengths)

    def percentile(p):
        idx = min(int(len(lengths_sorted) * p / 100), len(lengths_sorted) - 1)
        return lengths_sorted[idx]

    print("\n=========================")
    print(f"Token length stats: {name}")
    print("=========================")
    print("count:", len(lengths))
    print("min:", min(lengths))
    print("avg:", sum(lengths) / len(lengths))
    print("p50:", percentile(50))
    print("p90:", percentile(90))
    print("p95:", percentile(95))
    print("p99:", percentile(99))
    print("max:", max(lengths))
    print("over MAX_SEQ_LENGTH:", sum(1 for length in lengths if length > MAX_SEQ_LENGTH))
    print("=========================\n")


# =========================
# Training log callback
# =========================
class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    @staticmethod
    def jsonable(value):
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    def write_record(self, event, args, state, logs=None):
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "global_step": state.global_step,
            "epoch": state.epoch,
        }
        if logs:
            record.update(logs)

        record = {key: self.jsonable(value) for key, value in record.items()}

        with open(self.path, "a", encoding="utf-8", buffering=1) as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def on_train_begin(self, args, state, control, **kwargs):
        self.write_record(
            "train_begin",
            args,
            state,
            {
                "output_dir": args.output_dir,
                "num_train_epochs": args.num_train_epochs,
                "learning_rate": args.learning_rate,
                "logging_steps": args.logging_steps,
                "eval_steps": args.eval_steps,
                "save_steps": args.save_steps,
                "key_qa_repeat_factor": KEY_QA_REPEAT_FACTOR,
                "note": "Now tracking key_qa in validation set",
            },
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.write_record("log", args, state, logs or {})

    def on_train_end(self, args, state, control, **kwargs):
        self.write_record("train_end", args, state)


# =========================
# Main
# =========================
def main():
    print("Output dir:", OUTPUT_DIR)
    print("Training metrics log:", TRAINING_LOG_PATH)
    print("General data:", GENERAL_DATA_PATH)
    print("Key QA expansion data:", KEY_QA_EXPANSION_DATA_PATH)
    print("Validation ratio:", VAL_RATIO)
    print("KEY_QA REPEAT FACTOR:", KEY_QA_REPEAT_FACTOR)

    training_args = SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=LOGGING_STEPS,
        output_dir=OUTPUT_DIR,
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        remove_unused_columns=True,
        gradient_checkpointing=False,
        neftune_noise_alpha=0,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=GREATER_IS_BETTER,
        report_to="none",
    )

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    model = FastModel.get_peft_model(
        model=model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        random_state=SHUFFLE_SEED,
    )

    processor = get_chat_template(tokenizer, chat_template="gemma-3")
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, val_dataset = build_train_val_datasets()

    train_dataset = train_dataset.map(
        process,
        batched=True,
        num_proc=MAP_NUM_PROC,
        fn_kwargs={"tokenizer": tokenizer},
    )
    val_dataset = val_dataset.map(
        process,
        batched=True,
        num_proc=MAP_NUM_PROC,
        fn_kwargs={"tokenizer": tokenizer},
    )

    train_dataset = train_dataset.filter(
        lambda x: x["text"] is not None and str(x["text"]).strip() != ""
    )
    val_dataset = val_dataset.filter(
        lambda x: x["text"] is not None and str(x["text"]).strip() != ""
    )

    print("Train size after processing:", len(train_dataset))
    print("Val size after processing:", len(val_dataset))

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty after processing")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty after processing")

    debug_dataset_text(train_dataset, tokenizer, n=DEBUG_PRINT_TRAIN_SAMPLES, name="train")
    debug_dataset_text(val_dataset, tokenizer, n=DEBUG_PRINT_VAL_SAMPLES, name="validation")
    sanity_check_markers(train_dataset, sample_size=100)
    sanity_check_markers(val_dataset, sample_size=min(100, len(val_dataset)))
    print_length_stats(train_dataset, tokenizer, name="train")
    print_length_stats(val_dataset, tokenizer, name="validation")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=[JsonlMetricsCallback(TRAINING_LOG_PATH)],
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    trainer.model.config.use_cache = False
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
