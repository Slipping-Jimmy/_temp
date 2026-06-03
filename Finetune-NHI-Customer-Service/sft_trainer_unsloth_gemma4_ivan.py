import os
import json
import ast

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🔥 一定要最早 import
from unsloth import FastModel

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template, train_on_responses_only


# =========================
# 設定
# =========================
output_dir = "./checkpoints/wingeneai--gemma-4-31b-20260413-reflection-sft-ivan"
model_path = "./models/google--gemma-4-31B-it"
train_path = "./training_data/202603/train_gemma_3_reflected_20260323_sft.csv"
val_path   = "./training_data/202603/val_gemma_3_reflected_20260323_sft.csv"


# =========================
# 安全解析
# =========================
def safe_parse(x):
    if x is None:
        return None

    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return None

    try:
        return json.loads(x)
    except:
        pass

    try:
        return ast.literal_eval(x)
    except:
        return None


# =========================
# system → user + assistant
# =========================
def transform_conversation(conv):

    new_conv = []
    i = 0

    if len(conv) > 0 and conv[0]["role"] == "system":
        system_msg = conv[0]["content"]

        new_conv.append({
            "role": "user",
            "content": system_msg
        })

        new_conv.append({
            "role": "assistant",
            "content": "好的！我會遵守您的提示。"
        })

        i = 1

    for msg in conv[i:]:

        role = msg.get("role")
        content = msg.get("content")

        if role not in ["user", "assistant"]:
            continue

        if content is None or str(content).strip() == "":
            continue

        if len(new_conv) > 0 and new_conv[-1]["role"] == role:
            new_conv[-1]["content"] += "\n" + content
            continue

        new_conv.append({
            "role": role,
            "content": content
        })

    if len(new_conv) > 0 and new_conv[0]["role"] != "user":
        new_conv = new_conv[1:]

    if len(new_conv) < 2:
        return None

    return new_conv


# =========================
# dataset processing
# =========================
def process(batch, tokenizer):

    texts = []

    for x in batch["content"]:

        conv = safe_parse(x)

        if conv is None or not isinstance(conv, list):
            texts.append("")
            continue

        conv = transform_conversation(conv)

        if conv is None:
            texts.append("")
            continue

        try:
            # Gemma-4 的 processor.tokenizer 在 apply_chat_template 時會自動加 <bos>
            # 訓練時建議用 removeprefix('<bos>') 避免重複（Unsloth 官方推薦做法）
            text = tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                enable_thinking=False,  # 🔥 純 SFT 回答，不走 thinking 模式
            ).removeprefix('<bos>')
            texts.append(text)
        except Exception as e:
            print(f"Template 轉換錯誤: {e}")
            texts.append("")

    return {"text": texts}


# =========================
# 主程式
# =========================
def main():

    training_args = SFTConfig(
        dataset_text_field="text",

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,

        learning_rate=5e-5,
        lr_scheduler_type="cosine",

        max_steps=5000,

        eval_strategy="steps",
        eval_steps=200,

        save_strategy="steps",
        save_steps=200,

        logging_steps=10,

        output_dir=output_dir,
        optim="adamw_torch",

        warmup_steps=100,

        bf16=True,
        fp16=False,

        remove_unused_columns=True,
        gradient_checkpointing=True,

        neftune_noise_alpha=0,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # =========================
    # 載入模型（H200 配置：不量化，bf16 LoRA）
    # =========================
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=8192,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        device_map={"": 0},   # 🔥 繞過 Gemma-4 31B vision tower 的 device_map bug
        attn_implementation="sdpa",   # 🔥 強制用 SDPA 而非 FA2
        use_gradient_checkpointing="unsloth", # Unsloth 優化版的 grad ckpt
    )

    model = FastModel.get_peft_model(
        model=model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=32,
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        random_state=3407,
    )

    # =========================
    # 🔥 Gemma-4 chat template
    # 31B 用 thinking 版本（官方推薦）
    # 若你不想要 thinking 能力，可改成 "gemma-4"
    # =========================
    processor = get_chat_template(
        tokenizer,
        chat_template="gemma-4-thinking",
    )
    tokenizer = processor.tokenizer

    # =========================
    # 載入資料
    # =========================
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_path,
            "validation": val_path,
        }
    )

    train_dataset = dataset["train"].filter(
        lambda x: x["content"] is not None and str(x["content"]).strip() != ""
    )

    val_dataset = dataset["validation"].filter(
        lambda x: x["content"] is not None and str(x["content"]).strip() != ""
    )

    train_dataset = train_dataset.map(
        process,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )

    val_dataset = val_dataset.map(
        process,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )

    train_dataset = train_dataset.filter(lambda x: x["text"] != "")
    val_dataset   = val_dataset.filter(lambda x: x["text"] != "")

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    # 🔍 建議：印出一筆樣本確認 chat template 格式正確
    print("=" * 50)
    print("Sample text (first 1000 chars):")
    print(train_dataset[0]["text"][:1000])
    print("=" * 50)

    if len(train_dataset) == 0:
        print("❌ 錯誤：訓練資料集大小為 0，請檢查資料處理邏輯或原始資料格式。")
        return

    # =========================
    # Trainer
    # =========================
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    # 🔥 Gemma-4 的分隔符格式（和 Gemma-3 完全不同！）
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    trainer.model.config.use_cache = False

    # =========================
    # Train
    # =========================
    trainer.train()

    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
