"""
LoRA fine-tuning script for text-to-SQL on Spider, using
Qwen2.5-Coder-3B-Instruct. Not quantized, see README.

Run: python -m src.train
"""

import math

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

from .data import build_training_datasets

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
MAX_LENGTH = 768
OUTPUT_DIR = "checkpoints"
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 3
WARMUP_RATIO = 0.03


def build_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", dtype=torch.float16
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # measured: 29,933,568 / 3,115,872,256 = 0.96%
    return model


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds, _ = build_training_datasets(tokenizer, max_length=MAX_LENGTH)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # masking-boundary mismatch flag, measured as 0 for this tokenizer
    mismatch_count = sum(train_ds["mismatch"]) + sum(val_ds["mismatch"])
    if mismatch_count > 0:
        print(
            f"WARNING: {mismatch_count} examples have a prompt/full tokenization "
            "mismatch. so their loss masking boundary may be off by a token or two."
            " Consider investigating before trusting training results."
        )

    model = build_model()
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100
    )

    # warmup_steps computed from the loaded dataset size, 
    # because hardcoded value would go stale if anything changed
    effective_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch_size)
    total_steps = steps_per_epoch * NUM_TRAIN_EPOCHS
    warmup_steps = max(1, round(total_steps * WARMUP_RATIO))
    print(f"Steps/epoch: {steps_per_epoch}  Total steps: {total_steps}  Warmup: {warmup_steps}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=2e-4,
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        max_length=MAX_LENGTH,
        loss_type="nll",  # trl's default "chunked_nll" crashes here
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )
    trainer.train()

    # NOTE: validation loss rose after epoch 1 in the reference run
    # (0.229 -> 0.289 -> 0.361) while training loss kept falling (overfitting)
    # The final model here is whatever epoch trainer.train() ends on. 
    # check eval_loss across `{OUTPUT_DIR}/checkpoint-*` and load the best one
    # (via peft.PeftModel.from_pretrained on the best checkpoint
    # directory) for the correct model.
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
