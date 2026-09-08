"""
Data loading and preprocessing for Spider text-to-SQL fine-tuning.

Joins the xlangai/spider dataset (question, gold SQL, db_id) with
richardr1126/spider-schema (schema info per db_id) using a db_id lookup dict
(built once for O(1) joins rather than re-scanning per example), formats
each example into a Qwen ChatML prompt, and builds loss-masked labels so
training only grades the model on the assistant's SQL response and not the
prompt tokens it was given.
"""

from datasets import load_dataset

from .prompts import (
    SYSTEM_PROMPT,  # noqa: F401  (unused here but re-exported for callers)
    build_messages,
)


def load_spider_with_schemas():
    """Load Spider train/validation splits and a db_id -> schema lookup dict."""
    spider = load_dataset("xlangai/spider")
    schemas = load_dataset("richardr1126/spider-schema")
    schema_lookup = {row["db_id"]: row for row in schemas["train"]}
    return spider, schema_lookup


def format_example(example: dict, tokenizer, schema_lookup: dict) -> dict:
    """
    Format one Spider example into tokenized input_ids + loss-masked labels.

    Prompt tokens (system + user) are masked with -100 so the model is only
    graded on predicting the assistant's SQL response. Flags
    `mismatch=True` if the prompt-only tokenization doesn't exactly
    prefix-match the full tokenization. `train.py::main` checks this 
    across the whole dataset before training.
    """
    schema = schema_lookup[example["db_id"]]["Schema (values (type))"]
    messages = build_messages(example["question"], schema, example["query"])

    prompt_text = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    prompt_len = len(prompt_ids)
    mismatch = full_ids[:prompt_len] != prompt_ids

    labels = full_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": full_ids,
        "labels": labels,
        "length": len(full_ids),
        "mismatch": mismatch,
    }


def build_training_datasets(tokenizer, max_length: int = 768):
    """
    Load, format, and length-filter the full train/validation splits.

    Examples longer than max_length are dropped from the splits entirely rather 
    than truncated as truncation risks silently cutting off the gold SQL answer
    itself for long-schema examples, which would teach the model
    an incomplete target. Dropping the long tail (~2.3% of
    training examples at max_length=768) keeps compute time small
    while still ensuring every remaining example's target is intact.
    """
    spider, schema_lookup = load_spider_with_schemas()

    def _format(ex):
        return format_example(ex, tokenizer, schema_lookup)

    train = spider["train"].map(_format).filter(lambda x: x["length"] <= max_length)
    val = spider["validation"].map(_format).filter(lambda x: x["length"] <= max_length)
    return train, val, schema_lookup
