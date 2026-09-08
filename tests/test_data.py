"""
Tests for prompt formatting and loss-masking logic.

Uses the Qwen2.5-Coder-3B-Instruct tokenizer (a small, fast download bc its
just the tokenizer/vocab files) so the masking boundary is checked against 
the model's actual chat template, not a mocked approximation of it.
"""

import pytest
from transformers import AutoTokenizer

from src.data import build_messages, format_example

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def test_build_messages_with_gold_has_assistant_turn():
    messages = build_messages(
        "How many singers?", "singer : id (number)", "SELECT count(*) FROM singer"
    )
    assert len(messages) == 3
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "SELECT count(*) FROM singer"


def test_build_messages_without_gold_is_generation_prompt():
    # inference time message shape
    messages = build_messages("How many singers?", "singer : id (number)")
    assert len(messages) == 2
    assert messages[-1]["role"] == "user"


def test_format_example_masks_prompt_only(tokenizer):
    schema_lookup = {
        "singer": {"Schema (values (type))": "singer : id (number), name (text)"}
    }
    example = {
        "db_id": "singer",
        "question": "How many singers?",
        "query": "SELECT count(*) FROM singer",
    }

    result = format_example(example, tokenizer, schema_lookup)

    assert result["mismatch"] is False

    labels = result["labels"]
    input_ids = result["input_ids"]

    # Every masked position must be -100, every unmasked position must
    # exactly reproduce the real input_ids token
    unmasked_count = 0
    for label, token in zip(labels, input_ids):
        if label == -100:
            continue
        assert label == token
        unmasked_count += 1

    # The assistant's SQL response should never be anywhere near the majority 
    # of a prompt+schema example built from a much longer system+user turn.
    assert 0 < unmasked_count < len(input_ids) // 2
