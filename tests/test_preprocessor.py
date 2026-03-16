"""Tests for src/preprocessor.py."""

import pytest
from src.preprocessor import format_sft, format_grpo, apply_chat_template, SYSTEM_PROMPT


# ── format_sft ────────────────────────────────────────────────────────────────

def test_sft_has_three_messages(sample_row):
    messages = format_sft(sample_row)
    assert len(messages) == 3


def test_sft_roles(sample_row):
    messages = format_sft(sample_row)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_sft_system_prompt(sample_row):
    messages = format_sft(sample_row)
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_sft_user_content(sample_row):
    messages = format_sft(sample_row)
    assert messages[1]["content"] == sample_row["instruction"]


def test_sft_assistant_content(sample_row):
    messages = format_sft(sample_row)
    assert messages[2]["content"] == sample_row["output"]


# ── format_grpo ───────────────────────────────────────────────────────────────

def test_grpo_has_two_messages(sample_row):
    messages = format_grpo(sample_row)
    assert len(messages) == 2


def test_grpo_roles(sample_row):
    messages = format_grpo(sample_row)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_grpo_no_assistant_message(sample_row):
    messages = format_grpo(sample_row)
    assert not any(m["role"] == "assistant" for m in messages)


def test_grpo_user_content_matches_sft(sample_row):
    sft = format_sft(sample_row)
    grpo = format_grpo(sample_row)
    assert grpo[1]["content"] == sft[1]["content"]


# ── apply_chat_template ───────────────────────────────────────────────────────

def test_apply_chat_template_returns_string(sample_row, mock_tokenizer):
    messages = format_sft(sample_row)
    result = apply_chat_template(messages, mock_tokenizer)
    assert isinstance(result, str)


def test_apply_chat_template_contains_content(sample_row, mock_tokenizer):
    messages = format_sft(sample_row)
    result = apply_chat_template(messages, mock_tokenizer)
    assert sample_row["instruction"] in result
    assert sample_row["output"] in result


def test_apply_chat_template_grpo_has_generation_prompt(sample_row, mock_tokenizer):
    messages = format_grpo(sample_row)
    result = apply_chat_template(messages, mock_tokenizer, add_generation_prompt=True)
    assert "assistant" in result.lower()


def test_apply_chat_template_sft_no_generation_prompt(sample_row, mock_tokenizer):
    messages = format_sft(sample_row)
    without = apply_chat_template(messages, mock_tokenizer, add_generation_prompt=False)
    with_prompt = apply_chat_template(messages, mock_tokenizer, add_generation_prompt=True)
    assert len(with_prompt) > len(without)
