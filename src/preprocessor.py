"""Format PokerBench rows into Qwen3 chat messages for SFT and GRPO training."""

SYSTEM_PROMPT = (
    "You are a poker decision engine. Given a game scenario, output only the "
    "optimal action (check, fold, call, bet X, or raise X). Do not explain."
)


def format_sft(row: dict) -> list[dict]:
    """Return Qwen3 chat messages for SFT (includes assistant response)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": row["output"]},
    ]


def format_grpo(row: dict) -> list[dict]:
    """Return Qwen3 chat messages for GRPO (prompt only, no assistant message)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["instruction"]},
    ]


def apply_chat_template(messages: list[dict], tokenizer, add_generation_prompt: bool = False) -> str:
    """Apply the Qwen3 tokenizer chat template to a message list."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def preprocess_sft(dataset, tokenizer):
    """Map a HuggingFace dataset split to SFT-formatted text strings."""
    def _format(row):
        return {"text": apply_chat_template(format_sft(row), tokenizer)}
    return dataset.map(_format)


def preprocess_grpo(dataset, tokenizer):
    """Map a HuggingFace dataset split to GRPO prompt strings."""
    def _format(row):
        return {"prompt": apply_chat_template(format_grpo(row), tokenizer, add_generation_prompt=True)}
    return dataset.map(_format)
