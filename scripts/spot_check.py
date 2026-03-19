"""
Local spot check — compares GRPO adapter against SFT baseline.

Usage:
    python scripts/spot_check.py \
        --sft-adapter  models/sft-adapter \
        --grpo-adapter models/grpo-adapter \
        --n-samples 20
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessor import format_grpo, apply_chat_template
from src.reward import poker_reward

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
DATASET_ID = "RZ412/PokerBench"
BASE_MODEL  = "Qwen/Qwen3-8B"


def load_model(sft_adapter: str, grpo_adapter: str | None):
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    print(f"Loading base model + SFT adapter from {sft_adapter} ...")
    model = AutoModelForCausalLM.from_pretrained(
        sft_adapter,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter, trust_remote_code=True)

    if grpo_adapter:
        print(f"Applying GRPO adapter from {grpo_adapter} ...")
        model = PeftModel.from_pretrained(model, grpo_adapter)

    model.eval()
    return model, tokenizer


def run_spot_check(model, tokenizer, n_samples: int, cache_dir: str):
    print(f"\nLoading {n_samples} test samples from PokerBench ...")
    dataset = load_dataset(DATASET_ID, split="test", cache_dir=cache_dir)
    test_samples = dataset.select(range(n_samples))

    correct = 0
    for i, row in enumerate(test_samples):
        prompt = apply_chat_template(
            format_grpo(row), tokenizer, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        generated = _THINK_RE.sub("", generated).strip()

        reward = poker_reward(generated, row["output"])
        if reward == 1.0:
            correct += 1

        print(f"[{i+1:02d}] Expected: {row['output']:<14} Predicted: {generated:<14} Reward: {reward}")

    pct = correct * 100 // n_samples
    print(f"\nAccuracy : {correct}/{n_samples} ({pct}%)")
    print(f"SFT baseline was 75% — improvement: {pct - 75:+}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-adapter",  default="models/sft-adapter")
    parser.add_argument("--grpo-adapter", default=None,
                        help="Path to GRPO checkpoint/adapter. Omit to test SFT only.")
    parser.add_argument("--n-samples",    type=int, default=20)
    parser.add_argument("--cache-dir",    default="data")
    args = parser.parse_args()

    model, tokenizer = load_model(args.sft_adapter, args.grpo_adapter)
    run_spot_check(model, tokenizer, args.n_samples, args.cache_dir)
