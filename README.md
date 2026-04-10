# Poker Decision Advisor — LLM Fine-Tuning Project

An AI poker decision engine built by fine-tuning Qwen3-8B on the [PokerBench](https://huggingface.co/datasets/RZ412/PokerBench) dataset. The model takes a poker game scenario and outputs the optimal action (check, fold, call, bet X, or raise X).

This project demonstrates a two-stage post-training pipeline: **Supervised Fine-Tuning (SFT)** followed by **Group Relative Policy Optimization (GRPO)**, applied to a domain-specific decision-making task.

## Results

Evaluated on 1,000 held-out PokerBench test examples:

| Metric | Base Qwen3-8B | SFT |
|---|---|---|
| Overall accuracy | ~40–55%* | **90.8%** |
| Average reward | n/a* | **0.815** |

*\*Base model evaluation was unreliable. Qwen3-8B's default thinking mode (`<think>...</think>`) consumed most of the generation budget, leaving little room for the actual answer. After disabling thinking with `/no_think` and using a flexible parser, accuracy was estimated at 40–55%, but this number should be treated as approximate. The reward metric is not comparable since the base model's output format rarely matched the expected action format.*

**Per-action breakdown (SFT):**

| Action | Accuracy |
|---|---|
| Fold | 94.1% |
| Check | 95.6% |
| Call | 91.6% |
| Raise | 85.5% |
| Bet | 74.4% |

The SFT model achieves strong performance across all action types, with bet and raise being the weakest — these require predicting the correct sizing (e.g., "bet 18") in addition to the action type.

## Model

The SFT adapter is available on HuggingFace: **[Dominicvdb/pokerqwen](https://huggingface.co/Dominicvdb/pokerqwen)**

## Architecture

- **Base model:** [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (Apache 2.0)
- **Fine-tuning method:** QLoRA (4-bit quantization + LoRA adapters, r=32, alpha=64)
- **Dataset:** [RZ412/PokerBench](https://huggingface.co/datasets/RZ412/PokerBench) — 563K training examples, 11K test
- **Training compute:** Cloud GPUs (Kaggle T4, Vast.ai RTX 3090)

## Project Structure

```
├── src/
│   ├── cards.py           # Card representation and deck utilities
│   ├── evaluator.py       # Hand evaluation (treys wrapper)
│   ├── data_loader.py     # PokerBench dataset loading
│   ├── preprocessor.py    # Chat template formatting for Qwen3
│   └── reward.py          # Tiered reward function for GRPO
├── notebooks/
│   ├── 01_sft_training.ipynb
│   ├── 02_grpo_training.ipynb
│   └── 03_evaluate.ipynb
├── scripts/
│   └── train_grpo.py
├── tests/                 # Unit tests (63 tests, 100% coverage on core modules)
└── tools/                 # Poker analysis utilities
```

## Training Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

Fine-tuned Qwen3-8B on the full PokerBench training set using QLoRA. The model learns to map game scenarios to optimal actions. Training used the Qwen3 chat template with a system prompt:

> *"You are a poker decision engine. Given a game scenario, output only the optimal action (check, fold, call, bet X, or raise X). Do not explain."*

### Stage 2: GRPO (In Progress)

Applied Group Relative Policy Optimization to push beyond SFT accuracy, particularly on bet/raise actions where the model is weakest. GRPO generates multiple completions per prompt and reinforces high-reward outputs relative to the group.

Key components:
- **Reward function:** Tiered scoring based on action type match and bet sizing accuracy
- **Reference model:** SFT weights (via LoRA adapter toggling)
- **KL penalty:** Prevents catastrophic forgetting of fold/call/check performance

**Current status:** GRPO has not yet produced a model that outperforms SFT overall. The primary challenge is reward signal sparsity — the SFT model is already correct on ~90% of examples, leaving few training samples with reward variance for GRPO to learn from. Potential solutions being explored include stricter reward functions that create more granular scoring on bet sizing, and online difficulty filtering to focus training on examples at the model's decision boundary.

## Reward Function

The GRPO reward function uses tiered scoring:

| Condition | Reward |
|---|---|
| Correct action + exact bet size | 1.0 |
| Correct action + bet within 10% | 0.7 |
| Correct action + bet within 20% | 0.4 |
| Correct action + bet within 50% | 0.1 |
| Wrong action, same category (e.g., bet vs raise) | -0.3 |
| Wrong action, different category | -1.0 |

## Key Learnings

- **SFT is remarkably effective** for structured decision tasks — a single epoch of QLoRA fine-tuning brought accuracy from ~50% to 91%
- **GRPO requires careful KL anchoring** — without a proper reference model, the policy diverges and forgets previously learned behaviors
- **Reward variance is essential** for GRPO — when the model is already highly accurate, most training steps produce zero gradient signal, limiting improvement

## Next Steps

- [ ] Implement stricter reward function to increase reward variance on bet/raise examples
- [ ] Online difficulty filtering (train only on examples where model has 25-75% solve rate)
- [ ] Add reasoning traces (chain-of-thought) to explain decisions
- [ ] Build tool-calling capabilities (equity calculator, pot odds) inspired by [ToolPoker](https://arxiv.org/abs/2602.00528)
- [ ] Deploy as a Streamlit app for interactive poker training
- [ ] Compare with Qwen3.5-9B as the base model

## Running

```bash
# Install dependencies
conda env create -f environment.yml
conda activate poker

# Run tests
pytest tests/ -v --cov=src

# Training notebooks are in notebooks/ — designed for Kaggle/Colab/Vast.ai
```

## References

- [PokerBench: Training Large Language Models to be Professional Poker Players](https://huggingface.co/datasets/RZ412/PokerBench)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) — introduces GRPO
- [ToolPoker: Solving Poker with LLM Tool Calling](https://arxiv.org/abs/2602.00528)
- [Unsloth](https://unsloth.ai/) — fast fine-tuning framework
- [TRL](https://huggingface.co/docs/trl) — Transformer Reinforcement Learning library
