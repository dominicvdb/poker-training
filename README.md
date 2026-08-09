# Poker Decision Advisor

Qwen3-8B fine-tuned on [PokerBench](https://arxiv.org/abs/2501.08328) to output game-theory-optimal actions for No-Limit Hold'em. Given a game state, the model returns a single action — `fold`, `check`, `call`, `bet X`, `raise X` — including sizing.

QLoRA, single rented RTX 3090. Adapter weights: **[Dominicvdb/pokerqwen](https://huggingface.co/Dominicvdb/pokerqwen)**.

## Results

Exact match: action and bet size must both be correct.

| Model | Exact match | Evaluated on |
|---|---|---|
| GPT-4 | 53.6% | PokerBench test set (11K) |
| Llama-3-8B, authors' SFT | 78.3% | PokerBench test set (11K) |
| **Qwen3-8B + QLoRA (this repo)** | **90.8%** | 1,000 held-out training examples |
| Qwen3-8B, no fine-tuning | ~40–55% | 1,000 held-out training examples |

Baselines from Zhuang et al. (2025).

Two limitations. The rows are not head-to-head: this model was evaluated on a held-out slice of the PokerBench *training* split, while the published baselines use the curated 11K test set, which concentrates on harder spots. Re-running on the official split is item 1 on the roadmap. The un-fine-tuned baseline is also an estimate rather than a measurement — Qwen3-8B's thinking mode consumed most of the generation budget before producing an answer; the 40–55% range comes from disabling it with `/no_think` and parsing leniently.

**Per-action accuracy:**

| Check | Fold | Call | Raise | Bet |
|---|---|---|---|---|
| 95.6% | 94.1% | 91.6% | 85.5% | 74.4% |

`fold`/`check`/`call` are three-way classification. `bet` and `raise` additionally require a continuous sizing prediction, and under exact match a bet of 17 where the solver says 18 scores zero. Residual error is predominantly sizing error, not action error.

## Setup

| | |
|---|---|
| Base model | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), Apache 2.0 |
| Method | QLoRA — 4-bit NF4, LoRA r=32, α=64 |
| Data | [RZ412/PokerBench](https://huggingface.co/datasets/RZ412/PokerBench), 563K train / 11K test |
| Compute | Kaggle T4 (prototyping), Vast.ai RTX 3090 (full runs) |
| Frameworks | Unsloth, TRL, PyTorch |

QLoRA rather than full fine-tuning because 4-bit base weights plus rank-32 adapters fit an 8B model into 24GB with a usable batch size.

## Usage

The adapter loads on top of stock Qwen3-8B:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto", load_in_4bit=True)
model = PeftModel.from_pretrained(base, "Dominicvdb/pokerqwen")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
```

Prompts use the Qwen3 chat template with the training system prompt (see `src/preprocessor.py`). Example:

<!-- TODO: replace with a real prompt/output pair from a run, in the exact PokerBench format -->

```
input:  <PokerBench scenario>
output: raise 34
```

No inference script is included; evaluation runs from `notebooks/03_evaluate.ipynb`.

## Training

**Stage 1 — SFT.** One epoch over the full training set, Qwen3 chat template, system prompt constraining output to a bare action:

> *"You are a poker decision engine. Given a game scenario, output only the optimal action (check, fold, call, bet X, or raise X). Do not explain."*

Suppressing explanation is load-bearing: PokerBench targets are single actions, so emitted prose is unscored tokens competing for the generation budget — the same failure mode that made the un-fine-tuned baseline hard to measure.

**Stage 2 — GRPO, which did not beat SFT.** [GRPO](https://arxiv.org/abs/2402.03300) applied on top of the SFT checkpoint, targeting bet/raise sizing. Reference model held at SFT weights via LoRA adapter toggling, KL penalty against that reference, tiered sizing reward.

The failure is reward variance, not implementation. GRPO's advantage is computed relative to the group mean, so a prompt where every sampled completion is correct yields zero advantage across the group and contributes no gradient. At ~91% SFT accuracy most sampled groups are unanimous, leaving few usable training signals. Two fixes follow from this and are the open work:

- Continuous rather than bucketed sizing penalty, restoring variance within groups that are already action-correct.
- Online difficulty filtering — sample each prompt before training on it, keep only those in a 25–75% solve band.

Secondary finding: without KL anchoring to a reference model the policy degraded on fold/call/check while chasing sizing reward. Adapter toggling was added in response.

**Reward function:**

| Condition | Reward |
|---|---|
| Correct action, exact size | 1.0 |
| Correct action, within 10% | 0.7 |
| Correct action, within 20% | 0.4 |
| Correct action, within 50% | 0.1 |
| Wrong action, same family (bet ↔ raise) | −0.3 |
| Wrong action, different family | −1.0 |

## Roadmap

Paused for now. Every remaining item needs rented GPU hours, which are self-funded, so progress is gated on budget rather than direction.

**1. Evaluate on the official PokerBench test split.** Prerequisite for reading the results table as head-to-head.

**2. Tool calling.** Design constraint: the model never emits a number it did not receive from a deterministic tool. Equity from eval7 Monte Carlo, pot odds from arithmetic, preflop frequencies from chart lookup, postflop solutions from a cached TexasSolver subprocess. Game state is validated through a Pydantic schema before any tool sees it, so malformed states fail loudly rather than returning a plausible number. Same direction as [Lin et al. (ICLR 2026)](https://arxiv.org/abs/2602.00528), but applied to an 8B model that already has the action distribution from SFT and needs tools only for arithmetic.

**3. Retrieval over hand histories and range charts,** so decisions are grounded in documented ranges rather than recalled, with retrieved context cited in the output.

## Layout

```
src/
  cards.py         Card and deck representation
  evaluator.py     Hand strength (treys wrapper)
  data_loader.py   PokerBench loading and splitting
  preprocessor.py  Qwen3 chat-template formatting
  reward.py        Tiered reward function for GRPO
notebooks/         SFT, GRPO, evaluation
scripts/train_grpo.py
tests/             63 tests, 100% coverage on src/
tools/             Poker analysis utilities
```

```bash
conda env create -f environment.yml && conda activate poker
pytest tests/ -v --cov=src
```

Training notebooks were developed across Kaggle, Colab and Vast.ai; paths and install cells need adjusting per environment.

## References

- Zhuang et al. (2025). [PokerBench: Training Large Language Models to become Professional Poker Players](https://arxiv.org/abs/2501.08328). AAAI 2025.
- Shao et al. (2024). [DeepSeekMath](https://arxiv.org/abs/2402.03300) — introduces GRPO.
- Lin et al. (2026). [How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](https://arxiv.org/abs/2602.00528). ICLR 2026 — proposes ToolPoker.
- [Unsloth](https://unsloth.ai/) · [TRL](https://huggingface.co/docs/trl)
