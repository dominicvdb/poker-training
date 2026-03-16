"""
GRPO fine-tuning script — runs on any Linux GPU machine (Google Cloud, etc.)

Usage:
    python scripts/train_grpo.py \
        --sft-adapter-dir /path/to/sft-adapter \
        --output-dir      /path/to/grpo-adapter \
        --checkpoint-dir  /path/to/checkpoints \
        --max-steps       2000

Defaults assume the repo is cloned to the working directory and
the SFT adapter is at ./models/sft-adapter.
"""

import argparse
import os
import re
import sys
import time

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GRPO training for PokerApp")
    p.add_argument("--sft-adapter-dir", default="./models/sft-adapter",
                   help="Path to the Stage 1 SFT adapter")
    p.add_argument("--output-dir",      default="./models/grpo-adapter",
                   help="Where to save the final GRPO adapter")
    p.add_argument("--checkpoint-dir",  default="./models/grpo-checkpoints",
                   help="Where to save training checkpoints")
    p.add_argument("--data-cache-dir",  default="./data",
                   help="Where to cache the PokerBench dataset")
    p.add_argument("--max-steps",       type=int, default=2000)
    p.add_argument("--batch-size",      type=int, default=2)
    p.add_argument("--grad-accum",      type=int, default=4)
    p.add_argument("--num-generations", type=int, default=6)
    p.add_argument("--lr",              type=float, default=4e-5)
    p.add_argument("--beta",            type=float, default=0.05)
    p.add_argument("--max-grad-norm",   type=float, default=0.3)
    p.add_argument("--skip-spot-check", action="store_true",
                   help="Skip the 20-example spot check after training")
    return p.parse_args()


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str):
    """Print with timestamp — shows up cleanly in GCloud logs."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # GPU check
    import torch
    assert torch.cuda.is_available(), "No GPU found — provision a GPU instance"
    gpu = torch.cuda.get_device_properties(0)
    log(f"GPU : {gpu.name}")
    log(f"VRAM: {gpu.total_memory / 1e9:.1f} GB")

    # Validate SFT adapter exists
    assert os.path.exists(args.sft_adapter_dir), (
        f"SFT adapter not found at {args.sft_adapter_dir}\n"
        f"Copy it from Google Drive or GCS before running."
    )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    log("Loading dataset...")
    from src.data_loader import load_pokerbench
    dataset = load_pokerbench(cache_dir=args.data_cache_dir)
    train_ds = dataset["train"]
    log(f"Train: {len(train_ds):,}  Test: {len(dataset['test']):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    log("Loading SFT model...")
    from unsloth import FastLanguageModel

    MAX_SEQ_LENGTH = 1024
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.sft_adapter_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    log(f"Model loaded — VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    log("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
    )
    model.print_trainable_parameters()

    # ── Preprocess ────────────────────────────────────────────────────────────
    log("Preprocessing dataset...")
    from src.preprocessor import format_grpo, apply_chat_template

    def preprocess_for_grpo(row):
        return {
            "prompt": apply_chat_template(
                format_grpo(row), tokenizer, add_generation_prompt=True
            ),
            "answer": row["output"],
        }

    grpo_dataset = train_ds.map(
        preprocess_for_grpo,
        remove_columns=train_ds.column_names,
    )
    log(f"Preprocessed {len(grpo_dataset):,} rows")

    # ── Reward function ───────────────────────────────────────────────────────
    from src.reward import poker_reward

    def grpo_reward_fn(completions, answer=None, **kwargs):
        return [poker_reward(c, a) for c, a in zip(completions, answer)]

    # Sanity check
    assert grpo_reward_fn(["bet 18", "fold"], answer=["bet 18", "raise 10"]) == [1.0, -1.0]
    log("Reward function OK")

    # ── Completion logger callback ─────────────────────────────────────────────
    from transformers import TrainerCallback

    class CompletionLogger(TrainerCallback):
        def on_step_end(self, cb_args, state, control, **kwargs):
            if state.global_step % 100 == 0:
                model_ref = kwargs.get("model")
                if model_ref is None:
                    return
                sample = grpo_dataset[0]
                inputs = tokenizer(sample["prompt"], return_tensors="pt").to("cuda")
                with torch.no_grad():
                    completions = []
                    for _ in range(3):
                        out = model_ref.generate(
                            **inputs,
                            max_new_tokens=16,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        gen = tokenizer.decode(
                            out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True,
                        ).strip()
                        reward = poker_reward(gen, sample["answer"])
                        completions.append(f"'{gen}' → {reward:.1f}")
                log(f"Step {state.global_step} samples: {' | '.join(completions)} (answer: '{sample['answer']}')")

    # ── Training config ───────────────────────────────────────────────────────
    log(f"Configuring GRPO — max_steps={args.max_steps}")
    from trl import GRPOTrainer, GRPOConfig

    use_bf16 = torch.cuda.is_bf16_supported()
    log(f"Precision: {'bf16' if use_bf16 else 'fp16'}")

    config = GRPOConfig(
        output_dir=args.checkpoint_dir,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.num_generations,
        max_completion_length=32,
        max_prompt_length=512,
        max_grad_norm=args.max_grad_norm,
        temperature=0.7,
        top_p=0.9,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=False,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward_fn,
        args=config,
        train_dataset=grpo_dataset,
    )
    trainer.add_callback(CompletionLogger())

    effective_batch = args.batch_size * args.grad_accum * args.num_generations
    log(f"Effective batch: {args.batch_size}×{args.grad_accum} prompts × {args.num_generations} generations = {effective_batch} completions per update")

    # ── Train (with auto-resume) ───────────────────────────────────────────────
    checkpoints = [
        d for d in os.listdir(args.checkpoint_dir)
        if d.startswith("checkpoint-")
    ] if os.path.exists(args.checkpoint_dir) else []

    resume_from = args.checkpoint_dir if checkpoints else None
    if resume_from:
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        log(f"Resuming from checkpoint: {latest}")
    else:
        log("Starting from scratch")

    t0 = time.time()
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - t0

    log(f"Training complete — {elapsed / 60:.1f} min")
    log(f"Loss: {trainer_stats.metrics['train_loss']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    log(f"Saving adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log("Adapter saved")

    # ── Spot check ────────────────────────────────────────────────────────────
    if not args.skip_spot_check:
        log("Running spot check (20 examples)...")
        _think_re = re.compile(r"<think>.*?</think>", re.DOTALL)
        FastLanguageModel.for_inference(model)

        test_ds = dataset["test"].select(range(20))
        correct = 0
        for row in test_ds:
            prompt = apply_chat_template(
                format_grpo(row), tokenizer, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
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
            generated_clean = _think_re.sub("", generated).strip()
            reward = poker_reward(generated_clean, row["output"])
            if reward == 1.0:
                correct += 1
            log(f"  Expected: {row['output']:<12} Predicted: {generated_clean:<12} Reward: {reward}")

        log(f"Spot-check: {correct}/20 ({correct * 5}%) — SFT baseline was 75%")


if __name__ == "__main__":
    main()
