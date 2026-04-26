#!/usr/bin/env python3
"""
Post-training evaluation for ES fine-tuned quadratic models.

Loads a checkpoint (final or intermediate) and evaluates on a held-out
eval parquet, producing:
  - Console summary (accuracy, reward breakdown)
  - CSV with per-sample results
  - TensorBoard scalars (appended to existing logdir if provided)

Usage:
  python es_eval_quadratic.py \
    --model_dir <path_to_base_model> \
    --weights_path <path_to_pytorch_model.pth> \
    --eval_data <path_to_eval.parquet> \
    --out_dir <output_directory>

  # Or evaluate the base model (no checkpoint):
  python es_eval_quadratic.py \
    --model_dir <path_to_base_model> \
    --eval_data <path_to_eval.parquet> \
    --out_dir <output_directory>
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from quad.quadratic_task import reward_function

# ── System prompt (must match training) ───────────────────────────────────

SYSTEM_PROMPT = (
    "You are a math solver that finds integer roots of quadratic equations.\n"
    "\n"
    "Instructions:\n"
    "1. Work through the problem step-by-step (use 2-6 short lines of reasoning).\n"
    "2. End your response with EXACTLY ONE line in this format:\n"
    "   \\boxed{r1, r2}\n"
    "   where r1 and r2 are the two integer roots with r1 <= r2.\n"
    "\n"
    "Rules:\n"
    "- Both roots must be integers (no decimals, no fractions).\n"
    "- Write the smaller root first: r1 <= r2.\n"
    "- Do NOT use XML or HTML tags anywhere.\n"
    "- Do NOT write anything after the \\boxed{} line.\n"
    "- Do NOT write multiple \\boxed{} expressions.\n"
    "- Reasoning must come BEFORE the \\boxed{} line.\n"
    "\n"
    "Example output format:\n"
    "The equation x^2 - 5x + 6 = 0 has discriminant 25-24=1.\n"
    "Using the quadratic formula: x = (5 +/- 1)/2 giving x=3 and x=2.\n"
    "\\boxed{2, 3}\n"
)

CSV_FIELDNAMES = [
    "sample_idx", "equation", "a", "b", "c", "gt_r1", "gt_r2",
    "user_prompt", "model_output", "reward", "math_reward",
    "format_reward", "reasoning_reward", "deduction",
    "both_exact", "one_root", "parse_fail",
]


def _normalize_equation_text(eq):
    t = str(eq) if eq is not None else ""
    t = t.replace("\u2212", "-").replace("\u2013", "-")
    t = t.replace("\u00b2", "^2")
    t = t.replace("\u00d7", "*")
    t = t.replace("\u00A0", " ")
    return t


def load_eval_data(data_path, tokenizer):
    """Load eval parquet and build prompts."""
    print(f"Loading eval data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"  Total eval rows: {len(df)}")

    task_datas = []
    for _, row in df.iterrows():
        equation = _normalize_equation_text(row.get("equation", ""))
        user_message = row.get("user_message", "")
        if not user_message:
            user_message = f"Solve the quadratic equation for integer roots: {equation}"

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        if getattr(tokenizer, "chat_template", None):
            context = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            context = f"SYSTEM: {SYSTEM_PROMPT}\nUSER: {user_message}\nASSISTANT:"

        task_datas.append({
            "context": context,
            "a": int(row["a"]),
            "b": int(row["b"]),
            "c": int(row["c"]),
            "r1": int(row["r1"]),
            "r2": int(row["r2"]),
            "equation": equation,
            "user_message": user_message,
        })
    return task_datas


def run_eval(args):
    # Prepare model: if weights_path given, patch the base model with ES checkpoint
    model_path = args.model_dir
    if args.weights_path:
        print(f"Patching base model with ES weights: {args.weights_path}")
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=dtype_map[args.dtype]
        )
        es_state = torch.load(args.weights_path, map_location="cpu")
        base_model.load_state_dict(es_state, strict=False)

        # Save patched model to temp dir for vLLM
        patched_dir = os.path.join(args.out_dir, "_patched_model")
        os.makedirs(patched_dir, exist_ok=True)
        base_model.save_pretrained(patched_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        tokenizer.save_pretrained(patched_dir)
        del base_model
        torch.cuda.empty_cache()
        model_path = patched_dir
    else:
        print("No --weights_path given; evaluating base model as-is.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    task_datas = load_eval_data(args.eval_data, tokenizer)

    # Launch single vLLM engine for eval
    print(f"Launching vLLM for evaluation (dtype={args.dtype})...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype=args.dtype,
        enable_prefix_caching=False,
        enforce_eager=False,
    )

    prompts = [d["context"] for d in task_datas]
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=args.max_tokens,
    )

    print(f"Running inference on {len(prompts)} eval samples...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.1f}s")

    # Compute rewards
    both_exact_count = 0
    one_root_count = 0
    parse_fail_count = 0
    all_rewards = []
    csv_rows = []

    for idx, (output, data) in enumerate(zip(outputs, task_datas)):
        response = output.outputs[0].text
        r = reward_function(
            response,
            a=data["a"], b=data["b"], c=data["c"],
            r1_gt=data["r1"], r2_gt=data["r2"],
        )
        info = r.get("reward_info", {})
        all_rewards.append(r["reward"])

        if info.get("both_exact"):
            both_exact_count += 1
        elif info.get("one_root"):
            one_root_count += 1
        if info.get("parse_fail"):
            parse_fail_count += 1

        csv_rows.append({
            "sample_idx": idx,
            "equation": data["equation"],
            "a": data["a"],
            "b": data["b"],
            "c": data["c"],
            "gt_r1": data["r1"],
            "gt_r2": data["r2"],
            "user_prompt": data["user_message"],
            "model_output": response,
            "reward": r["reward"],
            "math_reward": info.get("math_reward", 0.0),
            "format_reward": info.get("format_reward", 0.0),
            "reasoning_reward": info.get("reasoning_reward", 0.0),
            "deduction": info.get("deduction", 0.0),
            "both_exact": info.get("both_exact", False),
            "one_root": info.get("one_root", False),
            "parse_fail": info.get("parse_fail", False),
        })

    n = len(task_datas)
    mean_reward = float(np.mean(all_rewards))
    both_rate = both_exact_count / n
    one_rate = one_root_count / n
    fail_rate = parse_fail_count / n

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({os.path.basename(args.eval_data)})")
    print(f"{'='*60}")
    print(f"  Samples:          {n}")
    print(f"  Mean Reward:      {mean_reward:.4f}")
    print(f"  Both Roots Exact: {both_exact_count}/{n} ({both_rate*100:.1f}%)")
    print(f"  One Root Only:    {one_root_count}/{n} ({one_rate*100:.1f}%)")
    print(f"  Parse Fail:       {parse_fail_count}/{n} ({fail_rate*100:.1f}%)")
    print(f"  Inference Time:   {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # Save CSV
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "eval_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"Per-sample results saved to: {csv_path}")

    # Save summary JSON
    summary = {
        "eval_data": args.eval_data,
        "model_dir": args.model_dir,
        "weights_path": args.weights_path,
        "num_samples": n,
        "mean_reward": mean_reward,
        "both_exact_rate": both_rate,
        "one_root_rate": one_rate,
        "parse_fail_rate": fail_rate,
        "inference_time_s": elapsed,
    }
    summary_path = os.path.join(args.out_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # TensorBoard (optional)
    if args.tb_logdir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=args.tb_logdir)
            writer.add_scalar("eval/mean_reward", mean_reward, 0)
            writer.add_scalar("eval/both_exact_rate", both_rate, 0)
            writer.add_scalar("eval/one_root_rate", one_rate, 0)
            writer.add_scalar("eval/parse_fail_rate", fail_rate, 0)
            writer.close()
            print(f"TensorBoard eval scalars written to: {args.tb_logdir}")
        except ImportError:
            print("tensorboard not available; skipping TB logging")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ES fine-tuned quadratic model")
    parser.add_argument("--model_dir", required=True,
                        help="Path to base HF model directory")
    parser.add_argument("--weights_path", type=str, default="",
                        help="Path to ES checkpoint pytorch_model.pth (optional; omit to eval base model)")
    parser.add_argument("--eval_data", required=True,
                        help="Path to eval parquet file")
    parser.add_argument("--out_dir", required=True,
                        help="Directory for eval outputs (CSV, JSON)")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--tb_logdir", type=str, default="",
                        help="TensorBoard logdir to append eval metrics to")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
