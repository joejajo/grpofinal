#!/usr/bin/env python3
"""
ES Fine-Tuning for Quadratic Integer Roots — Accelerated (vLLM + Ray + NCCL).

Adapted from VsonicV/es-fine-tuning-paper (es_fine-tuning_countdown_accl.py)
for the quadratic root-finding task used in grpofinal.

Architecture:
  - N vLLM engines, each pinned to one GPU via Ray placement groups
  - WorkerExtension provides GPU-resident perturbation/restore/broadcast
  - ES update computed on engine 0, then NCCL broadcast syncs to all engines
  - Round-robin scheduling of seeds across engines for parallelism

Task:
  Given ax^2 + bx + c = 0 with integer roots, the model must output
  \\boxed{r1, r2} with r1 <= r2. Reward function mirrors grpo_quad_train_v12.

Usage:
  python es_finetuning_quadratic_accl.py \\
    --model_name Qwen/Qwen2.5-0.5B-Instruct \\
    --cuda_devices 0,1,2,3 \\
    --num_engines 4 \\
    --population_size 30 \\
    --num_iterations 500

  For single-GPU:
  python es_finetuning_quadratic_accl.py \\
    --model_name Qwen/Qwen2.5-0.5B-Instruct \\
    --cuda_devices 0 \\
    --num_engines 1 \\
    --population_size 20 \\
    --num_iterations 250
"""

import argparse
import csv
from datetime import datetime
import gc
import glob
import json
import os
import random
import shutil
import signal
import socket
import sys
import time

import numpy as np
import pandas as pd
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def get_ip():
    return socket.gethostbyname(socket.gethostname())


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

from quad.quadratic_task import reward_function, format_grade, parse_roots_from_answer, verify_roots

# ── CSV logging ──────────────────────────────────────────────────────────────

CSV_FIELDNAMES = [
    "iteration", "seed", "sample_idx", "system_prompt", "equation",
    "a", "b", "c", "gt_r1", "gt_r2", "user_prompt", "model_output",
    "reward", "math_reward", "format_reward", "reasoning_reward",
    "deduction", "both_exact", "one_root", "parse_fail",
    # Format accuracy fields
    "format_valid_clean", "format_strict_ok", "format_mid_ok",
    "format_loose_ok", "format_bad_tags", "format_multiple_boxed",
    "format_trailing_junk",
    # Math correctness fields
    "math_root1_correct", "math_root2_correct", "math_sum_ok",
    "math_prod_ok", "parsed_r1", "parsed_r2",
]


EVAL_CSV_FIELDNAMES = [
    "iteration", "sample_idx", "equation",
    "a", "b", "c", "gt_r1", "gt_r2", "user_prompt", "model_output",
    "reward", "math_reward", "format_reward", "reasoning_reward",
    "deduction", "both_exact", "one_root", "parse_fail",
    # Format accuracy fields
    "format_valid_clean", "format_strict_ok", "format_mid_ok",
    "format_loose_ok", "format_bad_tags", "format_multiple_boxed",
    "format_trailing_junk",
    # Math correctness fields
    "math_root1_correct", "math_root2_correct", "math_sum_ok",
    "math_prod_ok", "parsed_r1", "parsed_r2",
]


def _compute_format_math_fields(response, a, b, c):
    """Compute granular format accuracy and math correctness fields for a response."""
    from quad.quadratic_task import BOXED_PAIR_RE
    strict_ok, mid_ok, loose_ok, bad_tags = format_grade(response)
    boxed_matches = list(BOXED_PAIR_RE.finditer(response or ""))
    has_boxed = bool(boxed_matches)
    multiple_boxed = len(boxed_matches) > 1
    trailing_junk = False
    if has_boxed:
        trailing_junk = (response[boxed_matches[-1].end():].strip() != "")
    valid_clean = bool(mid_ok and has_boxed and (not multiple_boxed)
                       and (not bad_tags) and (not trailing_junk))

    parsed = parse_roots_from_answer(response)
    r1_ok = r2_ok = sum_ok = prod_ok = False
    pr1 = pr2 = None
    if parsed is not None:
        pr1, pr2, _, _ = parsed
        ok1, ok2, _, _ = verify_roots(a, b, c, pr1, pr2)
        r1_ok = bool(ok1)
        r2_ok = bool(ok2)
        sum_ok = (abs(a * (pr1 + pr2) + b) == 0)
        prod_ok = (abs(a * pr1 * pr2 - c) == 0)

    return {
        "format_valid_clean": valid_clean,
        "format_strict_ok": strict_ok,
        "format_mid_ok": mid_ok,
        "format_loose_ok": loose_ok,
        "format_bad_tags": bad_tags,
        "format_multiple_boxed": multiple_boxed,
        "format_trailing_junk": trailing_junk,
        "math_root1_correct": r1_ok,
        "math_root2_correct": r2_ok,
        "math_sum_ok": sum_ok,
        "math_prod_ok": prod_ok,
        "parsed_r1": pr1,
        "parsed_r2": pr2,
    }


def _append_training_csv(path: str, rows: list) -> None:
    """Append training-time generation rows to CSV with system prompt included."""
    if not path or not rows:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_eval_csv(path: str, rows: list) -> None:
    """Append eval-time generation rows to CSV."""
    if not path or not rows:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_periodic_eval(engine, eval_task_datas, iteration, max_tokens, eval_csv_path, writer):
    """Run evaluation on held-out eval set using engine 0 (unperturbed weights)."""
    prompts = [d["context"] for d in eval_task_datas]
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=max_tokens,
    )
    t0 = time.time()
    outputs = ray.get(engine.generate.remote(prompts, sampling_params, use_tqdm=False))
    elapsed = time.time() - t0

    csv_rows = []
    all_rewards = []
    both_exact_count = 0
    one_root_count = 0
    parse_fail_count = 0
    # Format accuracy counters
    valid_clean_count = 0
    strict_ok_count = 0
    bad_tags_count = 0
    multiple_boxed_count = 0
    trailing_junk_count = 0
    # Math correctness counters
    root1_correct_count = 0
    root2_correct_count = 0
    sum_ok_count = 0
    prod_ok_count = 0

    for idx, (output, data) in enumerate(zip(outputs, eval_task_datas)):
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

        fm = _compute_format_math_fields(response, data["a"], data["b"], data["c"])
        if fm["format_valid_clean"]:
            valid_clean_count += 1
        if fm["format_strict_ok"]:
            strict_ok_count += 1
        if fm["format_bad_tags"]:
            bad_tags_count += 1
        if fm["format_multiple_boxed"]:
            multiple_boxed_count += 1
        if fm["format_trailing_junk"]:
            trailing_junk_count += 1
        if fm["math_root1_correct"]:
            root1_correct_count += 1
        if fm["math_root2_correct"]:
            root2_correct_count += 1
        if fm["math_sum_ok"]:
            sum_ok_count += 1
        if fm["math_prod_ok"]:
            prod_ok_count += 1

        row = {
            "iteration": iteration,
            "sample_idx": idx,
            "equation": data["equation"],
            "a": data["a"],
            "b": data["b"],
            "c": data["c"],
            "gt_r1": data["r1"],
            "gt_r2": data["r2"],
            "user_prompt": data["context"],
            "model_output": response,
            "reward": r["reward"],
            "math_reward": info.get("math_reward", 0.0),
            "format_reward": info.get("format_reward", 0.0),
            "reasoning_reward": info.get("reasoning_reward", 0.0),
            "deduction": info.get("deduction", 0.0),
            "both_exact": info.get("both_exact", False),
            "one_root": info.get("one_root", False),
            "parse_fail": info.get("parse_fail", False),
        }
        row.update(fm)
        csv_rows.append(row)

    n = len(eval_task_datas)
    mean_reward = float(np.mean(all_rewards))
    both_rate = both_exact_count / n
    one_rate = one_root_count / n
    fail_rate = parse_fail_count / n
    valid_clean_rate = valid_clean_count / n
    r1_rate = root1_correct_count / n
    r2_rate = root2_correct_count / n
    sum_rate = sum_ok_count / n
    prod_rate = prod_ok_count / n

    # TensorBoard — core metrics
    writer.add_scalar("eval/mean_reward", mean_reward, iteration)
    writer.add_scalar("eval/both_exact_rate", both_rate, iteration)
    writer.add_scalar("eval/one_root_rate", one_rate, iteration)
    writer.add_scalar("eval/parse_fail_rate", fail_rate, iteration)

    # TensorBoard — format accuracy
    writer.add_scalar("eval_format/valid_clean_rate", valid_clean_rate, iteration)
    writer.add_scalar("eval_format/strict_ok_rate", strict_ok_count / n, iteration)
    writer.add_scalar("eval_format/bad_tags_rate", bad_tags_count / n, iteration)
    writer.add_scalar("eval_format/multiple_boxed_rate", multiple_boxed_count / n, iteration)
    writer.add_scalar("eval_format/trailing_junk_rate", trailing_junk_count / n, iteration)

    writer.add_scalars("Eval Format Accuracy", {
        "Valid Clean": valid_clean_rate,
        "Bad Tags": bad_tags_count / n,
        "Multiple Boxed": multiple_boxed_count / n,
        "Trailing Junk": trailing_junk_count / n,
    }, iteration)

    # TensorBoard — math correctness
    writer.add_scalar("eval_math/root1_correct_rate", r1_rate, iteration)
    writer.add_scalar("eval_math/root2_correct_rate", r2_rate, iteration)
    writer.add_scalar("eval_math/sum_ok_rate", sum_rate, iteration)
    writer.add_scalar("eval_math/prod_ok_rate", prod_rate, iteration)

    writer.add_scalars("Eval Math Correctness", {
        "Root 1 Correct": r1_rate,
        "Root 2 Correct": r2_rate,
        "Sum (Vieta) OK": sum_rate,
        "Product (Vieta) OK": prod_rate,
    }, iteration)

    # CSV
    _append_eval_csv(eval_csv_path, csv_rows)

    print(f"[EVAL] iter={iteration} | reward={mean_reward:.4f} | "
          f"both_exact={both_rate*100:.1f}% | format_clean={valid_clean_rate*100:.1f}% | "
          f"math_r1={r1_rate*100:.1f}% r2={r2_rate*100:.1f}% | "
          f"parse_fail={fail_rate*100:.1f}% | {elapsed:.1f}s")

    return mean_reward


# ── Default Hyperparameters ──────────────────────────────────────────────────

SIGMA = 0.001           # noise standard deviation for perturbation
ALPHA = 0.0005          # learning rate for ES update
POPULATION_SIZE = 30    # number of perturbation seeds per iteration
NUM_ENGINES = 4         # number of vLLM engines (one per GPU)
NUM_ITERATIONS = 500   # total ES iterations
EXPERIMENT_DIR = "es-ft-quadratic-experiment"
DATA_SAMPLE = 200       # number of dataset samples to evaluate per seed

# ── System prompt matching grpo_quad_train_v12 ──────────────────────────────

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="ES Fine-tuning for Quadratic Task with multi-engine NCCL sync"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--data_sample", type=int, default=DATA_SAMPLE,
                        help="Number of dataset samples to evaluate per perturbation")
    parser.add_argument("--data_path", type=str, default="",
                        help="Path to parquet dataset; default uses Dataset/quad_medhard_train.parquet")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens for vLLM generation")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],
                        help="Model dtype for vLLM engines")
    parser.add_argument("--global_seed", type=int, help="Global random seed")
    parser.add_argument("--csv_every_iters", type=int, default=50,
                        help="Write training CSV every N iterations; 0 disables")
    parser.add_argument("--csv_sample_n", type=int, default=0,
                        help="Max samples to log per CSV write; 0 logs all")
    parser.add_argument("--eval_data", type=str, default="",
                        help="Path to eval parquet for periodic evaluation during training")
    parser.add_argument("--eval_every_iters", type=int, default=50,
                        help="Run evaluation every N iterations; 0 disables")
    args = parser.parse_args()

    # Scope host GPU visibility; vLLM actors use placement groups for device assignment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Set global random seed
    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)

    return args


# ── vLLM engine subclass ────────────────────────────────────────────────────

class ESNcclLLM(LLM):
    """LLM subclass that unsets CUDA_VISIBLE_DEVICES so Ray PG controls device."""
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


def launch_engines(num_engines, model_name, dtype="float16"):
    """Launch num_engines vLLM instances, each on its own GPU via placement groups."""
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype=dtype,
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for strategy in strategies
    ]
    return engines, pgs


# ── Dataset loading ─────────────────────────────────────────────────────────

def _normalize_equation_text(eq):
    t = str(eq) if eq is not None else ""
    t = t.replace("\u2212", "-").replace("\u2013", "-")
    t = t.replace("\u00b2", "^2")
    t = t.replace("\u00d7", "*")
    t = t.replace("\u00A0", " ")
    return t


def load_quadratic_data(data_path, tokenizer, data_sample=200):
    """
    Load quadratic dataset from parquet and build prompts.

    Returns list of dicts with keys: context (prompt string), a, b, c, r1, r2, equation
    """
    if not data_path:
        # Default to the medhard train dataset
        candidates = [
            os.path.join(os.path.dirname(__file__), "Dataset", "quad_medhard_train.parquet"),
            os.path.join(os.path.dirname(__file__), "Dataset", "quad_diverse_train.parquet"),
        ]
        data_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    print(f"Loading dataset from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"  Total rows: {len(df)}")

    task_datas = []
    for _, row in df.iterrows():
        equation = _normalize_equation_text(row.get("equation", ""))
        user_message = row.get("user_message", "")
        if not user_message:
            user_message = f"Solve the quadratic equation for integer roots: {equation}"

        # Build chat prompt using tokenizer template
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        if getattr(tokenizer, "chat_template", None):
            context = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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
        })

    # Subsample if needed
    if data_sample > 0 and len(task_datas) > data_sample:
        random.shuffle(task_datas)
        task_datas = task_datas[:data_sample]
        print(f"  Subsampled to {len(task_datas)} rows")

    return task_datas


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_quadratic_handle(llm, task_datas, max_tokens=1024):
    """Submit async generation for all prompts on one engine."""
    prompts = [d["context"] for d in task_datas]
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=max_tokens,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, time.time()


def _postprocess_outputs(outputs, task_datas):
    """Compute rewards for all generated outputs."""
    rewards = []
    avg_rewards = []
    responses = []
    both_exact_count = 0
    one_root_count = 0
    parse_fail_count = 0
    # Format accuracy counters
    valid_clean_count = 0
    strict_ok_count = 0
    bad_tags_count = 0
    multiple_boxed_count = 0
    trailing_junk_count = 0
    # Math correctness counters
    root1_correct_count = 0
    root2_correct_count = 0
    sum_ok_count = 0
    prod_ok_count = 0
    fmt_math_fields = []

    for output, data in zip(outputs, task_datas):
        response = output.outputs[0].text
        responses.append(response)
        r = reward_function(
            response,
            a=data["a"],
            b=data["b"],
            c=data["c"],
            r1_gt=data["r1"],
            r2_gt=data["r2"],
        )
        rewards.append(r)
        avg_rewards.append(r["reward"])

        info = r.get("reward_info", {})
        if info.get("both_exact"):
            both_exact_count += 1
        elif info.get("one_root"):
            one_root_count += 1
        if info.get("parse_fail"):
            parse_fail_count += 1

        fm = _compute_format_math_fields(response, data["a"], data["b"], data["c"])
        fmt_math_fields.append(fm)
        if fm["format_valid_clean"]:
            valid_clean_count += 1
        if fm["format_strict_ok"]:
            strict_ok_count += 1
        if fm["format_bad_tags"]:
            bad_tags_count += 1
        if fm["format_multiple_boxed"]:
            multiple_boxed_count += 1
        if fm["format_trailing_junk"]:
            trailing_junk_count += 1
        if fm["math_root1_correct"]:
            root1_correct_count += 1
        if fm["math_root2_correct"]:
            root2_correct_count += 1
        if fm["math_sum_ok"]:
            sum_ok_count += 1
        if fm["math_prod_ok"]:
            prod_ok_count += 1

    n = len(task_datas) or 1

    # Aggregate reward components across samples
    math_rewards = [r.get("reward_info", {}).get("math_reward", 0.0) for r in rewards]
    format_rewards = [r.get("reward_info", {}).get("format_reward", 0.0) for r in rewards]
    reasoning_rewards = [r.get("reward_info", {}).get("reasoning_reward", 0.0) for r in rewards]
    deductions = [r.get("reward_info", {}).get("deduction", 0.0) for r in rewards]

    return {
        "rewards": rewards,
        "responses": responses,
        "fmt_math_fields": fmt_math_fields,
        "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
        "both_exact_rate": both_exact_count / n,
        "one_root_rate": one_root_count / n,
        "parse_fail_rate": parse_fail_count / n,
        "avg_math_reward": float(np.mean(math_rewards)),
        "avg_format_reward": float(np.mean(format_rewards)),
        "avg_reasoning_reward": float(np.mean(reasoning_rewards)),
        "avg_deduction": float(np.mean(deductions)),
        # Format accuracy rates
        "format_valid_clean_rate": valid_clean_count / n,
        "format_strict_ok_rate": strict_ok_count / n,
        "format_bad_tags_rate": bad_tags_count / n,
        "format_multiple_boxed_rate": multiple_boxed_count / n,
        "format_trailing_junk_rate": trailing_junk_count / n,
        # Math correctness rates
        "math_root1_correct_rate": root1_correct_count / n,
        "math_root2_correct_rate": root2_correct_count / n,
        "math_sum_ok_rate": sum_ok_count / n,
        "math_prod_ok_rate": prod_ok_count / n,
    }


# ── Main training loop ─────────────────────────────────────────────────────

def main(args):
    # Ensure local Ray (no cluster)
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    # Logging
    logging_dir = f"{args.experiment_dir}/quadratic_nccl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logging_dir)

    # Prepare HF checkpoint for vLLM to load
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    print(f"Loading base model: {args.model_name}")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype_map[args.dtype]
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    base_model_path = f"{model_saves_dir}/base_model"
    if os.path.exists(base_model_path):
        shutil.rmtree(base_model_path)
    os.makedirs(base_model_path, exist_ok=True)
    tokenizer.save_pretrained(base_model_path)
    base_model.save_pretrained(base_model_path)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load quadratic data
    task_datas = load_quadratic_data(args.data_path, tokenizer, args.data_sample)
    print(f"Evaluating on {len(task_datas)} quadratic problems per perturbation")

    # Launch vLLM engines
    print(f"Launching {args.num_engines} vLLM engines...")
    engines, pgs = launch_engines(args.num_engines, base_model_path, dtype=args.dtype)

    # Init inter-engine NCCL communicator
    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])
    print("NCCL inter-engine communication initialized")

    def cleanup():
        for llm in engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        for pg in pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()

    def sig_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Save config for reproducibility
    config = {
        "model_name": args.model_name,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "population_size": args.population_size,
        "num_engines": args.num_engines,
        "num_iterations": args.num_iterations,
        "data_sample": args.data_sample,
        "max_tokens": args.max_tokens,
        "dtype": args.dtype,
        "global_seed": args.global_seed,
        "eval_data": args.eval_data,
        "eval_every_iters": args.eval_every_iters,
        "task": "quadratic_integer_roots",
    }
    with open(f"{logging_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # CSV setup
    train_csv_path = os.path.join(logging_dir, "train_generations.csv")
    csv_every = max(0, args.csv_every_iters)
    csv_sample_n = args.csv_sample_n
    if csv_every > 0:
        print(f"CSV logging enabled: every {csv_every} iters -> {train_csv_path}")

    # Eval setup
    eval_task_datas = None
    eval_csv_path = os.path.join(logging_dir, "eval_generations.csv")
    eval_every = max(0, args.eval_every_iters)
    if args.eval_data and os.path.exists(args.eval_data):
        eval_task_datas = load_quadratic_data(args.eval_data, tokenizer, data_sample=0)
        print(f"Eval data loaded: {len(eval_task_datas)} samples from {args.eval_data}")
        print(f"Eval CSV: every {eval_every} iters -> {eval_csv_path}")
    elif args.eval_data:
        print(f"WARNING: eval_data path not found: {args.eval_data}")

    print(f"\nConfig saved to {logging_dir}/config.json")
    print(f"TensorBoard logs: {logging_dir}")
    print(f"\n{'='*80}")
    print(f"Starting ES fine-tuning: {args.num_iterations} iterations, "
          f"pop_size={args.population_size}, sigma={args.sigma}, alpha={args.alpha}")
    print(f"{'='*80}\n")

    # ── ES training loop ────────────────────────────────────────────────────
    best_reward = -float("inf")

    for i in range(args.num_iterations):
        print(f"\n\n=== Iteration {i} ===")
        total_iter_start = time.time()

        # Random seeds for population
        seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
        seeds_perf = {}

        # Round-robin scheduling across engines
        seed_iter = iter(seeds)
        inflight = {}
        results_this_gen = []

        # Kick off initial eval on each engine
        for eng_idx, llm in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            # Add exploration noise
            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(seed, args.sigma, False)))
            handle, start_ts = evaluate_quadratic_handle(llm, task_datas, args.max_tokens)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": eng_idx,
                "seed": seed,
                "start_ts": start_ts,
            }

        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            metrics = _postprocess_outputs(outputs, task_datas)
            elapsed = time.time() - meta["start_ts"]

            seeds_perf[meta["seed"]] = metrics
            results_this_gen.append(
                {"seed": meta["seed"], "avg_reward": metrics["avg_reward"], "time": elapsed}
            )

            llm = meta["engine"]
            # Remove exploration noise (restore original weights)
            ray.get(llm.collective_rpc.remote("restore_self_weights", args=(meta["seed"], args.sigma)))

            # Schedule next seed on this engine
            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue

            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(next_seed, args.sigma, False)))
            handle, start_ts = evaluate_quadratic_handle(llm, task_datas, args.max_tokens)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": meta["engine_idx"],
                "seed": next_seed,
                "start_ts": start_ts,
            }
            if args.verbose:
                print(f"Scheduled seed {next_seed} on engine {meta['engine_idx']}")

        # Normalize rewards across population
        all_avg_rewards = [v["avg_reward"] for v in seeds_perf.values()]
        mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
        std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
        min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
        max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0

        # Aggregate task-specific metrics
        avg_both_exact = float(np.mean([v["both_exact_rate"] for v in seeds_perf.values()]))
        avg_one_root = float(np.mean([v["one_root_rate"] for v in seeds_perf.values()]))
        avg_parse_fail = float(np.mean([v["parse_fail_rate"] for v in seeds_perf.values()]))

        print(f"Mean reward: {mean_reward:.4f}, std: {std_reward:.4f}, "
              f"min: {min_reward:.4f}, max: {max_reward:.4f}")
        print(f"Avg both_exact: {avg_both_exact:.3f}, one_root: {avg_one_root:.3f}, "
              f"parse_fail: {avg_parse_fail:.3f}")

        for k in seeds_perf:
            seeds_perf[k]["norm_reward"] = (seeds_perf[k]["avg_reward"] - mean_reward) / (std_reward + 1e-8)
            if args.verbose:
                print(f"Seed {k} normalized reward: {seeds_perf[k]['norm_reward']:.4f}")

        # ── TensorBoard logging ──────────────────────────────────────
        # Individual scalars (for clean single-line plots)
        writer.add_scalar("reward/mean", mean_reward, i)
        writer.add_scalar("reward/std", std_reward, i)
        writer.add_scalar("reward/min", min_reward, i)
        writer.add_scalar("reward/max", max_reward, i)
        writer.add_scalar("task/both_exact_rate", avg_both_exact, i)
        writer.add_scalar("task/one_root_rate", avg_one_root, i)
        writer.add_scalar("task/parse_fail_rate", avg_parse_fail, i)

        if mean_reward > best_reward:
            best_reward = mean_reward
            writer.add_scalar("reward/best", best_reward, i)

        # Grouped scalars (multi-line charts with legends in TensorBoard)
        writer.add_scalars("Reward Overview", {
            "Mean Reward": mean_reward,
            "Best Reward": best_reward,
            "Min Reward": min_reward,
            "Max Reward": max_reward,
        }, i)

        writer.add_scalars("Accuracy Rates", {
            "Both Roots Exact": avg_both_exact,
            "One Root Only": avg_one_root,
            "Parse Fail": avg_parse_fail,
        }, i)

        # Reward component breakdown (averaged across seeds)
        avg_math = float(np.mean([v["avg_math_reward"] for v in seeds_perf.values()]))
        avg_fmt = float(np.mean([v["avg_format_reward"] for v in seeds_perf.values()]))
        avg_reas = float(np.mean([v["avg_reasoning_reward"] for v in seeds_perf.values()]))
        avg_ded = float(np.mean([v["avg_deduction"] for v in seeds_perf.values()]))

        writer.add_scalar("components/math_reward", avg_math, i)
        writer.add_scalar("components/format_reward", avg_fmt, i)
        writer.add_scalar("components/reasoning_reward", avg_reas, i)
        writer.add_scalar("components/deduction", avg_ded, i)

        writer.add_scalars("Reward Components", {
            "Math": avg_math,
            "Format": avg_fmt,
            "Reasoning": avg_reas,
            "Deduction (penalty)": avg_ded,
        }, i)

        # Format accuracy metrics (averaged across seeds)
        avg_valid_clean = float(np.mean([v["format_valid_clean_rate"] for v in seeds_perf.values()]))
        avg_strict_ok = float(np.mean([v["format_strict_ok_rate"] for v in seeds_perf.values()]))
        avg_bad_tags = float(np.mean([v["format_bad_tags_rate"] for v in seeds_perf.values()]))
        avg_multi_boxed = float(np.mean([v["format_multiple_boxed_rate"] for v in seeds_perf.values()]))
        avg_trailing = float(np.mean([v["format_trailing_junk_rate"] for v in seeds_perf.values()]))

        writer.add_scalar("format/valid_clean_rate", avg_valid_clean, i)
        writer.add_scalar("format/strict_ok_rate", avg_strict_ok, i)
        writer.add_scalar("format/bad_tags_rate", avg_bad_tags, i)
        writer.add_scalar("format/multiple_boxed_rate", avg_multi_boxed, i)
        writer.add_scalar("format/trailing_junk_rate", avg_trailing, i)

        writer.add_scalars("Format Accuracy", {
            "Valid Clean": avg_valid_clean,
            "Strict OK": avg_strict_ok,
            "Bad Tags": avg_bad_tags,
            "Multiple Boxed": avg_multi_boxed,
            "Trailing Junk": avg_trailing,
        }, i)

        # Math correctness metrics (averaged across seeds)
        avg_r1_ok = float(np.mean([v["math_root1_correct_rate"] for v in seeds_perf.values()]))
        avg_r2_ok = float(np.mean([v["math_root2_correct_rate"] for v in seeds_perf.values()]))
        avg_sum_ok = float(np.mean([v["math_sum_ok_rate"] for v in seeds_perf.values()]))
        avg_prod_ok = float(np.mean([v["math_prod_ok_rate"] for v in seeds_perf.values()]))

        writer.add_scalar("math/root1_correct_rate", avg_r1_ok, i)
        writer.add_scalar("math/root2_correct_rate", avg_r2_ok, i)
        writer.add_scalar("math/sum_ok_rate", avg_sum_ok, i)
        writer.add_scalar("math/prod_ok_rate", avg_prod_ok, i)

        writer.add_scalars("Math Correctness", {
            "Root 1 Correct": avg_r1_ok,
            "Root 2 Correct": avg_r2_ok,
            "Sum (Vieta) OK": avg_sum_ok,
            "Product (Vieta) OK": avg_prod_ok,
        }, i)

        # CSV logging: write per-sample outputs from the best seed this iteration
        if csv_every > 0 and (i % csv_every == 0 or i == args.num_iterations - 1):
            best_seed = max(seeds_perf.keys(), key=lambda s: seeds_perf[s]["avg_reward"])
            best_metrics = seeds_perf[best_seed]
            csv_rows = []
            n_log = len(task_datas)
            if csv_sample_n > 0:
                n_log = min(csv_sample_n, n_log)
            for si in range(n_log):
                info = best_metrics["rewards"][si].get("reward_info", {})
                fm = best_metrics["fmt_math_fields"][si]
                row = {
                    "iteration": i,
                    "seed": best_seed,
                    "sample_idx": si,
                    "system_prompt": SYSTEM_PROMPT,
                    "equation": task_datas[si]["equation"],
                    "a": task_datas[si]["a"],
                    "b": task_datas[si]["b"],
                    "c": task_datas[si]["c"],
                    "gt_r1": task_datas[si]["r1"],
                    "gt_r2": task_datas[si]["r2"],
                    "user_prompt": task_datas[si]["context"],
                    "model_output": best_metrics["responses"][si],
                    "reward": best_metrics["rewards"][si]["reward"],
                    "math_reward": info.get("math_reward", 0.0),
                    "format_reward": info.get("format_reward", 0.0),
                    "reasoning_reward": info.get("reasoning_reward", 0.0),
                    "deduction": info.get("deduction", 0.0),
                    "both_exact": info.get("both_exact", False),
                    "one_root": info.get("one_root", False),
                    "parse_fail": info.get("parse_fail", False),
                }
                row.update(fm)
                csv_rows.append(row)
            _append_training_csv(train_csv_path, csv_rows)
            print(f"[CSV] wrote {len(csv_rows)} rows at iter={i} (best seed={best_seed}) -> {train_csv_path}")

        # Periodic evaluation on held-out eval set
        if eval_task_datas and eval_every > 0 and (i % eval_every == 0 or i == args.num_iterations - 1):
            _run_periodic_eval(engines[0], eval_task_datas, i, args.max_tokens, eval_csv_path, writer)

        # Compute ES update ONLY on engine 0
        per_seed_coeffs = [
            (seed, (args.alpha / args.population_size) * float(seeds_perf[seed]["norm_reward"]))
            for seed in seeds
        ]

        perturb_start = time.time()
        handles = []
        for seed, coeff in per_seed_coeffs:
            handles.append(engines[0].collective_rpc.remote(
                "perturb_self_weights", args=(seed, coeff, False)
            ))
        ray.get(handles)
        if args.verbose:
            print(f"Applied perturbations in {time.time() - perturb_start:.2f}s")
        writer.add_scalar("time/perturbation_application", time.time() - perturb_start, i)

        # Broadcast updated weights from engine 0 to all engines via NCCL
        broadcast_start = time.time()
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])
        if args.verbose:
            print(f"Broadcasted updated weights in {time.time() - broadcast_start:.2f}s")
        writer.add_scalar("time/broadcast", time.time() - broadcast_start, i)

        # Per-result logging
        if args.verbose:
            for res_idx, res in enumerate(results_this_gen):
                print(f"IDX:{res_idx} Seed {res['seed']} avg_reward: {res['avg_reward']:.4f}, "
                      f"time: {res['time']:.2f}s")

        total_iter_end = time.time()
        t_perturb = time.time() - perturb_start
        t_broadcast = time.time() - broadcast_start
        t_total = total_iter_end - total_iter_start
        writer.add_scalar("time/iteration", t_total, i)

        writer.add_scalars("Timing Breakdown (s)", {
            "Total Iteration": t_total,
            "Perturbation": t_perturb,
            "Broadcast": t_broadcast,
        }, i)
        print(f"Wall clock time for iteration {i}: {t_total:.2f}s")

        # Periodic checkpointing
        if (i + 1) % 100 == 0 or i == args.num_iterations - 1:
            ckpt_path = f"{model_saves_dir}/checkpoint_iter_{i+1}"
            os.makedirs(ckpt_path, exist_ok=True)
            ray.get(
                engines[0].collective_rpc.remote(
                    "save_self_weights_to_disk", args=(f"{ckpt_path}/pytorch_model.pth",)
                )
            )
            print(f"Checkpoint saved: {ckpt_path}")
            # Keep only the latest checkpoint — delete all older ones
            for old_ckpt in sorted(glob.glob(f"{model_saves_dir}/checkpoint_iter_*")):
                if old_ckpt != ckpt_path:
                    shutil.rmtree(old_ckpt)
                    print(f"Deleted old checkpoint: {old_ckpt}")

        print(f"=== Iteration {i} finished ===\n")

    # Save final model weights
    final_model_path = f"{model_saves_dir}/final_model_iteration_{args.num_iterations}"
    os.makedirs(final_model_path, exist_ok=True)
    ray.get(
        engines[0].collective_rpc.remote(
            "save_self_weights_to_disk", args=(f"{final_model_path}/pytorch_model.pth",)
        )
    )
    # Also save tokenizer alongside final weights
    tokenizer.save_pretrained(final_model_path)
    print(f"\nFinal model weights saved to {final_model_path}")
    print(f"Best mean reward achieved: {best_reward:.4f}")

    writer.close()
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
