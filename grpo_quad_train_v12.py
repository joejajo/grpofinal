#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO (Countdown/R1-style) for Quadratic Integer Roots  [v12]
Compatible with: TRL 0.24.x + vLLM 0.10.2

v12 changes vs v11:
  - Targets medium+hard only dataset (quad_medhard_train.parquet / quad_medhard_eval.parquet)
    * No easy zone (|a|=1, roots [-5,5]) — model already saturated those in v11
    * Medium: |a| in {1,2}, roots in [-9,9]   (~760 samples)
    * Hard:   |a| in {1,2,3,4}, roots in [-15,15] (~1800 samples)
    * Total train: 2560 rows; eval: 500 rows (|a| up to 5, held-out templates)
  - max_steps raised 1500 -> 2000 (harder problems need more training signal)
  - num_generations raised 8 -> 12 (wider group for better advantage estimation)
  - per_device_batch_size raised 1 -> 2 so effective=32, prompts/step=32/12~=2
  - temperature raised 0.70 -> 0.80 (harder problems need more diversity to avoid zero-std collapse)
  - learning_rate lowered 2e-6 -> 1.5e-6 (conservative; harder dataset = noisier gradients)
  - warmup_steps raised 60 -> 100 (longer warmup for harder task)
  - beta raised 0.001 -> 0.002 (stronger KL anchor on harder task)
  - lora_r raised 32 -> 48, lora_alpha 64 -> 96 (more capacity for harder arithmetic)

Dataset columns required:
  equation, a, b, c, r1, r2
(we create/overwrite column "prompt")

Single combined reward in [0,1]:
  - math (0.0 / 0.40 / 0.82) via substitution + Vieta shaping
  - format bonus gated by correctness (prevents farming format with wrong roots)
  - reasoning bonus only when math >= 0.40
  - penalties for XML tags, multiple boxed pairs, trailing junk, excessive length
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import statistics
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger("quad_grpo")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# -----------------------
# Debug globals (set from args)
# -----------------------
REWARD_CALLS = 0
DEBUG_EVERY = 0
DEBUG_SHOW_N = 8
DEBUG_MAX_CHARS = 0          # 0 => print FULL completion
DEBUG_PRINT_PROMPT = True
DEBUG_PRINT_EQUATION = True
DEBUG_PRINT_EFFECTIVE = True

# Reasoning control (set from args)
MIN_THINK_CHARS = 80         # minimum reasoning characters before the final boxed answer (v11: 120->80)
REASONING_KEYWORDS = (
    "discriminant",
    "delta",
    "sqrt",
    "quadratic formula",
    "factor",
    "vieta",
    "b^2-4ac",
    "+/-",
    # v11 additions
    "formula",
    "x =",
    "solve",
    "coefficient",
    "product of roots",
    "sum of roots",
    "roots are",
)

LAST_REWARD_STATS: Dict[str, float] = {}
LAST_EVAL_STATS: Dict[str, float] = {}
GEN_CSV_PATH = ""
GEN_CSV_EVERY = 50
GEN_CSV_LAST_STEP = -1
EVAL_CSV_PATH = ""
def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0

class RewardBreakdownCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not LAST_REWARD_STATS:
            return
        logs["rewards/equation_reward_func"] = LAST_REWARD_STATS.get("equation_reward_func", 0.0)
        logs["rewards/format_reward_func"] = LAST_REWARD_STATS.get("format_reward_func", 0.0)
        logs["rewards/reasoning_reward_func"] = LAST_REWARD_STATS.get("reasoning_reward_func", 0.0)
        logs["rewards/penalty_term"] = LAST_REWARD_STATS.get("penalty_term", 0.0)
        logs["rewards/parse_fail_rate"] = LAST_REWARD_STATS.get("parse_fail_rate", 0.0)
        logs["rewards/both_roots_rate"] = LAST_REWARD_STATS.get("both_roots_rate", 0.0)
        logs["rewards/one_root_rate"] = LAST_REWARD_STATS.get("one_root_rate", 0.0)
        logs["completion_length_chars"] = LAST_REWARD_STATS.get("completion_length_chars", 0.0)
        for k, v in LAST_EVAL_STATS.items():
            logs[k] = v

class ExactAccuracyEvalCallback(TrainerCallback):
    """
    Periodic exact-eval on held-out parquet split.
    Logs metrics into TensorBoard via Trainer's regular on_log flow.
    """
    def __init__(
        self,
        eval_ds,
        tokenizer,
        eval_every_steps: int,
        eval_batch_size: int,
        eval_max_samples: int,
        max_prompt_length: int,
        max_new_tokens: int,
        eval_csv_path: str = "",
    ):
        self.eval_ds = eval_ds
        self.tokenizer = tokenizer
        self.eval_every_steps = max(0, int(eval_every_steps))
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.eval_max_samples = int(eval_max_samples)
        self.max_prompt_length = int(max_prompt_length)
        self.max_new_tokens = int(max_new_tokens)
        self.eval_csv_path = eval_csv_path
        self._last_run_step = -1

    def _compute_exact_metrics(self, model, step: int = 0) -> Dict[str, float]:
        total_rows = len(self.eval_ds)
        if total_rows == 0:
            return {
                "eval/exact_both_roots_accuracy": 0.0,
                "eval/one_root_rate": 0.0,
                "eval/parse_fail_rate": 1.0,
                "eval/gt_match_rate": 0.0,
                "eval/samples": 0.0,
            }

        n = total_rows if self.eval_max_samples <= 0 else min(total_rows, self.eval_max_samples)
        exact = 0
        one = 0
        parse_fail = 0
        gt_match = 0
        seen = 0
        csv_rows: List[Dict[str, Any]] = []

        was_training = model.training
        device = next(model.parameters()).device
        model.eval()
        try:
            with torch.inference_mode():
                for start in range(0, n, self.eval_batch_size):
                    end = min(n, start + self.eval_batch_size)
                    batch = self.eval_ds[start:end]
                    prompts = batch["prompt"]

                    enc = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_prompt_length,
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}

                    out = generate_with_answer_stop(
                        model=model,
                        tokenizer=self.tokenizer,
                        enc=enc,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )

                    prompt_len = int(enc["input_ids"].shape[1])
                    comp_ids = out[:, prompt_len:]
                    completions = self.tokenizer.batch_decode(comp_ids, skip_special_tokens=True)
                    completions = [truncate_after_answer(x) for x in completions]

                    for i, txt in enumerate(completions):
                        ai = safe_int(batch["a"][i]) if "a" in batch else None
                        bi = safe_int(batch["b"][i]) if "b" in batch else None
                        ci = safe_int(batch["c"][i]) if "c" in batch else None
                        gt1 = safe_int(batch["r1"][i]) if "r1" in batch else None
                        gt2 = safe_int(batch["r2"][i]) if "r2" in batch else None

                        # build eval CSV row defaults
                        csv_row: Dict[str, Any] = {
                            "step": step,
                            "prompt": str(prompts[i]),
                            "model_output": txt,
                            "final_boxed": "",
                            "gt_r1": gt1, "gt_r2": gt2,
                            "parsed_r1": None, "parsed_r2": None,
                            "both_exact": False, "one_root": False,
                            "parse_fail": False, "gt_match": False,
                        }

                        if ai is None or bi is None or ci is None:
                            parse_fail += 1
                            seen += 1
                            csv_row["parse_fail"] = True
                            csv_rows.append(csv_row)
                            continue

                        parsed = parse_roots_from_answer(txt)
                        if parsed is None:
                            parse_fail += 1
                            seen += 1
                            csv_row["parse_fail"] = True
                            csv_rows.append(csv_row)
                            continue

                        pr1, pr2, _src, _raw = parsed
                        ok1, ok2, _res1, _res2 = verify_roots(ai, bi, ci, pr1, pr2)
                        sum_ok = (ai * (pr1 + pr2) + bi == 0)
                        prod_ok = (ai * pr1 * pr2 - ci == 0)
                        both_exact = bool(ok1 and ok2 and sum_ok and prod_ok)
                        one_root = bool((ok1 or ok2) and not both_exact)
                        if both_exact:
                            exact += 1
                        elif ok1 or ok2:
                            one += 1

                        is_gt_match = False
                        if gt1 is not None and gt2 is not None:
                            if (pr1, pr2) == (min(gt1, gt2), max(gt1, gt2)):
                                gt_match += 1
                                is_gt_match = True

                        csv_row.update({
                            "final_boxed": f"\\boxed{{{pr1}, {pr2}}}",
                            "parsed_r1": pr1, "parsed_r2": pr2,
                            "both_exact": both_exact, "one_root": one_root,
                            "parse_fail": False, "gt_match": is_gt_match,
                        })
                        csv_rows.append(csv_row)
                        seen += 1
        finally:
            if was_training:
                model.train()

        if self.eval_csv_path and csv_rows and is_rank0():
            _append_eval_csv(self.eval_csv_path, csv_rows)

        denom = float(max(1, seen))
        return {
            "eval/exact_both_roots_accuracy": float(exact / denom),
            "eval/one_root_rate": float(one / denom),
            "eval/parse_fail_rate": float(parse_fail / denom),
            "eval/gt_match_rate": float(gt_match / denom),
            "eval/samples": float(seen),
        }

    def on_step_end(self, args, state, control, **kwargs):
        global LAST_EVAL_STATS
        if self.eval_every_steps <= 0:
            return
        if state.global_step <= 0:
            return
        if state.global_step % self.eval_every_steps != 0:
            return
        if state.global_step == self._last_run_step:
            return
        if not is_rank0():
            return

        model = kwargs.get("model", None)
        if model is None:
            return

        self._last_run_step = int(state.global_step)
        metrics = self._compute_exact_metrics(model, step=int(state.global_step))
        LAST_EVAL_STATS = metrics
        log.info(
            "[EVAL] step=%d exact=%.4f one=%.4f parse_fail=%.4f gt_match=%.4f n=%d",
            state.global_step,
            metrics.get("eval/exact_both_roots_accuracy", 0.0),
            metrics.get("eval/one_root_rate", 0.0),
            metrics.get("eval/parse_fail_rate", 0.0),
            metrics.get("eval/gt_match_rate", 0.0),
            int(metrics.get("eval/samples", 0.0)),
        )

# -----------------------
# Helpers
# -----------------------
def is_rank0() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0

def get_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x and isinstance(x[-1], dict) and "content" in x[-1]:
        return str(x[-1]["content"])
    if isinstance(x, dict) and "content" in x:
        return str(x["content"])
    return ""

def normalize_equation_text(eq: Any) -> str:
    t = str(eq) if eq is not None else ""
    t = t.replace("\u2212", "-").replace("\u2013", "-")
    t = t.replace("\u00b2", "^2")
    t = t.replace("\u00d7", "*")
    t = t.replace("\u00A0", " ")
    return t

def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def col_get(col, i):
    if col is None:
        return None
    try:
        return col[i]
    except Exception:
        return col

# -----------------------
# Regex / Parsing
# -----------------------
BOXED_PAIR_RE = re.compile(
    r"\\boxed\{\s*\(?\s*([-+]?\d+)(?:\.0+)?\s*,\s*([-+]?\d+)(?:\.0+)?\s*\)?\s*\}"
)
ROOT_LABEL_PAIR_RE = re.compile(r"r1\s*=\s*([-+]?\d+)(?!\.\d).{0,80}?r2\s*=\s*([-+]?\d+)(?!\.\d)", re.IGNORECASE | re.DOTALL)
ROOT_LABEL_PAIR_RE_REV = re.compile(r"r2\s*=\s*([-+]?\d+)(?!\.\d).{0,80}?r1\s*=\s*([-+]?\d+)(?!\.\d)", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
ESCAPED_TAG_RE = re.compile(r"\\</?[a-zA-Z][^>]*>")

def truncate_after_answer(txt: str) -> str:
    s = txt or ""
    ms = list(BOXED_PAIR_RE.finditer(s))
    if not ms:
        return s
    return s[: ms[-1].end()]

def _build_generate_stop_kwargs(model, tokenizer) -> Tuple[Dict[str, Any], str]:
    kwargs: Dict[str, Any] = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    return kwargs, "none"

def generate_with_answer_stop(model, tokenizer, enc: Dict[str, torch.Tensor], max_new_tokens: int, do_sample: bool):
    stop_kwargs, _stop_mode = _build_generate_stop_kwargs(model, tokenizer)
    try:
        return model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **stop_kwargs,
        )
    except TypeError:
        return model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

def parse_roots_from_answer(txt: str) -> Optional[Tuple[int, int, str, Tuple[int, int]]]:
    """
    Strict parser for the target format: \boxed{r1, r2}
    Returns: (sorted_r1, sorted_r2, src, raw_pair)
    """
    ms = list(BOXED_PAIR_RE.finditer(txt or ""))
    if not ms:
        return None
    r1_raw = int(ms[-1].group(1))
    r2_raw = int(ms[-1].group(2))
    return (min(r1_raw, r2_raw), max(r1_raw, r2_raw), "boxed_pair", (r1_raw, r2_raw))

def parse_roots_from_labeled_text(txt: str) -> Optional[Tuple[int, int, str, Tuple[int, int]]]:
    """Fallback parser for outputs like 'r1 = -5, r2 = 10' without boxed format."""
    ms = list(ROOT_LABEL_PAIR_RE.finditer(txt))
    if ms:
        m = ms[-1]
        r1_raw = int(m.group(1))
        r2_raw = int(m.group(2))
        return (min(r1_raw, r2_raw), max(r1_raw, r2_raw), "label_pair_text", (r1_raw, r2_raw))

    ms2 = list(ROOT_LABEL_PAIR_RE_REV.finditer(txt))
    if ms2:
        m2 = ms2[-1]
        r2_raw = int(m2.group(1))
        r1_raw = int(m2.group(2))
        return (min(r1_raw, r2_raw), max(r1_raw, r2_raw), "label_pair_text", (r1_raw, r2_raw))

    return None

def verify_roots(a: int, b: int, c: int, r1: int, r2: int) -> Tuple[bool, bool, int, int]:
    res1 = a * r1 * r1 + b * r1 + c
    res2 = a * r2 * r2 + b * r2 + c
    return (res1 == 0), (res2 == 0), res1, res2

def format_grade(txt: str) -> Tuple[bool, bool, bool, bool]:
    """
    strict_ok: exactly one boxed pair and nothing else
    mid_ok: one boxed pair with optional reasoning before it, nothing after
    loose_ok: at least one boxed pair exists
    bad_tags: XML/HTML-like tags or escaped tags are present
    """
    s = txt or ""
    ms = list(BOXED_PAIR_RE.finditer(s))
    loose_ok = bool(ms)
    strict_ok = False
    mid_ok = False

    if loose_ok:
        first = ms[0]
        last = ms[-1]
        prefix = s[: first.start()].strip()
        suffix = s[last.end():].strip()
        strict_ok = (len(ms) == 1 and prefix == "" and suffix == "")
        mid_ok = (len(ms) == 1 and suffix == "")

    bad_tags = bool(TAG_RE.search(s) or ESCAPED_TAG_RE.search(s))
    return strict_ok, mid_ok, loose_ok, bad_tags

def split_reasoning(txt: str) -> str:
    s = txt or ""
    ms = list(BOXED_PAIR_RE.finditer(s))
    if not ms:
        return s
    return s[: ms[-1].start()]

def has_reasoning_signal(reason_text: str) -> bool:
    t = (reason_text or "").lower()
    return any(k in t for k in REASONING_KEYWORDS)

def _to_json_scalar(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    try:
        return int(x)
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        pass
    return str(x)


def _append_generation_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Append training-time generation rows to train_generations.csv."""
    if not path or not rows:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "prompt", "model_output", "reasoning", "final_boxed", "reward"],
        )
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "step": int(row.get("step", 0)),
                    "prompt": str(row.get("prompt", "")),
                    "model_output": str(row.get("model_output", "")),
                    "reasoning": str(row.get("reasoning", "")),
                    "final_boxed": str(row.get("final_boxed", "")),
                    "reward": float(row.get("reward", 0.0)),
                }
            )


def _append_eval_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Append eval-time rows to eval_generations.csv."""
    if not path or not rows:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step", "prompt", "model_output", "final_boxed",
                "gt_r1", "gt_r2", "parsed_r1", "parsed_r2",
                "both_exact", "one_root", "parse_fail", "gt_match",
            ],
        )
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "step": int(row.get("step", 0)),
                    "prompt": str(row.get("prompt", "")),
                    "model_output": str(row.get("model_output", "")),
                    "final_boxed": str(row.get("final_boxed", "")),
                    "gt_r1": row.get("gt_r1", ""),
                    "gt_r2": row.get("gt_r2", ""),
                    "parsed_r1": row.get("parsed_r1", ""),
                    "parsed_r2": row.get("parsed_r2", ""),
                    "both_exact": bool(row.get("both_exact", False)),
                    "one_root": bool(row.get("one_root", False)),
                    "parse_fail": bool(row.get("parse_fail", False)),
                    "gt_match": bool(row.get("gt_match", False)),
                }
            )

def _write_jsonl(path: str, rows: List[Dict[str, Any]], append: bool) -> None:
    if not rows:
        return
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_probe_outputs(
    model,
    tokenizer,
    ds,
    split_name: str,
    phase: str,
    step: int,
    max_prompt_length: int,
    max_new_tokens: int,
    batch_size: int,
    num_examples: int,
) -> List[Dict[str, Any]]:
    if ds is None or len(ds) == 0 or num_examples <= 0:
        return []

    n = min(int(num_examples), len(ds))
    out_rows: List[Dict[str, Any]] = []
    was_training = model.training
    device = next(model.parameters()).device

    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                batch = ds[start:end]
                prompts = batch["prompt"]

                enc = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_length,
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                out = generate_with_answer_stop(
                    model=model,
                    tokenizer=tokenizer,
                    enc=enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

                prompt_len = int(enc["input_ids"].shape[1])
                comp_ids = out[:, prompt_len:]
                completions = tokenizer.batch_decode(comp_ids, skip_special_tokens=True)
                completions = [truncate_after_answer(x) for x in completions]

                for i, completion in enumerate(completions):
                    idx = start + i
                    ai = safe_int(batch["a"][i]) if "a" in batch else None
                    bi = safe_int(batch["b"][i]) if "b" in batch else None
                    ci = safe_int(batch["c"][i]) if "c" in batch else None
                    gt1 = safe_int(batch["r1"][i]) if "r1" in batch else None
                    gt2 = safe_int(batch["r2"][i]) if "r2" in batch else None

                    parsed = parse_roots_from_answer(completion)
                    parsed_r1 = parsed[0] if parsed is not None else None
                    parsed_r2 = parsed[1] if parsed is not None else None
                    parsed_src = parsed[2] if parsed is not None else None

                    parse_fail = True
                    one_root = False
                    both_exact = False
                    gt_match = False
                    res1 = None
                    res2 = None

                    if parsed is not None and ai is not None and bi is not None and ci is not None:
                        parse_fail = False
                        ok1, ok2, res1, res2 = verify_roots(ai, bi, ci, parsed_r1, parsed_r2)
                        sum_ok = (ai * (parsed_r1 + parsed_r2) + bi == 0)
                        prod_ok = (ai * parsed_r1 * parsed_r2 - ci == 0)
                        both_exact = bool(ok1 and ok2 and sum_ok and prod_ok)
                        one_root = bool((ok1 or ok2) and not both_exact)
                        if gt1 is not None and gt2 is not None:
                            gt_match = bool((parsed_r1, parsed_r2) == (min(gt1, gt2), max(gt1, gt2)))

                    strict_ok, mid_ok, loose_ok, bad_tags = format_grade(completion)
                    row = {
                        "phase": phase,
                        "step": int(step),
                        "split": split_name,
                        "example_index": int(idx),
                        "id": _to_json_scalar(batch["id"][i]) if "id" in batch else None,
                        "equation": str(batch["equation"][i]) if "equation" in batch else "",
                        "a": ai,
                        "b": bi,
                        "c": ci,
                        "gt_r1": gt1,
                        "gt_r2": gt2,
                        "prompt": str(prompts[i]),
                        "completion": str(completion),
                        "parsed_r1": parsed_r1,
                        "parsed_r2": parsed_r2,
                        "parsed_source": parsed_src,
                        "parse_fail": bool(parse_fail),
                        "one_root": bool(one_root),
                        "both_exact": bool(both_exact),
                        "gt_match": bool(gt_match),
                        "residual_r1": safe_int(res1) if res1 is not None else None,
                        "residual_r2": safe_int(res2) if res2 is not None else None,
                        "strict_format": bool(strict_ok),
                        "mid_format": bool(mid_ok),
                        "loose_format": bool(loose_ok),
                        "bad_tags": bool(bad_tags),
                    }
                    out_rows.append(row)
    finally:
        if was_training:
            model.train()

    return out_rows


class JsonlTrainingProbeCallback(TrainerCallback):
    def __init__(
        self,
        out_path: str,
        probe_ds,
        probe_split: str,
        tokenizer,
        every_steps: int,
        max_prompt_length: int,
        max_new_tokens: int,
        batch_size: int,
        num_examples: int,
    ):
        self.out_path = out_path
        self.probe_ds = probe_ds
        self.probe_split = probe_split
        self.tokenizer = tokenizer
        self.every_steps = max(0, int(every_steps))
        self.max_prompt_length = int(max_prompt_length)
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = max(1, int(batch_size))
        self.num_examples = int(num_examples)
        self._last_run_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_steps <= 0:
            return
        if state.global_step <= 0:
            return
        if state.global_step % self.every_steps != 0:
            return
        if state.global_step == self._last_run_step:
            return
        if not is_rank0():
            return

        model = kwargs.get("model", None)
        if model is None:
            return

        rows = collect_probe_outputs(
            model=model,
            tokenizer=self.tokenizer,
            ds=self.probe_ds,
            split_name=self.probe_split,
            phase="training",
            step=int(state.global_step),
            max_prompt_length=self.max_prompt_length,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            num_examples=self.num_examples,
        )
        _write_jsonl(self.out_path, rows, append=True)
        self._last_run_step = int(state.global_step)
        if rows:
            log.info("[JSONL] wrote %d training rows at step=%d -> %s", len(rows), state.global_step, self.out_path)


# -----------------------
# Combined reward (single function)
# -----------------------
def combined_reward_func(
    completions,
    a=None, b=None, c=None,
    r1=None, r2=None,
    equation=None,
    prompt=None,
    trainer_state=None,
    **kwargs
) -> List[float]:
    """
    Total reward in [0,1]:
      math: 0 / 0.4 / 0.82 (+ dense near-miss shaping capped inside math)
      format bonus: gated by correctness and clean final boxed output
      reasoning bonus: only when math >= 0.4
      deductions: malformed output patterns that correlate with reward hacking
    """
    global REWARD_CALLS, LAST_REWARD_STATS, GEN_CSV_PATH, GEN_CSV_EVERY, GEN_CSV_LAST_STEP
    REWARD_CALLS += 1

    n = len(completions)
    rewards: List[float] = [0.0] * n
    eq_rewards: List[float] = []
    fmt_rewards: List[float] = []
    reason_rewards: List[float] = []
    penalties: List[float] = []
    completion_lengths: List[float] = []

    parsed_cache: List[Optional[Tuple[int,int,str,Tuple[int,int]]]] = [None] * n
    ok_cache: List[Tuple[bool,bool,int,int]] = [(False, False, 0, 0)] * n
    fmt_cache: List[Tuple[bool,bool,bool,bool]] = [(False, False, False, False)] * n

    both = one = parse_fail = gt_match = 0
    step_now = int(getattr(trainer_state, "global_step", 0) or 0) if trainer_state is not None else 0
    # v11: extended bootstrap (0-149) and delayed hard phase (600+)
    # gives the model more room to learn the task before format enforcement tightens
    bootstrap_phase = step_now < 150
    strict_phase = step_now >= 150
    hard_phase = step_now >= 600

    for i, comp in enumerate(completions):
        txt = get_text(comp)
        completion_lengths.append(float(len(txt)))

        r_math = 0.0
        r_fmt = 0.0
        r_reason = 0.0
        deduction = 0.0

        ai = safe_int(col_get(a, i))
        bi = safe_int(col_get(b, i))
        ci = safe_int(col_get(c, i))
        if ai is None or bi is None or ci is None:
            rewards[i] = 0.0
            parse_fail += 1
            eq_rewards.append(0.0)
            fmt_rewards.append(0.0)
            reason_rewards.append(0.0)
            penalties.append(0.0)
            continue

        strict_ok, mid_ok, loose_ok, bad = format_grade(txt)
        fmt_cache[i] = (strict_ok, mid_ok, loose_ok, bad)

        boxed_matches = list(BOXED_PAIR_RE.finditer(txt or ""))
        has_boxed_pair = bool(boxed_matches)
        has_multiple_pairs = len(boxed_matches) > 1
        trailing_after_boxed = False
        if has_boxed_pair:
            trailing_after_boxed = (txt[boxed_matches[-1].end():].strip() != "")

        parsed = parse_roots_from_answer(txt)
        if parsed is None and bootstrap_phase:
            parsed = parse_roots_from_labeled_text(txt)
        parsed_cache[i] = parsed

        if parsed is None:
            rewards[i] = 0.005 if loose_ok else 0.0
            parse_fail += 1
            r_fmt = rewards[i]
            eq_rewards.append(r_math)
            fmt_rewards.append(r_fmt)
            reason_rewards.append(r_reason)
            penalties.append(-deduction)
            continue

        pr1, pr2, src, raw = parsed
        ok1, ok2, res1, res2 = verify_roots(ai, bi, ci, pr1, pr2)
        ok_cache[i] = (ok1, ok2, res1, res2)

        sum_err = abs(ai * (pr1 + pr2) + bi)
        prod_err = abs(ai * pr1 * pr2 - ci)
        sum_ok = (sum_err == 0)
        prod_ok = (prod_err == 0)
        both_exact = bool(ok1 and ok2 and sum_ok and prod_ok)

        if both_exact:
            r_math = 0.82
            both += 1
        elif ok1 or ok2:
            r_math = 0.40
            one += 1

        gt1 = safe_int(col_get(r1, i)) if r1 is not None else None
        gt2 = safe_int(col_get(r2, i)) if r2 is not None else None
        if gt1 is not None and gt2 is not None:
            if (pr1, pr2) == (min(gt1, gt2), max(gt1, gt2)):
                gt_match += 1

        # Dense math shaping capped inside math budget.
        r_shape = 0.04 * (1 if sum_ok else 0) + 0.04 * (1 if prod_ok else 0)
        if not (ok1 or ok2):
            residual_err = abs(res1) + abs(res2)
            near_sub = 0.07 / (1.0 + 0.05 * residual_err)
            near_vieta = 0.07 / (1.0 + 0.20 * (sum_err + prod_err))
            r_math = max(r_math, min(0.14, near_sub + near_vieta))
        elif ok1 ^ ok2:
            wrong_residual = abs(res2 if ok1 else res1)
            r_math += min(0.04, 0.04 / (1.0 + 0.05 * wrong_residual))
        r_math = min(0.82, r_math + r_shape)

        # Clean final format means: one boxed pair, optional reasoning before it, nothing after it.
        valid_clean = bool(mid_ok and has_boxed_pair and (not has_multiple_pairs) and (not bad) and (not trailing_after_boxed))

        if both_exact:
            if valid_clean:
                r_fmt = 0.15
            elif loose_ok:
                r_fmt = 0.04
        elif ok1 or ok2:
            if valid_clean:
                r_fmt = 0.06
            elif loose_ok:
                r_fmt = 0.02
        else:
            if valid_clean:
                r_fmt = 0.01

        reason_text = split_reasoning(txt)
        reason_len = len((reason_text or "").strip())
        if r_math >= 0.40:
            if reason_len >= MIN_THINK_CHARS:
                r_reason += 0.02
            if has_reasoning_signal(reason_text):
                r_reason += 0.01
            r_reason = min(0.03, r_reason)

        # Explicit negative deductions.
        low = (txt or "").lower()
        if bad:
            deduction -= 0.12
        if has_multiple_pairs:
            deduction -= 0.12
        if trailing_after_boxed:
            deduction -= 0.10
        if "\\</" in low:
            deduction -= 0.08
        if not has_boxed_pair:
            deduction -= 0.15
        if raw[0] > raw[1]:
            deduction -= 0.02

        # Rambling discouragement: long outputs often correlate with format drift.
        # v11: threshold raised 700->1200 to allow more reasoning space without penalty.
        if len(txt) > 1200:
            deduction -= 0.05

        total = r_math + r_fmt + r_reason + deduction

        # Anti-hack caps: high reward requires clean format.
        if not valid_clean:
            if both_exact:
                total = min(total, 0.65)
            elif (ok1 or ok2):
                total = min(total, 0.35)
            else:
                total = min(total, 0.18)

        # Labeled fallback is bootstrap-only and is heavily capped.
        # v11: bootstrap caps raised to give better early signal for correct-but-unformatted outputs.
        if src == "label_pair_text":
            if hard_phase:
                total = min(total, 0.005)
            elif strict_phase:
                if both_exact:
                    total = min(total, 0.04)
                elif (ok1 or ok2):
                    total = min(total, 0.02)
                else:
                    total = min(total, 0.005)
            else:
                # bootstrap: more generous so the model isn't starved of signal early on
                if both_exact:
                    total = min(total, 0.18)
                elif (ok1 or ok2):
                    total = min(total, 0.10)
                else:
                    total = min(total, 0.03)

        rewards[i] = float(min(1.0, max(0.0, total)))
        eq_rewards.append(r_math)
        fmt_rewards.append(r_fmt)
        reason_rewards.append(r_reason)
        penalties.append(-deduction)

    LAST_REWARD_STATS = {
        "equation_reward_func": _mean(eq_rewards),
        "format_reward_func": _mean(fmt_rewards),
        "reasoning_reward_func": _mean(reason_rewards),
        "penalty_term": _mean(penalties),
        "parse_fail_rate": (parse_fail / n) if n else 0.0,
        "both_roots_rate": (both / n) if n else 0.0,
        "one_root_rate": (one / n) if n else 0.0,
        "completion_length_chars": _mean(completion_lengths),
        "reward_std": float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
    }

    if is_rank0() and GEN_CSV_PATH and GEN_CSV_EVERY > 0 and step_now > 0 and (step_now % GEN_CSV_EVERY == 0) and (step_now != GEN_CSV_LAST_STEP):
        csv_rows: List[Dict[str, Any]] = []
        for i, comp in enumerate(completions):
            ptxt = col_get(prompt, i) if prompt is not None else ""
            out_txt = get_text(comp)
            reasoning_txt = split_reasoning(out_txt).strip()
            final_boxed = ""
            boxed_match = BOXED_PAIR_RE.findall(out_txt)
            if boxed_match:
                last_r1, last_r2 = boxed_match[-1]
                final_boxed = f"\\boxed{{{last_r1}, {last_r2}}}"
            csv_rows.append(
                {
                    "step": step_now,
                    "prompt": get_text(ptxt) if ptxt is not None else "",
                    "model_output": out_txt,
                    "reasoning": reasoning_txt,
                    "final_boxed": final_boxed,
                    "reward": float(rewards[i]),
                }
            )
        _append_generation_csv(GEN_CSV_PATH, csv_rows)
        GEN_CSV_LAST_STEP = step_now
        log.info("[CSV] wrote %d generation rows at step=%d -> %s", len(csv_rows), step_now, GEN_CSV_PATH)
    if is_rank0() and DEBUG_EVERY and (REWARD_CALLS % DEBUG_EVERY == 0):
        step = getattr(trainer_state, "global_step", None)
        epoch = getattr(trainer_state, "epoch", None)
        step_s = "?" if step is None else str(step)
        epoch_s = "?" if epoch is None else f"{epoch:.4f}"

        log.info(
            f"[REWARD call={REWARD_CALLS}] step={step_s} epoch={epoch_s} "
            f"both={both/n:.1%} one={one/n:.1%} parse_fail={parse_fail/n:.1%} gt_match={gt_match/n:.1%}"
        )

        show_n = min(DEBUG_SHOW_N, n)
        print("\n" + "=" * 120, flush=True)
        print(f"[FULL COMPLETIONS] call={REWARD_CALLS} step={step_s} epoch={epoch_s} showing {show_n}/{n}", flush=True)
        print("=" * 120, flush=True)

        for j in range(show_n):
            txt = get_text(completions[j])

            ai = safe_int(col_get(a, j))
            bi = safe_int(col_get(b, j))
            ci = safe_int(col_get(c, j))
            eq = col_get(equation, j) if equation is not None else None

            gt1 = safe_int(col_get(r1, j)) if r1 is not None else None
            gt2 = safe_int(col_get(r2, j)) if r2 is not None else None
            gt = None if (gt1 is None or gt2 is None) else (min(gt1, gt2), max(gt1, gt2))

            strict_ok, mid_ok, loose_ok, bad = fmt_cache[j]
            parsed = parsed_cache[j]
            if parsed is None:
                header = f"gen[{j}] strict={strict_ok} mid={mid_ok} loose={loose_ok} bad={bad} parsed=None reward={rewards[j]:.3f}"
            else:
                pr1, pr2, src, raw = parsed
                ok1, ok2, res1, res2 = ok_cache[j]
                header = (
                    f"gen[{j}] strict={strict_ok} mid={mid_ok} loose={loose_ok} bad={bad} "
                    f"parsed_sorted=({pr1},{pr2}) raw={raw} src={src} ok1={ok1} ok2={ok2} res1={res1} res2={res2} "
                    f"reward={rewards[j]:.3f}"
                )

            print(f"\n--- {header} ---", flush=True)

            if DEBUG_PRINT_EQUATION:
                print(f"a,b,c=({ai},{bi},{ci})  GT={gt}", flush=True)
                if eq is not None:
                    print(f"equation: {eq}", flush=True)

            if DEBUG_PRINT_PROMPT:
                ptxt = col_get(prompt, j) if prompt is not None else None
                if ptxt is not None:
                    print("\n[PROMPT]", flush=True)
                    print(get_text(ptxt), flush=True)

            if DEBUG_PRINT_EFFECTIVE:
                print("\n[EFFECTIVE ASSISTANT MESSAGE]", flush=True)
                print(txt, flush=True)

            print("\n[COMPLETION ONLY]", flush=True)
            if DEBUG_MAX_CHARS and DEBUG_MAX_CHARS > 0 and len(txt) > DEBUG_MAX_CHARS:
                print(txt[:DEBUG_MAX_CHARS], flush=True)
                print("\n... [TRUNCATED] ...\n", flush=True)
            else:
                print(txt, flush=True)

        print("=" * 120 + "\n", flush=True)

    return rewards

# -----------------------
# Main
# -----------------------
def main():
    global DEBUG_EVERY, DEBUG_SHOW_N, DEBUG_MAX_CHARS, DEBUG_PRINT_PROMPT, DEBUG_PRINT_EQUATION, DEBUG_PRINT_EFFECTIVE
    global MIN_THINK_CHARS, LAST_EVAL_STATS, GEN_CSV_PATH, GEN_CSV_EVERY, GEN_CSV_LAST_STEP, EVAL_CSV_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--train_parquet", type=str, required=True)
    ap.add_argument("--test_parquet", type=str, default="", help="Optional test/eval parquet for schema check and logging.")
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--learning_rate", type=float, default=1e-6)
    ap.add_argument("--warmup_steps", type=int, default=30)
    ap.add_argument("--beta", type=float, default=0.002)  # v12: 0.001->0.002 for harder task stability
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=16)
    ap.add_argument("--num_generations", type=int, default=4)

    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--max_completion_length", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.80)  # v12: 0.70->0.80 for harder-task diversity
    ap.add_argument("--top_p", type=float, default=0.90)

    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.45)  # v11: 0.30->0.45

    ap.add_argument("--grpo_max_samples", type=int, default=0)
    ap.add_argument("--tb_log_dir", type=str, default="", help="TensorBoard log directory; default is <output_dir>/tb")
    ap.add_argument("--disable_tensorboard", action="store_true")
    ap.add_argument("--eval_every_steps", type=int, default=100, help="Run periodic exact eval every N train steps; 0 disables periodic eval.")
    ap.add_argument("--eval_max_samples", type=int, default=256, help="Max test samples per periodic exact eval; 0 uses full test split.")
    ap.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for periodic exact eval generation.")
    ap.add_argument("--eval_max_new_tokens", type=int, default=0, help="Max new tokens for periodic exact eval; 0 uses max_completion_length.")
    ap.add_argument("--jsonl_export_dir", type=str, default="", help="Directory for thesis JSONL outputs; default is <output_dir>/jsonl")
    ap.add_argument("--jsonl_num_examples", type=int, default=24, help="Number of fixed probe examples to export per phase")
    ap.add_argument("--jsonl_batch_size", type=int, default=8, help="Batch size for JSONL probe generation")
    ap.add_argument("--jsonl_train_every_steps", type=int, default=200, help="Write training-phase probe outputs every N steps; 0 disables")
    ap.add_argument("--jsonl_split", type=str, default="test", choices=["test", "train"], help="Probe split for JSONL exports")
    ap.add_argument("--disable_jsonl_export", action="store_true")

    # Debug args
    ap.add_argument("--reward_debug_every", type=int, default=100)
    ap.add_argument("--debug_show_n", type=int, default=8)
    ap.add_argument("--debug_max_chars", type=int, default=0, help="0 => FULL completion")
    ap.add_argument("--debug_print_prompt", action="store_true")
    ap.add_argument("--debug_print_equation", action="store_true")
    ap.add_argument("--debug_print_effective", action="store_true")

    # Reasoning control
    ap.add_argument("--min_think_chars", type=int, default=80)  # v12: kept at 80
    ap.add_argument("--train_csv_path", type=str, default="", help="CSV file for training-time prompt/model_output/reward rows (default: <output_dir>/train_generations.csv)")
    ap.add_argument("--eval_csv_path", type=str, default="", help="CSV file for eval-time prompt/model_output/correctness rows (default: <output_dir>/eval_generations.csv)")
    ap.add_argument("--generation_csv_every_steps", type=int, default=50, help="Write train generation CSV every N global steps; 0 disables")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=48)      # v12: 32->48 for harder arithmetic
    ap.add_argument("--lora_alpha", type=int, default=96)  # v12: 64->96 (keep ratio 2:1)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tb_log_dir = args.tb_log_dir if args.tb_log_dir else os.path.join(args.output_dir, "tb")
    os.makedirs(tb_log_dir, exist_ok=True)
    report_to = [] if args.disable_tensorboard else ["tensorboard"]

    jsonl_export_enabled = not bool(args.disable_jsonl_export)
    jsonl_export_dir = args.jsonl_export_dir if args.jsonl_export_dir else os.path.join(args.output_dir, "jsonl")
    baseline_jsonl_path = os.path.join(jsonl_export_dir, "baseline_outputs.jsonl")
    training_jsonl_path = os.path.join(jsonl_export_dir, "training_outputs.jsonl")
    post_jsonl_path = os.path.join(jsonl_export_dir, "post_eval_outputs.jsonl")
    jsonl_meta_path = os.path.join(jsonl_export_dir, "metadata.json")
    if jsonl_export_enabled:
        os.makedirs(jsonl_export_dir, exist_ok=True)

    # set globals
    DEBUG_EVERY = int(args.reward_debug_every)
    DEBUG_SHOW_N = int(args.debug_show_n)
    DEBUG_MAX_CHARS = int(args.debug_max_chars)
    DEBUG_PRINT_PROMPT = bool(args.debug_print_prompt)
    DEBUG_PRINT_EQUATION = bool(args.debug_print_equation)
    DEBUG_PRINT_EFFECTIVE = bool(args.debug_print_effective)
    MIN_THINK_CHARS = int(args.min_think_chars)
    GEN_CSV_PATH = args.train_csv_path if args.train_csv_path else os.path.join(args.output_dir, "train_generations.csv")
    EVAL_CSV_PATH = args.eval_csv_path if args.eval_csv_path else os.path.join(args.output_dir, "eval_generations.csv")
    GEN_CSV_EVERY = max(0, int(args.generation_csv_every_steps))
    GEN_CSV_LAST_STEP = -1
    if is_rank0() and GEN_CSV_EVERY > 0:
        log.info("[CSV] train generation export enabled every=%d path=%s", GEN_CSV_EVERY, GEN_CSV_PATH)
    if is_rank0():
        log.info("[CSV] eval generation export path=%s", EVAL_CSV_PATH)

    # divisibility rule (single GPU)
    effective = args.per_device_batch_size * args.grad_accum_steps
    if effective % args.num_generations != 0:
        raise ValueError(
            f"(per_device_batch_size * grad_accum_steps)={effective} must be divisible by num_generations={args.num_generations}"
        )
    prompts_per_step = effective // args.num_generations
    if is_rank0():
        log.info(
            "[BATCH] effective=%d num_generations=%d prompts_per_step=%d",
            effective,
            args.num_generations,
            prompts_per_step,
        )
        if prompts_per_step < 2:
            log.warning(
                "[BATCH] prompts_per_step < 2 can hurt accuracy due to high-variance policy updates."
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # v11: clearer format contract - explicit single boxed line, sort order, no trailing text
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
    def build_prompt(equation: str, user_message: str = None) -> str:
        # If the dataset provides a pre-baked user message (diverse phrasing),
        # use it directly; otherwise fall back to the original default.
        if user_message:
            user_content = user_message
        else:
            equation = normalize_equation_text(equation)
            user_content = f"Solve the quadratic equation for integer roots: {equation}"
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return f"SYSTEM: {SYSTEM_PROMPT}\nUSER: {user_content}\nASSISTANT:"

    def to_prompt(ex: Dict[str, Any]) -> Dict[str, Any]:
        # user_message column is optional (backward-compatible with old parquets)
        ex["prompt"] = build_prompt(ex.get("equation", ""), ex.get("user_message"))
        return ex

    def validate_required_columns(ds, split_name: str) -> None:
        required = {"equation", "a", "b", "c", "r1", "r2"}
        missing = required - set(ds.column_names)
        if missing:
            raise ValueError(f"{split_name} dataset missing required columns: {missing}")

    train_ds = load_dataset("parquet", data_files={"train": args.train_parquet})["train"]
    validate_required_columns(train_ds, "train")
    if is_rank0():
        log.info(f"[DATA] loaded train rows={len(train_ds)}, cols={train_ds.column_names}")

    test_ds = None

    if args.test_parquet:
        test_ds = load_dataset("parquet", data_files={"test": args.test_parquet})["test"]
        validate_required_columns(test_ds, "test")
        if is_rank0():
            log.info(f"[DATA] loaded test rows={len(test_ds)}, cols={test_ds.column_names}")

    train_ds = train_ds.map(to_prompt)
    if test_ds is not None:
        test_ds = test_ds.map(to_prompt)
        if is_rank0():
            log.info(f"[DATA] prepared test prompts rows={len(test_ds)}")

    if args.grpo_max_samples and args.grpo_max_samples > 0 and len(train_ds) > args.grpo_max_samples:
        train_ds = train_ds.select(range(args.grpo_max_samples))
        if is_rank0():
            log.info(f"[DATA] truncated to {len(train_ds)} samples (grpo_max_samples)")

    probe_ds = None
    probe_split = ""
    if args.jsonl_split == "test" and test_ds is not None:
        probe_ds = test_ds
        probe_split = "test"
    else:
        probe_ds = train_ds
        probe_split = "train"

    if is_rank0() and jsonl_export_enabled:
        log.info("[JSONL] enabled dir=%s split=%s examples=%d train_every=%d", jsonl_export_dir, probe_split, args.jsonl_num_examples, args.jsonl_train_every_steps)

    # v11: added MLP layers (gate_proj, up_proj, down_proj) - essential for math reasoning
    # Qwen2.5 uses SwiGLU MLP: gate*up projected then down-projected.
    # Covering these dramatically increases the model's ability to learn numeric computation.
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,   # v11-fix: Qwen2.5 is bf16-native; fp16 risks NaN on Ampere
        attn_implementation=attn_impl,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    jsonl_max_new_tokens = args.eval_max_new_tokens if args.eval_max_new_tokens > 0 else args.max_completion_length
    if is_rank0() and jsonl_export_enabled and probe_ds is not None:
        if next(model.parameters()).device.type == "cpu" and torch.cuda.is_available():
            model = model.to("cuda")
            log.info("[JSONL] moved model to CUDA for baseline probe generation")
        baseline_rows = collect_probe_outputs(
            model=model,
            tokenizer=tokenizer,
            ds=probe_ds,
            split_name=probe_split,
            phase="baseline",
            step=0,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=jsonl_max_new_tokens,
            batch_size=args.jsonl_batch_size,
            num_examples=args.jsonl_num_examples,
        )
        _write_jsonl(baseline_jsonl_path, baseline_rows, append=False)
        open(training_jsonl_path, "w", encoding="utf-8").close()
        if baseline_rows:
            log.info("[JSONL] wrote baseline rows=%d -> %s", len(baseline_rows), baseline_jsonl_path)

    if is_rank0() and args.beta == 0.0:
        log.warning(
            "[CONFIG] beta=0.0 disables KL penalty entirely. "
            "The policy can drift arbitrarily from the reference model. "
            "Consider --beta 0.001 for stability (v11 default)."
        )

    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        logging_dir=tb_log_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        seed=args.seed,

        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_generations=args.num_generations,

        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,

        temperature=args.temperature,
        top_p=args.top_p,

        fp16=False,
        bf16=True,    # v11-fix: Qwen2.5 is bf16-native; RTX 3080 (Ampere) supports bf16

        loss_type="dr_grpo",
        beta=args.beta,
        scale_rewards=True,

        remove_unused_columns=False,
        logging_steps=10,
        logging_first_step=True,
        save_steps=200,
        save_total_limit=3,
        report_to=report_to,

        # v11: avoid tokenizer forking issues on multi-process HPC nodes
        dataloader_num_workers=0,

        use_vllm=bool(args.use_vllm),
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    if is_rank0():
        log.info("[START] GRPOTrainer init...")
        log.info(f"  use_vllm={grpo_cfg.use_vllm} vllm_mode={grpo_cfg.vllm_mode} vllm_gpu_mem={grpo_cfg.vllm_gpu_memory_utilization}")
        log.info(f"  debug_every={DEBUG_EVERY} show_n={DEBUG_SHOW_N} max_chars={DEBUG_MAX_CHARS} min_think_chars={MIN_THINK_CHARS}")
        log.info(f"  report_to={report_to} logging_dir={tb_log_dir}")

    callbacks = [RewardBreakdownCallback()]
    if jsonl_export_enabled and is_rank0() and probe_ds is not None and args.jsonl_train_every_steps > 0:
        callbacks.append(
            JsonlTrainingProbeCallback(
                out_path=training_jsonl_path,
                probe_ds=probe_ds,
                probe_split=probe_split,
                tokenizer=tokenizer,
                every_steps=args.jsonl_train_every_steps,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=jsonl_max_new_tokens,
                batch_size=args.jsonl_batch_size,
                num_examples=args.jsonl_num_examples,
            )
        )

    if test_ds is not None and args.eval_every_steps > 0:
        eval_max_new_tokens = args.eval_max_new_tokens if args.eval_max_new_tokens > 0 else args.max_completion_length
        callbacks.append(
            ExactAccuracyEvalCallback(
                eval_ds=test_ds,
                tokenizer=tokenizer,
                eval_every_steps=args.eval_every_steps,
                eval_batch_size=args.eval_batch_size,
                eval_max_samples=args.eval_max_samples,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=eval_max_new_tokens,
                eval_csv_path=EVAL_CSV_PATH,
            )
        )
        if is_rank0():
            log.info(
                "[EVAL] exact eval enabled every=%d max_samples=%d batch=%d max_new_tokens=%d",
                args.eval_every_steps,
                args.eval_max_samples,
                args.eval_batch_size,
                eval_max_new_tokens,
            )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=train_ds,
        reward_funcs=[combined_reward_func],   # single reward
        processing_class=tokenizer,
        peft_config=peft_cfg,
        callbacks=callbacks,
    )

    # ── Baseline evaluation (step 0, before any training) ────────────────────
    baseline_eval_metrics: Dict[str, float] = {}
    if test_ds is not None and is_rank0():
        eval_max_new_tokens_bl = args.eval_max_new_tokens if args.eval_max_new_tokens > 0 else args.max_completion_length
        _bl_cb = ExactAccuracyEvalCallback(
            eval_ds=test_ds,
            tokenizer=tokenizer,
            eval_every_steps=0,
            eval_batch_size=args.eval_batch_size,
            eval_max_samples=args.eval_max_samples,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=eval_max_new_tokens_bl,
            eval_csv_path=EVAL_CSV_PATH,
        )
        baseline_eval_metrics = _bl_cb._compute_exact_metrics(model, step=0)
        log.info(
            "\n" + "=" * 80 + "\n"
            "[BASELINE EVAL] step=0 (before training)\n"
            "  exact_accuracy : %.4f\n"
            "  one_root_rate  : %.4f\n"
            "  parse_fail_rate: %.4f\n"
            "  gt_match_rate  : %.4f\n"
            "  samples        : %d\n"
            "=" * 80,
            baseline_eval_metrics.get("eval/exact_both_roots_accuracy", 0.0),
            baseline_eval_metrics.get("eval/one_root_rate", 0.0),
            baseline_eval_metrics.get("eval/parse_fail_rate", 0.0),
            baseline_eval_metrics.get("eval/gt_match_rate", 0.0),
            int(baseline_eval_metrics.get("eval/samples", 0.0)),
        )

    trainer.train()

    if test_ds is not None:
        eval_max_new_tokens = args.eval_max_new_tokens if args.eval_max_new_tokens > 0 else args.max_completion_length
        final_eval = ExactAccuracyEvalCallback(
            eval_ds=test_ds,
            tokenizer=tokenizer,
            eval_every_steps=0,
            eval_batch_size=args.eval_batch_size,
            eval_max_samples=args.eval_max_samples,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=eval_max_new_tokens,
            eval_csv_path=EVAL_CSV_PATH,
        )
        final_step = int(getattr(trainer.state, "global_step", args.max_steps))
        LAST_EVAL_STATS = final_eval._compute_exact_metrics(trainer.model, step=final_step)
        trainer.log(LAST_EVAL_STATS)
        if is_rank0():
            bl_exact = baseline_eval_metrics.get("eval/exact_both_roots_accuracy", None)
            fn_exact = LAST_EVAL_STATS.get("eval/exact_both_roots_accuracy", 0.0)
            delta_str = ""
            if bl_exact is not None:
                delta = fn_exact - bl_exact
                delta_str = f"  delta vs baseline: {delta:+.4f} ({delta*100:+.1f}pp)\n"
            log.info(
                "\n" + "=" * 80 + "\n"
                "[FINAL EVAL] step=%d (after training)\n"
                "  exact_accuracy : %.4f\n"
                "  one_root_rate  : %.4f\n"
                "  parse_fail_rate: %.4f\n"
                "  gt_match_rate  : %.4f\n"
                "  samples        : %d\n"
                "%s"
                "[BASELINE vs FINAL]\n"
                "  baseline exact : %.4f\n"
                "  final    exact : %.4f\n"
                "=" * 80,
                final_step,
                fn_exact,
                LAST_EVAL_STATS.get("eval/one_root_rate", 0.0),
                LAST_EVAL_STATS.get("eval/parse_fail_rate", 0.0),
                LAST_EVAL_STATS.get("eval/gt_match_rate", 0.0),
                int(LAST_EVAL_STATS.get("eval/samples", 0.0)),
                delta_str,
                bl_exact if bl_exact is not None else float("nan"),
                fn_exact,
            )

    if is_rank0() and jsonl_export_enabled and probe_ds is not None:
        post_rows = collect_probe_outputs(
            model=trainer.model,
            tokenizer=tokenizer,
            ds=probe_ds,
            split_name=probe_split,
            phase="post_eval",
            step=int(getattr(trainer.state, "global_step", 0)),
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=jsonl_max_new_tokens,
            batch_size=args.jsonl_batch_size,
            num_examples=args.jsonl_num_examples,
        )
        _write_jsonl(post_jsonl_path, post_rows, append=False)
        meta = {
            "baseline_jsonl": baseline_jsonl_path,
            "training_jsonl": training_jsonl_path,
            "post_jsonl": post_jsonl_path,
            "probe_split": probe_split,
            "probe_examples": int(args.jsonl_num_examples),
            "train_every_steps": int(args.jsonl_train_every_steps),
            "global_step": int(getattr(trainer.state, "global_step", 0)),
        }
        with open(jsonl_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if post_rows:
            log.info("[JSONL] wrote post-eval rows=%d -> %s", len(post_rows), post_jsonl_path)

    trainer.save_model(args.output_dir)
    if is_rank0():
        tokenizer.save_pretrained(args.output_dir)
        log.info(f"[DONE] saved to {args.output_dir}")

if __name__ == "__main__":
    main()








