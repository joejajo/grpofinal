#!/usr/bin/env python3
"""
Convert existing quadratic parquet files into VERL RLHF parquet format.

Input schema expected (current repo):
  equation, a, b, c, r1, r2, user_message (optional)

Output schema (VERL RLHF):
  data_source, prompt, ability, reward_model, extra_info
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import datasets


SYSTEM_PROMPT = (
    "You are a math solver. Solve quadratic equations with integer roots. "
    "Show brief reasoning, then output exactly one final answer line in this format: "
    "\\boxed{r1, r2} where r1 <= r2."
)


def _to_int(x: Any) -> int:
    if isinstance(x, bool):
        raise ValueError("Invalid input: boolean values are not supported for integer conversion")
    return int(x)


def build_prompt(example: Dict[str, Any]) -> list[Dict[str, str]]:
    equation = str(example.get("equation", "")).strip()
    user_message = str(example.get("user_message", "")).strip()
    if not user_message:
        user_message = f"Solve the quadratic equation for integer roots: {equation}"

    user_content = (
        f"{user_message}\n"
        "Return the final answer exactly as \\boxed{r1, r2} with integers and r1 <= r2."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def map_fn(split: str):
    def _fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        a = _to_int(example["a"])
        b = _to_int(example["b"])
        c = _to_int(example["c"])
        r1 = _to_int(example["r1"])
        r2 = _to_int(example["r2"])
        lo, hi = (r1, r2) if r1 <= r2 else (r2, r1)

        return {
            "data_source": "quadratic/roots",
            "prompt": build_prompt(example),
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"r1": lo, "r2": hi},
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "equation": str(example.get("equation", "")),
                "a": a,
                "b": b,
                "c": c,
                "r1": lo,
                "r2": hi,
            },
        }

    return _fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", required=True, help="Input train parquet")
    ap.add_argument("--val_in", required=True, help="Input eval/val parquet")
    ap.add_argument("--out_dir", required=True, help="Output directory for VERL parquet files")
    args = ap.parse_args()

    ds_train = datasets.load_dataset("parquet", data_files={"train": args.train_in})["train"]
    ds_val = datasets.load_dataset("parquet", data_files={"val": args.val_in})["val"]

    required = {"equation", "a", "b", "c", "r1", "r2"}
    for name, ds in (("train", ds_train), ("val", ds_val)):
        missing = sorted(required - set(ds.column_names))
        if missing:
            raise ValueError(f"{name} parquet missing required columns: {missing}")

    ds_train = ds_train.map(map_fn("train"), with_indices=True, remove_columns=ds_train.column_names)
    ds_val = ds_val.map(map_fn("val"), with_indices=True, remove_columns=ds_val.column_names)

    os.makedirs(args.out_dir, exist_ok=True)
    train_out = os.path.join(args.out_dir, "train.parquet")
    val_out = os.path.join(args.out_dir, "val.parquet")

    ds_train.to_parquet(train_out)
    ds_val.to_parquet(val_out)

    print(f"Wrote {len(ds_train)} train rows -> {train_out}")
    print(f"Wrote {len(ds_val)} val rows   -> {val_out}")


if __name__ == "__main__":
    main()
