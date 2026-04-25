#!/usr/bin/env python3
"""Custom VERL reward function for quadratic integer roots.

Hooked via:
  reward.custom_reward_function.path=/abs/path/to/quadratic_reward.py
  reward.custom_reward_function.name=compute_score
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}")
PAIR_RE = re.compile(r"(-?\d+)\s*[,; ]\s*(-?\d+)")


def _parse_pair(solution_str: str) -> Optional[Tuple[int, int]]:
    text = solution_str.strip()

    m = BOXED_RE.search(text)
    if m:
        text = m.group(1)

    p = PAIR_RE.search(text)
    if not p:
        nums = re.findall(r"-?\d+", text)
        if len(nums) < 2:
            return None
        a, b = int(nums[0]), int(nums[1])
    else:
        a, b = int(p.group(1)), int(p.group(2))

    return (a, b) if a <= b else (b, a)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> float:
    # Restrict to this task; return 0 for unexpected datasource to avoid accidental leakage.
    if data_source != "quadratic/roots":
        return 0.0

    pred = _parse_pair(solution_str)
    if pred is None:
        return 0.0

    try:
        gt1 = int(ground_truth["r1"]) if isinstance(ground_truth, dict) else int(ground_truth[0])
        gt2 = int(ground_truth["r2"]) if isinstance(ground_truth, dict) else int(ground_truth[1])
    except Exception:
        return 0.0

    gt = (gt1, gt2) if gt1 <= gt2 else (gt2, gt1)

    if pred == gt:
        return 1.0

    # partial credit if exactly one root matches
    overlap = len(set(pred) & set(gt))
    return 0.5 if overlap == 1 else 0.0
