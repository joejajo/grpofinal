"""
Quadratic Task Reward Functions for ES Fine-Tuning.

Adapted from the GRPO quadratic reward (grpo_quad_train_v12.py) to work
with the ES fine-tuning framework (where each sample is evaluated independently
and returns a dict with 'reward' and 'reward_info').

The quadratic task asks the model to find integer roots of ax^2 + bx + c = 0
and output them as \\boxed{r1, r2} with r1 <= r2.

Reward structure (single combined reward in [0, 1]):
  - math:    0.0 / 0.40 / 0.82  (substitution + Vieta verification)
  - format:  bonus gated by correctness
  - reasoning: bonus when math >= 0.40
  - deductions: XML tags, multiple boxed, trailing junk, etc.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# ── Regex patterns ──────────────────────────────────────────────────────────

BOXED_PAIR_RE = re.compile(
    r"\\boxed\{\s*\(?\s*([-+]?\d+)(?:\.0+)?\s*,\s*([-+]?\d+)(?:\.0+)?\s*\)?\s*\}"
)
TAG_RE = re.compile(r"<[^>]+>")
ESCAPED_TAG_RE = re.compile(r"\\</?[a-zA-Z][^>]*>")

MIN_THINK_CHARS = 80
REASONING_KEYWORDS = (
    "discriminant", "delta", "sqrt", "quadratic formula", "factor", "vieta",
    "b^2-4ac", "+/-", "formula", "x =", "solve", "coefficient",
    "product of roots", "sum of roots", "roots are",
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def parse_roots_from_answer(txt: str) -> Optional[Tuple[int, int, str, Tuple[int, int]]]:
    """Parse \\boxed{r1, r2} from model output. Returns (sorted_r1, sorted_r2, src, raw_pair)."""
    ms = list(BOXED_PAIR_RE.finditer(txt or ""))
    if not ms:
        return None
    r1_raw = int(ms[-1].group(1))
    r2_raw = int(ms[-1].group(2))
    return (min(r1_raw, r2_raw), max(r1_raw, r2_raw), "boxed_pair", (r1_raw, r2_raw))


def verify_roots(a: int, b: int, c: int, r1: int, r2: int) -> Tuple[bool, bool, int, int]:
    """Verify roots by substitution: returns (ok1, ok2, residual1, residual2)."""
    res1 = a * r1 * r1 + b * r1 + c
    res2 = a * r2 * r2 + b * r2 + c
    return (res1 == 0), (res2 == 0), res1, res2


def format_grade(txt: str) -> Tuple[bool, bool, bool, bool]:
    """Grade format quality: (strict_ok, mid_ok, loose_ok, bad_tags)."""
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
    """Extract reasoning text (everything before the last boxed answer)."""
    s = txt or ""
    ms = list(BOXED_PAIR_RE.finditer(s))
    if not ms:
        return s
    return s[: ms[-1].start()]


def has_reasoning_signal(reason_text: str) -> bool:
    t = (reason_text or "").lower()
    return any(k in t for k in REASONING_KEYWORDS)


# ── Main reward function ───────────────────────────────────────────────────

def reward_function(
    response: str,
    a: int,
    b: int,
    c: int,
    r1_gt: Optional[int] = None,
    r2_gt: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute reward for a single quadratic task response.

    Args:
        response: Model-generated text
        a, b, c: Coefficients of ax^2 + bx + c = 0
        r1_gt, r2_gt: Optional ground-truth roots (for logging, not used in reward)

    Returns:
        dict with 'reward' (float in [0,1]) and 'reward_info' (breakdown dict)
    """
    txt = response
    r_math = 0.0
    r_fmt = 0.0
    r_reason = 0.0
    deduction = 0.0

    strict_ok, mid_ok, loose_ok, bad = format_grade(txt)

    boxed_matches = list(BOXED_PAIR_RE.finditer(txt or ""))
    has_boxed_pair = bool(boxed_matches)
    has_multiple_pairs = len(boxed_matches) > 1
    trailing_after_boxed = False
    if has_boxed_pair:
        trailing_after_boxed = (txt[boxed_matches[-1].end():].strip() != "")

    parsed = parse_roots_from_answer(txt)

    if parsed is None:
        total = 0.005 if loose_ok else 0.0
        return {
            "reward": float(total),
            "reward_info": {
                "math_reward": 0.0,
                "format_reward": float(total),
                "reasoning_reward": 0.0,
                "deduction": 0.0,
                "parse_fail": True,
                "both_exact": False,
                "one_root": False,
            },
        }

    pr1, pr2, src, raw = parsed
    ok1, ok2, res1, res2 = verify_roots(a, b, c, pr1, pr2)

    sum_err = abs(a * (pr1 + pr2) + b)
    prod_err = abs(a * pr1 * pr2 - c)
    sum_ok = (sum_err == 0)
    prod_ok = (prod_err == 0)
    both_exact = bool(ok1 and ok2 and sum_ok and prod_ok)

    if both_exact:
        r_math = 0.82
    elif ok1 or ok2:
        r_math = 0.40

    # Dense math shaping
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

    # Format bonus gated by correctness
    valid_clean = bool(
        mid_ok and has_boxed_pair and (not has_multiple_pairs)
        and (not bad) and (not trailing_after_boxed)
    )

    if both_exact:
        r_fmt = 0.15 if valid_clean else (0.04 if loose_ok else 0.0)
    elif ok1 or ok2:
        r_fmt = 0.06 if valid_clean else (0.02 if loose_ok else 0.0)
    else:
        r_fmt = 0.01 if valid_clean else 0.0

    # Reasoning bonus
    reason_text = split_reasoning(txt)
    reason_len = len((reason_text or "").strip())
    if r_math >= 0.40:
        if reason_len >= MIN_THINK_CHARS:
            r_reason += 0.02
        if has_reasoning_signal(reason_text):
            r_reason += 0.01
        r_reason = min(0.03, r_reason)

    # Deductions
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
    if len(txt) > 1200:
        deduction -= 0.05

    total = r_math + r_fmt + r_reason + deduction

    # Anti-hack caps
    if not valid_clean:
        if both_exact:
            total = min(total, 0.65)
        elif ok1 or ok2:
            total = min(total, 0.35)
        else:
            total = min(total, 0.18)

    total = float(min(1.0, max(0.0, total)))

    return {
        "reward": total,
        "reward_info": {
            "math_reward": r_math,
            "format_reward": r_fmt,
            "reasoning_reward": r_reason,
            "deduction": deduction,
            "parse_fail": False,
            "both_exact": both_exact,
            "one_root": bool((ok1 or ok2) and not both_exact),
            "parsed_roots": (pr1, pr2),
            "gt_roots": (r1_gt, r2_gt) if r1_gt is not None else None,
        },
    }
