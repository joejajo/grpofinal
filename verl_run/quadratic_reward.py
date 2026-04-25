#!/usr/bin/env python3
"""Custom VERL reward function for quadratic integer roots.

Ported from grpo_quad_train_v12.py:combined_reward_func — same scoring logic,
adapted to verl's per-sample compute_score signature.

Hooked via:
  custom_reward_function.path=/abs/path/to/quadratic_reward.py
  custom_reward_function.name=compute_score
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, Optional, Tuple

# ── Constants (ported from v12) ──────────────────────────────────────────────
MIN_THINK_CHARS = 80
REASONING_KEYWORDS = (
    "discriminant", "delta", "sqrt", "quadratic formula", "factor", "vieta",
    "b^2-4ac", "+/-", "formula", "x =", "solve", "coefficient",
    "product of roots", "sum of roots", "roots are",
)

# ── Regexes (ported from v12) ────────────────────────────────────────────────
BOXED_PAIR_RE = re.compile(
    r"\\boxed\{\s*\(?\s*([-+]?\d+)(?:\.0+)?\s*,\s*([-+]?\d+)(?:\.0+)?\s*\)?\s*\}"
)
ROOT_LABEL_PAIR_RE = re.compile(
    r"r1\s*=\s*([-+]?\d+)(?!\.\d).{0,80}?r2\s*=\s*([-+]?\d+)(?!\.\d)",
    re.IGNORECASE | re.DOTALL,
)
ROOT_LABEL_PAIR_RE_REV = re.compile(
    r"r2\s*=\s*([-+]?\d+)(?!\.\d).{0,80}?r1\s*=\s*([-+]?\d+)(?!\.\d)",
    re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")
ESCAPED_TAG_RE = re.compile(r"\\</?[a-zA-Z][^>]*>")


# ── Parsing helpers (ported from v12) ────────────────────────────────────────
def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _parse_roots_from_answer(
    txt: str,
) -> Optional[Tuple[int, int, str, Tuple[int, int]]]:
    ms = list(BOXED_PAIR_RE.finditer(txt or ""))
    if not ms:
        return None
    r1_raw = int(ms[-1].group(1))
    r2_raw = int(ms[-1].group(2))
    return (min(r1_raw, r2_raw), max(r1_raw, r2_raw), "boxed_pair", (r1_raw, r2_raw))


def _parse_roots_from_labeled_text(
    txt: str,
) -> Optional[Tuple[int, int, str, Tuple[int, int]]]:
    ms = list(ROOT_LABEL_PAIR_RE.finditer(txt))
    if ms:
        m = ms[-1]
        r1_raw = int(m.group(1))
        r2_raw = int(m.group(2))
        return (min(r1_raw, r2_raw), max(r1_raw, r2_raw), "label_pair_text",
                (r1_raw, r2_raw))
    ms2 = list(ROOT_LABEL_PAIR_RE_REV.finditer(txt))
    if ms2:
        m2 = ms2[-1]
        r2_raw = int(m2.group(1))
        r1_raw = int(m2.group(2))
        return (min(r1_raw, r2_raw), max(r1_raw, r2_raw), "label_pair_text",
                (r1_raw, r2_raw))
    return None


def _verify_roots(a: int, b: int, c: int, r1: int, r2: int) -> Tuple[bool, bool, int, int]:
    res1 = a * r1 * r1 + b * r1 + c
    res2 = a * r2 * r2 + b * r2 + c
    return (res1 == 0), (res2 == 0), res1, res2


def _format_grade(txt: str) -> Tuple[bool, bool, bool, bool]:
    s = txt or ""
    ms = list(BOXED_PAIR_RE.finditer(s))
    loose_ok = bool(ms)
    strict_ok = False
    mid_ok = False
    if loose_ok:
        first, last = ms[0], ms[-1]
        prefix = s[: first.start()].strip()
        suffix = s[last.end():].strip()
        strict_ok = (len(ms) == 1 and prefix == "" and suffix == "")
        mid_ok = (len(ms) == 1 and suffix == "")
    bad_tags = bool(TAG_RE.search(s) or ESCAPED_TAG_RE.search(s))
    return strict_ok, mid_ok, loose_ok, bad_tags


def _split_reasoning(txt: str) -> str:
    s = txt or ""
    ms = list(BOXED_PAIR_RE.finditer(s))
    if not ms:
        return s
    return s[: ms[-1].start()]


def _has_reasoning_signal(reason_text: str) -> bool:
    t = (reason_text or "").lower()
    return any(k in t for k in REASONING_KEYWORDS)


# ── verl entry point ─────────────────────────────────────────────────────────
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> float:
    """Per-sample reward in [0,1]. Behaviour matches v12 'strict_phase' (post step 150),
    since verl does not pass a step index to this function.

      - math: 0 / 0.40 / 0.82 (+ dense near-miss shaping capped inside math budget)
      - format bonus: gated by correctness and clean final boxed output
      - reasoning bonus: only when math >= 0.40
      - deductions: malformed output patterns
      - labeled-text fallback (r1=, r2=): heavily capped (strict_phase caps).
    """
    if data_source != "quadratic/roots":
        return 0.0

    # Coefficients from extra_info (set in prepare_quadratic_verl_data.py).
    info = extra_info or {}
    ai = _safe_int(info.get("a"))
    bi = _safe_int(info.get("b"))
    ci = _safe_int(info.get("c"))
    if ai is None or bi is None or ci is None:
        return 0.0

    txt = solution_str or ""

    strict_ok, mid_ok, loose_ok, bad = _format_grade(txt)
    boxed_matches = list(BOXED_PAIR_RE.finditer(txt))
    has_boxed_pair = bool(boxed_matches)
    has_multiple_pairs = len(boxed_matches) > 1
    trailing_after_boxed = bool(
        has_boxed_pair and txt[boxed_matches[-1].end():].strip() != ""
    )

    parsed = _parse_roots_from_answer(txt)
    # strict_phase: do NOT use labeled-text fallback for parsing (caps would zero it anyway).
    if parsed is None:
        # Tiny crumb for loose format only — matches v12 behaviour for parse_fail with loose_ok.
        return 0.005 if loose_ok else 0.0

    pr1, pr2, src, raw = parsed
    ok1, ok2, res1, res2 = _verify_roots(ai, bi, ci, pr1, pr2)

    sum_err = abs(ai * (pr1 + pr2) + bi)
    prod_err = abs(ai * pr1 * pr2 - ci)
    sum_ok = (sum_err == 0)
    prod_ok = (prod_err == 0)
    both_exact = bool(ok1 and ok2 and sum_ok and prod_ok)

    r_math = 0.0
    if both_exact:
        r_math = 0.82
    elif ok1 or ok2:
        r_math = 0.40

    # Dense math shaping — capped within the math budget.
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

    # Format bonus.
    valid_clean = bool(
        mid_ok and has_boxed_pair and (not has_multiple_pairs)
        and (not bad) and (not trailing_after_boxed)
    )
    r_fmt = 0.0
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

    # Reasoning bonus — only when math is non-trivial.
    r_reason = 0.0
    if r_math >= 0.40:
        reason_text = _split_reasoning(txt)
        reason_len = len((reason_text or "").strip())
        if reason_len >= MIN_THINK_CHARS:
            r_reason += 0.02
        if _has_reasoning_signal(reason_text):
            r_reason += 0.01
        r_reason = min(0.03, r_reason)

    # Negative deductions for hack-prone patterns.
    deduction = 0.0
    low = txt.lower()
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

    # Anti-hack caps: high reward requires clean format.
    if not valid_clean:
        if both_exact:
            total = min(total, 0.65)
        elif (ok1 or ok2):
            total = min(total, 0.35)
        else:
            total = min(total, 0.18)

    # Strict-phase cap on labeled-text fallback (v12 lines 952–958).
    if src == "label_pair_text":
        if both_exact:
            total = min(total, 0.04)
        elif (ok1 or ok2):
            total = min(total, 0.02)
        else:
            total = min(total, 0.005)

    return float(min(1.0, max(0.0, total)))
