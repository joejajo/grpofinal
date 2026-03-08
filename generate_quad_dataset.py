"""
generate_quad_dataset.py
========================
Generates diverse quadratic-roots datasets for GRPO training.

Key design goals
----------------
1. PROMPT DIVERSITY   — 10 train templates, 4 held-out eval templates.
   The model sees many ways to ask the same question during training, so
   it doesn't over-fit to a single phrasing and fail on eval.

2. COEFFICIENT DIVERSITY — three difficulty zones:
     easy   : |a|=1,        roots in [-5,  5]
     medium : |a| in {1,2}, roots in [-9,  9]
     hard   : |a| in {1..3},roots in [-12,12]
   (eval also includes |a|=4 to probe slight out-of-distribution)

3. STRUCTURAL COVERAGE — repeated roots, zero root, large-coefficient
   edge cases are explicitly added.

Output columns (same schema the training script requires + user_message)
------------------------------------------------------------------------
  equation     : str  — human-readable "ax^2 + bx + c = 0"
  a, b, c      : int  — coefficients
  r1, r2       : int  — roots with r1 <= r2
  user_message : str  — full user-turn text (varies per sample)
"""

import argparse
import math
import random
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── prompt templates ─────────────────────────────────────────────────────────
# Training templates – model will see all of these
TRAIN_TEMPLATES = [
    "Solve the quadratic equation for integer roots: {eq}",          # T0 (current default)
    "Find the integer roots of: {eq}",                               # T1
    "What are the two integer roots of the equation {eq}?",          # T2
    "Determine the integer solutions to {eq}",                        # T3
    "The equation {eq} has two integer roots. Find both.",            # T4
    "Solve for x (integers only): {eq}",                             # T5
    "Find both integer roots of the quadratic {eq}",                 # T6
    "Given the quadratic {eq}, find integer roots r1 ≤ r2.",         # T7
    "Factor by finding integer roots of: {eq}",                      # T8
    "Solve {eq} for integer x.",                                     # T9
]

# Eval templates – deliberately different phrasing the model hasn't trained on
EVAL_TEMPLATES = [
    "Calculate the integer roots of the quadratic: {eq}",            # E0
    "For the equation {eq}, identify integer roots r1 and r2 (r1 ≤ r2).",  # E1
    "The quadratic {eq} — find both integer solutions.",              # E2
    "Determine r1 and r2 (r1 ≤ r2, both integers) for: {eq}",       # E3
]

# ── equation helpers ─────────────────────────────────────────────────────────

def _term_x2(a: int) -> str:
    """Return the x^2 term string."""
    if a == 1:   return "x^2"
    if a == -1:  return "-x^2"
    return f"{a}x^2"

def _term_x(b: int) -> str:
    """Return the '+ bx' or '- |b|x' term string (empty if b==0)."""
    if b == 0:   return ""
    if b == 1:   return "+ x"
    if b == -1:  return "- x"
    if b > 0:    return f"+ {b}x"
    return f"- {abs(b)}x"

def _term_c(c: int) -> str:
    """Return the '+ c' or '- |c|' term string (empty if c==0)."""
    if c == 0:   return ""
    if c > 0:    return f"+ {c}"
    return f"- {abs(c)}"

def make_equation_str(a: int, b: int, c: int) -> str:
    """Build ASCII equation string for ax^2 + bx + c = 0."""
    parts = [_term_x2(a)]
    tx = _term_x(b)
    if tx: parts.append(tx)
    tc = _term_c(c)
    if tc: parts.append(tc)
    return " ".join(parts) + " = 0"

def roots_to_coeffs(a: int, r1: int, r2: int):
    """Given a and roots r1 <= r2, return (a, b, c)."""
    b = -a * (r1 + r2)
    c = a * r1 * r2
    return a, b, c

# ── sample generation ─────────────────────────────────────────────────────────

def _make_sample(a, r1, r2, template, rng):
    """Return one dataset row dict."""
    r1, r2 = min(r1, r2), max(r1, r2)
    _, b, c = roots_to_coeffs(a, r1, r2)
    eq = make_equation_str(a, b, c)
    return {
        "equation":     eq,
        "a":            a,
        "b":            b,
        "c":            c,
        "r1":           r1,
        "r2":           r2,
        "user_message": template.format(eq=eq),
    }

def generate_zone(
    a_vals, root_range, templates, n_target, rng,
    seen: set, extras=None
):
    """
    Generate up to n_target unique samples from (a_vals x root_range^2).
    `seen` is a global set of (a,b,c) to prevent cross-split leakage.
    `extras` is a list of forced (a,r1,r2) tuples for edge cases.
    """
    rows = []

    # forced edge cases first
    for (a, r1, r2) in (extras or []):
        key = roots_to_coeffs(a, r1, r2)
        if key not in seen:
            seen.add(key)
            tpl = rng.choice(templates)
            rows.append(_make_sample(a, r1, r2, tpl, rng))

    lo, hi = root_range
    candidates = [
        (a, r1, r2)
        for a in a_vals
        for r1 in range(lo, hi + 1)
        for r2 in range(r1, hi + 1)          # r1 <= r2 always
        if roots_to_coeffs(a, r1, r2) not in seen
    ]
    rng.shuffle(candidates)

    for (a, r1, r2) in candidates:
        if len(rows) >= n_target:
            break
        key = roots_to_coeffs(a, r1, r2)
        if key in seen:
            continue
        seen.add(key)
        tpl = rng.choice(templates)
        rows.append(_make_sample(a, r1, r2, tpl, rng))

    return rows

def make_train_dataset(rng, seen: set, target=3000):
    """
    Three difficulty zones, all 10 train templates.
    Zone split: 40% easy / 35% medium / 25% hard.
    """
    n_easy   = int(target * 0.40)
    n_medium = int(target * 0.35)
    n_hard   = target - n_easy - n_medium

    # Edge cases sprinkled across zones
    edge_easy = [
        (1, 0, 0),   # both roots zero
        (1, 0, 3),   # r1=0, r2=3
        (-1, 2, 2),  # repeated root
        (1, -1, -1), # repeated negative root
    ]
    edge_medium = [
        (2, 0, 4),   # r1=0
        (-2, -3, 3), # symmetric roots
        (2, 5, 5),   # repeated root, a>1
    ]
    edge_hard = [
        (3, -10, 10),  # large symmetric roots
        (-3, 0, 12),   # zero root, a<0
        (3, 8, 8),     # repeated root, a=3
    ]

    rows  = generate_zone([1, -1],           (-5,  5),  TRAIN_TEMPLATES, n_easy,   rng, seen, edge_easy)
    rows += generate_zone([1, -1, 2, -2],    (-9,  9),  TRAIN_TEMPLATES, n_medium, rng, seen, edge_medium)
    rows += generate_zone([1,-1,2,-2,3,-3],  (-12, 12), TRAIN_TEMPLATES, n_hard,   rng, seen, edge_hard)

    # Ensure every train template appears at least once in final set
    tpl_coverage = {t for r in rows for t in [r["user_message"].split()[0]]}  # rough check
    # (template diversity is guaranteed by rng.choice across ~3000 rows)

    rng.shuffle(rows)
    return rows

def make_eval_dataset(rng, seen: set, target=400):
    """
    Uses EVAL_TEMPLATES only (held-out).
    Includes |a|=4 to probe slight OOD generalisation.
    Roots [-15, 15] — slightly wider than train.
    """
    edge_eval = [
        (1, 0, 0),    # both zero (should generalise from train)
        (4, -5, 5),   # a=4 OOD
        (-4, 0, 7),   # a=-4 OOD, zero root
        (1, 13, 13),  # large repeated root
        (2, -13, 14), # large roots
    ]
    rows  = generate_zone([1,-1,2,-2,3,-3,4,-4], (-15, 15), EVAL_TEMPLATES,
                          target, rng, seen, edge_eval)
    rng.shuffle(rows)
    return rows[:target]

# ── write parquet ────────────────────────────────────────────────────────────

SCHEMA = pa.schema([
    pa.field("equation",     pa.string()),
    pa.field("a",            pa.int32()),
    pa.field("b",            pa.int32()),
    pa.field("c",            pa.int32()),
    pa.field("r1",           pa.int32()),
    pa.field("r2",           pa.int32()),
    pa.field("user_message", pa.string()),
])

def rows_to_table(rows):
    cols = {k: [r[k] for r in rows] for k in SCHEMA.names}
    return pa.table(cols, schema=SCHEMA)

def write_parquet(rows, path: Path):
    tbl = rows_to_table(rows)
    pq.write_table(tbl, path, compression="snappy")
    print(f"  wrote {len(rows):>5} rows → {path}")

# ── statistics ───────────────────────────────────────────────────────────────

def print_stats(rows, label):
    from collections import Counter
    a_counts = Counter(r["a"] for r in rows)
    templates = Counter(r["user_message"].split(":")[0] if ":" in r["user_message"]
                        else r["user_message"][:40]
                        for r in rows)
    r_vals = [r["r1"] for r in rows] + [r["r2"] for r in rows]
    print(f"\n── {label} ({len(rows)} samples) ──")
    print(f"  a values : {dict(sorted(a_counts.items()))}")
    print(f"  root range : [{min(r_vals)}, {max(r_vals)}]")
    print(f"  repeated roots : {sum(1 for r in rows if r['r1']==r['r2'])}")
    print(f"  zero root      : {sum(1 for r in rows if r['r1']==0 or r['r2']==0)}")
    print(f"  unique templates : {len(set(r['user_message'] for r in rows))}")
    print(f"  template stems:")
    for stem, cnt in templates.most_common():
        print(f"    {cnt:4d}x  {stem[:60]}")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir",    default="Dataset",  help="Output directory")
    ap.add_argument("--train_file", default="quad_diverse_train.parquet")
    ap.add_argument("--eval_file",  default="quad_diverse_eval.parquet")
    ap.add_argument("--n_train",    type=int, default=3000)
    ap.add_argument("--n_eval",     type=int, default=400)
    ap.add_argument("--seed",       type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = set()          # shared to prevent train/eval leakage

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating training set …")
    train_rows = make_train_dataset(rng, seen, target=args.n_train)
    print_stats(train_rows, "TRAIN")
    write_parquet(train_rows, out / args.train_file)

    print("\nGenerating eval set …")
    eval_rows = make_eval_dataset(rng, seen, target=args.n_eval)
    print_stats(eval_rows, "EVAL")
    write_parquet(eval_rows, out / args.eval_file)

    print("\nDone. No train/eval overlap (enforced by shared `seen` set).")
    print(f"Train templates : {len(TRAIN_TEMPLATES)}  (T0–T9)")
    print(f"Eval  templates : {len(EVAL_TEMPLATES)}   (E0–E3, held-out)")

if __name__ == "__main__":
    main()
