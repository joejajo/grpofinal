#!/usr/bin/env python3
"""
Plot GRPO finetuning results for thesis presentation.

Reads TensorBoard event files and optionally CSV logs from a GRPO experiment
directory and generates publication-quality figures:
  1. Total Reward vs Step
  2. Reward Components Breakdown vs Step
  3. Training Accuracy Rates vs Step (both_roots, one_root, parse_fail)
  4. Evaluation Accuracy vs Step (exact_both_roots on held-out set)
  5. Completion Length vs Step
  6. Loss vs Step
  7. Reward Distribution Histogram (early vs late)
  8. Combined Summary Dashboard (2x3 grid)

Usage:
    python plot_grpo_results.py --exp_dir <path_to_grpo_output>
    python plot_grpo_results.py --exp_dir <path> --save_dir <output_dir>
    python plot_grpo_results.py --exp_dir <path> --csv   # also read CSVs
"""

import argparse
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── TensorBoard reading ─────────────────────────────────────────────────────
def read_tensorboard_scalars(logdir: str) -> dict[str, list[tuple[int, float]]]:
    """Return {tag: [(step, value), ...]} from all event files under logdir."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError("tensorboard is required: pip install tensorboard")

    ea = EventAccumulator(logdir)
    ea.Reload()

    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    return data


def extract(data, tag):
    """Return (steps, values) numpy arrays for a given tag."""
    if tag not in data:
        return None, None
    pairs = data[tag]
    steps = np.array([p[0] for p in pairs])
    vals = np.array([p[1] for p in pairs])
    return steps, vals


def smooth(values, weight=0.6):
    """Exponential moving average for smoother curves."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


# ── CSV reading ──────────────────────────────────────────────────────────────
def read_csv(exp_dir: str, filename: str):
    """Read a CSV file from exp_dir, return list of dicts or None."""
    import csv
    csv_path = os.path.join(exp_dir, filename)
    if not os.path.isfile(csv_path):
        # Also check subdirectories
        candidates = glob.glob(os.path.join(exp_dir, "**", filename), recursive=True)
        if not candidates:
            return None
        csv_path = candidates[0]
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── Styling ──────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}

COLORS = {
    "total_reward": "#2196F3",
    "equation": "#1565C0",
    "format": "#4CAF50",
    "reasoning": "#FF9800",
    "penalty": "#F44336",
    "both": "#4CAF50",
    "one": "#FF9800",
    "parse_fail": "#F44336",
    "eval_acc": "#9C27B0",
    "eval_one": "#FF9800",
    "eval_pf": "#F44336",
    "eval_gt": "#1565C0",
    "loss": "#E91E63",
    "comp_len": "#607D8B",
    "smooth": "#B0BEC5",
}


def apply_style():
    plt.rcParams.update(STYLE)


# ── Individual plots ─────────────────────────────────────────────────────────
def plot_total_reward(data, save_dir):
    """Total reward (mean of reward function outputs) vs step."""
    # The trainer logs 'reward' or we can sum components
    s, eq = extract(data, "rewards/equation_reward_func")
    _, fm = extract(data, "rewards/format_reward_func")
    _, re = extract(data, "rewards/reasoning_reward_func")
    _, pn = extract(data, "rewards/penalty_term")

    if s is None:
        # Try the built-in reward tag
        s, total = extract(data, "reward")
        if s is None:
            print("  [skip] no reward tags found")
            return
    else:
        total = eq
        if fm is not None:
            total = total + fm
        if re is not None:
            total = total + re
        if pn is not None:
            total = total + pn

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s, total, color=COLORS["total_reward"], alpha=0.3, linewidth=1)
    ax.plot(s, smooth(total), color=COLORS["total_reward"], label="Total Reward (smoothed)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")
    ax.set_title("GRPO Training — Total Reward vs Step")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.savefig(os.path.join(save_dir, "grpo_reward_curve.png"))
    plt.close(fig)
    print("  ✓ grpo_reward_curve.png")


def plot_reward_components(data, save_dir):
    """Reward component breakdown: equation, format, reasoning, penalty."""
    s, eq = extract(data, "rewards/equation_reward_func")
    if s is None:
        print("  [skip] reward component tags not found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s, smooth(eq), color=COLORS["equation"], label="Equation (math)")

    _, fm = extract(data, "rewards/format_reward_func")
    if fm is not None:
        ax.plot(s, smooth(fm), color=COLORS["format"], label="Format")

    _, re = extract(data, "rewards/reasoning_reward_func")
    if re is not None:
        ax.plot(s, smooth(re), color=COLORS["reasoning"], label="Reasoning")

    _, pn = extract(data, "rewards/penalty_term")
    if pn is not None:
        ax.plot(s, smooth(pn), color=COLORS["penalty"], label="Penalty")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward Component")
    ax.set_title("GRPO Training — Reward Components Breakdown")
    ax.legend()
    fig.savefig(os.path.join(save_dir, "grpo_reward_components.png"))
    plt.close(fig)
    print("  ✓ grpo_reward_components.png")


def plot_training_accuracy(data, save_dir):
    """Training accuracy rates (both_roots, one_root, parse_fail) vs step."""
    s, both = extract(data, "rewards/both_roots_rate")
    _, one = extract(data, "rewards/one_root_rate")
    _, pf = extract(data, "rewards/parse_fail_rate")

    if s is None:
        print("  [skip] training accuracy tags not found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if both is not None:
        ax.plot(s, smooth(both * 100), color=COLORS["both"], label="Both Roots Correct")
    if one is not None:
        ax.plot(s, smooth(one * 100), color=COLORS["one"], label="One Root Only")
    if pf is not None:
        ax.plot(s, smooth(pf * 100), color=COLORS["parse_fail"], label="Parse Fail")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate (%)")
    ax.set_title("GRPO Training — Accuracy Rates vs Step")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend()
    ax.set_ylim(0, 105)
    fig.savefig(os.path.join(save_dir, "grpo_training_accuracy.png"))
    plt.close(fig)
    print("  ✓ grpo_training_accuracy.png")


def plot_eval_accuracy(data, save_dir):
    """Evaluation accuracy on held-out dataset vs step."""
    s, acc = extract(data, "eval/exact_both_roots_accuracy")
    _, e_one = extract(data, "eval/one_root_rate")
    _, e_pf = extract(data, "eval/parse_fail_rate")
    _, e_gt = extract(data, "eval/gt_match_rate")

    if s is None:
        print("  [skip] eval accuracy tags not found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s, acc * 100, color=COLORS["eval_acc"], marker="o", markersize=4,
            label="Both Roots Exact (Eval)")
    if e_one is not None:
        ax.plot(s, e_one * 100, color=COLORS["eval_one"], marker="s", markersize=3,
                label="One Root (Eval)")
    if e_pf is not None:
        ax.plot(s, e_pf * 100, color=COLORS["eval_pf"], marker="^", markersize=3,
                label="Parse Fail (Eval)")
    if e_gt is not None:
        ax.plot(s, e_gt * 100, color=COLORS["eval_gt"], marker="D", markersize=3,
                linestyle="--", label="GT Match (Eval)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate (%)")
    ax.set_title("GRPO — Evaluation Accuracy on Held-Out Set")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend()
    ax.set_ylim(0, 105)
    fig.savefig(os.path.join(save_dir, "grpo_eval_accuracy.png"))
    plt.close(fig)
    print("  ✓ grpo_eval_accuracy.png")


def plot_loss(data, save_dir):
    """Training loss vs step."""
    s, loss = extract(data, "loss")
    if s is None:
        # Try train/loss
        s, loss = extract(data, "train/loss")
    if s is None:
        print("  [skip] loss tag not found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s, loss, color=COLORS["loss"], alpha=0.3, linewidth=1)
    ax.plot(s, smooth(loss), color=COLORS["loss"], label="Loss (smoothed)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("GRPO Training — Loss vs Step")
    ax.legend()
    fig.savefig(os.path.join(save_dir, "grpo_loss.png"))
    plt.close(fig)
    print("  ✓ grpo_loss.png")


def plot_completion_length(data, save_dir):
    """Completion length (chars) vs step."""
    s, clen = extract(data, "completion_length_chars")
    if s is None:
        print("  [skip] completion_length_chars not found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s, clen, color=COLORS["comp_len"], alpha=0.3, linewidth=1)
    ax.plot(s, smooth(clen), color=COLORS["comp_len"], label="Avg Completion Length")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Characters")
    ax.set_title("GRPO Training — Completion Length vs Step")
    ax.legend()
    fig.savefig(os.path.join(save_dir, "grpo_completion_length.png"))
    plt.close(fig)
    print("  ✓ grpo_completion_length.png")


def plot_reward_histogram_from_csv(csv_rows, save_dir):
    """Histogram of per-sample rewards at early vs late training steps."""
    if csv_rows is None:
        print("  [skip] train CSV not available for reward histogram")
        return

    steps = sorted(set(int(r["step"]) for r in csv_rows))
    if len(steps) < 2:
        print("  [skip] need >= 2 logged steps for histogram")
        return

    first_step, last_step = steps[0], steps[-1]
    rewards_first = [float(r["reward"]) for r in csv_rows if int(r["step"]) == first_step]
    rewards_last = [float(r["reward"]) for r in csv_rows if int(r["step"]) == last_step]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 25)
    ax.hist(rewards_first, bins=bins, alpha=0.5, label=f"Step {first_step}", color="#90CAF9")
    ax.hist(rewards_last, bins=bins, alpha=0.5, label=f"Step {last_step}", color="#4CAF50")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title("GRPO — Reward Distribution: Early vs Late Training")
    ax.legend()
    fig.savefig(os.path.join(save_dir, "grpo_reward_histogram.png"))
    plt.close(fig)
    print("  ✓ grpo_reward_histogram.png")


def plot_eval_progression_from_csv(csv_rows, save_dir):
    """Per-step eval accuracy computed from eval_generations.csv."""
    if csv_rows is None:
        print("  [skip] eval CSV not available")
        return

    steps = sorted(set(int(r["step"]) for r in csv_rows))
    both_rates, one_rates, pf_rates = [], [], []
    for step in steps:
        rows_s = [r for r in csv_rows if int(r["step"]) == step]
        n = len(rows_s)
        if n == 0:
            continue
        both_rates.append(sum(1 for r in rows_s if r["both_exact"].lower() == "true") / n * 100)
        one_rates.append(sum(1 for r in rows_s if r["one_root"].lower() == "true") / n * 100)
        pf_rates.append(sum(1 for r in rows_s if r["parse_fail"].lower() == "true") / n * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, both_rates, color=COLORS["eval_acc"], marker="o", markersize=4,
            label="Both Roots Exact")
    ax.plot(steps, one_rates, color=COLORS["eval_one"], marker="s", markersize=3,
            label="One Root Only")
    ax.plot(steps, pf_rates, color=COLORS["eval_pf"], marker="^", markersize=3,
            label="Parse Fail")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate (%)")
    ax.set_title("GRPO — Eval Accuracy Progression (from CSV)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend()
    ax.set_ylim(0, 105)
    fig.savefig(os.path.join(save_dir, "grpo_eval_progression_csv.png"))
    plt.close(fig)
    print("  ✓ grpo_eval_progression_csv.png")


def plot_dashboard(data, save_dir):
    """Combined 2x3 dashboard for thesis slides."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("GRPO Finetuning — Training Summary", fontsize=16, fontweight="bold")

    # (0,0) Total reward
    s, eq = extract(data, "rewards/equation_reward_func")
    _, fm = extract(data, "rewards/format_reward_func")
    _, re = extract(data, "rewards/reasoning_reward_func")
    _, pn = extract(data, "rewards/penalty_term")
    if s is not None:
        total = eq + (fm if fm is not None else 0) + (re if re is not None else 0) + (pn if pn is not None else 0)
        axes[0, 0].plot(s, smooth(total), color=COLORS["total_reward"])
    axes[0, 0].set_title("Total Reward")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylim(bottom=0)

    # (0,1) Reward components
    if s is not None:
        axes[0, 1].plot(s, smooth(eq), color=COLORS["equation"], label="Math")
        if fm is not None:
            axes[0, 1].plot(s, smooth(fm), color=COLORS["format"], label="Format")
        if re is not None:
            axes[0, 1].plot(s, smooth(re), color=COLORS["reasoning"], label="Reasoning")
        if pn is not None:
            axes[0, 1].plot(s, smooth(pn), color=COLORS["penalty"], label="Penalty")
        axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("Reward Components")
    axes[0, 1].set_xlabel("Step")

    # (0,2) Training accuracy
    s2, both = extract(data, "rewards/both_roots_rate")
    _, one = extract(data, "rewards/one_root_rate")
    if s2 is not None:
        axes[0, 2].plot(s2, smooth(both * 100), color=COLORS["both"], label="Both Roots")
        if one is not None:
            axes[0, 2].plot(s2, smooth(one * 100), color=COLORS["one"], label="One Root")
        axes[0, 2].legend(fontsize=8)
        axes[0, 2].yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    axes[0, 2].set_title("Training Accuracy (%)")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylim(0, 105)

    # (1,0) Eval accuracy
    s3, eacc = extract(data, "eval/exact_both_roots_accuracy")
    if s3 is not None:
        axes[1, 0].plot(s3, eacc * 100, color=COLORS["eval_acc"], marker="o", markersize=3)
        axes[1, 0].yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    axes[1, 0].set_title("Eval Accuracy (%)")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylim(0, 105)

    # (1,1) Loss
    s4, loss = extract(data, "loss")
    if s4 is None:
        s4, loss = extract(data, "train/loss")
    if s4 is not None:
        axes[1, 1].plot(s4, smooth(loss), color=COLORS["loss"])
    axes[1, 1].set_title("Training Loss")
    axes[1, 1].set_xlabel("Step")

    # (1,2) Completion length
    s5, clen = extract(data, "completion_length_chars")
    if s5 is not None:
        axes[1, 2].plot(s5, smooth(clen), color=COLORS["comp_len"])
    axes[1, 2].set_title("Avg Completion Length")
    axes[1, 2].set_xlabel("Step")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(save_dir, "grpo_dashboard.png"))
    plt.close(fig)
    print("  ✓ grpo_dashboard.png")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot GRPO finetuning results")
    parser.add_argument("--exp_dir", required=True,
                        help="Path to GRPO experiment/output directory")
    parser.add_argument("--save_dir", default=None,
                        help="Where to save plots (default: <exp_dir>/plots)")
    parser.add_argument("--csv", action="store_true",
                        help="Also read CSV files for per-sample plots")
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(args.exp_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    apply_style()

    # Find TensorBoard logdir
    event_files = glob.glob(os.path.join(args.exp_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        print(f"ERROR: No TensorBoard event files found under {args.exp_dir}")
        return
    logdir = os.path.dirname(event_files[0])
    print(f"Reading TensorBoard events from: {logdir}")

    data = read_tensorboard_scalars(logdir)
    print(f"Found tags: {list(data.keys())}")

    train_csv = read_csv(args.exp_dir, "train_generations.csv") if args.csv else None
    eval_csv = read_csv(args.exp_dir, "eval_generations.csv") if args.csv else None
    if args.csv:
        if train_csv:
            print(f"Read {len(train_csv)} rows from train_generations.csv")
        if eval_csv:
            print(f"Read {len(eval_csv)} rows from eval_generations.csv")

    print("\nGenerating plots:")
    plot_total_reward(data, save_dir)
    plot_reward_components(data, save_dir)
    plot_training_accuracy(data, save_dir)
    plot_eval_accuracy(data, save_dir)
    plot_loss(data, save_dir)
    plot_completion_length(data, save_dir)
    plot_dashboard(data, save_dir)

    if train_csv:
        plot_reward_histogram_from_csv(train_csv, save_dir)
    if eval_csv:
        plot_eval_progression_from_csv(eval_csv, save_dir)

    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
