#!/usr/bin/env python3
"""
Plot ES (Evolutionary Strategies) finetuning results for thesis presentation.

Reads TensorBoard event files from an ES experiment directory and generates
publication-quality figures:
  1. Mean Reward vs Iteration
  2. Accuracy Rates vs Iteration (both_exact, one_root, parse_fail)
  3. Best Reward vs Iteration
  4. Reward Distribution (mean +/- std ribbon)
  5. Timing Breakdown per Iteration
  6. Combined Summary Dashboard (2x3 grid)
  7. Eval Results Summary (from post-training eval_summary.json)

Usage:
    python plot_es_results.py --exp_dir <path_to_es_experiment>
    python plot_es_results.py --exp_dir <path> --save_dir <output_dir>
    python plot_es_results.py --exp_dir <path> --csv   # also read CSV for per-sample plots
    python plot_es_results.py --exp_dir <path> --eval_dir <path_to_eval_results>
"""

import argparse
import json
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
        raise ImportError(
            "tensorboard is required: pip install tensorboard"
        )

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


# ── CSV reading (optional, for per-sample reward histograms) ─────────────────
def read_csv(exp_dir: str):
    """Read train_generations.csv if it exists."""
    import csv
    csv_path = os.path.join(exp_dir, "train_generations.csv")
    if not os.path.isfile(csv_path):
        return None
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
    "reward": "#2196F3",
    "best": "#4CAF50",
    "both": "#4CAF50",
    "one": "#FF9800",
    "parse_fail": "#F44336",
    "std_fill": "#BBDEFB",
    "perturb": "#9C27B0",
    "broadcast": "#FF5722",
    "iter_time": "#607D8B",
}


def apply_style():
    plt.rcParams.update(STYLE)


# ── Individual plots ─────────────────────────────────────────────────────────
def plot_reward_curve(data, save_dir):
    """Mean reward vs iteration with std ribbon."""
    steps, mean = extract(data, "reward/mean")
    _, std = extract(data, "reward/std")
    if steps is None:
        print("  [skip] reward/mean not found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, mean, color=COLORS["reward"], label="Mean Reward")
    if std is not None:
        ax.fill_between(steps, mean - std, mean + std,
                        alpha=0.2, color=COLORS["std_fill"], label="± 1 Std Dev")
    _, best = extract(data, "reward/best")
    if best is not None:
        ax.plot(steps, best, color=COLORS["best"], linestyle="--", label="Best Reward")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.set_title("ES Training — Reward vs Iteration")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.savefig(os.path.join(save_dir, "es_reward_curve.png"))
    plt.close(fig)
    print("  ✓ es_reward_curve.png")


def plot_accuracy_rates(data, save_dir):
    """Accuracy rates (both_exact, one_root, parse_fail) vs iteration."""
    steps, both = extract(data, "task/both_exact_rate")
    _, one = extract(data, "task/one_root_rate")
    _, pf = extract(data, "task/parse_fail_rate")

    if steps is None and both is None:
        print("  [skip] accuracy rate tags not found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if both is not None:
        ax.plot(steps, both * 100, color=COLORS["both"], label="Both Roots Exact")
    if one is not None:
        ax.plot(steps, one * 100, color=COLORS["one"], label="One Root Only")
    if pf is not None:
        ax.plot(steps, pf * 100, color=COLORS["parse_fail"], label="Parse Fail")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Rate (%)")
    ax.set_title("ES Training — Accuracy Rates vs Iteration")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend()
    ax.set_ylim(0, 105)
    fig.savefig(os.path.join(save_dir, "es_accuracy_rates.png"))
    plt.close(fig)
    print("  ✓ es_accuracy_rates.png")


def plot_reward_components_from_csv(csv_rows, save_dir):
    """Stacked area of reward components from CSV (math, format, reasoning, deduction)."""
    if csv_rows is None:
        print("  [skip] CSV not available for reward components")
        return

    iters = sorted(set(int(r["iteration"]) for r in csv_rows))
    math_avg, fmt_avg, reas_avg, ded_avg = [], [], [], []
    for it in iters:
        rows_it = [r for r in csv_rows if int(r["iteration"]) == it]
        math_avg.append(np.mean([float(r["math_reward"]) for r in rows_it]))
        fmt_avg.append(np.mean([float(r["format_reward"]) for r in rows_it]))
        reas_avg.append(np.mean([float(r["reasoning_reward"]) for r in rows_it]))
        ded_avg.append(np.mean([float(r["deduction"]) for r in rows_it]))

    iters = np.array(iters)
    math_avg = np.array(math_avg)
    fmt_avg = np.array(fmt_avg)
    reas_avg = np.array(reas_avg)
    ded_avg = np.array(ded_avg)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stackplot(iters, math_avg, fmt_avg, reas_avg,
                 labels=["Math Reward", "Format Reward", "Reasoning Reward"],
                 colors=["#2196F3", "#4CAF50", "#FF9800"], alpha=0.7)
    ax.plot(iters, ded_avg, color=COLORS["parse_fail"], linewidth=2,
            label="Deductions (penalty)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward Component")
    ax.set_title("ES Training — Reward Component Breakdown")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(save_dir, "es_reward_components.png"))
    plt.close(fig)
    print("  ✓ es_reward_components.png")


def plot_baseline_vs_final(data, save_dir):
    """Bar chart comparing first iteration vs last iteration metrics."""
    s, both = extract(data, "task/both_exact_rate")
    _, one = extract(data, "task/one_root_rate")
    _, pf = extract(data, "task/parse_fail_rate")
    s_r, mean_r = extract(data, "reward/mean")

    if s is None or len(s) < 2:
        print("  [skip] need >= 2 iterations for baseline vs final")
        return

    labels = ["Both Roots\nExact", "One Root\nOnly", "Parse\nFail", "Mean\nReward"]
    bl_vals = [
        both[0] * 100,
        one[0] * 100 if one is not None else 0,
        pf[0] * 100 if pf is not None else 0,
        mean_r[0] * 100 if mean_r is not None else 0,
    ]
    fn_vals = [
        both[-1] * 100,
        one[-1] * 100 if one is not None else 0,
        pf[-1] * 100 if pf is not None else 0,
        mean_r[-1] * 100 if mean_r is not None else 0,
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width / 2, bl_vals, width,
                   label=f"Iteration {int(s[0])} (start)",
                   color="#90CAF9", edgecolor="white")
    bars2 = ax.bar(x + width / 2, fn_vals, width,
                   label=f"Iteration {int(s[-1])} (final)",
                   color="#4CAF50", edgecolor="white")

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold")

    ax.set_ylabel("Rate / Reward (%)")
    ax.set_title("ES Training — Baseline vs Post-Training")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, max(max(bl_vals), max(fn_vals)) * 1.2 + 5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    fig.savefig(os.path.join(save_dir, "es_baseline_vs_final.png"))
    plt.close(fig)
    print("  ✓ es_baseline_vs_final.png")


def plot_reward_histogram_from_csv(csv_rows, save_dir):
    """Histogram of per-sample rewards at first and last logged iteration."""
    if csv_rows is None:
        print("  [skip] CSV not available for reward histogram")
        return

    iters = sorted(set(int(r["iteration"]) for r in csv_rows))
    if len(iters) < 2:
        print("  [skip] need >= 2 logged iterations for histogram")
        return

    first_it, last_it = iters[0], iters[-1]
    rewards_first = [float(r["reward"]) for r in csv_rows if int(r["iteration"]) == first_it]
    rewards_last = [float(r["reward"]) for r in csv_rows if int(r["iteration"]) == last_it]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 25)
    ax.hist(rewards_first, bins=bins, alpha=0.5, label=f"Iteration {first_it}", color="#90CAF9")
    ax.hist(rewards_last, bins=bins, alpha=0.5, label=f"Iteration {last_it}", color="#4CAF50")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title("ES — Reward Distribution: Early vs Late")
    ax.legend()
    fig.savefig(os.path.join(save_dir, "es_reward_histogram.png"))
    plt.close(fig)
    print("  ✓ es_reward_histogram.png")


def plot_timing(data, save_dir):
    """Timing breakdown per iteration."""
    steps, t_perturb = extract(data, "time/perturbation_application")
    _, t_broad = extract(data, "time/broadcast")
    _, t_iter = extract(data, "time/iteration")

    if steps is None:
        print("  [skip] timing tags not found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if t_iter is not None:
        ax.plot(steps, t_iter, color=COLORS["iter_time"], label="Total Iteration")
    if t_perturb is not None:
        ax.plot(steps, t_perturb, color=COLORS["perturb"], label="Perturbation")
    if t_broad is not None:
        ax.plot(steps, t_broad, color=COLORS["broadcast"], label="Broadcast")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time (s)")
    ax.set_title("ES Training — Timing per Iteration")
    ax.legend()
    fig.savefig(os.path.join(save_dir, "es_timing.png"))
    plt.close(fig)
    print("  ✓ es_timing.png")


def plot_dashboard(data, save_dir):
    """Combined 2x3 dashboard for thesis slides."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("ES Finetuning — Training Summary", fontsize=16, fontweight="bold")

    # (0,0) Mean reward
    s, m = extract(data, "reward/mean")
    _, sd = extract(data, "reward/std")
    if s is not None:
        axes[0, 0].plot(s, m, color=COLORS["reward"])
        if sd is not None:
            axes[0, 0].fill_between(s, m - sd, m + sd, alpha=0.15, color=COLORS["std_fill"])
    axes[0, 0].set_title("Mean Reward")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylim(bottom=0)

    # (0,1) Best reward
    s2, b = extract(data, "reward/best")
    if s2 is not None:
        axes[0, 1].plot(s2, b, color=COLORS["best"])
    axes[0, 1].set_title("Best Reward")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylim(bottom=0)

    # (0,2) Both exact rate
    s3, both = extract(data, "task/both_exact_rate")
    if s3 is not None:
        axes[0, 2].plot(s3, both * 100, color=COLORS["both"])
        axes[0, 2].yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    axes[0, 2].set_title("Both Roots Exact (%)")
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylim(0, 105)

    # (1,0) One root rate
    _, one = extract(data, "task/one_root_rate")
    if s3 is not None and one is not None:
        axes[1, 0].plot(s3, one * 100, color=COLORS["one"])
        axes[1, 0].yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    axes[1, 0].set_title("One Root Only (%)")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylim(0, 105)

    # (1,1) Parse fail rate
    _, pf = extract(data, "task/parse_fail_rate")
    if s3 is not None and pf is not None:
        axes[1, 1].plot(s3, pf * 100, color=COLORS["parse_fail"])
        axes[1, 1].yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    axes[1, 1].set_title("Parse Fail Rate (%)")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylim(0, 105)

    # (1,2) Timing
    s4, t_iter = extract(data, "time/iteration")
    if s4 is not None:
        axes[1, 2].plot(s4, t_iter, color=COLORS["iter_time"])
    axes[1, 2].set_title("Iteration Time (s)")
    axes[1, 2].set_xlabel("Iteration")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(save_dir, "es_dashboard.png"))
    plt.close(fig)
    print("  ✓ es_dashboard.png")


# ── Eval results plotting ─────────────────────────────────────────────────────
def read_eval_summary(eval_dir: str):
    """Read eval_summary.json from eval output directory."""
    path = os.path.join(eval_dir, "eval_summary.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def read_eval_csv(eval_dir: str):
    """Read eval_results.csv from eval output directory."""
    import csv as csvmod
    path = os.path.join(eval_dir, "eval_results.csv")
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path, "r") as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_eval_summary(eval_summary, save_dir):
    """Bar chart of eval metrics from eval_summary.json."""
    if eval_summary is None:
        print("  [skip] eval_summary.json not found")
        return

    labels = ["Both Roots\nExact", "One Root\nOnly", "Parse\nFail", "Mean\nReward"]
    values = [
        eval_summary["both_exact_rate"] * 100,
        eval_summary["one_root_rate"] * 100,
        eval_summary["parse_fail_rate"] * 100,
        eval_summary["mean_reward"] * 100,
    ]
    bar_colors = [COLORS["both"], COLORS["one"], COLORS["parse_fail"], COLORS["reward"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", width=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=11, fontweight="bold")
    ax.set_ylabel("Rate / Reward (%)")
    ax.set_title(f"ES Post-Training Eval — {eval_summary.get('num_samples', '?')} samples")
    ax.set_ylim(0, max(values) * 1.25 + 5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    fig.savefig(os.path.join(save_dir, "es_eval_summary.png"))
    plt.close(fig)
    print("  ✓ es_eval_summary.png")


def plot_eval_reward_histogram(eval_csv_rows, save_dir):
    """Histogram of per-sample rewards from eval."""
    if eval_csv_rows is None:
        print("  [skip] eval_results.csv not found for histogram")
        return

    rewards = [float(r["reward"]) for r in eval_csv_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 25)
    ax.hist(rewards, bins=bins, color=COLORS["reward"], edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(rewards), color=COLORS["parse_fail"], linestyle="--",
               label=f"Mean: {np.mean(rewards):.3f}")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title("ES Eval — Per-Sample Reward Distribution")
    ax.legend()
    fig.savefig(os.path.join(save_dir, "es_eval_reward_histogram.png"))
    plt.close(fig)
    print("  ✓ es_eval_reward_histogram.png")


def plot_train_vs_eval(data, eval_summary, save_dir):
    """Side-by-side bar chart comparing final training metrics vs eval metrics."""
    if eval_summary is None:
        print("  [skip] no eval summary for train-vs-eval comparison")
        return

    s, both_train = extract(data, "task/both_exact_rate")
    _, one_train = extract(data, "task/one_root_rate")
    _, pf_train = extract(data, "task/parse_fail_rate")
    _, mean_r_train = extract(data, "reward/mean")

    if s is None or len(s) < 1:
        print("  [skip] no training data for train-vs-eval comparison")
        return

    labels = ["Both Roots\nExact", "One Root\nOnly", "Parse\nFail", "Mean\nReward"]
    train_vals = [
        both_train[-1] * 100,
        one_train[-1] * 100 if one_train is not None else 0,
        pf_train[-1] * 100 if pf_train is not None else 0,
        mean_r_train[-1] * 100 if mean_r_train is not None else 0,
    ]
    eval_vals = [
        eval_summary["both_exact_rate"] * 100,
        eval_summary["one_root_rate"] * 100,
        eval_summary["parse_fail_rate"] * 100,
        eval_summary["mean_reward"] * 100,
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width / 2, train_vals, width,
                   label="Train (final iter)", color="#90CAF9", edgecolor="white")
    bars2 = ax.bar(x + width / 2, eval_vals, width,
                   label="Eval (held-out)", color="#4CAF50", edgecolor="white")

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold")

    ax.set_ylabel("Rate / Reward (%)")
    ax.set_title("ES — Training vs Held-Out Eval")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    all_vals = train_vals + eval_vals
    ax.set_ylim(0, max(all_vals) * 1.2 + 5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    fig.savefig(os.path.join(save_dir, "es_train_vs_eval.png"))
    plt.close(fig)
    print("  ✓ es_train_vs_eval.png")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot ES finetuning results")
    parser.add_argument("--exp_dir", required=True,
                        help="Path to ES experiment directory (contains events.out.tfevents.*)")
    parser.add_argument("--save_dir", default=None,
                        help="Where to save plots (default: <exp_dir>/plots)")
    parser.add_argument("--csv", action="store_true",
                        help="Also read train_generations.csv for per-sample plots")
    parser.add_argument("--eval_dir", default=None,
                        help="Path to eval results directory (contains eval_summary.json, eval_results.csv)")
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(args.exp_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    apply_style()

    # Find TensorBoard logdir (could be exp_dir itself or a subdirectory)
    event_files = glob.glob(os.path.join(args.exp_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        print(f"ERROR: No TensorBoard event files found under {args.exp_dir}")
        return
    logdir = os.path.dirname(event_files[0])
    print(f"Reading TensorBoard events from: {logdir}")

    data = read_tensorboard_scalars(logdir)
    print(f"Found tags: {list(data.keys())}")

    csv_rows = read_csv(args.exp_dir) if args.csv else None
    if args.csv and csv_rows:
        print(f"Read {len(csv_rows)} rows from train_generations.csv")

    # Auto-detect eval_dir: check sibling eval_results/ directory
    eval_dir = args.eval_dir
    if eval_dir is None:
        parent = os.path.dirname(args.exp_dir)
        candidate = os.path.join(parent, "eval_results")
        if os.path.isdir(candidate):
            eval_dir = candidate
            print(f"Auto-detected eval results: {eval_dir}")

    eval_summary = read_eval_summary(eval_dir) if eval_dir else None
    eval_csv_rows = read_eval_csv(eval_dir) if eval_dir else None

    if eval_summary:
        print(f"Eval summary: {eval_summary['num_samples']} samples, "
              f"mean_reward={eval_summary['mean_reward']:.4f}, "
              f"both_exact={eval_summary['both_exact_rate']:.3f}")

    print("\nGenerating plots:")
    plot_reward_curve(data, save_dir)
    plot_accuracy_rates(data, save_dir)
    plot_timing(data, save_dir)
    plot_baseline_vs_final(data, save_dir)
    plot_dashboard(data, save_dir)

    if csv_rows:
        plot_reward_components_from_csv(csv_rows, save_dir)
        plot_reward_histogram_from_csv(csv_rows, save_dir)

    # Eval plots
    if eval_summary or eval_csv_rows:
        print("\nGenerating eval plots:")
        plot_eval_summary(eval_summary, save_dir)
        plot_eval_reward_histogram(eval_csv_rows, save_dir)
        plot_train_vs_eval(data, eval_summary, save_dir)

    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
