"""Generate plots for Experiment 1 — per dataset in separate sub-folders."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc, average_precision_score

# ── Colour palette ───────────────────────────────────────────────────────────
_C_CORRECT   = "#2ecc71"   # green
_C_INCORRECT = "#e74c3c"   # red
_C_COMBINED  = "#3498db"   # blue

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11})


# ── Helpers ──────────────────────────────────────────────────────────────────

def _split(data: list[dict], key: str):
    correct   = [d[key] for d in data if d["correct"] == 1]
    incorrect = [d[key] for d in data if d["correct"] == 0]
    return correct, incorrect


def _title_suffix(label: str) -> str:
    return f" [{label}]" if label and label != "all" else ""


# ── 1. Discrimination bar chart ──────────────────────────────────────────────

def plot_discrimination_bars(data: list[dict], save_dir: str, label: str = "all"):
    """Grouped bar chart: mean ± std of each signal for correct vs incorrect."""
    signals = {
        "Hybrid Cycle\n(dist↓)":      "hybrid_cycle",
        "Answer Match\n(agree↑)":     "answer_match",
        "Combined\nReward (↑)":       "combined_reward",
    }

    n_sig = len(signals)
    fig, axes = plt.subplots(1, n_sig, figsize=(4 * n_sig, 5), sharey=False)
    if n_sig == 1:
        axes = [axes]

    for ax, (sig_label, key) in zip(axes, signals.items()):
        correct_vals   = np.array([d[key] for d in data if d["correct"] == 1], dtype=float)
        incorrect_vals = np.array([d[key] for d in data if d["correct"] == 0], dtype=float)

        means  = [incorrect_vals.mean(), correct_vals.mean()]
        stds   = [incorrect_vals.std(),  correct_vals.std()]
        labels = ["Incorrect", "Correct"]
        colors = [_C_INCORRECT, _C_CORRECT]

        bars = ax.bar(labels, means, yerr=stds, capsize=6,
                      color=colors, alpha=0.82, edgecolor="white", linewidth=1.2,
                      error_kw={"elinewidth": 1.5, "ecolor": "grey"})

        # Annotate mean value on each bar
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.05,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        ax.set_title(sig_label, fontsize=11)
        ax.set_ylabel("Mean value" if ax == axes[0] else "")
        ax.tick_params(axis="x", labelsize=10)

        # Annotate gap
        gap = correct_vals.mean() - incorrect_vals.mean()
        ax.set_xlabel(f"Δ = {gap:+.3f}", fontsize=9, labelpad=4)

    suf = _title_suffix(label)
    fig.suptitle(f"Signal Discrimination: Correct vs Incorrect{suf}", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "discrimination_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── 2. Violin plot ───────────────────────────────────────────────────────────

def plot_violin(data: list[dict], save_dir: str, label: str = "all"):
    """Violin plots of hybrid_cycle and combined_reward, by correctness."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    suf = _title_suffix(label)

    for ax, key, title in zip(
        axes,
        ["hybrid_cycle", "combined_reward"],
        ["Hybrid Cycle Distance (dist↓)", "Combined Reward (↑)"],
    ):
        correct_vals, incorrect_vals = _split(data, key)

        parts = ax.violinplot(
            [incorrect_vals, correct_vals],
            positions=[0, 1],
            showmeans=True,
            showmedians=True,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor([_C_INCORRECT, _C_CORRECT][i])
            pc.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            f"Incorrect\n(n={len(incorrect_vals)})",
            f"Correct\n(n={len(correct_vals)})"
        ])
        ax.set_title(f"{title}{suf}")
        ax.set_ylabel(key)

        # Annotate means
        for pos, vals, c in [(0, incorrect_vals, _C_INCORRECT), (1, correct_vals, _C_CORRECT)]:
            ax.hlines(np.mean(vals), pos - 0.1, pos + 0.1,
                      colors=c, linewidths=2.5, zorder=5)

    plt.tight_layout()
    path = os.path.join(save_dir, "violin.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── 3. Histogram ─────────────────────────────────────────────────────────────

def plot_histogram(data: list[dict], save_dir: str, label: str = "all"):
    """Overlapping histogram of hybrid_cycle for correct vs incorrect."""
    correct_vals, incorrect_vals = _split(data, "hybrid_cycle")
    suf = _title_suffix(label)

    plt.figure(figsize=(8, 5))
    all_vals = correct_vals + incorrect_vals
    bins = np.linspace(0, max(all_vals) if all_vals else 1, 50)
    plt.hist(correct_vals,   bins=bins, alpha=0.6, label=f"Correct (n={len(correct_vals)})",
             color=_C_CORRECT,   density=True)
    plt.hist(incorrect_vals, bins=bins, alpha=0.6, label=f"Incorrect (n={len(incorrect_vals)})",
             color=_C_INCORRECT, density=True)
    plt.axvline(np.mean(correct_vals),   color=_C_CORRECT,   linestyle="--", lw=1.8,
                label=f"mean correct={np.mean(correct_vals):.3f}")
    plt.axvline(np.mean(incorrect_vals), color=_C_INCORRECT, linestyle="--", lw=1.8,
                label=f"mean incorrect={np.mean(incorrect_vals):.3f}")
    plt.xlabel("hybrid_cycle  (0.4·embed + 0.4·num_jaccard + 0.2·chrF)")
    plt.ylabel("Density")
    plt.title(f"Hybrid Cycle Distance Distribution{suf}")
    plt.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(save_dir, "histogram_hybrid_cycle.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── 4. ROC + Precision-Recall curves ─────────────────────────────────────────

def plot_roc(data: list[dict], save_dir: str, label: str = "all"):
    """ROC and Precision-Recall curves for combined_reward."""
    y_true  = np.array([d["correct"]         for d in data])
    y_score = np.array([d["combined_reward"]  for d in data])
    suf = _title_suffix(label)

    if len(set(y_true)) < 2:
        print(f"  [skip] ROC/PR for {label}: only one class")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # — ROC —
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=_C_COMBINED, lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    axes[0].fill_between(fpr, tpr, alpha=0.08, color=_C_COMBINED)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve — combined_reward{suf}")
    axes[0].legend(loc="lower right")

    # — Precision-Recall —
    from sklearn.metrics import precision_recall_curve
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_score)
    pr_auc  = average_precision_score(y_true, y_score)
    baseline = y_true.mean()  # random classifier baseline
    axes[1].plot(rec_arr, prec_arr, color=_C_COMBINED, lw=2, label=f"AUPRC = {pr_auc:.3f}")
    axes[1].axhline(baseline, color="grey", linestyle="--", lw=1.2,
                    label=f"Random baseline = {baseline:.3f}")
    axes[1].fill_between(rec_arr, prec_arr, baseline, where=(prec_arr >= baseline),
                         alpha=0.08, color=_C_COMBINED)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall — combined_reward{suf}")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(save_dir, "roc_pr_combined_reward.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── 5. Summary dashboard (all signals, one figure) ───────────────────────────

def plot_summary_dashboard(data: list[dict], save_dir: str, label: str = "all"):
    """One-page figure: discrimination bars + score histograms side by side."""
    y = np.array([d["correct"] for d in data])
    suf = _title_suffix(label)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    signal_cfg = [
        ("hybrid_cycle",    "Hybrid Cycle Distance (dist↓)",     "lower"),
        ("answer_match",    "Answer Match Agreement (↑)",        "higher"),
        ("combined_reward", "Combined Reward (↑)",               "higher"),
    ]

    for col, (key, title, direction) in enumerate(signal_cfg):
        vals        = np.array([d[key] for d in data], dtype=float)
        correct_v   = vals[y == 1]
        incorrect_v = vals[y == 0]

        # — Top row: bar chart —
        ax_bar = fig.add_subplot(gs[0, col])
        means  = [incorrect_v.mean(), correct_v.mean()]
        stds   = [incorrect_v.std(),  correct_v.std()]
        bars   = ax_bar.bar(
            ["Incorrect", "Correct"], means, yerr=stds,
            color=[_C_INCORRECT, _C_CORRECT], alpha=0.82,
            capsize=5, edgecolor="white",
            error_kw={"elinewidth": 1.5, "ecolor": "grey"},
        )
        for bar, m in zip(bars, means):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(stds) * 0.08,
                        f"{m:.3f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
        gap = correct_v.mean() - incorrect_v.mean()
        ax_bar.set_title(f"{title}\nΔ = {gap:+.3f}", fontsize=9.5)
        ax_bar.tick_params(axis="x", labelsize=8.5)

        # — Bottom row: histogram —
        ax_hist = fig.add_subplot(gs[1, col])
        all_v = np.concatenate([correct_v, incorrect_v])
        bins  = np.linspace(all_v.min(), all_v.max(), 40)
        ax_hist.hist(correct_v,   bins=bins, alpha=0.6, color=_C_CORRECT,
                     density=True, label=f"Correct (n={len(correct_v)})")
        ax_hist.hist(incorrect_v, bins=bins, alpha=0.6, color=_C_INCORRECT,
                     density=True, label=f"Incorrect (n={len(incorrect_v)})")
        ax_hist.axvline(correct_v.mean(),   color=_C_CORRECT,   linestyle="--", lw=1.5)
        ax_hist.axvline(incorrect_v.mean(), color=_C_INCORRECT, linestyle="--", lw=1.5)
        ax_hist.set_xlabel(key, fontsize=8.5)
        ax_hist.set_ylabel("Density", fontsize=8.5)
        ax_hist.legend(fontsize=7)

    fig.suptitle(
        f"Cycle-Consistency Signal Discrimination{suf}  "
        f"(N={len(data)}, acc={y.mean():.1%})",
        fontsize=13, y=1.01,
    )
    path = os.path.join(save_dir, "summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Public entry point ────────────────────────────────────────────────────────

def generate_all_plots(data: list[dict], base_plots_dir: str) -> None:
    """Generate plots for ALL data and each dataset subset in separate folders."""
    # Determine which sub-datasets are present
    datasets_present = sorted(set(d["dataset"] for d in data))

    subsets: dict[str, list[dict]] = {"all": data}
    for ds in datasets_present:
        subsets[ds] = [d for d in data if d["dataset"] == ds]

    for label, subset in subsets.items():
        if len(subset) < 2:
            print(f"  [skip] {label}: too few samples")
            continue

        save_dir = os.path.join(base_plots_dir, label)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n  ── Plots: {label} (N={len(subset)}) ──")

        plot_summary_dashboard(subset, save_dir, label=label)
        plot_discrimination_bars(subset, save_dir, label=label)
        plot_violin(subset, save_dir, label=label)
        plot_histogram(subset, save_dir, label=label)
        plot_roc(subset, save_dir, label=label)
