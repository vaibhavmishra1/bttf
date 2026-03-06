"""Generate all plots for Exp 2 — includes a head-to-head exp1 vs exp2 ROC comparison.

New vs exp1/visualize.py:
  + plot_roc_comparison()    — overlays old and new combined_reward on same ROC/PR axes.
  + plot_cycle_comparison()  — question_cycle vs bleu_cycle vs hybrid_cycle histograms.
  + plot_answer_match_flip() — samples that changed from answer_match=0 to _llm=1.
  = summary_dashboard updated to include all 7 signals (4 rows × 2 cols layout).
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve

_C_CORRECT   = "#2ecc71"
_C_INCORRECT = "#e74c3c"
_C_EXP1      = "#3498db"   # blue
_C_EXP2      = "#e67e22"   # orange

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split(data: list[dict], key: str):
    correct   = [d[key] for d in data if d["correct"] == 1]
    incorrect = [d[key] for d in data if d["correct"] == 0]
    return correct, incorrect


def _title_suffix(label: str) -> str:
    return f" [{label}]" if label and label != "all" else ""


# ── 1. Summary dashboard (7 signals, 2-row layout) ───────────────────────────

def plot_summary_dashboard(data: list[dict], save_dir: str, label: str = "all"):
    """Two-row figure: exp1 baseline (top) + exp2 new signals (bottom)."""
    y   = np.array([d["correct"] for d in data])
    suf = _title_suffix(label)

    # 3 exp1 signals + 4 exp2 signals
    signal_cfg = [
        ("question_cycle",    "Q-Cycle (dist↓) [e1]",       "lower"),
        ("answer_match",      "Ans-Match Regex (↑) [e1]",   "higher"),
        ("combined_reward",   "Combined Rwd (↑) [e1]",      "higher"),
        ("bleu_cycle",        "BLEU-Cycle (dist↓) [e2]",    "lower"),
        ("hybrid_cycle",      "Hybrid Cycle (dist↓) [e2]",  "lower"),
        ("answer_match_llm",  "Ans-Match LLM (↑) [e2]",     "higher"),
        ("combined_reward_v2","Combined Rwd v2 (↑) [e2]",   "higher"),
    ]

    n_cols = 4
    n_rows = 2  # top row: exp1 (3 signals), bottom row: exp2 (4 signals)
    fig = plt.figure(figsize=(5 * n_cols, 9))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.38)

    # Row 0: exp1 signals (3), leave col 3 blank
    row0_signals = signal_cfg[:3]
    # Row 1: exp2 signals (4)
    row1_signals = signal_cfg[3:]

    for col, (key, title, direction) in enumerate(row0_signals):
        _draw_bar_panel(fig.add_subplot(gs[0, col]), data, y, key, title)

    for col, (key, title, direction) in enumerate(row1_signals):
        _draw_bar_panel(fig.add_subplot(gs[1, col]), data, y, key, title)

    fig.suptitle(
        f"Cycle-Consistency Signal Discrimination{suf}  (N={len(data)}, acc={y.mean():.1%})\n"
        f"Top row = exp1 baseline · Bottom row = exp2 improvements",
        fontsize=12, y=1.01,
    )
    path = os.path.join(save_dir, "summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def _draw_bar_panel(ax, data, y, key, title):
    correct_v   = np.array([d.get(key, 0) for d in data if d["correct"] == 1], dtype=float)
    incorrect_v = np.array([d.get(key, 0) for d in data if d["correct"] == 0], dtype=float)
    means  = [incorrect_v.mean(), correct_v.mean()]
    stds   = [incorrect_v.std(),  correct_v.std()]
    bars   = ax.bar(
        ["Incorrect", "Correct"], means, yerr=stds,
        color=[_C_INCORRECT, _C_CORRECT], alpha=0.82,
        capsize=5, edgecolor="white",
        error_kw={"elinewidth": 1.5, "ecolor": "grey"},
    )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.08,
            f"{m:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    gap = correct_v.mean() - incorrect_v.mean()
    ax.set_title(f"{title}\nΔ={gap:+.3f}", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8)


# ── 2. ROC + PR comparison: exp1 vs exp2 ─────────────────────────────────────

def plot_roc_comparison(data: list[dict], save_dir: str, label: str = "all"):
    """Single figure: ROC and PR curves for combined_reward (e1) vs v2 (e2)."""
    y_true = np.array([d["correct"] for d in data])
    suf    = _title_suffix(label)

    if len(set(y_true)) < 2:
        print(f"  [skip] ROC comparison for {label}: only one class")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    pairs = [
        ("combined_reward",    "combined_reward [exp1]",    _C_EXP1, "--"),
        ("combined_reward_v2", "combined_reward_v2 [exp2]", _C_EXP2, "-"),
    ]

    for key, legend_label, color, ls in pairs:
        if key not in data[0]:
            continue
        y_score = np.array([d[key] for d in data])

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc     = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2.5, linestyle=ls,
                     label=f"{legend_label}  AUC={roc_auc:.3f}")
        axes[0].fill_between(fpr, tpr, alpha=0.04, color=color)

        # PR
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        axes[1].plot(rec_arr, prec_arr, color=color, lw=2.5, linestyle=ls,
                     label=f"{legend_label}  AUPRC={pr_auc:.3f}")

    baseline = y_true.mean()
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve — exp1 vs exp2{suf}")
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].axhline(baseline, color="grey", linestyle=":", lw=1.5,
                    label=f"Random baseline={baseline:.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall — exp1 vs exp2{suf}")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(save_dir, "roc_pr_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── 3. Cycle distance comparison: embedding vs BLEU vs hybrid ─────────────────

def plot_cycle_comparison(data: list[dict], save_dir: str, label: str = "all"):
    """Three histograms side by side: question_cycle | bleu_cycle | hybrid_cycle."""
    y   = np.array([d["correct"] for d in data])
    suf = _title_suffix(label)

    cycle_keys = [
        ("question_cycle", "Embedding Cycle  [e1]"),
        ("bleu_cycle",     "BLEU Cycle  [e2]"),
        ("hybrid_cycle",   "Hybrid Cycle  [e2]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (key, title) in zip(axes, cycle_keys):
        if key not in data[0]:
            ax.set_visible(False)
            continue

        vals       = np.array([d[key] for d in data], dtype=float)
        correct_v  = vals[y == 1]
        incorrect_v= vals[y == 0]
        all_vals   = np.concatenate([correct_v, incorrect_v])
        bins       = np.linspace(all_vals.min(), all_vals.max(), 50)

        ax.hist(correct_v,   bins=bins, alpha=0.6, color=_C_CORRECT,
                density=True, label=f"Correct (n={len(correct_v)})")
        ax.hist(incorrect_v, bins=bins, alpha=0.6, color=_C_INCORRECT,
                density=True, label=f"Incorrect (n={len(incorrect_v)})")
        ax.axvline(correct_v.mean(),   color=_C_CORRECT,   lw=1.8, linestyle="--",
                   label=f"mean correct={correct_v.mean():.3f}")
        ax.axvline(incorrect_v.mean(), color=_C_INCORRECT, lw=1.8, linestyle="--",
                   label=f"mean incorrect={incorrect_v.mean():.3f}")

        gap = correct_v.mean() - incorrect_v.mean()
        ax.set_title(f"{title}\n(Δ = {gap:+.3f})", fontsize=10)
        ax.set_xlabel("Cycle distance (1 = max drift)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle(f"Cycle Distance Comparison{suf}  (N={len(data)})", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "cycle_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── 4. Answer-match flip analysis ─────────────────────────────────────────────

def plot_answer_match_flip(data: list[dict], save_dir: str, label: str = "all"):
    """Bar chart showing how many answer_match decisions flipped with the LLM judge.

    Quadrants:
      regex=0, llm=0  — both say no match   (true negatives in both)
      regex=0, llm=1  — Fix 1 recovered a match (key improvement)
      regex=1, llm=0  — Fix 1 rejected a match  (false positives reduced)
      regex=1, llm=1  — both say match        (stable)
    """
    suf = _title_suffix(label)
    if "answer_match_llm" not in data[0]:
        return

    y   = np.array([d["correct"] for d in data])
    am  = np.array([d["answer_match"]     for d in data])
    aml = np.array([d["answer_match_llm"] for d in data])

    cases = {
        "regex=0\nllm=0\n(both disagree)": ((am==0) & (aml==0)).sum(),
        "regex=0\nllm=1\n★ Fix 1 recovered": ((am==0) & (aml==1)).sum(),
        "regex=1\nllm=0\n★ Fix 1 corrected": ((am==1) & (aml==0)).sum(),
        "regex=1\nllm=1\n(both agree)":       ((am==1) & (aml==1)).sum(),
    }

    colors = ["#bdc3c7", _C_EXP2, _C_EXP1, "#2c3e50"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(list(cases.keys()), list(cases.values()), color=colors,
                  alpha=0.85, edgecolor="white", linewidth=1.3)
    for bar, v in zip(bars, cases.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Count")
    ax.set_title(f"Answer-Match: Regex vs LLM Judge{suf}  (N={len(data)})", fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, "answer_match_flip.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── 5. Violin — combined_reward v1 vs v2 ────────────────────────────────────

def plot_violin(data: list[dict], save_dir: str, label: str = "all"):
    """Violin plots: combined_reward (e1) and combined_reward_v2 (e2)."""
    suf = _title_suffix(label)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, title in zip(
        axes,
        ["combined_reward", "combined_reward_v2"],
        ["Combined Reward [exp1]", "Combined Reward v2 [exp2]"],
    ):
        if key not in data[0]:
            ax.set_visible(False)
            continue
        correct_vals, incorrect_vals = _split(data, key)
        parts = ax.violinplot(
            [incorrect_vals, correct_vals], positions=[0, 1],
            showmeans=True, showmedians=True,
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
        for pos, vals, c in [(0, incorrect_vals, _C_INCORRECT), (1, correct_vals, _C_CORRECT)]:
            ax.hlines(np.mean(vals), pos-0.1, pos+0.1, colors=c, linewidths=2.5, zorder=5)

    plt.tight_layout()
    path = os.path.join(save_dir, "violin.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Public entry point ────────────────────────────────────────────────────────

def generate_all_plots(data: list[dict], base_plots_dir: str) -> None:
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
        plot_roc_comparison(subset, save_dir, label=label)
        plot_cycle_comparison(subset, save_dir, label=label)
        plot_answer_match_flip(subset, save_dir, label=label)
        plot_violin(subset, save_dir, label=label)
