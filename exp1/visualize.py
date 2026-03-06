"""Generate all plots for Experiment 1."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def _split_by_correctness(data: list[dict], key: str):
    correct = [d[key] for d in data if d["correct"] == 1]
    incorrect = [d[key] for d in data if d["correct"] == 0]
    return correct, incorrect


# ── Violin plot ─────────────────────────────────────────────────────────────

def plot_violin(data: list[dict], save_dir: str):
    """Violin plot of question_cycle and combined_reward, split by correctness."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, title in zip(
        axes,
        ["question_cycle", "combined_reward"],
        ["Question Cycle Distance", "Combined Reward"],
    ):
        correct_vals, incorrect_vals = _split_by_correctness(data, key)

        parts = ax.violinplot(
            [incorrect_vals, correct_vals],
            positions=[0, 1],
            showmeans=True,
            showmedians=True,
        )

        # Colour the violins
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(["#e74c3c", "#2ecc71"][i])
            pc.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Incorrect", "Correct"])
        ax.set_title(title)
        ax.set_ylabel(key)

    plt.tight_layout()
    path = os.path.join(save_dir, "violin.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Histogram ───────────────────────────────────────────────────────────────

def plot_histogram(data: list[dict], save_dir: str):
    """Overlapping histogram of question_cycle (correct vs incorrect)."""
    correct_vals, incorrect_vals = _split_by_correctness(data, "question_cycle")

    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, max(max(correct_vals, default=1), max(incorrect_vals, default=1)), 50)
    plt.hist(correct_vals, bins=bins, alpha=0.6, label="Correct", color="#2ecc71")
    plt.hist(incorrect_vals, bins=bins, alpha=0.6, label="Incorrect", color="#e74c3c")
    plt.xlabel("question_cycle  (1 − cosine similarity)")
    plt.ylabel("Count")
    plt.title("Question Cycle Distance — Correct vs Incorrect")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, "histogram_question_cycle.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── ROC Curve ───────────────────────────────────────────────────────────────

def plot_roc(data: list[dict], save_dir: str):
    """ROC curve for combined_reward → correctness."""
    y_true = np.array([d["correct"] for d in data])
    y_score = np.array([d["combined_reward"] for d in data])

    if len(set(y_true)) < 2:
        print("  [skip] ROC curve: only one class present")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — combined_reward → correctness")
    plt.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(save_dir, "roc_combined_reward.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Public entry point ──────────────────────────────────────────────────────

def generate_all_plots(data: list[dict], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    plot_violin(data, save_dir)
    plot_histogram(data, save_dir)
    plot_roc(data, save_dir)
