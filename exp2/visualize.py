"""Visualizations for Experiment 2 — MV vs CC comparison plots."""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ── Colour palette ───────────────────────────────────────────────────────────
_C_MV       = "#3498db"   # blue  — majority voting
_C_CC       = "#e67e22"   # orange — cycle consistency
_C_FUSION   = "#9b59b6"   # purple — fusion
_C_CORRECT  = "#2ecc71"   # green
_C_INCORRECT = "#e74c3c"  # red

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11})


# ── 1. AUROC comparison bar chart ────────────────────────────────────────────

def plot_auroc_bars(
    results_by_dataset: dict[str, dict],
    save_dir: str,
    k: int = 8,
):
    """Side-by-side AUROC bars for MV, CC, and Fusion, per dataset.

    results_by_dataset: {dataset_name: {signal_name: auroc_value, ...}}
    """
    datasets = list(results_by_dataset.keys())
    signals = [
        (f"voting_confidence_K{k}", f"MV (K={k})", _C_MV),
        ("combined_reward", "Cycle Consistency", _C_CC),
        (f"fused_best_K{k}", f"Fusion (best α, K={k})", _C_FUSION),
    ]

    n_ds = len(datasets)
    n_sig = len(signals)
    x = np.arange(n_ds)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, 3 * n_ds), 6))

    for i, (key, label, color) in enumerate(signals):
        vals = []
        for ds in datasets:
            v = results_by_dataset[ds].get(key)
            vals.append(v if v is not None else 0.5)
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Dataset")
    ax.set_ylabel("AUROC")
    ax.set_title(f"AUROC Comparison: Majority Voting vs Cycle Consistency vs Fusion (K={k})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0.45, min(1.0, max(
        max(results_by_dataset[ds].get(k, 0.5) for ds in datasets for k in
            [s[0] for s in signals])
    , 0.8) + 0.05))
    ax.axhline(0.5, color="grey", linestyle="--", lw=1, alpha=0.5, label="Random")

    plt.tight_layout()
    path = os.path.join(save_dir, f"auroc_comparison_K{k}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── 2. K-sensitivity curve ──────────────────────────────────────────────────

def plot_k_sensitivity(
    k_aurocs: dict[int, float],
    cc_auroc: float,
    fusion_aurocs: dict[int, float] | None = None,
    save_dir: str = ".",
    dataset_label: str = "all",
):
    """AUROC vs K for majority voting, with CC as horizontal reference line."""
    ks = sorted(k_aurocs.keys())
    mv_vals = [k_aurocs[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))

    # MV curve
    ax.plot(ks, mv_vals, "o-", color=_C_MV, lw=2.5, markersize=8, label="Majority Voting")
    for k, v in zip(ks, mv_vals):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5)

    # CC reference line
    ax.axhline(cc_auroc, color=_C_CC, linestyle="--", lw=2, label=f"Cycle Consistency ({cc_auroc:.3f})")

    # Fusion curve
    if fusion_aurocs:
        fks = sorted(fusion_aurocs.keys())
        fvals = [fusion_aurocs[k] for k in fks]
        ax.plot(fks, fvals, "s--", color=_C_FUSION, lw=2, markersize=7, label="MV+CC Fusion")
        for k, v in zip(fks, fvals):
            ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=8.5, color=_C_FUSION)

    ax.set_xlabel("K (number of samples)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"K-Sensitivity: AUROC vs Number of Majority Voting Samples [{dataset_label}]")
    ax.set_xticks(ks)
    ax.legend()
    ax.axhline(0.5, color="grey", linestyle=":", lw=1, alpha=0.4)
    ax.set_ylim(0.45, max(max(mv_vals), cc_auroc, max(fusion_aurocs.values()) if fusion_aurocs else 0) + 0.05)

    plt.tight_layout()
    path = os.path.join(save_dir, f"k_sensitivity_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── 3. Cost-accuracy Pareto curve ────────────────────────────────────────────

def plot_cost_accuracy(
    points: list[dict],
    save_dir: str,
    dataset_label: str = "all",
):
    """AUROC vs compute cost for all methods.

    Each point: {name, cost, auroc, color, marker}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for p in points:
        ax.scatter(p["cost"], p["auroc"], color=p.get("color", "grey"),
                   marker=p.get("marker", "o"), s=120, zorder=5)
        ax.annotate(p["name"], (p["cost"], p["auroc"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    # Connect MV points
    mv_points = [p for p in points if p.get("group") == "mv"]
    if mv_points:
        mv_points.sort(key=lambda p: p["cost"])
        ax.plot([p["cost"] for p in mv_points], [p["auroc"] for p in mv_points],
                "-", color=_C_MV, alpha=0.4, lw=1.5)

    ax.set_xlabel("Relative Compute Cost (solver-equivalent passes)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Cost-Accuracy Tradeoff [{dataset_label}]")
    ax.axhline(0.5, color="grey", linestyle=":", lw=1, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(save_dir, f"cost_accuracy_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── 4. Failure overlap Venn-style diagram ────────────────────────────────────

def plot_failure_overlap(
    overlap: dict,
    save_dir: str,
    dataset_label: str = "all",
):
    """Stacked bar showing MV-only, overlap, CC-only false positives."""
    fp_mv_only = overlap["fp_mv_only"]
    fp_cc_only = overlap["fp_cc_only"]
    n_overlap = overlap["n_overlap"]

    fig, ax = plt.subplots(figsize=(8, 4))

    categories = ["MV only", "Overlap", "CC only"]
    counts = [fp_mv_only, n_overlap, fp_cc_only]
    colors = [_C_MV, "#95a5a6", _C_CC]

    bars = ax.barh(["False Positives"], [fp_mv_only], color=_C_MV, alpha=0.85, label=f"MV only ({fp_mv_only})")
    ax.barh(["False Positives"], [n_overlap], left=[fp_mv_only], color="#95a5a6", alpha=0.85,
            label=f"Overlap ({n_overlap})")
    ax.barh(["False Positives"], [fp_cc_only], left=[fp_mv_only + n_overlap], color=_C_CC, alpha=0.85,
            label=f"CC only ({fp_cc_only})")

    total = fp_mv_only + n_overlap + fp_cc_only
    ax.set_xlabel("Number of False Positives")
    ax.set_title(
        f"False Positive Overlap [{dataset_label}]\n"
        f"Jaccard = {overlap['jaccard']:.3f}  |  "
        f"Overlap Coeff = {overlap['overlap_coefficient']:.3f}  |  "
        f"Total incorrect = {overlap['n_incorrect']}"
    )
    ax.legend(loc="lower right")

    plt.tight_layout()
    path = os.path.join(save_dir, f"failure_overlap_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── 5. Voting confidence distributions ──────────────────────────────────────

def plot_vote_distribution(
    data: list[dict],
    k: int,
    save_dir: str,
    dataset_label: str = "all",
):
    """Distribution of voting_confidence split by correct/incorrect."""
    key = f"voting_confidence_K{k}"
    correct_vals = [d[key] for d in data if d["correct"] == 1 and key in d]
    incorrect_vals = [d[key] for d in data if d["correct"] == 0 and key in d]

    if not correct_vals or not incorrect_vals:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(correct_vals, bins=bins, alpha=0.6, color=_C_CORRECT, density=True,
            label=f"Correct (n={len(correct_vals)})")
    ax.hist(incorrect_vals, bins=bins, alpha=0.6, color=_C_INCORRECT, density=True,
            label=f"Incorrect (n={len(incorrect_vals)})")
    ax.axvline(np.mean(correct_vals), color=_C_CORRECT, linestyle="--", lw=2,
               label=f"Mean correct = {np.mean(correct_vals):.3f}")
    ax.axvline(np.mean(incorrect_vals), color=_C_INCORRECT, linestyle="--", lw=2,
               label=f"Mean incorrect = {np.mean(incorrect_vals):.3f}")

    ax.set_xlabel(f"Voting Confidence (K={k})")
    ax.set_ylabel("Density")
    ax.set_title(f"Voting Confidence Distribution [{dataset_label}]")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f"vote_distribution_K{k}_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── 6. Scatter: MV vs CC ────────────────────────────────────────────────────

def plot_scatter_mv_cc(
    data: list[dict],
    k: int,
    save_dir: str,
    dataset_label: str = "all",
):
    """Scatter of voting_confidence vs combined_reward, coloured by correctness."""
    vc_key = f"voting_confidence_K{k}"
    correct = np.array([d["correct"] for d in data if vc_key in d])
    mv_vals = np.array([d[vc_key] for d in data if vc_key in d])
    cc_vals = np.array([d["combined_reward"] for d in data if vc_key in d])

    fig, ax = plt.subplots(figsize=(8, 7))

    mask_c = correct == 1
    mask_i = correct == 0
    ax.scatter(mv_vals[mask_i], cc_vals[mask_i], c=_C_INCORRECT, alpha=0.35,
               s=20, label=f"Incorrect (n={mask_i.sum()})")
    ax.scatter(mv_vals[mask_c], cc_vals[mask_c], c=_C_CORRECT, alpha=0.35,
               s=20, label=f"Correct (n={mask_c.sum()})")

    ax.set_xlabel(f"Voting Confidence (K={k})")
    ax.set_ylabel("Combined Reward (CC)")
    ax.set_title(f"MV vs CC Signal Space [{dataset_label}]")
    ax.legend()

    # Add quadrant labels
    ax.axvline(0.5, color="grey", linestyle=":", alpha=0.3)
    ax.axhline(0.0, color="grey", linestyle=":", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"scatter_mv_cc_K{k}_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── 7. Overlaid ROC curves ──────────────────────────────────────────────────

def plot_roc_overlaid(
    data: list[dict],
    k: int,
    save_dir: str,
    dataset_label: str = "all",
    fused_scores: np.ndarray | None = None,
):
    """Overlaid ROC curves for MV, CC, and optionally Fusion."""
    y_true = np.array([d["correct"] for d in data])
    if len(set(y_true)) < 2:
        return

    vc_key = f"voting_confidence_K{k}"
    mv_scores = np.array([d.get(vc_key, 0.5) for d in data])
    cc_scores = np.array([d["combined_reward"] for d in data])

    fig, ax = plt.subplots(figsize=(8, 7))

    # MV ROC
    fpr_mv, tpr_mv, _ = roc_curve(y_true, mv_scores)
    auc_mv = auc(fpr_mv, tpr_mv)
    ax.plot(fpr_mv, tpr_mv, color=_C_MV, lw=2, label=f"MV K={k} (AUC={auc_mv:.3f})")

    # CC ROC
    fpr_cc, tpr_cc, _ = roc_curve(y_true, cc_scores)
    auc_cc = auc(fpr_cc, tpr_cc)
    ax.plot(fpr_cc, tpr_cc, color=_C_CC, lw=2, label=f"CC (AUC={auc_cc:.3f})")

    # Fusion ROC
    if fused_scores is not None:
        fpr_f, tpr_f, _ = roc_curve(y_true, fused_scores)
        auc_f = auc(fpr_f, tpr_f)
        ax.plot(fpr_f, tpr_f, color=_C_FUSION, lw=2, label=f"Fusion (AUC={auc_f:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves: MV vs CC vs Fusion [{dataset_label}]")
    ax.legend(loc="lower right")

    plt.tight_layout()
    path = os.path.join(save_dir, f"roc_overlaid_K{k}_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── 8. Alpha sweep plot ─────────────────────────────────────────────────────

def plot_alpha_sweep(
    alpha_results: list[dict],
    save_dir: str,
    k: int,
    dataset_label: str = "all",
):
    """AUROC vs α for fused_linear."""
    alphas = [r["alpha"] for r in alpha_results]
    aurocs = [r["auroc"] for r in alpha_results]

    best_idx = int(np.argmax(aurocs))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, aurocs, "o-", color=_C_FUSION, lw=2, markersize=5)
    ax.scatter([alphas[best_idx]], [aurocs[best_idx]], color="red", s=150,
               zorder=5, marker="*", label=f"Best α={alphas[best_idx]:.2f} (AUROC={aurocs[best_idx]:.3f})")

    ax.axhline(aurocs[0], color=_C_CC, linestyle="--", lw=1.5, alpha=0.6,
               label=f"α=0 (CC only): {aurocs[0]:.3f}")
    ax.axhline(aurocs[-1], color=_C_MV, linestyle="--", lw=1.5, alpha=0.6,
               label=f"α=1 (MV only): {aurocs[-1]:.3f}")

    ax.set_xlabel("α  (0 = CC only, 1 = MV only)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Fusion Weight Sweep [{dataset_label}, K={k}]")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f"alpha_sweep_K{k}_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── 9. Summary dashboard ────────────────────────────────────────────────────

def plot_summary_dashboard(
    results_by_dataset: dict[str, dict],
    k_aurocs_all: dict[int, float],
    cc_auroc_all: float,
    best_fusion_all: float,
    save_dir: str,
):
    """One-page summary: AUROC bars + K-sensitivity + key numbers."""
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    # Left: AUROC bar chart by dataset
    ax_bar = fig.add_subplot(gs[0, 0])
    datasets = list(results_by_dataset.keys())
    n_ds = len(datasets)
    x = np.arange(n_ds)
    width = 0.25

    k = max(k_aurocs_all.keys())  # use largest K for bars
    for i, (key_pat, label, color) in enumerate([
        (f"voting_confidence_K{k}", f"MV K={k}", _C_MV),
        ("combined_reward", "CC", _C_CC),
        (f"fused_best_K{k}", f"Fusion", _C_FUSION),
    ]):
        vals = [results_by_dataset[ds].get(key_pat, 0.5) for ds in datasets]
        bars = ax_bar.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_bar.set_xticks(x + width)
    ax_bar.set_xticklabels(datasets)
    ax_bar.set_ylabel("AUROC")
    ax_bar.set_title(f"AUROC by Dataset (K={k})")
    ax_bar.legend(fontsize=9)
    ax_bar.axhline(0.5, color="grey", linestyle=":", lw=1, alpha=0.4)

    # Right: K-sensitivity
    ax_k = fig.add_subplot(gs[0, 1])
    ks = sorted(k_aurocs_all.keys())
    mv_vals = [k_aurocs_all[k] for k in ks]
    ax_k.plot(ks, mv_vals, "o-", color=_C_MV, lw=2.5, markersize=8, label="Majority Voting")
    ax_k.axhline(cc_auroc_all, color=_C_CC, linestyle="--", lw=2,
                 label=f"Cycle Consistency ({cc_auroc_all:.3f})")
    ax_k.axhline(best_fusion_all, color=_C_FUSION, linestyle=":", lw=2,
                 label=f"Best Fusion ({best_fusion_all:.3f})")
    for k_val, v in zip(ks, mv_vals):
        ax_k.annotate(f"{v:.3f}", (k_val, v), textcoords="offset points",
                      xytext=(0, 10), ha="center", fontsize=9)
    ax_k.set_xlabel("K (number of samples)")
    ax_k.set_ylabel("AUROC")
    ax_k.set_title("K-Sensitivity (All Datasets)")
    ax_k.set_xticks(ks)
    ax_k.legend(fontsize=9)

    fig.suptitle("Experiment 2: Majority Voting vs Cycle Consistency — Summary", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
