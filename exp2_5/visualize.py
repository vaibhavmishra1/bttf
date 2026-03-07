"""Visualizations for Experiment 2.5 — Per-Trajectory Composite Reward Analysis."""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ── Colour palette ───────────────────────────────────────────────────────────
_C_MV = "#3498db"          # blue  — majority voting
_C_CC = "#e67e22"          # orange — cycle consistency
_C_COMPOSITE = "#9b59b6"   # purple — composite
_C_CORRECT = "#2ecc71"     # green
_C_INCORRECT = "#e74c3c"   # red
_C_ORACLE = "#1abc9c"      # teal

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11})


# ═════════════════════════════════════════════════════════════════════════════
# 1. Per-Trajectory AUROC Bar Chart
# ═════════════════════════════════════════════════════════════════════════════

def plot_trajectory_auroc_bars(
    auroc_results: dict,
    save_dir: str,
    dataset_label: str = "all",
):
    """Grouped bar chart: AUROC of each signal for predicting trajectory correctness."""
    signals = auroc_results.get("signals", {})
    if not signals:
        return

    names = list(signals.keys())
    aurocs = [signals[n]["auroc"] for n in names]

    # Colour mapping
    color_map = {
        "mv_agreement_K16": _C_MV,
        "mv_agreement_soft_K16": _C_MV,
        "hybrid_cycle": _C_CC,
        "answer_match": _C_CC,
        "combined_reward": _C_CC,
    }
    colors = [color_map.get(n, "#95a5a6") for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, aurocs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    for bar, v in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_K16", "\n(K=16)") for n in names], fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title(f"Per-Trajectory Correctness Prediction [{dataset_label}]")
    ax.axhline(0.5, color="grey", linestyle="--", lw=1, alpha=0.5)
    ax.set_ylim(0.45, max(aurocs) + 0.05)

    plt.tight_layout()
    path = os.path.join(save_dir, f"trajectory_auroc_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Alpha-Sweep Curve (Trajectory-Level)
# ═════════════════════════════════════════════════════════════════════════════

def plot_alpha_sweep_trajectory(
    sweep_results: list[dict],
    save_dir: str,
    dataset_label: str = "all",
):
    """AUROC vs α for trajectory-level composite reward."""
    alphas = [r["alpha"] for r in sweep_results]
    aurocs = [r["auroc"] for r in sweep_results]
    best_idx = int(np.argmax(aurocs))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alphas, aurocs, "o-", color=_C_COMPOSITE, lw=2, markersize=5)
    ax.scatter([alphas[best_idx]], [aurocs[best_idx]], color="red", s=150,
               zorder=5, marker="*",
               label=f"Best α={alphas[best_idx]:.2f} (AUROC={aurocs[best_idx]:.3f})")

    ax.axhline(aurocs[0], color=_C_CC, linestyle="--", lw=1.5, alpha=0.6,
               label=f"α=0 (CC only): {aurocs[0]:.3f}")
    ax.axhline(aurocs[-1], color=_C_MV, linestyle="--", lw=1.5, alpha=0.6,
               label=f"α=1 (MV only): {aurocs[-1]:.3f}")

    ax.set_xlabel("α  (0 = CC only, 1 = MV only)")
    ax.set_ylabel("Trajectory AUROC")
    ax.set_title(f"Composite Reward α-Sweep (Trajectory Level) [{dataset_label}]")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f"alpha_sweep_trajectory_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. CC Distribution Violin Plots
# ═════════════════════════════════════════════════════════════════════════════

def plot_cc_distribution(
    traj_data: list[dict],
    save_dir: str,
    dataset_label: str = "all",
):
    """Violin + strip plot of hybrid_cycle and combined_reward by correctness."""
    hc_correct = [t["hybrid_cycle"] for t in traj_data if t["trajectory_correct"]]
    hc_incorrect = [t["hybrid_cycle"] for t in traj_data if not t["trajectory_correct"]]
    cr_correct = [t["combined_reward"] for t in traj_data if t["trajectory_correct"]]
    cr_incorrect = [t["combined_reward"] for t in traj_data if not t["trajectory_correct"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # hybrid_cycle
    ax = axes[0]
    data_hc = [hc_correct, hc_incorrect]
    parts = ax.violinplot(data_hc, positions=[1, 2], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(_C_CORRECT if i == 0 else _C_INCORRECT)
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Correct", "Incorrect"])
    ax.set_ylabel("hybrid_cycle (distance)")
    ax.set_title(f"CC Distance by Correctness [{dataset_label}]")

    # combined_reward
    ax = axes[1]
    data_cr = [cr_correct, cr_incorrect]
    parts = ax.violinplot(data_cr, positions=[1, 2], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(_C_CORRECT if i == 0 else _C_INCORRECT)
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Correct", "Incorrect"])
    ax.set_ylabel("combined_reward")
    ax.set_title(f"Combined Reward by Correctness [{dataset_label}]")

    plt.tight_layout()
    path = os.path.join(save_dir, f"cc_distribution_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Voting Accuracy vs K
# ═════════════════════════════════════════════════════════════════════════════

def plot_voting_accuracy_vs_k(
    voting_results: dict,
    save_dir: str,
    dataset_label: str = "all",
):
    """Lines for standard MV, CC-weighted MV, composite-weighted MV."""
    ks = sorted(voting_results.keys())
    std_acc = [voting_results[k]["standard_mv_acc"] for k in ks]
    ccw_acc = [voting_results[k]["cc_weighted_acc"] for k in ks]
    compw_acc = [voting_results[k]["composite_weighted_acc"] for k in ks]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, std_acc, "o-", color=_C_MV, lw=2.5, markersize=8,
            label="Standard MV")
    ax.plot(ks, ccw_acc, "s--", color=_C_CC, lw=2, markersize=7,
            label="CC-Weighted MV")
    ax.plot(ks, compw_acc, "^:", color=_C_COMPOSITE, lw=2, markersize=7,
            label="Composite-Weighted MV")

    for k, v1, v2, v3 in zip(ks, std_acc, ccw_acc, compw_acc):
        ax.annotate(f"{v1:.3f}", (k, v1), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color=_C_MV)
        ax.annotate(f"{v2:.3f}", (k, v2), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8, color=_C_CC)

    ax.set_xlabel("K (number of samples)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Voting Accuracy: Standard vs CC-Weighted vs Composite [{dataset_label}]")
    ax.set_xticks(ks)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f"voting_accuracy_vs_k_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. Best-of-K Accuracy vs K
# ═════════════════════════════════════════════════════════════════════════════

def plot_best_of_k(
    bok_results: dict,
    save_dir: str,
    dataset_label: str = "all",
):
    """Lines for each selection strategy."""
    ks = sorted(bok_results.keys())
    strategies = ["random", "mv_aligned", "best_cc", "best_composite", "oracle"]
    colors = ["#95a5a6", _C_MV, _C_CC, _C_COMPOSITE, _C_ORACLE]
    markers = ["x", "o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for strat, color, marker in zip(strategies, colors, markers):
        vals = [bok_results[k].get(strat, 0) for k in ks]
        linestyle = "--" if strat == "oracle" else "-"
        ax.plot(ks, vals, f"{marker}{linestyle}", color=color, lw=2, markersize=7,
                label=strat.replace("_", " ").title())

    ax.set_xlabel("K (number of samples)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Best-of-K Selection Strategies [{dataset_label}]")
    ax.set_xticks(ks)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f"best_of_k_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 6. Filtered Voting Accuracy vs Threshold
# ═════════════════════════════════════════════════════════════════════════════

def plot_filtered_voting(
    filtered_results: dict,
    save_dir: str,
    dataset_label: str = "all",
):
    """Accuracy vs CC threshold for filtered voting, one line per K."""
    fig, ax = plt.subplots(figsize=(10, 6))
    k_colors = {2: "#e74c3c", 4: "#e67e22", 8: "#3498db", 16: "#9b59b6"}

    for K in sorted(filtered_results.keys()):
        entry = filtered_results[K]
        baseline = entry["baseline_acc"]
        thresholds = [r["threshold"] for r in entry["thresholds"]]
        accs = [r["accuracy"] for r in entry["thresholds"]]

        color = k_colors.get(K, "#95a5a6")
        ax.plot(thresholds, accs, "o-", color=color, lw=2, markersize=6,
                label=f"K={K}")
        ax.axhline(baseline, color=color, linestyle=":", lw=1, alpha=0.5)

    ax.set_xlabel("CC Threshold (keep trajectories with hybrid_cycle < threshold)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Filtered Voting: Accuracy vs CC Threshold [{dataset_label}]")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f"filtered_voting_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. Wrong-Majority Rescue Scatter
# ═════════════════════════════════════════════════════════════════════════════

def plot_wrong_majority_rescue(
    traj_data: list[dict],
    K: int,
    save_dir: str,
    dataset_label: str = "all",
):
    """For wrong-majority questions: scatter of CC scores (correct minority vs incorrect majority)."""
    from collections import defaultdict
    from utils import answers_equivalent
    from voting import cluster_answers

    groups = defaultdict(list)
    for t in traj_data:
        groups[t["question_idx"]].append(t)

    cc_correct = []
    cc_incorrect = []

    for q_idx, trajs in groups.items():
        trajs_k = sorted(trajs, key=lambda t: t["traj_idx"])[:K]
        if len(trajs_k) < K:
            continue

        gt = trajs_k[0]["ground_truth"]
        answers = [t["answer"] for t in trajs_k]
        clusters = cluster_answers(answers)
        majority_answer = answers[clusters[0][0]]

        if answers_equivalent(majority_answer, gt):
            continue  # majority is right

        for t in trajs_k:
            if answers_equivalent(t["answer"], gt):
                cc_correct.append(t["hybrid_cycle"])
            else:
                cc_incorrect.append(t["hybrid_cycle"])

    if not cc_correct or not cc_incorrect:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max(max(cc_correct), max(cc_incorrect)) + 0.05, 40)

    ax.hist(cc_correct, bins=bins, alpha=0.6, color=_C_CORRECT, density=True,
            label=f"Correct minority (n={len(cc_correct)})")
    ax.hist(cc_incorrect, bins=bins, alpha=0.6, color=_C_INCORRECT, density=True,
            label=f"Incorrect majority (n={len(cc_incorrect)})")
    ax.axvline(np.mean(cc_correct), color=_C_CORRECT, linestyle="--", lw=2)
    ax.axvline(np.mean(cc_incorrect), color=_C_INCORRECT, linestyle="--", lw=2)

    ax.set_xlabel("hybrid_cycle (lower = more consistent)")
    ax.set_ylabel("Density")
    ax.set_title(f"CC Distribution in Wrong-Majority Questions (K={K}) [{dataset_label}]")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f"wrong_majority_rescue_K{K}_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 8. Simulated GRPO Advantage Histogram
# ═════════════════════════════════════════════════════════════════════════════

def plot_grpo_advantage(
    grpo_results: dict,
    K: int,
    save_dir: str,
    dataset_label: str = "all",
):
    """Histogram of advantages for correct vs incorrect trajectories."""
    if K not in grpo_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, reward_type, title in [
        (axes[0], "composite", "Composite Reward (MV+CC)"),
        (axes[1], "mv_only", "MV-Only Reward"),
    ]:
        entry = grpo_results[K][reward_type]
        adv_c = np.array(entry["adv_correct"])
        adv_i = np.array(entry["adv_incorrect"])

        if len(adv_c) == 0 and len(adv_i) == 0:
            continue

        all_advs = np.concatenate([adv_c, adv_i])
        lo, hi = np.percentile(all_advs, [1, 99])
        bins = np.linspace(lo, hi, 50)

        ax.hist(adv_c, bins=bins, alpha=0.6, color=_C_CORRECT, density=True,
                label=f"Correct ({entry['pct_correct_positive']:.1%} positive)")
        ax.hist(adv_i, bins=bins, alpha=0.6, color=_C_INCORRECT, density=True,
                label=f"Incorrect ({entry['pct_incorrect_negative']:.1%} negative)")
        ax.axvline(0, color="black", linestyle="-", lw=1, alpha=0.5)
        ax.axvline(np.mean(adv_c), color=_C_CORRECT, linestyle="--", lw=2)
        ax.axvline(np.mean(adv_i), color=_C_INCORRECT, linestyle="--", lw=2)

        ax.set_xlabel("Advantage (standardised)")
        ax.set_ylabel("Density")
        ax.set_title(f"{title} [{dataset_label}]")
        ax.legend(fontsize=9)

    plt.suptitle(f"Simulated GRPO Advantage Distribution (K={K})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, f"grpo_advantage_K{K}_{dataset_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 9. Summary Dashboard
# ═════════════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(
    auroc_results: dict,
    voting_results: dict,
    bok_results: dict,
    grpo_results: dict,
    save_dir: str,
):
    """2×2 grid: trajectory AUROC, voting accuracy, best-of-K, GRPO advantage."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.35)

    # (0,0) Trajectory AUROC bars
    ax1 = fig.add_subplot(gs[0, 0])
    signals = auroc_results.get("signals", {})
    if signals:
        names = list(signals.keys())
        aurocs = [signals[n]["auroc"] for n in names]
        color_map = {
            "mv_agreement_K16": _C_MV, "mv_agreement_soft_K16": _C_MV,
            "hybrid_cycle": _C_CC, "answer_match": _C_CC, "combined_reward": _C_CC,
        }
        colors = [color_map.get(n, "#95a5a6") for n in names]
        x = np.arange(len(names))
        bars = ax1.bar(x, aurocs, color=colors, alpha=0.85)
        for bar, v in zip(bars, aurocs):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([n.replace("_K16", "") for n in names], fontsize=7, rotation=15)
        ax1.set_ylabel("AUROC")
        ax1.set_title("Per-Trajectory Correctness AUROC")
        ax1.axhline(0.5, color="grey", linestyle="--", lw=1, alpha=0.5)

    # (0,1) Voting accuracy vs K
    ax2 = fig.add_subplot(gs[0, 1])
    if voting_results:
        ks = sorted(voting_results.keys())
        ax2.plot(ks, [voting_results[k]["standard_mv_acc"] for k in ks],
                 "o-", color=_C_MV, lw=2, label="Standard MV")
        ax2.plot(ks, [voting_results[k]["cc_weighted_acc"] for k in ks],
                 "s--", color=_C_CC, lw=2, label="CC-Weighted")
        ax2.plot(ks, [voting_results[k]["composite_weighted_acc"] for k in ks],
                 "^:", color=_C_COMPOSITE, lw=2, label="Composite-Weighted")
        ax2.set_xlabel("K")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Voting Accuracy vs K")
        ax2.set_xticks(ks)
        ax2.legend(fontsize=8)

    # (1,0) Best-of-K
    ax3 = fig.add_subplot(gs[1, 0])
    if bok_results:
        ks = sorted(bok_results.keys())
        for strat, color, marker in [
            ("random", "#95a5a6", "x"), ("mv_aligned", _C_MV, "o"),
            ("best_cc", _C_CC, "s"), ("best_composite", _C_COMPOSITE, "^"),
            ("oracle", _C_ORACLE, "D"),
        ]:
            vals = [bok_results[k].get(strat, 0) for k in ks]
            ls = "--" if strat == "oracle" else "-"
            ax3.plot(ks, vals, f"{marker}{ls}", color=color, lw=2, markersize=6,
                     label=strat.replace("_", " ").title())
        ax3.set_xlabel("K")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Best-of-K Selection")
        ax3.set_xticks(ks)
        ax3.legend(fontsize=7, ncol=2)

    # (1,1) GRPO advantage
    ax4 = fig.add_subplot(gs[1, 1])
    if grpo_results and 16 in grpo_results:
        entry = grpo_results[16]
        labels = ["Composite", "MV-Only"]
        pct_cp = [entry["composite"]["pct_correct_positive"],
                   entry["mv_only"]["pct_correct_positive"]]
        pct_in = [entry["composite"]["pct_incorrect_negative"],
                   entry["mv_only"]["pct_incorrect_negative"]]

        x = np.arange(len(labels))
        width = 0.35
        ax4.bar(x - width / 2, pct_cp, width, label="% Correct → +Advantage",
                color=_C_CORRECT, alpha=0.8)
        ax4.bar(x + width / 2, pct_in, width, label="% Incorrect → −Advantage",
                color=_C_INCORRECT, alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.set_ylabel("Fraction")
        ax4.set_title("GRPO Signal Quality (K=16)")
        ax4.legend(fontsize=8)
        ax4.set_ylim(0, 1)

    fig.suptitle("Experiment 2.5: Per-Trajectory Composite Reward — Summary",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")
