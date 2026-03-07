"""All 7 analyses for Experiment 2.5.

Each analysis function takes the flat trajectory dataset (list of dicts,
one per trajectory) and returns a results dict.

NOTE: exp1/ and exp2/ must be on sys.path (set up by run.py).
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import answers_equivalent
from voting import extract_final_answer, cluster_answers
from signals import (
    compute_mv_agreement,
    cc_weighted_vote,
    composite_weighted_vote,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two groups."""
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def _safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(set(y_true)) < 2:
        return float(y_true.mean())
    return float(average_precision_score(y_true, scores))


def _group_by_question(traj_data: list[dict]) -> dict[int, list[dict]]:
    """Group trajectory rows by question_idx."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for t in traj_data:
        groups[t["question_idx"]].append(t)
    return dict(groups)


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 1: Per-Trajectory Correctness Prediction (AUROC)
# ═════════════════════════════════════════════════════════════════════════════

def analysis_1_trajectory_auroc(traj_data: list[dict]) -> dict:
    """Evaluate how well each signal predicts trajectory correctness.

    Returns per-signal AUROC, AUPRC, Cohen's d.
    """
    y = np.array([t["trajectory_correct"] for t in traj_data])
    n_correct = int(y.sum())
    n_incorrect = len(y) - n_correct

    signals = {
        "mv_agreement_K16": ("mv_agreement_K16", "higher"),
        "mv_agreement_soft_K16": ("mv_agreement_soft_K16", "higher"),
        "hybrid_cycle": ("hybrid_cycle", "lower"),
        "answer_match": ("answer_match", "higher"),
        "combined_reward": ("combined_reward", "higher"),
    }

    results = {
        "n_trajectories": len(traj_data),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "accuracy": float(y.mean()),
        "signals": {},
    }

    for name, (key, direction) in signals.items():
        scores = np.array([t[key] for t in traj_data], dtype=float)
        eval_scores = -scores if direction == "lower" else scores

        y_c = scores[y == 1]
        y_i = scores[y == 0]

        entry = {
            "auroc": _safe_auroc(y, eval_scores),
            "auprc": _safe_auprc(y, eval_scores),
            "cohen_d": _cohen_d(y_c, y_i) if len(y_c) and len(y_i) else 0.0,
            "mean_correct": float(y_c.mean()) if len(y_c) else 0.0,
            "mean_incorrect": float(y_i.mean()) if len(y_i) else 0.0,
            "direction": direction,
        }
        results["signals"][name] = entry

    return results


def analysis_1_alpha_sweep(
    traj_data: list[dict],
    alphas: list[float],
) -> list[dict]:
    """Sweep α for composite reward → trajectory AUROC.

    composite(α) = α * mv_agreement_soft_K16 + (1-α) * combined_reward
    """
    y = np.array([t["trajectory_correct"] for t in traj_data])
    mv = np.array([t["mv_agreement_soft_K16"] for t in traj_data], dtype=float)
    cc = np.array([t["combined_reward"] for t in traj_data], dtype=float)

    sweep = []
    for alpha in alphas:
        comp = alpha * mv + (1.0 - alpha) * cc
        auroc = _safe_auroc(y, comp)
        sweep.append({"alpha": alpha, "auroc": auroc})
    return sweep


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 2: Reward-Weighted Voting
# ═════════════════════════════════════════════════════════════════════════════

def analysis_2_weighted_voting(
    traj_data: list[dict],
    k_values: list[int],
) -> dict:
    """Compare standard MV vs CC-weighted MV vs composite-weighted MV."""
    groups = _group_by_question(traj_data)

    results = {}
    for K in k_values:
        n_total = 0
        n_standard_correct = 0
        n_cc_weighted_correct = 0
        n_composite_weighted_correct = 0

        for q_idx, trajs in groups.items():
            trajs_k = sorted(trajs, key=lambda t: t["traj_idx"])[:K]
            if len(trajs_k) < K:
                continue
            n_total += 1
            gt = trajs_k[0]["ground_truth"]
            answers = [t["answer"] for t in trajs_k]
            cc_scores = [t["hybrid_cycle"] for t in trajs_k]
            comp_rewards = [t["combined_reward"] for t in trajs_k]

            # Standard MV
            clusters = cluster_answers(answers)
            std_answer = answers[clusters[0][0]]
            if answers_equivalent(std_answer, gt):
                n_standard_correct += 1

            # CC-weighted MV
            ccw_answer = cc_weighted_vote(answers, cc_scores, weight_fn="linear")
            if answers_equivalent(ccw_answer, gt):
                n_cc_weighted_correct += 1

            # Composite-weighted MV
            compw_answer = composite_weighted_vote(answers, comp_rewards)
            if answers_equivalent(compw_answer, gt):
                n_composite_weighted_correct += 1

        results[K] = {
            "n_questions": n_total,
            "standard_mv_acc": n_standard_correct / max(1, n_total),
            "cc_weighted_acc": n_cc_weighted_correct / max(1, n_total),
            "composite_weighted_acc": n_composite_weighted_correct / max(1, n_total),
        }
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 3: Best-of-K Selection
# ═════════════════════════════════════════════════════════════════════════════

def analysis_3_best_of_k(
    traj_data: list[dict],
    k_values: list[int],
    seed: int = 42,
) -> dict:
    """Compare selection strategies for picking the single best solution."""
    groups = _group_by_question(traj_data)
    rng = random.Random(seed)

    results = {}
    for K in k_values:
        counts = {
            "random": 0, "mv_aligned": 0, "best_cc": 0,
            "best_composite": 0, "oracle": 0,
        }
        n_total = 0

        for q_idx, trajs in groups.items():
            trajs_k = sorted(trajs, key=lambda t: t["traj_idx"])[:K]
            if len(trajs_k) < K:
                continue
            n_total += 1
            gt = trajs_k[0]["ground_truth"]

            # Random
            pick = rng.choice(trajs_k)
            if answers_equivalent(pick["answer"], gt):
                counts["random"] += 1

            # MV-aligned: pick one that agrees with majority, break ties by CC
            maj = compute_mv_agreement(
                [t["answer"] for t in trajs_k], K=K
            )["majority_answer"]
            aligned = [t for t in trajs_k if answers_equivalent(t["answer"], maj)]
            if aligned:
                aligned.sort(key=lambda t: t["hybrid_cycle"])
                pick_mv = aligned[0]
            else:
                pick_mv = min(trajs_k, key=lambda t: t["hybrid_cycle"])
            if answers_equivalent(pick_mv["answer"], gt):
                counts["mv_aligned"] += 1

            # Best-CC: lowest hybrid_cycle
            pick_cc = min(trajs_k, key=lambda t: t["hybrid_cycle"])
            if answers_equivalent(pick_cc["answer"], gt):
                counts["best_cc"] += 1

            # Best-composite: highest combined_reward
            pick_comp = max(trajs_k, key=lambda t: t["combined_reward"])
            if answers_equivalent(pick_comp["answer"], gt):
                counts["best_composite"] += 1

            # Oracle: any correct solution exists
            any_correct = any(
                answers_equivalent(t["answer"], gt) for t in trajs_k
            )
            if any_correct:
                counts["oracle"] += 1

        results[K] = {
            "n_questions": n_total,
            **{k: v / max(1, n_total) for k, v in counts.items()},
        }
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 4: Filtered Voting
# ═════════════════════════════════════════════════════════════════════════════

def analysis_4_filtered_voting(
    traj_data: list[dict],
    k_values: list[int],
    thresholds: list[float],
) -> dict:
    """Remove high-CC (cycle-inconsistent) trajectories before voting."""
    groups = _group_by_question(traj_data)

    results = {}
    for K in k_values:
        threshold_results = []
        for threshold in thresholds:
            n_total = 0
            n_correct = 0
            n_filtered_empty = 0  # fell back to standard MV

            for q_idx, trajs in groups.items():
                trajs_k = sorted(trajs, key=lambda t: t["traj_idx"])[:K]
                if len(trajs_k) < K:
                    continue
                n_total += 1
                gt = trajs_k[0]["ground_truth"]

                # Filter: keep solutions with hybrid_cycle < threshold
                filtered = [t for t in trajs_k if t["hybrid_cycle"] < threshold]
                if not filtered:
                    # Fall back to standard MV
                    filtered = trajs_k
                    n_filtered_empty += 1

                answers = [t["answer"] for t in filtered]
                clusters = cluster_answers(answers)
                winner = answers[clusters[0][0]]
                if answers_equivalent(winner, gt):
                    n_correct += 1

            threshold_results.append({
                "threshold": threshold,
                "accuracy": n_correct / max(1, n_total),
                "n_questions": n_total,
                "n_fallback": n_filtered_empty,
            })

        # Also add unfiltered baseline (threshold=inf)
        n_total = 0
        n_correct = 0
        for q_idx, trajs in groups.items():
            trajs_k = sorted(trajs, key=lambda t: t["traj_idx"])[:K]
            if len(trajs_k) < K:
                continue
            n_total += 1
            gt = trajs_k[0]["ground_truth"]
            answers = [t["answer"] for t in trajs_k]
            clusters = cluster_answers(answers)
            winner = answers[clusters[0][0]]
            if answers_equivalent(winner, gt):
                n_correct += 1
        baseline_acc = n_correct / max(1, n_total)

        results[K] = {
            "baseline_acc": baseline_acc,
            "thresholds": threshold_results,
        }
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 5: Wrong-Majority Rescue
# ═════════════════════════════════════════════════════════════════════════════

def analysis_5_wrong_majority_rescue(
    traj_data: list[dict],
    k_values: list[int],
) -> dict:
    """When the majority is wrong, can CC identify correct minority solutions?"""
    groups = _group_by_question(traj_data)

    results = {}
    for K in k_values:
        n_wrong_majority = 0
        n_has_correct_minority = 0
        n_rescued_by_cc = 0
        n_rescued_by_composite = 0

        # For AUROC within wrong-majority groups
        within_correct_labels = []
        within_cc_scores = []
        within_composite_scores = []

        cc_correct_vals = []
        cc_incorrect_vals = []

        for q_idx, trajs in groups.items():
            trajs_k = sorted(trajs, key=lambda t: t["traj_idx"])[:K]
            if len(trajs_k) < K:
                continue

            gt = trajs_k[0]["ground_truth"]
            answers = [t["answer"] for t in trajs_k]
            clusters = cluster_answers(answers)
            majority_answer = answers[clusters[0][0]]

            # Is the majority answer correct?
            if answers_equivalent(majority_answer, gt):
                continue  # Skip — majority is right

            n_wrong_majority += 1

            # Is there a correct minority?
            correct_trajs = [
                t for t in trajs_k
                if answers_equivalent(t["answer"], gt)
            ]
            incorrect_trajs = [
                t for t in trajs_k
                if not answers_equivalent(t["answer"], gt)
            ]

            if correct_trajs:
                n_has_correct_minority += 1

                # Track CC scores for correct vs incorrect within wrong-majority
                for t in correct_trajs:
                    cc_correct_vals.append(t["hybrid_cycle"])
                for t in incorrect_trajs:
                    cc_incorrect_vals.append(t["hybrid_cycle"])

            # Gather labels/scores for AUROC within wrong-majority groups
            for t in trajs_k:
                is_c = int(answers_equivalent(t["answer"], gt))
                within_correct_labels.append(is_c)
                within_cc_scores.append(1.0 - t["hybrid_cycle"])  # higher = better
                within_composite_scores.append(t["combined_reward"])

            # Rescue attempt: would best-CC selection pick a correct answer?
            best_cc_traj = min(trajs_k, key=lambda t: t["hybrid_cycle"])
            if answers_equivalent(best_cc_traj["answer"], gt):
                n_rescued_by_cc += 1

            # Rescue attempt: would best-composite pick a correct answer?
            best_comp_traj = max(trajs_k, key=lambda t: t["combined_reward"])
            if answers_equivalent(best_comp_traj["answer"], gt):
                n_rescued_by_composite += 1

        y_within = np.array(within_correct_labels)
        cc_within = np.array(within_cc_scores)
        comp_within = np.array(within_composite_scores)

        entry = {
            "n_wrong_majority": n_wrong_majority,
            "n_has_correct_minority": n_has_correct_minority,
            "pct_has_correct_minority": (
                n_has_correct_minority / max(1, n_wrong_majority)
            ),
            "n_rescued_by_cc": n_rescued_by_cc,
            "n_rescued_by_composite": n_rescued_by_composite,
            "rescue_rate_cc": n_rescued_by_cc / max(1, n_wrong_majority),
            "rescue_rate_composite": n_rescued_by_composite / max(1, n_wrong_majority),
            "within_auroc_cc": _safe_auroc(y_within, cc_within),
            "within_auroc_composite": _safe_auroc(y_within, comp_within),
        }

        if cc_correct_vals and cc_incorrect_vals:
            entry["mean_cc_correct_minority"] = float(np.mean(cc_correct_vals))
            entry["mean_cc_incorrect_majority"] = float(np.mean(cc_incorrect_vals))
            entry["cohen_d_rescue"] = _cohen_d(
                np.array(cc_incorrect_vals),  # "higher" group for d
                np.array(cc_correct_vals),     # "lower" group
            )

        results[K] = entry
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 6: CC Distribution by Correctness
# ═════════════════════════════════════════════════════════════════════════════

def analysis_6_cc_distribution(traj_data: list[dict]) -> dict:
    """Distribution stats for hybrid_cycle by trajectory correctness."""
    y = np.array([t["trajectory_correct"] for t in traj_data])
    hc = np.array([t["hybrid_cycle"] for t in traj_data])
    cr = np.array([t["combined_reward"] for t in traj_data])

    hc_correct = hc[y == 1]
    hc_incorrect = hc[y == 0]
    cr_correct = cr[y == 1]
    cr_incorrect = cr[y == 0]

    result = {
        "hybrid_cycle": {
            "mean_correct": float(hc_correct.mean()) if len(hc_correct) else 0.0,
            "mean_incorrect": float(hc_incorrect.mean()) if len(hc_incorrect) else 0.0,
            "std_correct": float(hc_correct.std()) if len(hc_correct) else 0.0,
            "std_incorrect": float(hc_incorrect.std()) if len(hc_incorrect) else 0.0,
            "median_correct": float(np.median(hc_correct)) if len(hc_correct) else 0.0,
            "median_incorrect": float(np.median(hc_incorrect)) if len(hc_incorrect) else 0.0,
            "cohen_d": _cohen_d(hc_incorrect, hc_correct),  # expect d > 0 (incorrect higher)
        },
        "combined_reward": {
            "mean_correct": float(cr_correct.mean()) if len(cr_correct) else 0.0,
            "mean_incorrect": float(cr_incorrect.mean()) if len(cr_incorrect) else 0.0,
            "std_correct": float(cr_correct.std()) if len(cr_correct) else 0.0,
            "std_incorrect": float(cr_incorrect.std()) if len(cr_incorrect) else 0.0,
            "cohen_d": _cohen_d(cr_correct, cr_incorrect),  # expect d > 0 (correct higher)
        },
    }

    # Within-question analysis: is CC better at separating within the same question?
    groups = _group_by_question(traj_data)
    within_ds = []
    for q_idx, trajs in groups.items():
        hc_c = [t["hybrid_cycle"] for t in trajs if t["trajectory_correct"]]
        hc_i = [t["hybrid_cycle"] for t in trajs if not t["trajectory_correct"]]
        if hc_c and hc_i:
            within_ds.append(
                _cohen_d(np.array(hc_i), np.array(hc_c))
            )
    result["within_question_cohen_d_mean"] = float(np.mean(within_ds)) if within_ds else 0.0
    result["within_question_cohen_d_std"] = float(np.std(within_ds)) if within_ds else 0.0
    result["n_questions_with_mixed"] = len(within_ds)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 7: Simulated GRPO Advantage
# ═════════════════════════════════════════════════════════════════════════════

def analysis_7_grpo_advantage(
    traj_data: list[dict],
    k_values: list[int] | None = None,
) -> dict:
    """Simulate GRPO advantage signal for each trajectory.

    advantage_i = (reward_i - mean(rewards)) / std(rewards)
    """
    if k_values is None:
        k_values = [16]

    groups = _group_by_question(traj_data)

    results = {}
    for K in k_values:
        # Composite reward advantages
        adv_correct = []
        adv_incorrect = []
        # MV-only reward advantages (for comparison)
        adv_mv_correct = []
        adv_mv_incorrect = []

        n_correct_positive_advantage = 0
        n_correct_total = 0
        n_incorrect_negative_advantage = 0
        n_incorrect_total = 0

        n_mv_correct_positive = 0
        n_mv_correct_total = 0
        n_mv_incorrect_negative = 0
        n_mv_incorrect_total = 0

        for q_idx, trajs in groups.items():
            trajs_k = sorted(trajs, key=lambda t: t["traj_idx"])[:K]
            if len(trajs_k) < K:
                continue

            # Composite rewards
            rewards = np.array([t["combined_reward"] for t in trajs_k])
            mv_rewards = np.array([t["mv_agreement_soft_K16"] for t in trajs_k], dtype=float)

            r_mean, r_std = rewards.mean(), rewards.std()
            mv_mean, mv_std = mv_rewards.mean(), mv_rewards.std()

            for t, r, mvr in zip(trajs_k, rewards, mv_rewards):
                # Composite advantage
                adv = (r - r_mean) / (r_std + 1e-8)
                is_correct = t["trajectory_correct"]

                if is_correct:
                    adv_correct.append(float(adv))
                    n_correct_total += 1
                    if adv > 0:
                        n_correct_positive_advantage += 1
                else:
                    adv_incorrect.append(float(adv))
                    n_incorrect_total += 1
                    if adv < 0:
                        n_incorrect_negative_advantage += 1

                # MV-only advantage
                mv_adv = (mvr - mv_mean) / (mv_std + 1e-8)
                if is_correct:
                    adv_mv_correct.append(float(mv_adv))
                    n_mv_correct_total += 1
                    if mv_adv > 0:
                        n_mv_correct_positive += 1
                else:
                    adv_mv_incorrect.append(float(mv_adv))
                    n_mv_incorrect_total += 1
                    if mv_adv < 0:
                        n_mv_incorrect_negative += 1

        results[K] = {
            "composite": {
                "mean_adv_correct": float(np.mean(adv_correct)) if adv_correct else 0.0,
                "mean_adv_incorrect": float(np.mean(adv_incorrect)) if adv_incorrect else 0.0,
                "std_adv_correct": float(np.std(adv_correct)) if adv_correct else 0.0,
                "std_adv_incorrect": float(np.std(adv_incorrect)) if adv_incorrect else 0.0,
                "pct_correct_positive": (
                    n_correct_positive_advantage / max(1, n_correct_total)
                ),
                "pct_incorrect_negative": (
                    n_incorrect_negative_advantage / max(1, n_incorrect_total)
                ),
                "adv_correct": adv_correct,
                "adv_incorrect": adv_incorrect,
            },
            "mv_only": {
                "mean_adv_correct": float(np.mean(adv_mv_correct)) if adv_mv_correct else 0.0,
                "mean_adv_incorrect": float(np.mean(adv_mv_incorrect)) if adv_mv_incorrect else 0.0,
                "pct_correct_positive": (
                    n_mv_correct_positive / max(1, n_mv_correct_total)
                ),
                "pct_incorrect_negative": (
                    n_mv_incorrect_negative / max(1, n_mv_incorrect_total)
                ),
                "adv_correct": adv_mv_correct,
                "adv_incorrect": adv_mv_incorrect,
            },
        }

    return results
