"""Per-trajectory signal computation: MV agreement, composite rewards.

NOTE: This module expects exp1/ and exp2/ to be on sys.path (set up by run.py).
"""

from __future__ import annotations

import numpy as np

from utils import answers_equivalent
from voting import extract_final_answer, cluster_answers


# ── Per-trajectory MV agreement ──────────────────────────────────────────────

def compute_mv_agreement(
    answers: list[str],
    K: int | None = None,
) -> dict:
    """Compute majority answer + per-trajectory MV agreement for one question.

    Args:
        answers: List of K extracted answer strings (one per trajectory).
        K: If provided, use only the first K answers (nested subsets).

    Returns:
        Dict with:
            majority_answer: str
            mv_agreement: list[int]   — binary (1 if agrees with majority)
            mv_agreement_soft: list[float]  — fraction of *other* solutions agreeing
            majority_correct: None  (to be filled later by caller)
    """
    if K is not None:
        answers = answers[:K]
    actual_K = len(answers)

    clusters = cluster_answers(answers)
    majority_answer = answers[clusters[0][0]]

    # Binary agreement
    mv_agreement = [
        int(answers_equivalent(a, majority_answer)) for a in answers
    ]

    # Soft agreement: fraction of *other* solutions with same answer
    mv_agreement_soft = []
    for i, a in enumerate(answers):
        # Count how many others agree with this answer
        same_count = sum(
            1 for j, b in enumerate(answers)
            if j != i and answers_equivalent(a, b)
        )
        mv_agreement_soft.append(same_count / max(1, actual_K - 1))

    return {
        "majority_answer": majority_answer,
        "mv_agreement": mv_agreement,
        "mv_agreement_soft": mv_agreement_soft,
    }


# ── Composite rewards ────────────────────────────────────────────────────────

def compute_composite_rewards(
    mv_agreement_soft: list[float],
    combined_rewards: list[float],
    alpha: float = 0.5,
) -> dict:
    """Compute composite reward variants for one question's trajectories.

    Args:
        mv_agreement_soft: Per-trajectory soft MV agreement [0, 1].
        combined_rewards: Per-trajectory combined_reward_i (answer_match_i - hybrid_cycle_i).
        alpha: Weight for MV in weighted composite.

    Returns:
        Dict with lists:
            reward_soft: mv_agreement_soft + combined_reward
            reward_weighted: α * mv_agreement_soft + (1-α) * combined_reward
    """
    reward_soft = [
        mv + cr for mv, cr in zip(mv_agreement_soft, combined_rewards)
    ]
    reward_weighted = [
        alpha * mv + (1.0 - alpha) * cr
        for mv, cr in zip(mv_agreement_soft, combined_rewards)
    ]
    return {
        "reward_soft": reward_soft,
        "reward_weighted": reward_weighted,
    }


# ── CC-weighted voting ───────────────────────────────────────────────────────

def cc_weighted_vote(
    answers: list[str],
    cc_scores: list[float],
    weight_fn: str = "linear",
) -> str:
    """Majority voting where each vote is weighted by CC quality.

    Args:
        answers: Per-trajectory answer strings.
        cc_scores: Per-trajectory hybrid_cycle (distance; lower = better).
        weight_fn: "linear" → (1 - cc), "exp" → exp(-cc), "inv" → 1/(1+cc).

    Returns:
        The winning answer string.
    """
    if weight_fn == "linear":
        weights = [max(0.0, 1.0 - c) for c in cc_scores]
    elif weight_fn == "exp":
        weights = [np.exp(-c) for c in cc_scores]
    elif weight_fn == "inv":
        weights = [1.0 / (1.0 + c) for c in cc_scores]
    else:
        weights = [1.0] * len(answers)

    # Accumulate weighted votes per unique answer cluster
    clusters = cluster_answers(answers)
    best_score = -1.0
    best_answer = answers[0] if answers else ""

    for cluster in clusters:
        score = sum(weights[i] for i in cluster)
        if score > best_score:
            best_score = score
            best_answer = answers[cluster[0]]

    return best_answer


def composite_weighted_vote(
    answers: list[str],
    composite_rewards: list[float],
) -> str:
    """Vote weighted by composite reward (higher = more trusted)."""
    # Shift so all weights are non-negative
    min_r = min(composite_rewards) if composite_rewards else 0.0
    weights = [r - min_r + 1e-6 for r in composite_rewards]

    clusters = cluster_answers(answers)
    best_score = -1.0
    best_answer = answers[0] if answers else ""

    for cluster in clusters:
        score = sum(weights[i] for i in cluster)
        if score > best_score:
            best_score = score
            best_answer = answers[cluster[0]]

    return best_answer
