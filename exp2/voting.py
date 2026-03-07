"""Majority voting logic: answer clustering, vote counting, entropy.

NOTE: This module expects exp1/ to be on sys.path (set up by run.py).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from utils import extract_boxed_answer, extract_answer, answers_equivalent


# ── Answer extraction ────────────────────────────────────────────────────────

def extract_final_answer(solution: str) -> str:
    """Extract the final answer from a solution string (boxed → regex → last number)."""
    boxed = extract_boxed_answer(solution)
    if boxed:
        return boxed
    return extract_answer(solution)


# ── Answer clustering ────────────────────────────────────────────────────────

def cluster_answers(answers: list[str]) -> list[list[int]]:
    """Group answer indices into equivalence classes using answers_equivalent().

    Returns a list of clusters, each cluster is a list of indices into `answers`.
    Clusters are sorted by size (largest first), then by earliest member index.
    """
    clusters: list[list[int]] = []
    assigned = [False] * len(answers)

    for i, ans_i in enumerate(answers):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(answers)):
            if assigned[j]:
                continue
            if answers_equivalent(ans_i, answers[j]):
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    # Sort: largest cluster first; ties broken by earliest member
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters


# ── Voting signals ───────────────────────────────────────────────────────────

def compute_voting_signals(
    solutions: list[str],
    K: int | None = None,
) -> dict:
    """Compute all majority voting signals for a single question.

    Args:
        solutions: List of K solution strings (stochastic samples).
        K: If provided, use only the first K solutions. Otherwise use all.

    Returns:
        Dict with: majority_answer, voting_confidence, entropy, unique_ratio,
                   all_answers, cluster_sizes
    """
    if K is not None:
        solutions = solutions[:K]
    actual_K = len(solutions)

    # Extract answers
    answers = [extract_final_answer(s) for s in solutions]

    # Cluster
    clusters = cluster_answers(answers)

    # Majority = largest cluster
    majority_cluster = clusters[0]
    majority_answer = answers[majority_cluster[0]]
    voting_confidence = len(majority_cluster) / actual_K

    # Entropy over cluster distribution
    probs = np.array([len(c) / actual_K for c in clusters])
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    # Unique ratio
    unique_ratio = len(clusters) / actual_K

    # Cluster sizes for debugging
    cluster_sizes = [len(c) for c in clusters]

    return {
        "majority_answer": majority_answer,
        "voting_confidence": float(voting_confidence),
        "entropy": float(entropy),
        "unique_ratio": float(unique_ratio),
        "n_clusters": len(clusters),
        "cluster_sizes": cluster_sizes,
        "all_answers": answers,
    }


def compute_voting_signals_multi_k(
    solutions: list[str],
    k_values: list[int],
) -> dict[int, dict]:
    """Compute voting signals for multiple K values using nested subsets.

    The first K solutions are used for each K value (nested subsets ensure
    fair comparison across K).

    Returns:
        Dict mapping K → voting signals dict.
    """
    result = {}
    for k in k_values:
        if k > len(solutions):
            continue
        result[k] = compute_voting_signals(solutions, K=k)
    return result
