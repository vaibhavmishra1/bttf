"""Fusion strategies: linear, product, logistic regression."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ── Simple fusion functions ──────────────────────────────────────────────────

def fused_linear(
    voting_confidence: np.ndarray,
    combined_reward: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Linear combination: α · vote_conf + (1−α) · combined_reward.

    Both signals are normalised to [0, 1] before combining.
    """
    # Normalise combined_reward from [-1, 1] to [0, 1]
    cr_norm = (combined_reward + 1.0) / 2.0
    return alpha * voting_confidence + (1.0 - alpha) * cr_norm


def fused_product(
    voting_confidence: np.ndarray,
    combined_reward: np.ndarray,
) -> np.ndarray:
    """Multiplicative: vote_conf × (1 + combined_reward) / 2.

    Both must be high for the fused score to be high.
    """
    return voting_confidence * (1.0 + combined_reward) / 2.0


# ── Alpha sweep ──────────────────────────────────────────────────────────────

def sweep_alpha(
    voting_confidence: np.ndarray,
    combined_reward: np.ndarray,
    y_true: np.ndarray,
    alphas: list[float] | None = None,
) -> dict:
    """Sweep α for fused_linear, return best α and full results.

    Returns dict with:
        best_alpha, best_auroc, alpha_results (list of {alpha, auroc})
    """
    if alphas is None:
        alphas = [i / 20 for i in range(21)]

    results = []
    for a in alphas:
        scores = fused_linear(voting_confidence, combined_reward, a)
        if len(set(y_true)) < 2:
            auroc = 0.5
        else:
            auroc = roc_auc_score(y_true, scores)
        results.append({"alpha": a, "auroc": auroc})

    best = max(results, key=lambda r: r["auroc"])
    return {
        "best_alpha": best["alpha"],
        "best_auroc": best["auroc"],
        "alpha_results": results,
    }


# ── Logistic regression fusion ───────────────────────────────────────────────

def fused_logistic_cv(
    features: np.ndarray,
    y_true: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Train logistic regression on feature vectors, evaluate with cross-validation.

    Features should be shape (N, F) where F is the number of signals
    (e.g. [voting_confidence, combined_reward, hybrid_cycle, entropy]).

    Returns dict with:
        mean_auroc, std_auroc, fold_aurocs, oof_scores (out-of-fold predictions)
    """
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    oof_scores = np.zeros(len(y_true))
    fold_aurocs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y_true)):
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])
        y_train = y_true[train_idx]
        y_test = y_true[test_idx]

        clf = LogisticRegression(
            C=1.0, max_iter=1000, random_state=random_state, solver="lbfgs"
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        oof_scores[test_idx] = proba

        if len(set(y_test)) > 1:
            fold_aurocs.append(roc_auc_score(y_test, proba))

    overall_auroc = roc_auc_score(y_true, oof_scores) if len(set(y_true)) > 1 else 0.5

    return {
        "mean_auroc": float(np.mean(fold_aurocs)) if fold_aurocs else 0.5,
        "std_auroc": float(np.std(fold_aurocs)) if fold_aurocs else 0.0,
        "overall_auroc": float(overall_auroc),
        "fold_aurocs": fold_aurocs,
        "oof_scores": oof_scores,
    }


# ── Failure overlap analysis ────────────────────────────────────────────────

def compute_failure_overlap(
    y_true: np.ndarray,
    mv_scores: np.ndarray,
    cc_scores: np.ndarray,
    mv_threshold: float | None = None,
    cc_threshold: float | None = None,
) -> dict:
    """Compute false-positive overlap between MV and CC.

    A false positive is: method assigns high confidence but solver is actually wrong.
    Default thresholds: median of each score among incorrect samples.

    Returns:
        Dict with FP counts, overlap coefficient, Jaccard, and sample-level masks.
    """
    incorrect_mask = y_true == 0

    # Default threshold: median of scores for incorrect samples
    if mv_threshold is None:
        mv_threshold = float(np.median(mv_scores[incorrect_mask]))
    if cc_threshold is None:
        cc_threshold = float(np.median(cc_scores[incorrect_mask]))

    # False positives: incorrect AND high confidence
    fp_mv = incorrect_mask & (mv_scores >= mv_threshold)
    fp_cc = incorrect_mask & (cc_scores >= cc_threshold)

    n_fp_mv = int(fp_mv.sum())
    n_fp_cc = int(fp_cc.sum())
    n_overlap = int((fp_mv & fp_cc).sum())
    n_union = int((fp_mv | fp_cc).sum())

    # Overlap coefficient
    min_fp = min(n_fp_mv, n_fp_cc)
    overlap_coeff = n_overlap / min_fp if min_fp > 0 else 0.0

    # Jaccard
    jaccard = n_overlap / n_union if n_union > 0 else 0.0

    # Unique to each method
    fp_mv_only = int((fp_mv & ~fp_cc).sum())
    fp_cc_only = int((fp_cc & ~fp_mv).sum())

    return {
        "n_incorrect": int(incorrect_mask.sum()),
        "n_fp_mv": n_fp_mv,
        "n_fp_cc": n_fp_cc,
        "n_overlap": n_overlap,
        "n_union": n_union,
        "fp_mv_only": fp_mv_only,
        "fp_cc_only": fp_cc_only,
        "overlap_coefficient": float(overlap_coeff),
        "jaccard": float(jaccard),
        "mv_threshold": float(mv_threshold),
        "cc_threshold": float(cc_threshold),
    }
