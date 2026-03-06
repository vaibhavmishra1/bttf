"""Compute AUROC, AUPRC, and print a summary table."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def _safe_auroc(y_true, y_score, label: str) -> float | None:
    if len(set(y_true)) < 2:
        print(f"  [skip] {label}: only one class present — AUROC undefined")
        return None
    val = roc_auc_score(y_true, y_score)
    return val


def _safe_auprc(y_true, y_score, label: str) -> float | None:
    if len(set(y_true)) < 2:
        print(f"  [skip] {label}: only one class present — AUPRC undefined")
        return None
    val = average_precision_score(y_true, y_score)
    return val


def compute_all_metrics(data: list[dict]) -> dict:
    """Compute AUROC and AUPRC for the three cycle-consistency signals.

    Returns a dict of {signal_name: {auroc, auprc}}.
    """
    y_true = np.array([d["correct"] for d in data])

    # Higher combined_reward / answer_match → more likely correct
    # Lower question_cycle → more likely correct  (so negate for AUROC)
    signals = {
        "question_cycle (negated)": -np.array([d["question_cycle"] for d in data]),
        "answer_match": np.array([d["answer_match"] for d in data], dtype=float),
        "combined_reward": np.array([d["combined_reward"] for d in data]),
    }

    results = {}
    print(f"\n{'Signal':<30s}  {'AUROC':>8s}  {'AUPRC':>8s}")
    print("-" * 52)

    for name, scores in signals.items():
        auroc = _safe_auroc(y_true, scores, name)
        auprc = _safe_auprc(y_true, scores, name)
        results[name] = {"auroc": auroc, "auprc": auprc}
        a_str = f"{auroc:.4f}" if auroc is not None else "  n/a "
        p_str = f"{auprc:.4f}" if auprc is not None else "  n/a "
        print(f"{name:<30s}  {a_str:>8s}  {p_str:>8s}")

    # Overall solver accuracy
    acc = y_true.mean()
    print(f"\nSolver accuracy: {acc:.4f}  ({int(y_true.sum())}/{len(y_true)})")

    return results
