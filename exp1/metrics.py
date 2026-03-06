"""Simple discrimination metrics for cycle-consistency signals."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# ── Helpers ─────────────────────────────────────────────────────────────────

def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size: (mean_a - mean_b) / pooled_std."""
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


def _safe_roc(y_true, scores) -> float | None:
    return roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else None


def _safe_auprc(y_true, scores) -> float | None:
    return average_precision_score(y_true, scores) if len(set(y_true)) > 1 else None


# ── Core metrics ─────────────────────────────────────────────────────────────

def compute_all_metrics(data: list[dict], label: str = "all") -> dict:
    """Print a discrimination table and return a results dict.

    For each signal we show:
      - Mean ± std for correct and incorrect groups
      - Raw gap (correct − incorrect)
      - Cohen's d  (effect size; |d|>0.5 = medium, >0.8 = large)
      - AUROC  (area under ROC; 0.5 = random, 1.0 = perfect)
      - AUPRC  (area under precision-recall; baseline = prevalence)
    """
    y = np.array([d["correct"] for d in data])
    n_c, n_i = int(y.sum()), int((1 - y).sum())
    n_total  = len(y)
    acc      = y.mean()
    prev     = acc  # positive-class prevalence = AUPRC baseline

    print(f"\n{'─'*70}")
    print(f"  Dataset: {label}   |   N={n_total}  correct={n_c}  incorrect={n_i}  acc={acc:.1%}")
    print(f"{'─'*70}")

    # ── Three signals ──────────────────────────────────────────────────────
    # question_cycle: lower → better (correct), so negate for AUROC/AUPRC
    # answer_match  : higher → better (correct)
    # combined_reward: higher → better (correct)
    signals = {
        "question_cycle  (dist↓)": (
            np.array([d["question_cycle"]  for d in data]),
            "lower",        # direction: lower = better for the "correct" group
        ),
        "answer_match    (agree↑)": (
            np.array([d["answer_match"]    for d in data], dtype=float),
            "higher",
        ),
        "combined_reward (↑)": (
            np.array([d["combined_reward"] for d in data]),
            "higher",
        ),
    }

    # Header
    W = 26
    print(
        f"\n  {'Signal':<{W}}  {'Correct':>14}  {'Incorrect':>14}  "
        f"{'Gap':>7}  {'Cohen d':>7}  {'AUROC':>6}  {'AUPRC':>6}"
    )
    print(f"  {'─'*W}  {'─'*14}  {'─'*14}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}")

    results: dict[str, dict] = {}

    for name, (scores, direction) in signals.items():
        sc  = scores[y == 1]
        si  = scores[y == 0]
        gap = sc.mean() - si.mean()
        d   = _cohen_d(sc, si)
        # For "lower is better", negate so that higher score → more likely correct
        auroc_scores = -scores if direction == "lower" else scores
        auroc = _safe_roc(y, auroc_scores)
        auprc = _safe_auprc(y, auroc_scores)

        a_s = f"{auroc:.3f}" if auroc is not None else "  n/a"
        p_s = f"{auprc:.3f}" if auprc is not None else "  n/a"
        correct_s   = f"{sc.mean():.3f}±{sc.std():.3f}"
        incorrect_s = f"{si.mean():.3f}±{si.std():.3f}"

        print(
            f"  {name:<{W}}  {correct_s:>14}  {incorrect_s:>14}  "
            f"{gap:>+7.3f}  {d:>+7.2f}  {a_s:>6}  {p_s:>6}"
        )
        results[name] = dict(
            mean_correct=sc.mean(), std_correct=sc.std(),
            mean_incorrect=si.mean(), std_incorrect=si.std(),
            gap=gap, cohen_d=d, auroc=auroc, auprc=auprc,
        )

    # ── answer_match agreement rate ─────────────────────────────────────────
    am = np.array([d["answer_match"] for d in data], dtype=float)
    am_c = am[y == 1].mean()
    am_i = am[y == 0].mean()
    print(
        f"\n  Answer-agreement rate: correct={am_c:.1%}  "
        f"incorrect={am_i:.1%}  gap={am_c - am_i:+.1%}"
    )

    # ── answer_match as a hard binary threshold classifier ──────────────────
    tp = int(((am == 1) & (y == 1)).sum())
    fp = int(((am == 1) & (y == 0)).sum())
    tn = int(((am == 0) & (y == 0)).sum())
    fn = int(((am == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(
        f"  answer_match threshold: "
        f"Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  "
        f"TP={tp} FP={fp} TN={tn} FN={fn}"
    )
    print(f"  AUPRC baseline (prevalence): {prev:.3f}")

    results["accuracy"]        = float(acc)
    results["n_total"]         = n_total
    results["n_correct"]       = n_c
    results["n_incorrect"]     = n_i
    results["answer_match_precision"] = prec
    results["answer_match_recall"]    = rec
    results["answer_match_f1"]        = f1
    return results
