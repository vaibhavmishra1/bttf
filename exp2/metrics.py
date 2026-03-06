"""Discrimination metrics for exp2 — old signals + three new ones.

New vs exp1/metrics.py:
  + bleu_cycle          (Fix 3: BLEU distance, lower → correct)
  + hybrid_cycle        (Fix 3: 0.5*bleu_cycle + 0.5*question_cycle, lower → correct)
  + answer_match_llm    (Fix 1: LLM judge, higher → correct)
  + combined_reward_v2  (answer_match_llm - hybrid_cycle, higher → correct)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


def _safe_roc(y_true, scores) -> float | None:
    return roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else None


def _safe_auprc(y_true, scores) -> float | None:
    return average_precision_score(y_true, scores) if len(set(y_true)) > 1 else None


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_all_metrics(data: list[dict], label: str = "all") -> dict:
    """Print a full discrimination table for all exp2 signals and return results dict.

    Signals tracked:
      EXP-1 BASELINE (for comparison)
        question_cycle    (dist↓)
        answer_match      (regex ↑)
        combined_reward   (answer_match − question_cycle ↑)
      EXP-2 NEW
        bleu_cycle        (dist↓)  [Fix 3]
        hybrid_cycle      (dist↓)  [Fix 3]
        answer_match_llm  (↑)      [Fix 1]
        combined_reward_v2 (answer_match_llm − hybrid_cycle ↑) [Fix 1+3]
    """
    y        = np.array([d["correct"] for d in data])
    n_c, n_i = int(y.sum()), int((1 - y).sum())
    n_total  = len(y)
    acc      = y.mean()

    print(f"\n{'─'*80}")
    print(f"  Dataset: {label}   |   N={n_total}  correct={n_c}  incorrect={n_i}  acc={acc:.1%}")
    print(f"{'─'*80}")

    # ── Signal definitions ──────────────────────────────────────────────────
    # (name, key, direction)  direction: "lower" means lower → correct
    signals = [
        # exp1 baseline
        ("question_cycle   (dist↓)  [e1]",  "question_cycle",    "lower"),
        ("answer_match     (regex↑) [e1]",  "answer_match",      "higher"),
        ("combined_reward  (↑)      [e1]",  "combined_reward",   "higher"),
        # exp2 new
        ("bleu_cycle       (dist↓)  [e2]",  "bleu_cycle",        "lower"),
        ("hybrid_cycle     (dist↓)  [e2]",  "hybrid_cycle",      "lower"),
        ("answer_match_llm (↑)      [e2]",  "answer_match_llm",  "higher"),
        ("combined_rwd_v2  (↑)      [e2]",  "combined_reward_v2","higher"),
    ]

    W = 38
    print(
        f"\n  {'Signal':<{W}}  {'Correct':>14}  {'Incorrect':>14}  "
        f"{'Gap':>7}  {'Cohen d':>7}  {'AUROC':>6}  {'AUPRC':>6}"
    )
    print(f"  {'─'*W}  {'─'*14}  {'─'*14}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}")

    results: dict[str, dict] = {}
    divider_printed = False

    for name, key, direction in signals:
        if key not in data[0] and not divider_printed:
            continue

        # Print divider before exp2 block
        if "[e2]" in name and not divider_printed:
            print(f"  {'·'*W}  {'·'*14}  {'·'*14}  {'·'*7}  {'·'*7}  {'·'*6}  {'·'*6}")
            divider_printed = True

        scores = np.array([d.get(key, 0) for d in data], dtype=float)
        sc, si = scores[y == 1], scores[y == 0]
        gap    = sc.mean() - si.mean()
        d_eff  = _cohen_d(sc, si)

        auroc_scores = -scores if direction == "lower" else scores
        auroc  = _safe_roc(y, auroc_scores)
        auprc  = _safe_auprc(y, auroc_scores)

        a_s = f"{auroc:.3f}" if auroc is not None else "  n/a"
        p_s = f"{auprc:.3f}" if auprc is not None else "  n/a"
        print(
            f"  {name:<{W}}  {sc.mean():>6.3f}±{sc.std():.3f}  "
            f"{si.mean():>6.3f}±{si.std():.3f}  "
            f"{gap:>+7.3f}  {d_eff:>+7.2f}  {a_s:>6}  {p_s:>6}"
        )
        results[name] = dict(
            mean_correct=sc.mean(), std_correct=sc.std(),
            mean_incorrect=si.mean(), std_incorrect=si.std(),
            gap=gap, cohen_d=d_eff, auroc=auroc, auprc=auprc,
        )

    # ── Binary threshold stats for both answer_match variants ───────────────
    print()
    for am_key, label_str in [("answer_match", "regex [e1]"), ("answer_match_llm", "LLM   [e2]")]:
        am   = np.array([d.get(am_key, 0) for d in data], dtype=float)
        tp   = int(((am == 1) & (y == 1)).sum())
        fp   = int(((am == 1) & (y == 0)).sum())
        tn   = int(((am == 0) & (y == 0)).sum())
        fn   = int(((am == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        am_c = am[y == 1].mean()
        am_i = am[y == 0].mean()
        print(
            f"  answer_match ({label_str}): "
            f"correct={am_c:.1%}  incorrect={am_i:.1%}  "
            f"Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  "
            f"TP={tp} FP={fp} TN={tn} FN={fn}"
        )

    print(f"  AUPRC baseline (prevalence): {acc:.3f}")

    results["accuracy"]   = float(acc)
    results["n_total"]    = n_total
    results["n_correct"]  = n_c
    results["n_incorrect"] = n_i
    return results
