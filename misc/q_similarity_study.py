#!/usr/bin/env python3
"""
Q–Q' Similarity Study
=====================
For each (Q, Q') pair in exp1/outputs_v2/results.jsonl:

  Step 1 — GPT judge (cached)
    Ask GPT-4o-mini: "Do Q and Q' ask for the same numerical answer?"
    → gpt_q_match ∈ {0, 1}

  Step 2 — Compute similarity metrics between Q and Q'
    • question_cycle   — 1 − cosine_sim  (Qwen3-Embedding-4B, from results)
    • bleu             — sentence BLEU(Q, Q')
    • rouge1/2/L       — ROUGE F1 scores
    • number_jaccard   — Jaccard on numeric token sets
    • chrf             — character n-gram F-score (sacreBLEU)
    • token_jaccard    — word-level Jaccard (set overlap)
    • lev_ratio        — normalised Levenshtein edit distance

  Step 3 — Evaluate every metric as a predictor of:
    (a) correct        — solver got the right answer (the ultimate target)
    (b) answer_match   — regex-based A == A' (existing signal)
    (c) gpt_q_match    — GPT's Q≡Q' judgment (new signal)

  Step 4 — Report AUROC / AUPRC tables and plots
    • auroc_comparison.png    — bar chart, all metrics × all targets
    • distributions.png       — correct vs incorrect distributions per metric
    • correlation_heatmap.png — Pearson r between all metrics + labels

Usage:
    export OPENAI_API_KEY="sk-..."
    python q_similarity_study.py
    python q_similarity_study.py --skip-gpt   # skip GPT calls, use cached labels
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "..", "exp1", "outputs_v2", "results.jsonl")
OUT_DIR   = os.path.join(BASE, "outputs")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
GPT_CACHE = os.path.join(OUT_DIR, "gpt_labels.jsonl")
SCORES_OUT = os.path.join(OUT_DIR, "similarity_scores.jsonl")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

GPT_MODEL       = "gpt-4o-mini"
GPT_CONCURRENT  = 30


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def save_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


# ── Step 1 — GPT judge ────────────────────────────────────────────────────────

_GPT_SYSTEM = (
    "You are a math problem analyst. "
    "Decide whether two math questions are asking for the same numerical answer. "
    "Ignore wording differences — focus only on whether solving both would yield "
    "the same correct final number. "
    "Respond with ONLY the single word YES or NO."
)

_GPT_FEW_SHOT = [
    ("Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins. "
     "She sells the rest for $2 each. How much does she make daily?",
     "Janet's ducks lay 16 eggs per day. She uses 3 for breakfast and 4 for baking. "
     "She sells the remaining eggs at $2 each. What is her daily revenue?",
     "YES"),
    ("A train travels at 60 mph for 2 hours. How far does it go?",
     "A car travels at 60 mph for 3 hours. How far does it go?",
     "NO"),
    ("John has 5 red balls and 3 blue balls. How many balls total?",
     "John has 5 red and 3 blue marbles. What is the total number of marbles?",
     "YES"),
    ("What is 15% of 200?",
     "What is 20% of 200?",
     "NO"),
]


def _build_gpt_messages(q: str, q_prime: str) -> list[dict]:
    msgs = [{"role": "system", "content": _GPT_SYSTEM}]
    for q1, q2, verdict in _GPT_FEW_SHOT:
        msgs.append({"role": "user",
                     "content": f"Question 1: {q1}\nQuestion 2: {q2}\nSame answer?"})
        msgs.append({"role": "assistant", "content": verdict})
    msgs.append({"role": "user",
                 "content": f"Question 1: {q}\nQuestion 2: {q_prime}\nSame answer?"})
    return msgs


def _call_gpt(item: dict, client) -> dict:
    """Call GPT for one (Q, Q') pair. Returns {id, gpt_q_match, gpt_raw}."""
    msgs = _build_gpt_messages(item["question"], item["question_Q_prime"])
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=msgs,
        temperature=0.0,
        max_tokens=3,
    )
    raw = resp.choices[0].message.content.strip().upper()
    return {
        "id": item["id"],
        "gpt_q_match": 1 if "YES" in raw else 0,
        "gpt_raw": raw,
    }


def get_gpt_labels(data: list[dict], skip: bool = False) -> dict[int, int]:
    """Return {id: gpt_q_match} — from cache if available, else call API."""
    # Load existing cache
    cache: dict[int, int] = {}
    if os.path.exists(GPT_CACHE):
        for row in load_jsonl(GPT_CACHE):
            cache[row["id"]] = row["gpt_q_match"]
        print(f"  Loaded {len(cache)} cached GPT labels from {GPT_CACHE}")

    if skip:
        print("  --skip-gpt: using cached labels only")
        return cache

    missing = [d for d in data if d["id"] not in cache]
    if not missing:
        print("  All GPT labels cached — no API calls needed.")
        return cache

    print(f"  Calling GPT ({GPT_MODEL}) for {len(missing)} samples "
          f"[{GPT_CONCURRENT} concurrent] …")
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    new_rows: list[dict] = []
    n_yes = n_no = n_err = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=GPT_CONCURRENT) as pool:
        futures = {pool.submit(_call_gpt, d, client): d["id"] for d in missing}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            if done % 200 == 0 or done == len(missing):
                print(f"    {done}/{len(missing)} …", flush=True)
            try:
                row = fut.result()
                cache[row["id"]] = row["gpt_q_match"]
                new_rows.append(row)
                if row["gpt_q_match"]:
                    n_yes += 1
                else:
                    n_no += 1
            except Exception as e:
                idx = futures[fut]
                print(f"  [!] id={idx} error: {e}")
                cache[idx] = 0
                n_err += 1

    # Append new rows to cache file
    with open(GPT_CACHE, "a") as f:
        for row in new_rows:
            f.write(json.dumps(row) + "\n")

    print(f"  GPT done: YES={n_yes}  NO={n_no}  error→NO={n_err}")
    return cache


# ── Step 2 — Similarity metrics ───────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d+\.?\d*")


def _numbers(text: str) -> set[str]:
    return set(_NUM_RE.findall(text))


def _tokens(text: str) -> set[str]:
    return set(text.lower().split())


def number_jaccard(q: str, qp: str) -> float:
    """1 − Jaccard similarity on numeric tokens.  0 = identical numbers."""
    n_q, n_qp = _numbers(q), _numbers(qp)
    if not n_q and not n_qp:
        return 0.0
    union = n_q | n_qp
    if not union:
        return 0.0
    sim = len(n_q & n_qp) / len(union)
    return 1.0 - sim   # distance: lower = more preserved


def token_jaccard(q: str, qp: str) -> float:
    """1 − Jaccard similarity on word tokens.  0 = identical word sets."""
    t_q, t_qp = _tokens(q), _tokens(qp)
    union = t_q | t_qp
    if not union:
        return 0.0
    sim = len(t_q & t_qp) / len(union)
    return 1.0 - sim


def bleu_score(q: str, qp: str) -> float:
    """Sentence BLEU (smoothed).  Returns similarity [0,1]; higher = more similar."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref = q.lower().split()
    hyp = qp.lower().split()
    if not ref or not hyp:
        return 0.0
    return sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method1)


def rouge_scores(q: str, qp: str) -> dict[str, float]:
    """ROUGE-1/2/L F1.  Returns similarity [0,1]; higher = more similar."""
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(q.lower(), qp.lower())
    return {k: scores[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]}


def chrf_score(q: str, qp: str) -> float:
    """ChrF score (character n-gram F-score via sacreBLEU).  Higher = more similar."""
    import sacrebleu
    return sacrebleu.corpus_chrf([qp], [[q]]).score / 100.0


def lev_ratio(q: str, qp: str) -> float:
    """Normalised Levenshtein similarity (difflib SequenceMatcher).
    Returns similarity [0,1]; higher = more similar."""
    import difflib
    return difflib.SequenceMatcher(None, q.lower(), qp.lower()).ratio()


def compute_all_metrics(data: list[dict]) -> list[dict]:
    """Compute all similarity metrics for every (Q, Q') pair.
    Similarity metrics are returned as *distances* (lower = more similar)
    to match the direction of `question_cycle`.
    """
    rows = []
    n = len(data)
    print(f"  Computing similarity metrics for {n} pairs …")

    # Import heavy libs once
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    sf = SmoothingFunction().method1
    from rouge_score import rouge_scorer as rs
    rouge = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    import sacrebleu as sb
    import difflib

    for i, d in enumerate(data):
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n} …", flush=True)

        q  = d["question"]
        qp = d["question_Q_prime"]
        q_lo, qp_lo = q.lower(), qp.lower()

        # BLEU (similarity; convert to distance)
        ref = q_lo.split(); hyp = qp_lo.split()
        bleu = sentence_bleu([ref], hyp, smoothing_function=sf) if ref and hyp else 0.0

        # ROUGE
        r = rouge.score(q_lo, qp_lo)
        r1 = r["rouge1"].fmeasure
        r2 = r["rouge2"].fmeasure
        rL = r["rougeL"].fmeasure

        # ChrF
        chrf = sb.corpus_chrf([qp_lo], [[q_lo]]).score / 100.0

        # Levenshtein ratio
        lev = difflib.SequenceMatcher(None, q_lo, qp_lo).ratio()

        # Number Jaccard (kept as distance)
        num_jac = number_jaccard(q, qp)

        # Token Jaccard (kept as distance)
        tok_jac = token_jaccard(q, qp)

        rows.append({
            "id":              d["id"],
            "dataset":         d["dataset"],
            "correct":         d["correct"],
            "answer_match":    d["answer_match"],
            "question_cycle":  d["question_cycle"],   # distance (lower = closer)
            # similarity metrics → converted to distances (lower = more similar)
            "bleu_dist":       1.0 - bleu,
            "rouge1_dist":     1.0 - r1,
            "rouge2_dist":     1.0 - r2,
            "rougeL_dist":     1.0 - rL,
            "chrf_dist":       1.0 - chrf,
            "lev_dist":        1.0 - lev,
            "number_jaccard":  num_jac,
            "token_jaccard":   tok_jac,
        })

    return rows


# ── Step 3 — Evaluation ───────────────────────────────────────────────────────

def auroc(y_true, y_score):
    """Wilcoxon-Mann-Whitney AUROC (no sklearn needed)."""
    y_true  = np.array(y_true,  dtype=float)
    y_score = np.array(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    count = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
    return count / (len(pos) * len(neg))


def avg_precision(y_true, y_score):
    y_true  = np.array(y_true,  dtype=float)
    y_score = np.array(y_score, dtype=float)
    idx     = np.argsort(-y_score)
    ys = y_true[idx]
    if ys.sum() == 0:
        return float("nan")
    tp   = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float(np.sum(prec * (ys / ys.sum())))


def evaluate(scores_data: list[dict], gpt_labels: dict[int, int]) -> dict:
    """Compute AUROC / AUPRC for every metric predicting every target label."""

    # Merge GPT labels
    for d in scores_data:
        d["gpt_q_match"] = gpt_labels.get(d["id"], 0)

    # Distance metrics — lower = more similar → negate for AUROC
    # (higher score → more likely correct)
    METRIC_DISTS = [
        "question_cycle",
        "bleu_dist",
        "rouge1_dist",
        "rouge2_dist",
        "rougeL_dist",
        "chrf_dist",
        "lev_dist",
        "number_jaccard",
        "token_jaccard",
    ]
    # Non-distance binary signals (higher = more likely correct)
    METRIC_DIRECT = ["answer_match", "gpt_q_match"]

    TARGETS = ["correct", "answer_match", "gpt_q_match"]
    DATASETS = [None, "gsm8k", "math", "olympiadbench"]

    results = {}

    for ds in DATASETS:
        subset = scores_data if ds is None else [d for d in scores_data if d["dataset"] == ds]
        label  = ds or "all"
        n      = len(subset)
        y_correct = [d["correct"]      for d in subset]
        y_amatch  = [d["answer_match"] for d in subset]
        y_gpt     = [d["gpt_q_match"]  for d in subset]
        targets   = {"correct": y_correct, "answer_match": y_amatch, "gpt_q_match": y_gpt}

        results[label] = {}
        for target_name, y in targets.items():
            row = {}
            # Distance metrics → negate so "more similar → higher score"
            for m in METRIC_DISTS:
                scores = [-d[m] for d in subset]
                row[m] = {
                    "auroc": auroc(y, scores),
                    "auprc": avg_precision(y, scores),
                }
            # Direct binary signals
            for m in METRIC_DIRECT:
                if m == target_name:
                    continue   # skip predicting a signal with itself
                scores = [d[m] for d in subset]
                row[m] = {
                    "auroc": auroc(y, scores),
                    "auprc": avg_precision(y, scores),
                }
            results[label][target_name] = row

        # Print table
        _print_table(label, n, y_correct, results[label])

    return results


def _print_table(label: str, n: int, y_correct, results_for_ds: dict):
    acc = np.mean(y_correct)
    print(f"\n{'─'*90}")
    print(f"  Dataset: {label.upper()}  (n={n}, acc={acc:.1%})")
    print(f"{'─'*90}")

    targets = list(results_for_ds.keys())
    W = 20
    header = f"  {'Metric':<{W}}"
    for t in targets:
        header += f"  {'AUROC→'+t:>18}  {'AUPRC→'+t:>18}"
    print(header)
    print("  " + "─"*(W + len(targets)*40))

    # Collect all metrics present
    all_metrics = set()
    for t in targets:
        all_metrics.update(results_for_ds[t].keys())

    for m in sorted(all_metrics):
        row = f"  {m:<{W}}"
        for t in targets:
            if m in results_for_ds[t]:
                a = results_for_ds[t][m]["auroc"]
                p = results_for_ds[t][m]["auprc"]
                row += f"  {a:>18.4f}  {p:>18.4f}"
            else:
                row += f"  {'—':>18}  {'—':>18}"
        print(row)


# ── Step 4 — Plots ────────────────────────────────────────────────────────────

def make_plots(scores_data: list[dict], eval_results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 10})

    _C = {
        "question_cycle": "#3498db",
        "bleu_dist":      "#2ecc71",
        "rouge1_dist":    "#27ae60",
        "rouge2_dist":    "#1abc9c",
        "rougeL_dist":    "#16a085",
        "chrf_dist":      "#f39c12",
        "lev_dist":       "#e67e22",
        "number_jaccard": "#e74c3c",
        "token_jaccard":  "#9b59b6",
        "answer_match":   "#2c3e50",
        "gpt_q_match":    "#c0392b",
    }

    METRIC_ORDER = [
        "question_cycle", "bleu_dist", "rouge1_dist", "rouge2_dist",
        "rougeL_dist", "chrf_dist", "lev_dist", "number_jaccard",
        "token_jaccard", "answer_match", "gpt_q_match",
    ]

    # ── 1. AUROC comparison bars (predicting `correct`) ──────────────────────
    for ds in ["all", "gsm8k", "math", "olympiadbench"]:
        if ds not in eval_results:
            continue
        target = "correct"
        if target not in eval_results[ds]:
            continue

        metrics = [m for m in METRIC_ORDER if m in eval_results[ds][target]]
        aurocs  = [eval_results[ds][target][m]["auroc"]  for m in metrics]
        auprcs  = [eval_results[ds][target][m]["auprc"]  for m in metrics]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        suf = f"[{ds}]" if ds != "all" else "[all datasets]"

        for ax, vals, metric_name in zip(axes, [aurocs, auprcs], ["AUROC", "AUPRC"]):
            colors = [_C.get(m, "#95a5a6") for m in metrics]
            bars = ax.bar(metrics, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=1)
            ax.axhline(0.5 if metric_name == "AUROC" else np.mean([d["correct"] for d in scores_data]),
                       color="grey", linestyle="--", lw=1.2, label="Random baseline")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_title(f"{metric_name} → {target} {suf}", fontsize=11)
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.legend(fontsize=8)

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"auroc_comparison_{ds}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")

    # ── 2. Distribution plots: correct vs incorrect for key metrics ──────────
    key_metrics = ["question_cycle", "number_jaccard", "bleu_dist",
                   "chrf_dist", "rougeL_dist", "lev_dist"]
    y = np.array([d["correct"] for d in scores_data])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, m in zip(axes, key_metrics):
        vals    = np.array([d[m] for d in scores_data])
        correct_v   = vals[y == 1]
        incorrect_v = vals[y == 0]
        bins = np.linspace(vals.min(), vals.max(), 50)
        ax.hist(correct_v,   bins=bins, alpha=0.6, color="#2ecc71", density=True,
                label=f"Correct (n={len(correct_v)})")
        ax.hist(incorrect_v, bins=bins, alpha=0.6, color="#e74c3c", density=True,
                label=f"Incorrect (n={len(incorrect_v)})")
        ax.axvline(correct_v.mean(),   color="#2ecc71", lw=1.8, linestyle="--")
        ax.axvline(incorrect_v.mean(), color="#e74c3c", lw=1.8, linestyle="--")
        gap = correct_v.mean() - incorrect_v.mean()
        ax.set_title(f"{m}  (Δ={gap:+.3f})", fontsize=9.5)
        ax.set_xlabel("Distance (lower = more similar)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7.5)

    fig.suptitle("Similarity Metric Distributions — Correct vs Incorrect (all datasets)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    # ── 3. Correlation heatmap ────────────────────────────────────────────────
    corr_keys = METRIC_ORDER + ["correct"]
    mat = np.array([[d.get(k, 0) for k in corr_keys] for d in scores_data], dtype=float)
    corr = np.corrcoef(mat.T)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_keys))); ax.set_xticklabels(corr_keys, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(corr_keys))); ax.set_yticklabels(corr_keys, fontsize=8.5)
    for i in range(len(corr_keys)):
        for j in range(len(corr_keys)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if abs(corr[i,j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("Pearson Correlation Matrix — All Metrics + Labels", fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    # ── 4. Per-dataset AUROC heatmap (metric × dataset, target = correct) ────
    datasets = ["all", "gsm8k", "math", "olympiadbench"]
    metrics_plot = [m for m in METRIC_ORDER if m != "gpt_q_match"]
    try:
        gpt_present = any("gpt_q_match" in eval_results[ds].get("correct", {}) for ds in datasets)
        if gpt_present:
            metrics_plot = METRIC_ORDER
    except Exception:
        pass

    heat = np.zeros((len(metrics_plot), len(datasets)))
    for j, ds in enumerate(datasets):
        if ds not in eval_results or "correct" not in eval_results[ds]:
            heat[:, j] = float("nan")
            continue
        for i, m in enumerate(metrics_plot):
            heat[i, j] = eval_results[ds]["correct"].get(m, {}).get("auroc", float("nan"))

    fig, ax = plt.subplots(figsize=(8, len(metrics_plot) * 0.55 + 1.5))
    im2 = ax.imshow(heat, cmap="RdYlGn", vmin=0.45, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, fontsize=10)
    ax.set_yticks(range(len(metrics_plot))); ax.set_yticklabels(metrics_plot, fontsize=9)
    for i in range(len(metrics_plot)):
        for j in range(len(datasets)):
            v = heat[i, j]
            ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "—",
                    ha="center", va="center", fontsize=8,
                    color="white" if v > 0.72 else "black")
    plt.colorbar(im2, ax=ax, fraction=0.04, pad=0.02, label="AUROC → correct")
    ax.set_title("AUROC → Solver Correctness  (metric × dataset)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "auroc_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gpt",  action="store_true",
                        help="Skip OpenAI calls; use cached GPT labels only.")
    parser.add_argument("--skip-metrics", action="store_true",
                        help="Skip recomputing metrics; load from cache.")
    args = parser.parse_args()

    print("\n" + "="*70)
    print(" Q–Q' Similarity Study")
    print("="*70)

    # Load data
    print(f"\n[1] Loading data from {DATA_PATH} …")
    data = load_jsonl(DATA_PATH)
    print(f"  {len(data)} samples  |  datasets: "
          + ", ".join(f"{k}={sum(1 for d in data if d['dataset']==k)}"
                      for k in sorted(set(d['dataset'] for d in data))))

    # GPT labels
    print(f"\n[2] GPT labels (Q ≡ Q'? YES/NO) …")
    gpt_labels = get_gpt_labels(data, skip=args.skip_gpt)
    n_yes = sum(gpt_labels.values())
    n_lab = len(gpt_labels)
    print(f"  Coverage: {n_lab}/{len(data)}  |  YES={n_yes} ({100*n_yes/max(n_lab,1):.1f}%)")

    # Similarity metrics
    print(f"\n[3] Computing similarity metrics …")
    if args.skip_metrics and os.path.exists(SCORES_OUT):
        print(f"  Loading cached scores from {SCORES_OUT}")
        scores_data = load_jsonl(SCORES_OUT)
        # Re-attach gpt labels
        gpt_map = {d["id"]: gpt_labels.get(d["id"], 0) for d in scores_data}
        for d in scores_data:
            d["gpt_q_match"] = gpt_map[d["id"]]
    else:
        scores_data = compute_all_metrics(data)
        for d in scores_data:
            d["gpt_q_match"] = gpt_labels.get(d["id"], 0)
        save_jsonl(scores_data, SCORES_OUT)
        print(f"  Scores saved to {SCORES_OUT}")

    # Evaluation
    print(f"\n[4] Evaluation — AUROC / AUPRC per metric per target …")
    eval_results = evaluate(scores_data, gpt_labels)

    # Plots
    print(f"\n[5] Generating plots → {PLOTS_DIR} …")
    make_plots(scores_data, eval_results)

    print(f"\n✓ Done.  All outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
