#!/usr/bin/env python3
"""
Experiment 2 — Majority Voting vs Cycle Consistency

Pipeline:
  1. Load exp1 v2 results (cycle consistency signals already computed)
  2. Recompute hybrid_cycle if missing (for updated CC signals)
  3. Generate K=16 stochastic solutions per question via vLLM
  4. Extract answers, cluster, compute voting signals for K=2,4,8,16
  5. Compute fusion signals (linear sweep, product, logistic)
  6. Evaluate all signals (AUROC, AUPRC, selective prediction)
  7. Generate comparison plots

Run:
    python run.py               # full pipeline from scratch
    python run.py --resume      # skip steps whose checkpoint files exist
"""

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Path setup (exp1 for shared utils, exp2 for local modules) ───────────────
# IMPORTANT: exp1 must be inserted first, then exp2 at position 0
# so that exp2 modules (config, visualize, etc.) shadow exp1's.
_exp1_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp1")
_exp2_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _exp1_dir)
sys.path.insert(0, _exp2_dir)   # exp2 at position 0 → checked first

# exp1 utilities (no name clash with exp2 modules)
from utils import (                          # noqa: E402
    extract_boxed_answer, extract_answer,
    answers_equivalent, number_jaccard,
    chrf_distance, compute_hybrid_cycle,
)

# exp2 modules
from config import (                         # noqa: E402
    SOLVER_MODEL, K_MAX, K_VALUES,
    TEMPERATURE, TOP_P, MAX_NEW_TOKENS,
    ALPHA_SWEEP, LOGISTIC_CV_FOLDS,
    EXP1_RESULTS, OUTPUT_DIR, PLOTS_DIR, CC_COST,
)
from voting import (                         # noqa: E402
    compute_voting_signals_multi_k,
    extract_final_answer,
)
from fusion import (                         # noqa: E402
    fused_linear, fused_product, sweep_alpha,
    fused_logistic_cv, compute_failure_overlap,
)
from visualize import (                      # noqa: E402
    plot_auroc_bars, plot_k_sensitivity,
    plot_cost_accuracy, plot_failure_overlap,
    plot_vote_distribution, plot_scatter_mv_cc,
    plot_roc_overlaid, plot_alpha_sweep,
    plot_summary_dashboard,
)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _cached(path: str, resume: bool) -> list[dict] | None:
    if resume and os.path.exists(path):
        print(f"  ↳ cache hit: {os.path.basename(path)}")
        return load_jsonl(path)
    return None


# ── Cohen's d ────────────────────────────────────────────────────────────────

def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


# ── Selective prediction ─────────────────────────────────────────────────────

def selective_accuracy(y_true: np.ndarray, scores: np.ndarray, coverage: float) -> float:
    """Accuracy when keeping only the top-`coverage` fraction of samples."""
    n_keep = max(1, int(len(y_true) * coverage))
    idx = np.argsort(scores)[::-1][:n_keep]  # highest scores first
    return float(y_true[idx].mean())


# ── vLLM stochastic solver ──────────────────────────────────────────────────

def generate_k_solutions(
    questions: list[str],
    model_name: str,
    K: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> list[list[str]]:
    """Generate K stochastic solutions per question using vLLM's n= parameter.

    Returns: list of lists, outer list is per-question, inner is K solutions.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    SOLVER_PROMPT = (
        "Solve the following math problem step by step "
        "and give the final answer clearly.\n\n"
        "Problem:\n{question}\n\nSolution:"
    )

    print(f"  [vLLM] Loading solver: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager=False,
    )

    # Build prompts
    prompts = []
    for q in questions:
        messages = [
            {"role": "user", "content": SOLVER_PROMPT.format(question=q)}
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    # Generate K completions per prompt using n=K
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=K,
    )
    print(f"  Generating {len(prompts)} × {K} = {len(prompts) * K} solutions …")
    outputs = llm.generate(prompts, params)

    # Collect results
    all_solutions = []
    for output in outputs:
        solutions = [o.text.strip() for o in output.outputs]
        all_solutions.append(solutions)

    # Cleanup
    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("  [vLLM] solver destroyed, GPU memory freed.")

    return all_solutions


# ── Evaluation helper ────────────────────────────────────────────────────────

def evaluate_signal(
    y_true: np.ndarray,
    scores: np.ndarray,
    name: str,
    direction: str = "higher",
) -> dict:
    """Evaluate a single signal: AUROC, AUPRC, selective accuracy, Cohen's d."""
    if direction == "lower":
        eval_scores = -scores
    else:
        eval_scores = scores

    y_c = scores[y_true == 1]
    y_i = scores[y_true == 0]

    result = {
        "name": name,
        "direction": direction,
        "mean_correct": float(y_c.mean()) if len(y_c) else 0.0,
        "mean_incorrect": float(y_i.mean()) if len(y_i) else 0.0,
        "cohen_d": float(_cohen_d(y_c, y_i)) if len(y_c) and len(y_i) else 0.0,
    }

    if len(set(y_true)) > 1:
        result["auroc"] = float(roc_auc_score(y_true, eval_scores))
        result["auprc"] = float(average_precision_score(y_true, eval_scores))
        for cov in [0.1, 0.2, 0.5]:
            result[f"sel_acc@{int(cov*100)}%"] = float(selective_accuracy(y_true, eval_scores, cov))
    else:
        result["auroc"] = 0.5
        result["auprc"] = float(y_true.mean())

    return result


# ── Main pipeline ────────────────────────────────────────────────────────────

def main(resume: bool = False) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    t0 = time.time()
    sep = "\n" + "=" * 70

    # ══════════════════════════════════════════════════════════════════════
    # Step 0: Load exp1 v2 data (cycle consistency signals)
    # ══════════════════════════════════════════════════════════════════════
    print("[Step 0] Loading exp1 v2 results …")
    if not os.path.exists(EXP1_RESULTS):
        raise FileNotFoundError(
            f"exp1 results not found at {EXP1_RESULTS}.\n"
            "Run exp1 first: cd ../exp1 && python run.py"
        )
    data = load_jsonl(EXP1_RESULTS)
    print(f"  Loaded {len(data)} samples from exp1")

    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Recompute hybrid_cycle if not present
    # ══════════════════════════════════════════════════════════════════════
    step1_path = os.path.join(OUTPUT_DIR, "step1_cc_signals.jsonl")
    data_cc = _cached(step1_path, resume)
    if data_cc is not None:
        data = data_cc
    else:
        if "hybrid_cycle" not in data[0]:
            print("[Step 1] Computing hybrid_cycle (number_jaccard + chrf_dist) …")
            for d in data:
                q = d["question"]
                qp = d["question_Q_prime"]
                qc = d["question_cycle"]
                nj = number_jaccard(q, qp)
                cd = chrf_distance(q, qp)
                d["number_jaccard"] = nj
                d["chrf_dist"] = cd
                d["hybrid_cycle"] = compute_hybrid_cycle(qc, nj, cd)
                d["combined_reward"] = d["answer_match"] - d["hybrid_cycle"]
        else:
            print("[Step 1] hybrid_cycle already present, skipping.")
        save_jsonl(data, step1_path)

    # ══════════════════════════════════════════════════════════════════════
    # Step 2: Generate K=16 stochastic solutions per question
    # ══════════════════════════════════════════════════════════════════════
    step2_path = os.path.join(OUTPUT_DIR, "step2_k_solutions.jsonl")
    data = _cached(step2_path, resume)
    if data is None:
        data = load_jsonl(step1_path)
        print(f"[Step 2] Generating K={K_MAX} stochastic solutions per question …")
        questions = [d["question"] for d in data]

        all_solutions = generate_k_solutions(
            questions=questions,
            model_name=SOLVER_MODEL,
            K=K_MAX,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_NEW_TOKENS,
        )

        for d, sols in zip(data, all_solutions):
            d["k_solutions"] = sols

        save_jsonl(data, step2_path)

    # ══════════════════════════════════════════════════════════════════════
    # Step 3: Compute voting signals for each K value
    # ══════════════════════════════════════════════════════════════════════
    step3_path = os.path.join(OUTPUT_DIR, "step3_voting_signals.jsonl")
    data = _cached(step3_path, resume)
    if data is None:
        data = load_jsonl(step2_path)
        print(f"[Step 3] Computing voting signals for K ∈ {K_VALUES} …")

        for idx, d in enumerate(data):
            solutions = d["k_solutions"]
            multi_k = compute_voting_signals_multi_k(solutions, K_VALUES)

            for k, signals in multi_k.items():
                d[f"voting_confidence_K{k}"] = signals["voting_confidence"]
                d[f"entropy_K{k}"] = signals["entropy"]
                d[f"unique_ratio_K{k}"] = signals["unique_ratio"]
                d[f"majority_answer_K{k}"] = signals["majority_answer"]
                d[f"n_clusters_K{k}"] = signals["n_clusters"]

            # Also check if majority answer is correct (for each K)
            for k in K_VALUES:
                if k > len(solutions):
                    continue
                maj_key = f"majority_answer_K{k}"
                d[f"majority_correct_K{k}"] = int(
                    answers_equivalent(d[maj_key], d["ground_truth"])
                )

            if (idx + 1) % 200 == 0:
                print(f"    {idx + 1}/{len(data)} processed")

        # Drop k_solutions to save space in checkpoint (they're in step2)
        for d in data:
            d.pop("k_solutions", None)

        save_jsonl(data, step3_path)

    # ══════════════════════════════════════════════════════════════════════
    # Step 4: Compute fusion signals
    # ══════════════════════════════════════════════════════════════════════
    step4_path = os.path.join(OUTPUT_DIR, "step4_fusion.jsonl")
    data = _cached(step4_path, resume)
    if data is None:
        data = load_jsonl(step3_path)
        print("[Step 4] Computing fusion signals …")

        y_true = np.array([d["correct"] for d in data])
        cc_scores = np.array([d["combined_reward"] for d in data])

        for k in K_VALUES:
            vc_key = f"voting_confidence_K{k}"
            mv_scores = np.array([d[vc_key] for d in data])

            # Alpha sweep
            sweep = sweep_alpha(mv_scores, cc_scores, y_true, ALPHA_SWEEP)
            best_alpha = sweep["best_alpha"]

            # Best linear fusion
            fused_lin = fused_linear(mv_scores, cc_scores, best_alpha)
            # Product fusion
            fused_prod = fused_product(mv_scores, cc_scores)

            for i, d in enumerate(data):
                d[f"fused_linear_K{k}"] = float(fused_lin[i])
                d[f"fused_product_K{k}"] = float(fused_prod[i])
                d[f"best_alpha_K{k}"] = best_alpha

            # Logistic fusion
            ent_key = f"entropy_K{k}"
            features = np.column_stack([
                mv_scores,
                cc_scores,
                np.array([d.get("hybrid_cycle", d.get("question_cycle", 0.5)) for d in data]),
                np.array([d[ent_key] for d in data]),
            ])
            logistic_result = fused_logistic_cv(features, y_true, LOGISTIC_CV_FOLDS)

            for i, d in enumerate(data):
                d[f"fused_logistic_K{k}"] = float(logistic_result["oof_scores"][i])
                d[f"logistic_auroc_K{k}"] = logistic_result["overall_auroc"]

            print(f"  K={k}: best_α={best_alpha:.2f}  "
                  f"linear_AUROC={sweep['best_auroc']:.3f}  "
                  f"logistic_AUROC={logistic_result['overall_auroc']:.3f}  "
                  f"(±{logistic_result['std_auroc']:.3f})")

        save_jsonl(data, step4_path)

    # ══════════════════════════════════════════════════════════════════════
    # Step 5: Full evaluation
    # ══════════════════════════════════════════════════════════════════════
    print(sep)
    print(" Step 5 — Evaluation")
    print("=" * 70)

    y_true = np.array([d["correct"] for d in data])
    datasets = sorted(set(d["dataset"] for d in data))

    # Storage for plots
    results_by_dataset: dict[str, dict] = {}
    k_aurocs_all: dict[int, float] = {}
    fusion_aurocs_all: dict[int, float] = {}

    for ds_label in ["all"] + datasets:
        if ds_label == "all":
            subset = data
        else:
            subset = [d for d in data if d["dataset"] == ds_label]

        y_sub = np.array([d["correct"] for d in subset])
        n_c = int(y_sub.sum())
        n_i = len(y_sub) - n_c

        print(f"\n{'─'*70}")
        print(f"  Dataset: {ds_label.upper()}  |  N={len(subset)}  correct={n_c}  incorrect={n_i}  acc={y_sub.mean():.1%}")
        print(f"{'─'*70}")

        ds_results = {}

        # ── CC signals ──────────────────────────────────────────────────
        cc_cr = np.array([d["combined_reward"] for d in subset])
        cc_eval = evaluate_signal(y_sub, cc_cr, "combined_reward", "higher")
        print(f"  {'combined_reward (CC)':<35s}  AUROC={cc_eval['auroc']:.3f}  "
              f"AUPRC={cc_eval['auprc']:.3f}  d={cc_eval['cohen_d']:+.2f}  "
              f"sel@10%={cc_eval.get('sel_acc@10%', 'n/a')}")
        ds_results["combined_reward"] = cc_eval["auroc"]

        hc_key = "hybrid_cycle" if "hybrid_cycle" in subset[0] else "question_cycle"
        hc_scores = np.array([d[hc_key] for d in subset])
        hc_eval = evaluate_signal(y_sub, hc_scores, hc_key, "lower")
        print(f"  {hc_key + ' (CC)':<35s}  AUROC={hc_eval['auroc']:.3f}  "
              f"AUPRC={hc_eval['auprc']:.3f}  d={hc_eval['cohen_d']:+.2f}")

        am_scores = np.array([d["answer_match"] for d in subset], dtype=float)
        am_eval = evaluate_signal(y_sub, am_scores, "answer_match", "higher")
        print(f"  {'answer_match (CC)':<35s}  AUROC={am_eval['auroc']:.3f}")

        # ── MV signals per K ────────────────────────────────────────────
        for k in K_VALUES:
            vc_key = f"voting_confidence_K{k}"
            if vc_key not in subset[0]:
                continue
            mv_scores = np.array([d[vc_key] for d in subset])
            mv_eval = evaluate_signal(y_sub, mv_scores, vc_key, "higher")
            print(f"  {f'voting_confidence K={k}':<35s}  AUROC={mv_eval['auroc']:.3f}  "
                  f"AUPRC={mv_eval['auprc']:.3f}  d={mv_eval['cohen_d']:+.2f}  "
                  f"sel@10%={mv_eval.get('sel_acc@10%', 'n/a')}")
            ds_results[vc_key] = mv_eval["auroc"]

            # Majority vote accuracy
            mc_key = f"majority_correct_K{k}"
            if mc_key in subset[0]:
                maj_acc = np.mean([d[mc_key] for d in subset])
                greedy_acc = y_sub.mean()
                print(f"    majority_correct K={k}: {maj_acc:.3f}  (greedy: {greedy_acc:.3f}  Δ={maj_acc-greedy_acc:+.3f})")

            # Entropy
            ent_key = f"entropy_K{k}"
            if ent_key in subset[0]:
                ent_scores = np.array([d[ent_key] for d in subset])
                ent_eval = evaluate_signal(y_sub, ent_scores, ent_key, "lower")
                print(f"  {f'entropy K={k}':<35s}  AUROC={ent_eval['auroc']:.3f}")

            # Track for K-sensitivity
            if ds_label == "all":
                k_aurocs_all[k] = mv_eval["auroc"]

        # ── Fusion signals per K ────────────────────────────────────────
        for k in K_VALUES:
            fl_key = f"fused_linear_K{k}"
            if fl_key not in subset[0]:
                continue
            fl_scores = np.array([d[fl_key] for d in subset])
            fl_eval = evaluate_signal(y_sub, fl_scores, fl_key, "higher")
            alpha = subset[0].get(f"best_alpha_K{k}", "?")
            print(f"  {f'fused_linear K={k} (α={alpha})':<35s}  AUROC={fl_eval['auroc']:.3f}  "
                  f"AUPRC={fl_eval['auprc']:.3f}  "
                  f"sel@10%={fl_eval.get('sel_acc@10%', 'n/a')}")
            ds_results[f"fused_best_K{k}"] = fl_eval["auroc"]

            fp_key = f"fused_product_K{k}"
            fp_scores = np.array([d[fp_key] for d in subset])
            fp_eval = evaluate_signal(y_sub, fp_scores, fp_key, "higher")
            print(f"  {f'fused_product K={k}':<35s}  AUROC={fp_eval['auroc']:.3f}")

            flog_key = f"fused_logistic_K{k}"
            if flog_key in subset[0]:
                flog_scores = np.array([d[flog_key] for d in subset])
                flog_eval = evaluate_signal(y_sub, flog_scores, flog_key, "higher")
                print(f"  {f'fused_logistic K={k}':<35s}  AUROC={flog_eval['auroc']:.3f}  "
                      f"(CV: {subset[0].get(f'logistic_auroc_K{k}', '?')})")

            if ds_label == "all":
                fusion_aurocs_all[k] = fl_eval["auroc"]

        results_by_dataset[ds_label] = ds_results

    # ══════════════════════════════════════════════════════════════════════
    # Step 6: Failure overlap analysis
    # ══════════════════════════════════════════════════════════════════════
    print(sep)
    print(" Step 6 — Failure Overlap Analysis")
    print("=" * 70)

    for ds_label in ["all"] + datasets:
        subset = data if ds_label == "all" else [d for d in data if d["dataset"] == ds_label]
        y_sub = np.array([d["correct"] for d in subset])

        if len(set(y_sub)) < 2:
            continue

        for k in [8, 16]:
            vc_key = f"voting_confidence_K{k}"
            if vc_key not in subset[0]:
                continue
            mv_scores = np.array([d[vc_key] for d in subset])
            cc_scores = np.array([d["combined_reward"] for d in subset])

            overlap = compute_failure_overlap(y_sub, mv_scores, cc_scores)
            print(f"\n  [{ds_label.upper()}, K={k}]")
            print(f"    Incorrect: {overlap['n_incorrect']}  |  "
                  f"FP(MV): {overlap['n_fp_mv']}  FP(CC): {overlap['n_fp_cc']}  "
                  f"Overlap: {overlap['n_overlap']}")
            print(f"    MV-only: {overlap['fp_mv_only']}  CC-only: {overlap['fp_cc_only']}")
            print(f"    Jaccard: {overlap['jaccard']:.3f}  "
                  f"Overlap coeff: {overlap['overlap_coefficient']:.3f}")

            # Save overlap plot
            ds_plot_dir = os.path.join(PLOTS_DIR, ds_label)
            os.makedirs(ds_plot_dir, exist_ok=True)
            plot_failure_overlap(overlap, ds_plot_dir, f"{ds_label}_K{k}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 7: Generate all plots
    # ══════════════════════════════════════════════════════════════════════
    print(sep)
    print(" Step 7 — Visualizations")
    print("=" * 70)

    for ds_label in ["all"] + datasets:
        subset = data if ds_label == "all" else [d for d in data if d["dataset"] == ds_label]
        y_sub = np.array([d["correct"] for d in subset])
        ds_plot_dir = os.path.join(PLOTS_DIR, ds_label)
        os.makedirs(ds_plot_dir, exist_ok=True)

        print(f"\n  ── Plots: {ds_label} (N={len(subset)}) ──")

        for k in K_VALUES:
            vc_key = f"voting_confidence_K{k}"
            if vc_key not in subset[0]:
                continue

            # Vote distribution
            plot_vote_distribution(subset, k, ds_plot_dir, ds_label)

            # MV vs CC scatter
            plot_scatter_mv_cc(subset, k, ds_plot_dir, ds_label)

            # Overlaid ROC
            fl_key = f"fused_linear_K{k}"
            fused_scores = np.array([d.get(fl_key, 0.5) for d in subset]) if fl_key in subset[0] else None
            plot_roc_overlaid(subset, k, ds_plot_dir, ds_label, fused_scores=fused_scores)

            # Alpha sweep
            if len(set(y_sub)) > 1:
                mv_scores = np.array([d[vc_key] for d in subset])
                cc_cr = np.array([d["combined_reward"] for d in subset])
                sweep = sweep_alpha(mv_scores, cc_cr, y_sub, ALPHA_SWEEP)
                plot_alpha_sweep(sweep["alpha_results"], ds_plot_dir, k, ds_label)

        # K-sensitivity (per dataset)
        if len(set(y_sub)) > 1:
            ds_k_aurocs = {}
            ds_fusion_aurocs = {}
            cc_auroc_ds = evaluate_signal(
                y_sub,
                np.array([d["combined_reward"] for d in subset]),
                "cc", "higher"
            )["auroc"]

            for k in K_VALUES:
                vc_key = f"voting_confidence_K{k}"
                if vc_key not in subset[0]:
                    continue
                mv_s = np.array([d[vc_key] for d in subset])
                ds_k_aurocs[k] = evaluate_signal(y_sub, mv_s, vc_key, "higher")["auroc"]

                fl_key = f"fused_linear_K{k}"
                if fl_key in subset[0]:
                    fl_s = np.array([d[fl_key] for d in subset])
                    ds_fusion_aurocs[k] = evaluate_signal(y_sub, fl_s, fl_key, "higher")["auroc"]

            if ds_k_aurocs:
                plot_k_sensitivity(ds_k_aurocs, cc_auroc_ds, ds_fusion_aurocs, ds_plot_dir, ds_label)

    # ── Cost-accuracy Pareto (all datasets) ──────────────────────────────
    if k_aurocs_all:
        cc_auroc_overall = evaluate_signal(
            np.array([d["correct"] for d in data]),
            np.array([d["combined_reward"] for d in data]),
            "cc", "higher"
        )["auroc"]

        pareto_points = []
        for k in sorted(k_aurocs_all.keys()):
            pareto_points.append({
                "name": f"MV K={k}", "cost": k, "auroc": k_aurocs_all[k],
                "color": "#3498db", "marker": "o", "group": "mv",
            })
        pareto_points.append({
            "name": "CC", "cost": CC_COST, "auroc": cc_auroc_overall,
            "color": "#e67e22", "marker": "^", "group": "cc",
        })
        for k in sorted(fusion_aurocs_all.keys()):
            pareto_points.append({
                "name": f"MV{k}+CC", "cost": k + CC_COST, "auroc": fusion_aurocs_all[k],
                "color": "#9b59b6", "marker": "s", "group": "fusion",
            })

        all_plot_dir = os.path.join(PLOTS_DIR, "all")
        os.makedirs(all_plot_dir, exist_ok=True)
        plot_cost_accuracy(pareto_points, all_plot_dir, "all")

    # ── Summary dashboard ────────────────────────────────────────────────
    if k_aurocs_all and fusion_aurocs_all:
        best_fusion = max(fusion_aurocs_all.values()) if fusion_aurocs_all else 0.5
        plot_summary_dashboard(
            results_by_dataset, k_aurocs_all, cc_auroc_overall,
            best_fusion, PLOTS_DIR,
        )

    # ── Save final results ───────────────────────────────────────────────
    results_path = os.path.join(OUTPUT_DIR, "results.jsonl")
    save_jsonl(data, results_path)
    print(f"\n  Final results saved to {results_path}")

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed / 60:.1f} min.  Outputs → {OUTPUT_DIR}")
    print(f"  Plots → {PLOTS_DIR}/")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 2: MV vs CC comparison")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip steps whose checkpoint files already exist.",
    )
    args = parser.parse_args()
    main(resume=args.resume)
