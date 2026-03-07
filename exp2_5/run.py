#!/usr/bin/env python3
"""
Experiment 2.5 — Per-Trajectory Composite Reward Analysis

Pipeline:
  Step 0: Load Exp2 step2_k_solutions.jsonl (1800 × 16 solutions)
  Step 1: Flatten → 28,800 trajectories; extract answers; compute correctness
  Step 2: Reconstruct Q'_i for each trajectory  [GPU: Reconstructor]
  Step 3: Re-solve Q'_i → S'_i                  [GPU: Solver]
  Step 4: Embed Q & Q'_i + text CC → hybrid_cycle + combined_reward  [GPU: Embedder]
  Step 5: Compute per-trajectory MV agreement
  Step 6: Compute composite rewards
  Step 7: Run all 7 analyses
  Step 8: Generate all plots

Run:
    python run.py               # full pipeline from scratch
    python run.py --resume      # skip steps whose checkpoint files exist
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
# exp2_5 at position 0, then exp1, then exp2 (for voting.py)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_exp1_dir = os.path.join(_this_dir, "..", "exp1")
_exp2_dir = os.path.join(_this_dir, "..", "exp2")
sys.path.insert(0, _exp1_dir)
sys.path.insert(0, _exp2_dir)
sys.path.insert(0, _this_dir)

# Local config (shadows exp1/exp2 config)
from config import (
    SOLVER_MODEL, RECONSTRUCTOR_MODEL, EMBEDDING_MODEL,
    MAX_NEW_TOKENS, EMBEDDING_BATCH_SIZE,
    K_MAX, K_VALUES, ALPHA_SWEEP, FILTER_THRESHOLDS,
    EXP2_STEP2, EXP2_STEP3, OUTPUT_DIR, PLOTS_DIR,
)

# exp1 utilities
from utils import (
    extract_boxed_answer, extract_answer, answers_equivalent,
    number_jaccard, chrf_distance, compute_hybrid_cycle,
)

# exp2 voting utilities
from voting import extract_final_answer, cluster_answers

# Local modules
from signals import compute_mv_agreement, compute_composite_rewards
from analysis import (
    analysis_1_trajectory_auroc, analysis_1_alpha_sweep,
    analysis_2_weighted_voting, analysis_3_best_of_k,
    analysis_4_filtered_voting, analysis_5_wrong_majority_rescue,
    analysis_6_cc_distribution, analysis_7_grpo_advantage,
)
from visualize import (
    plot_trajectory_auroc_bars, plot_alpha_sweep_trajectory,
    plot_cc_distribution, plot_voting_accuracy_vs_k,
    plot_best_of_k, plot_filtered_voting,
    plot_wrong_majority_rescue, plot_grpo_advantage,
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


# ── vLLM cleanup helper ─────────────────────────────────────────────────────

def _destroy_vllm(llm) -> None:
    """Shut down a vLLM LLM instance and free GPU memory."""
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
    print("  [vLLM] model destroyed, GPU memory freed.")


# ═════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def main(resume: bool = False) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    t0 = time.time()
    sep = "\n" + "=" * 70

    # ══════════════════════════════════════════════════════════════════════
    # Step 0: Load Exp2 data
    # ══════════════════════════════════════════════════════════════════════
    print(sep)
    print(" Step 0 — Load Exp2 data")
    print("=" * 70)

    if not os.path.exists(EXP2_STEP2):
        raise FileNotFoundError(
            f"Exp2 step2 data not found at {EXP2_STEP2}.\n"
            "Run exp2 first: cd ../exp2 && python run.py"
        )

    raw_data = load_jsonl(EXP2_STEP2)
    print(f"  Loaded {len(raw_data)} questions from Exp2 step2")
    print(f"  Each question has {len(raw_data[0].get('k_solutions', []))} solutions")

    # Also load voting signals for majority answers
    mv_data = None
    if os.path.exists(EXP2_STEP3):
        mv_data = load_jsonl(EXP2_STEP3)
        print(f"  Loaded {len(mv_data)} questions with MV signals from Exp2 step3")

    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Flatten → 28,800 trajectory rows
    # ══════════════════════════════════════════════════════════════════════
    step1_path = os.path.join(OUTPUT_DIR, "step1_trajectories.jsonl")
    traj_data = _cached(step1_path, resume)
    if traj_data is None:
        print(sep)
        print(" Step 1 — Flatten trajectories & extract answers")
        print("=" * 70)

        traj_data = []
        for q_idx, d in enumerate(raw_data):
            question = d["question"]
            ground_truth = d["ground_truth"]
            dataset = d["dataset"]
            solutions = d.get("k_solutions", [])

            for s_idx, sol in enumerate(solutions[:K_MAX]):
                answer = extract_final_answer(sol)
                is_correct = int(answers_equivalent(answer, ground_truth))

                traj_data.append({
                    "question_idx": q_idx,
                    "traj_idx": s_idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "dataset": dataset,
                    "solution": sol,
                    "answer": answer,
                    "trajectory_correct": is_correct,
                })

            if (q_idx + 1) % 500 == 0:
                print(f"  {q_idx + 1}/{len(raw_data)} questions processed")

        n_correct = sum(t["trajectory_correct"] for t in traj_data)
        print(f"  Flattened to {len(traj_data)} trajectories")
        print(f"  Per-trajectory accuracy: {n_correct}/{len(traj_data)} = {n_correct/len(traj_data):.1%}")

        save_jsonl(traj_data, step1_path)

    # Free raw_data — we no longer need the full k_solutions
    del raw_data
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    # Step 2: Reconstruct Q'_i for each trajectory
    # ══════════════════════════════════════════════════════════════════════
    step2_path = os.path.join(OUTPUT_DIR, "step2_reconstructed.jsonl")
    traj_data = _cached(step2_path, resume)
    if traj_data is None:
        traj_data = load_jsonl(step1_path)
        print(sep)
        print(f" Step 2 — Reconstruct Q'_i ({len(traj_data)} calls)")
        print("=" * 70)

        from models import Reconstructor
        reconstructor = Reconstructor(RECONSTRUCTOR_MODEL)

        solutions = [t["solution"] for t in traj_data]
        reconstructed = reconstructor.reconstruct_batch(solutions)

        for t, qp in zip(traj_data, reconstructed):
            t["question_prime"] = qp

        reconstructor.destroy()
        del reconstructor
        gc.collect()
        torch.cuda.empty_cache()

        # Drop solution text to save checkpoint space
        for t in traj_data:
            t.pop("solution", None)

        save_jsonl(traj_data, step2_path)
        print(f"  Saved {len(traj_data)} reconstructed trajectories")

    # ══════════════════════════════════════════════════════════════════════
    # Step 3: Re-solve Q'_i → S'_i → answer_match
    # ══════════════════════════════════════════════════════════════════════
    step3_path = os.path.join(OUTPUT_DIR, "step3_resolved.jsonl")
    traj_data = _cached(step3_path, resume)
    if traj_data is None:
        traj_data = load_jsonl(step2_path)
        print(sep)
        print(f" Step 3 — Re-solve Q'_i → S'_i ({len(traj_data)} calls)")
        print("=" * 70)

        from models import Solver
        solver = Solver(SOLVER_MODEL)

        q_primes = [t["question_prime"] for t in traj_data]
        solutions_prime = solver.solve_batch(q_primes, MAX_NEW_TOKENS)

        # We also need the original solution answers — reload from step1
        step1_data = load_jsonl(step1_path)
        original_answers = {
            (d["question_idx"], d["traj_idx"]): d["answer"]
            for d in step1_data
        }
        del step1_data

        for t, sp in zip(traj_data, solutions_prime):
            # Extract answer from S'_i
            boxed_ap = extract_boxed_answer(sp)
            answer_ap = boxed_ap if boxed_ap else extract_answer(sp)
            t["answer_A_prime"] = answer_ap

            # answer_match: does S_i give the same answer as S'_i?
            orig_answer = original_answers.get(
                (t["question_idx"], t["traj_idx"]), t.get("answer", "")
            )
            t["answer_match"] = int(answers_equivalent(orig_answer, answer_ap))

        solver.destroy()
        del solver
        gc.collect()
        torch.cuda.empty_cache()

        save_jsonl(traj_data, step3_path)
        n_match = sum(t["answer_match"] for t in traj_data)
        print(f"  answer_match rate: {n_match}/{len(traj_data)} = {n_match/len(traj_data):.1%}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 4: Embed Q & Q'_i + text CC → hybrid_cycle + combined_reward
    # ══════════════════════════════════════════════════════════════════════
    step4_path = os.path.join(OUTPUT_DIR, "step4_cc_signals.jsonl")
    traj_data = _cached(step4_path, resume)
    if traj_data is None:
        traj_data = load_jsonl(step3_path)
        print(sep)
        print(f" Step 4 — Embeddings + CC signals ({len(traj_data)} trajectories)")
        print("=" * 70)

        # ── 4a: Compute embeddings ───────────────────────────────────────
        from models import Embedder
        embedder = Embedder(EMBEDDING_MODEL)

        # Unique questions (1800)
        q_idx_to_q = {}
        for t in traj_data:
            q_idx_to_q[t["question_idx"]] = t["question"]
        unique_q_idxs = sorted(q_idx_to_q.keys())
        unique_questions = [q_idx_to_q[i] for i in unique_q_idxs]

        print(f"  Embedding {len(unique_questions)} unique questions …")
        emb_q_all = embedder.embed(unique_questions, EMBEDDING_BATCH_SIZE)
        q_idx_to_emb_row = {qi: row for row, qi in enumerate(unique_q_idxs)}

        # Q'_i embeddings (28,800)
        q_primes = [t["question_prime"] for t in traj_data]
        print(f"  Embedding {len(q_primes)} reconstructed questions …")
        emb_qp_all = embedder.embed(q_primes, EMBEDDING_BATCH_SIZE)

        del embedder
        gc.collect()
        torch.cuda.empty_cache()

        # ── 4b: Compute question_cycle (cosine distance) ────────────────
        print("  Computing question_cycle (cosine distance) …")
        for i, t in enumerate(traj_data):
            q_row = q_idx_to_emb_row[t["question_idx"]]
            emb_q = emb_q_all[q_row]
            emb_qp = emb_qp_all[i]
            cos_sim = float(np.dot(emb_q, emb_qp))
            t["question_cycle"] = 1.0 - cos_sim

        del emb_q_all, emb_qp_all
        gc.collect()

        # ── 4c: Text-based CC metrics + hybrid_cycle + combined_reward ──
        print("  Computing text CC metrics + hybrid_cycle + combined_reward …")
        for idx, t in enumerate(traj_data):
            q = t["question"]
            qp = t["question_prime"]

            nj = number_jaccard(q, qp)
            cd = chrf_distance(q, qp)
            hc = compute_hybrid_cycle(t["question_cycle"], nj, cd)

            t["number_jaccard"] = nj
            t["chrf_dist"] = cd
            t["hybrid_cycle"] = hc
            t["combined_reward"] = t["answer_match"] - hc

            if (idx + 1) % 5000 == 0:
                print(f"    {idx + 1}/{len(traj_data)} done")

        save_jsonl(traj_data, step4_path)

        hc_vals = [t["hybrid_cycle"] for t in traj_data]
        cr_vals = [t["combined_reward"] for t in traj_data]
        print(f"  hybrid_cycle: mean={np.mean(hc_vals):.4f}  std={np.std(hc_vals):.4f}")
        print(f"  combined_reward: mean={np.mean(cr_vals):.4f}  std={np.std(cr_vals):.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 5: Per-trajectory MV agreement
    # ══════════════════════════════════════════════════════════════════════
    step5_path = os.path.join(OUTPUT_DIR, "step5_mv_signals.jsonl")
    traj_data = _cached(step5_path, resume)
    if traj_data is None:
        traj_data = load_jsonl(step4_path)
        print(sep)
        print(" Step 5 — Per-trajectory MV agreement signals")
        print("=" * 70)

        # We need original answers from step1 (since step4 may not have them)
        step1_data = load_jsonl(step1_path)
        answer_lookup = {}
        for d in step1_data:
            answer_lookup[(d["question_idx"], d["traj_idx"])] = d["answer"]
        del step1_data

        # Ensure each trajectory has its answer
        for t in traj_data:
            if "answer" not in t:
                t["answer"] = answer_lookup.get(
                    (t["question_idx"], t["traj_idx"]), ""
                )

        # Group by question
        from collections import defaultdict
        groups = defaultdict(list)
        for t in traj_data:
            groups[t["question_idx"]].append(t)

        for q_idx, trajs in groups.items():
            trajs.sort(key=lambda t: t["traj_idx"])
            answers = [t["answer"] for t in trajs]

            for K in K_VALUES:
                mv_result = compute_mv_agreement(answers, K=K)
                for i, t in enumerate(trajs[:K]):
                    t[f"mv_agreement_K{K}"] = mv_result["mv_agreement"][i]
                    t[f"mv_agreement_soft_K{K}"] = mv_result["mv_agreement_soft"][i]
                    t[f"majority_answer_K{K}"] = mv_result["majority_answer"]

                # For trajectories beyond K, set to None/0
                for t in trajs[K:]:
                    t[f"mv_agreement_K{K}"] = 0
                    t[f"mv_agreement_soft_K{K}"] = 0.0
                    t[f"majority_answer_K{K}"] = mv_result["majority_answer"]

                # Check if majority is correct
                maj_correct = int(answers_equivalent(
                    mv_result["majority_answer"],
                    trajs[0]["ground_truth"]
                ))
                for t in trajs:
                    t[f"majority_correct_K{K}"] = maj_correct

        if (q_idx + 1) % 500 == 0:
            print(f"    {q_idx + 1} questions processed")

        save_jsonl(traj_data, step5_path)
        print(f"  MV agreement signals computed for K ∈ {K_VALUES}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 6: Composite rewards
    # ══════════════════════════════════════════════════════════════════════
    step6_path = os.path.join(OUTPUT_DIR, "step6_composite.jsonl")
    traj_data = _cached(step6_path, resume)
    if traj_data is None:
        traj_data = load_jsonl(step5_path)
        print(sep)
        print(" Step 6 — Composite rewards")
        print("=" * 70)

        for t in traj_data:
            mv_soft = t.get("mv_agreement_soft_K16", 0.0)
            cr = t.get("combined_reward", 0.0)

            # Soft composite: MV + CC
            t["reward_soft"] = mv_soft + cr

            # Binary composite
            mv_bin = t.get("mv_agreement_K16", 0)
            t["reward_binary"] = mv_bin + int(cr > 0)

        save_jsonl(traj_data, step6_path)
        print("  Composite rewards computed")

    # ══════════════════════════════════════════════════════════════════════
    # Step 7: Run all 7 analyses
    # ══════════════════════════════════════════════════════════════════════
    print(sep)
    print(" Step 7 — Analyses")
    print("=" * 70)

    datasets = sorted(set(t["dataset"] for t in traj_data))
    all_analysis_results = {}

    for ds_label in ["all"] + datasets:
        if ds_label == "all":
            subset = traj_data
        else:
            subset = [t for t in traj_data if t["dataset"] == ds_label]

        n = len(subset)
        n_c = sum(t["trajectory_correct"] for t in subset)
        print(f"\n{'─'*70}")
        print(f"  Dataset: {ds_label.upper()}  |  N={n}  correct={n_c}  incorrect={n-n_c}  acc={n_c/n:.1%}")
        print(f"{'─'*70}")

        ds_results = {}

        # ── Analysis 1: Per-Trajectory AUROC ──────────────────────────
        print("\n  [A1] Per-Trajectory Correctness Prediction (AUROC)")
        a1 = analysis_1_trajectory_auroc(subset)
        for name, vals in a1["signals"].items():
            print(f"    {name:<30s}  AUROC={vals['auroc']:.3f}  "
                  f"AUPRC={vals['auprc']:.3f}  d={vals['cohen_d']:+.3f}")
        ds_results["a1"] = a1

        # Alpha sweep
        print("\n  [A1b] Alpha sweep for composite reward")
        sweep = analysis_1_alpha_sweep(subset, ALPHA_SWEEP)
        best = max(sweep, key=lambda r: r["auroc"])
        print(f"    Best α={best['alpha']:.2f}  AUROC={best['auroc']:.3f}")
        ds_results["a1_sweep"] = sweep

        # ── Analysis 2: Reward-Weighted Voting ───────────────────────
        print("\n  [A2] Reward-Weighted Voting")
        a2 = analysis_2_weighted_voting(subset, K_VALUES)
        for K, vals in sorted(a2.items()):
            print(f"    K={K:>2d}: Standard={vals['standard_mv_acc']:.3f}  "
                  f"CC-weighted={vals['cc_weighted_acc']:.3f}  "
                  f"Composite={vals['composite_weighted_acc']:.3f}  "
                  f"(Δcc={vals['cc_weighted_acc']-vals['standard_mv_acc']:+.3f}  "
                  f"Δcomp={vals['composite_weighted_acc']-vals['standard_mv_acc']:+.3f})")
        ds_results["a2"] = a2

        # ── Analysis 3: Best-of-K Selection ──────────────────────────
        print("\n  [A3] Best-of-K Selection")
        a3 = analysis_3_best_of_k(subset, K_VALUES)
        for K, vals in sorted(a3.items()):
            print(f"    K={K:>2d}: Random={vals['random']:.3f}  "
                  f"MV-aligned={vals['mv_aligned']:.3f}  "
                  f"Best-CC={vals['best_cc']:.3f}  "
                  f"Best-composite={vals['best_composite']:.3f}  "
                  f"Oracle={vals['oracle']:.3f}")
        ds_results["a3"] = a3

        # ── Analysis 4: Filtered Voting ──────────────────────────────
        print("\n  [A4] Filtered Voting")
        a4 = analysis_4_filtered_voting(subset, K_VALUES, FILTER_THRESHOLDS)
        for K in sorted(a4.keys()):
            baseline = a4[K]["baseline_acc"]
            best_t = max(a4[K]["thresholds"], key=lambda r: r["accuracy"])
            print(f"    K={K:>2d}: Baseline={baseline:.3f}  "
                  f"Best filtered={best_t['accuracy']:.3f} (θ={best_t['threshold']:.1f})  "
                  f"Δ={best_t['accuracy']-baseline:+.3f}")
        ds_results["a4"] = a4

        # ── Analysis 5: Wrong-Majority Rescue ────────────────────────
        print("\n  [A5] Wrong-Majority Rescue")
        a5 = analysis_5_wrong_majority_rescue(subset, K_VALUES)
        for K, vals in sorted(a5.items()):
            print(f"    K={K:>2d}: Wrong-majority={vals['n_wrong_majority']}  "
                  f"Has-correct={vals['n_has_correct_minority']} ({vals['pct_has_correct_minority']:.1%})  "
                  f"Rescue(CC)={vals['rescue_rate_cc']:.1%}  "
                  f"Rescue(comp)={vals['rescue_rate_composite']:.1%}")
            if "mean_cc_correct_minority" in vals:
                print(f"          CC(correct minority)={vals['mean_cc_correct_minority']:.4f}  "
                      f"CC(incorrect majority)={vals['mean_cc_incorrect_majority']:.4f}  "
                      f"d={vals.get('cohen_d_rescue', 0):.3f}")
        ds_results["a5"] = a5

        # ── Analysis 6: CC Distribution ──────────────────────────────
        print("\n  [A6] CC Distribution by Correctness")
        a6 = analysis_6_cc_distribution(subset)
        print(f"    hybrid_cycle:   correct={a6['hybrid_cycle']['mean_correct']:.4f}  "
              f"incorrect={a6['hybrid_cycle']['mean_incorrect']:.4f}  "
              f"d={a6['hybrid_cycle']['cohen_d']:.3f}")
        print(f"    combined_reward: correct={a6['combined_reward']['mean_correct']:.4f}  "
              f"incorrect={a6['combined_reward']['mean_incorrect']:.4f}  "
              f"d={a6['combined_reward']['cohen_d']:.3f}")
        print(f"    Within-question Cohen's d: {a6['within_question_cohen_d_mean']:.3f} "
              f"(±{a6['within_question_cohen_d_std']:.3f}, "
              f"n_mixed={a6['n_questions_with_mixed']})")
        ds_results["a6"] = a6

        # ── Analysis 7: Simulated GRPO Advantage ─────────────────────
        print("\n  [A7] Simulated GRPO Advantage")
        a7 = analysis_7_grpo_advantage(subset, K_VALUES)
        for K, vals in sorted(a7.items()):
            comp = vals["composite"]
            mv = vals["mv_only"]
            print(f"    K={K:>2d} Composite: correct→+adv={comp['pct_correct_positive']:.1%}  "
                  f"incorrect→−adv={comp['pct_incorrect_negative']:.1%}  "
                  f"mean(c)={comp['mean_adv_correct']:+.3f}  "
                  f"mean(i)={comp['mean_adv_incorrect']:+.3f}")
            print(f"          MV-only:  correct→+adv={mv['pct_correct_positive']:.1%}  "
                  f"incorrect→−adv={mv['pct_incorrect_negative']:.1%}  "
                  f"mean(c)={mv['mean_adv_correct']:+.3f}  "
                  f"mean(i)={mv['mean_adv_incorrect']:+.3f}")
        ds_results["a7"] = a7

        all_analysis_results[ds_label] = ds_results

    # Save analysis results as JSON
    analysis_json_path = os.path.join(OUTPUT_DIR, "analysis_results.json")
    # Strip large lists (adv_correct, adv_incorrect) for JSON serialization
    _saveable = {}
    for ds, res in all_analysis_results.items():
        _saveable[ds] = {}
        for k, v in res.items():
            if k == "a7":
                # Strip large arrays
                a7_clean = {}
                for kk, vv in v.items():
                    a7_clean[kk] = {}
                    for rtype in ["composite", "mv_only"]:
                        a7_clean[kk][rtype] = {
                            k2: v2 for k2, v2 in vv[rtype].items()
                            if k2 not in ("adv_correct", "adv_incorrect")
                        }
                _saveable[ds][k] = a7_clean
            else:
                _saveable[ds][k] = v
    with open(analysis_json_path, "w") as f:
        json.dump(_saveable, f, indent=2, default=str)
    print(f"\n  Analysis results saved to {analysis_json_path}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 8: Generate all plots
    # ══════════════════════════════════════════════════════════════════════
    print(sep)
    print(" Step 8 — Visualizations")
    print("=" * 70)

    for ds_label in ["all"] + datasets:
        if ds_label == "all":
            subset = traj_data
        else:
            subset = [t for t in traj_data if t["dataset"] == ds_label]

        ds_plot_dir = os.path.join(PLOTS_DIR, ds_label)
        os.makedirs(ds_plot_dir, exist_ok=True)
        ds_results = all_analysis_results[ds_label]

        print(f"\n  ── Plots: {ds_label} (N={len(subset)}) ──")

        # 1. Trajectory AUROC bars
        plot_trajectory_auroc_bars(ds_results["a1"], ds_plot_dir, ds_label)

        # 2. Alpha sweep
        plot_alpha_sweep_trajectory(ds_results["a1_sweep"], ds_plot_dir, ds_label)

        # 3. CC distribution violin
        plot_cc_distribution(subset, ds_plot_dir, ds_label)

        # 4. Voting accuracy vs K
        plot_voting_accuracy_vs_k(ds_results["a2"], ds_plot_dir, ds_label)

        # 5. Best-of-K
        plot_best_of_k(ds_results["a3"], ds_plot_dir, ds_label)

        # 6. Filtered voting
        plot_filtered_voting(ds_results["a4"], ds_plot_dir, ds_label)

        # 7. Wrong-majority rescue (K=8 and K=16)
        for K in [8, 16]:
            plot_wrong_majority_rescue(subset, K, ds_plot_dir, ds_label)

        # 8. GRPO advantage
        for K in [16]:
            plot_grpo_advantage(ds_results["a7"], K, ds_plot_dir, ds_label)

    # 9. Summary dashboard (all datasets)
    all_results = all_analysis_results.get("all", {})
    if all_results:
        plot_summary_dashboard(
            all_results.get("a1", {}),
            all_results.get("a2", {}),
            all_results.get("a3", {}),
            all_results.get("a7", {}),
            PLOTS_DIR,
        )

    # ── Save final results ──────────────────────────────────────────────
    results_path = os.path.join(OUTPUT_DIR, "results.jsonl")
    save_jsonl(traj_data, results_path)
    print(f"\n  Final trajectory data saved to {results_path}")

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed / 60:.1f} min.  Outputs → {OUTPUT_DIR}")
    print(f"  Plots → {PLOTS_DIR}/")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 2.5: Per-Trajectory Composite Reward Analysis")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip steps whose checkpoint files already exist.",
    )
    args = parser.parse_args()
    main(resume=args.resume)
