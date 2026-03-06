#!/usr/bin/env python3
"""
Experiment 2 — Cycle-Consistency with Three Fixes Applied.

  Fix 1  LLM answer judge  : AnswerJudge(A, A') → YES/NO instead of regex.
  Fix 2  Reconstructor      : few-shot + QUESTION: prefix + think=False.
  Fix 3  BLEU + embedding   : hybrid_cycle = 0.5·bleu_cycle + 0.5·question_cycle.

Pipeline:
  Q → Solver → S → Reconstructor → Q' → Solver → S'
  [Step 4a] Embed Q, Q'  → question_cycle
  [Step 4b] BLEU(Q, Q')  → bleu_cycle                          [Fix 3]
            hybrid_cycle  = blend of (4a) and (4b)
  [Step 5a] Extract A, A' (boxed → heuristic)
  [Step 5b] AnswerJudge(A, A')     → answer_match_llm           [Fix 1]
            answers_equivalent(A, A') → answer_match  (kept for comparison)
  [Step 6]  answers_equivalent(A, GT) → correct
  [Step 7]  combined_reward    = answer_match     - question_cycle  (exp1)
            combined_reward_v2 = answer_match_llm - hybrid_cycle    (exp2)

Run:
    python run.py                          # full pipeline from scratch
    python run.py --resume                 # skip completed steps
    python run.py --from-exp1 --resume     # bootstrap steps 0-4 from exp1 outputs
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from config import (
    SOLVER_MODEL,
    RECONSTRUCTOR_MODEL,
    EMBEDDING_MODEL,
    JUDGE_MODEL,
    GSM8K_SAMPLES,
    MATH_SAMPLES,
    OLYMPIADBENCH_SAMPLES,
    MAX_NEW_TOKENS,
    SOLVER_BATCH_SIZE,
    EMBEDDING_BATCH_SIZE,
    RECONSTRUCTOR_CONCURRENT,
    BLEU_WEIGHT,
    OUTPUT_DIR,
    PLOTS_DIR,
    EXP1_OUTPUTS,
)
from data import load_all_data
from models import Solver, Reconstructor, Embedder, AnswerJudge
from utils import (
    extract_answer,
    extract_boxed_answer,
    answers_equivalent,
    compute_bleu_cycles_batch,
    compute_hybrid_cycle,
)
from metrics import compute_all_metrics
from visualize import generate_all_plots


# ── I/O helpers ──────────────────────────────────────────────────────────────

def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _cached(path: str, resume: bool, fallback_path: str | None = None) -> list[dict] | None:
    """Return cached data if resume=True and file exists.

    If path doesn't exist but fallback_path does (exp1 output), use that instead
    so we don't re-run the expensive generation steps.
    """
    if resume and os.path.exists(path):
        print(f"  ↳ cache hit: {os.path.basename(path)}")
        return load_jsonl(path)
    if fallback_path and os.path.exists(fallback_path):
        print(f"  ↳ bootstrapping from exp1: {os.path.basename(fallback_path)}")
        data = load_jsonl(fallback_path)
        # Save into exp2/outputs so future --resume works
        save_jsonl(data, path)
        return data
    return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(resume: bool = False, from_exp1: bool = False) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    t0 = time.time()

    def _exp1(filename: str) -> str | None:
        """Return path to exp1 checkpoint if --from-exp1 and file exists."""
        if from_exp1 and EXP1_OUTPUTS and os.path.isdir(EXP1_OUTPUTS):
            p = os.path.join(EXP1_OUTPUTS, filename)
            return p if os.path.exists(p) else None
        return None

    # ── Step 0: Load data ────────────────────────────────────────────────
    data_path = os.path.join(OUTPUT_DIR, "data.jsonl")
    data = _cached(data_path, resume, _exp1("data.jsonl"))
    if data is None:
        print("[Step 0] Loading datasets …")
        data = load_all_data(GSM8K_SAMPLES, MATH_SAMPLES, OLYMPIADBENCH_SAMPLES)
        save_jsonl(data, data_path)
    ds_counts = {k: sum(1 for d in data if d["dataset"] == k)
                 for k in set(d["dataset"] for d in data)}
    print(f"  Total: {len(data)}  |  " + "  ".join(f"{k}={v}" for k, v in sorted(ds_counts.items())))

    # ── Step 1: Q → Solver → S ──────────────────────────────────────────
    step1_path = os.path.join(OUTPUT_DIR, "step1_solutions.jsonl")
    data = _cached(step1_path, resume, _exp1("step1_solutions.jsonl"))
    if data is None:
        data = load_jsonl(data_path)
        print("[Step 1] Solving questions (vLLM Solver) …")
        solver = Solver(SOLVER_MODEL)
        solutions = solver.solve_batch(
            [d["question"] for d in data], MAX_NEW_TOKENS, SOLVER_BATCH_SIZE
        )
        for d, s in zip(data, solutions):
            d["solution_S"] = s
        save_jsonl(data, step1_path)
        solver.destroy(); del solver

    # ── Step 2: S → Reconstructor → Q'  [Fix 2] ─────────────────────────
    step2_path = os.path.join(OUTPUT_DIR, "step2_reconstructed.jsonl")
    data = _cached(step2_path, resume)   # intentionally no exp1 fallback:
    # exp1 used GPT-4.1; exp2 uses Qwen3-4B — different outputs, run fresh.
    if data is None:
        data = load_jsonl(step1_path)
        print("[Step 2] Reconstructing questions (Fix 2: Qwen3-4B, few-shot, QUESTION: prefix) …")
        reconstructor = Reconstructor(RECONSTRUCTOR_MODEL)
        reconstructed = reconstructor.reconstruct_batch(
            [d["solution_S"] for d in data], RECONSTRUCTOR_CONCURRENT
        )
        for d, q in zip(data, reconstructed):
            d["question_Q_prime"] = q
        save_jsonl(data, step2_path)
        reconstructor.destroy(); del reconstructor

    # ── Step 3: Q' → Solver → S' ────────────────────────────────────────
    step3_path = os.path.join(OUTPUT_DIR, "step3_second_solutions.jsonl")
    data = _cached(step3_path, resume)   # no exp1 fallback: Q' is different
    if data is None:
        data = load_jsonl(step2_path)
        print("[Step 3] Solving reconstructed questions …")
        solver = Solver(SOLVER_MODEL)
        solutions_prime = solver.solve_batch(
            [d["question_Q_prime"] for d in data], MAX_NEW_TOKENS, SOLVER_BATCH_SIZE
        )
        for d, s in zip(data, solutions_prime):
            d["solution_S_prime"] = s
        save_jsonl(data, step3_path)
        solver.destroy(); del solver

    # ── Step 4a: Embed Q, Q' → question_cycle ───────────────────────────
    step4_path = os.path.join(OUTPUT_DIR, "step4_embeddings.jsonl")
    data = _cached(step4_path, resume)
    if data is None:
        data = load_jsonl(step3_path)
        print("[Step 4a] Computing embedding cycle (question_cycle) …")
        embedder = Embedder(EMBEDDING_MODEL)
        emb_q  = embedder.embed([d["question"]          for d in data], EMBEDDING_BATCH_SIZE)
        emb_qp = embedder.embed([d["question_Q_prime"]  for d in data], EMBEDDING_BATCH_SIZE)
        cosine_sims     = np.sum(emb_q * emb_qp, axis=1)
        question_cycles = 1.0 - cosine_sims
        for d, qc in zip(data, question_cycles):
            d["question_cycle"] = float(qc)
        save_jsonl(data, step4_path)
        del embedder; torch.cuda.empty_cache()

    # ── Step 4b: BLEU(Q, Q') → bleu_cycle + hybrid_cycle  [Fix 3] ───────
    step4b_path = os.path.join(OUTPUT_DIR, "step4b_bleu.jsonl")
    data = _cached(step4b_path, resume)
    if data is None:
        data = load_jsonl(step4_path)
        print("[Step 4b] Computing BLEU cycle (Fix 3) …")
        bleu_cycles = compute_bleu_cycles_batch(
            [d["question"]         for d in data],
            [d["question_Q_prime"] for d in data],
        )
        for d, bc in zip(data, bleu_cycles):
            d["bleu_cycle"]    = float(bc)
            d["hybrid_cycle"]  = float(
                compute_hybrid_cycle(d["question_cycle"], bc, BLEU_WEIGHT)
            )
        save_jsonl(data, step4b_path)

    # ── Step 5a: Extract answers ─────────────────────────────────────────
    step5a_path = os.path.join(OUTPUT_DIR, "step5a_answers.jsonl")
    data = _cached(step5a_path, resume)
    if data is None:
        data = load_jsonl(step4b_path)
        print("[Step 5a] Extracting answers from S and S' …")
        for d in data:
            boxed_A       = extract_boxed_answer(d["solution_S"])
            boxed_A_prime = extract_boxed_answer(d["solution_S_prime"])
            d["answer_A"]       = boxed_A       if boxed_A       else extract_answer(d["solution_S"])
            d["answer_A_prime"] = boxed_A_prime if boxed_A_prime else extract_answer(d["solution_S_prime"])
            # Regex equivalence (kept for comparison with exp1)
            d["answer_match"] = int(
                answers_equivalent(d["answer_A"], d["answer_A_prime"])
            )
        save_jsonl(data, step5a_path)

    # ── Step 5b: LLM answer judge  [Fix 1] ───────────────────────────────
    step5b_path = os.path.join(OUTPUT_DIR, "step5b_judge.jsonl")
    data = _cached(step5b_path, resume)
    if data is None:
        data = load_jsonl(step5a_path)
        print("[Step 5b] Judging answer equivalence with LLM (Fix 1) …")
        judge = AnswerJudge(JUDGE_MODEL)
        pairs   = [(d["answer_A"], d["answer_A_prime"]) for d in data]
        verdicts = judge.judge_batch(pairs)
        for d, v in zip(data, verdicts):
            d["answer_match_llm"] = v
        save_jsonl(data, step5b_path)
        judge.destroy(); del judge

    # ── Step 6: Solver correctness + combined rewards ────────────────────
    results_path = os.path.join(OUTPUT_DIR, "results.jsonl")
    data = _cached(results_path, resume)
    if data is None:
        data = load_jsonl(step5b_path)
        print("[Step 6] Computing correctness and combined rewards …")
        for d in data:
            d["correct"] = int(answers_equivalent(d["answer_A"], d["ground_truth"]))

            # exp1-style combined reward (for baseline comparison)
            d["combined_reward"]    = d["answer_match"]     - d["question_cycle"]

            # exp2 combined reward (Fix 1 + Fix 3)
            d["combined_reward_v2"] = d["answer_match_llm"] - d["hybrid_cycle"]

        save_jsonl(data, results_path)

    # ── Step 9: Metrics ───────────────────────────────────────────────────
    sep = "\n" + "=" * 85
    print(sep); print(" Step 9 — Discrimination Metrics"); print("=" * 85)
    datasets_present = sorted(set(d["dataset"] for d in data))
    compute_all_metrics(data, label="ALL datasets combined")
    for ds in datasets_present:
        compute_all_metrics([d for d in data if d["dataset"] == ds], label=ds.upper())

    # ── Step 10: Visualizations ───────────────────────────────────────────
    print(sep); print(" Step 10 — Visualizations"); print("=" * 85)
    generate_all_plots(data, PLOTS_DIR)

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed / 60:.1f} min.  Outputs → {OUTPUT_DIR}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 2: cycle-consistency with fixes")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip steps whose checkpoint files already exist.")
    parser.add_argument("--from-exp1", action="store_true",
                        help="Bootstrap steps 0, 1 (data + first-pass solutions) "
                             "from exp1/outputs/ to avoid re-running generation. "
                             "Steps 2-3 are always re-run (different reconstructor).")
    args = parser.parse_args()
    main(resume=args.resume, from_exp1=args.from_exp1)
