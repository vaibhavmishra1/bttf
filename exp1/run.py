#!/usr/bin/env python3
"""
Experiment 1 — Cycle-Consistency as a Self-Supervised Signal
for LLM Reasoning Correctness.

Pipeline:  Q → Solver → S → Reconstructor → Q' → Solver → S'

Datasets: GSM8K (easy) · MATH (medium) · OlympiadBench (hard)

Run:
    python run.py                  # full pipeline from scratch
    python run.py --resume         # skip steps whose output files exist
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
    GSM8K_SAMPLES,
    MATH_SAMPLES,
    OLYMPIADBENCH_SAMPLES,
    MAX_NEW_TOKENS,
    SOLVER_BATCH_SIZE,
    EMBEDDING_BATCH_SIZE,
    RECONSTRUCTOR_CONCURRENT,
    OUTPUT_DIR,
    PLOTS_DIR,
)
from data import load_all_data
from models import Solver, Reconstructor, Embedder
from utils import (
    extract_answer, extract_boxed_answer, answers_equivalent,
    number_jaccard, chrf_distance, compute_hybrid_cycle,
)
from metrics import compute_all_metrics
from visualize import generate_all_plots


# ── I/O helpers ─────────────────────────────────────────────────────────────

def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


# ── Main pipeline ───────────────────────────────────────────────────────────

def main(resume: bool = False) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    t0 = time.time()

    # ── Step 0: Load data ──────────────────────────────────────────────
    data_path = os.path.join(OUTPUT_DIR, "data.jsonl")
    data = _cached(data_path, resume)
    if data is None:
        print("[Step 0] Loading datasets …")
        data = load_all_data(GSM8K_SAMPLES, MATH_SAMPLES, OLYMPIADBENCH_SAMPLES)
        save_jsonl(data, data_path)
    ds_counts = {k: sum(1 for d in data if d["dataset"] == k) for k in set(d["dataset"] for d in data)}
    print(f"  Total samples: {len(data)}  |  " + "  ".join(f"{k}={v}" for k,v in sorted(ds_counts.items())))

    # ── Step 1: Solve Q → S ───────────────────────────────────────────
    step1_path = os.path.join(OUTPUT_DIR, "step1_solutions.jsonl")
    data = _cached(step1_path, resume)
    if data is None:
        data = load_jsonl(data_path)
        print("[Step 1] Solving questions with vLLM …")
        solver = Solver(SOLVER_MODEL)
        solutions = solver.solve_batch(
            [d["question"] for d in data], MAX_NEW_TOKENS, SOLVER_BATCH_SIZE
        )
        for d, s in zip(data, solutions):
            d["solution_S"] = s
        save_jsonl(data, step1_path)
        solver.destroy()
        del solver

    # ── Step 2: S → Reconstructor → Q' ────────────────────────────────
    step2_path = os.path.join(OUTPUT_DIR, "step2_reconstructed.jsonl")
    data = _cached(step2_path, resume)
    if data is None:
        data = load_jsonl(step1_path)
        print("[Step 2] Reconstructing questions via vLLM (Qwen3-4B) …")
        reconstructor = Reconstructor(RECONSTRUCTOR_MODEL)
        reconstructed = reconstructor.reconstruct_batch(
            [d["solution_S"] for d in data], RECONSTRUCTOR_CONCURRENT
        )
        for d, q in zip(data, reconstructed):
            d["question_Q_prime"] = q
        save_jsonl(data, step2_path)
        reconstructor.destroy()
        del reconstructor

    # ── Step 3: Q' → Solver → S' ─────────────────────────────────────
    step3_path = os.path.join(OUTPUT_DIR, "step3_second_solutions.jsonl")
    data = _cached(step3_path, resume)
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
        solver.destroy()
        del solver

    # ── Step 4: Embed Q, Q' → question_cycle + hybrid_cycle ─────────
    step4_path = os.path.join(OUTPUT_DIR, "step4_embeddings.jsonl")
    data = _cached(step4_path, resume)
    if data is None:
        data = load_jsonl(step3_path)
        print("[Step 4] Computing embeddings …")
        embedder = Embedder(EMBEDDING_MODEL)
        emb_q  = embedder.embed([d["question"]         for d in data], EMBEDDING_BATCH_SIZE)
        emb_qp = embedder.embed([d["question_Q_prime"] for d in data], EMBEDDING_BATCH_SIZE)
        cosine_sims   = np.sum(emb_q * emb_qp, axis=1)
        question_cycles = 1.0 - cosine_sims

        print("[Step 4b] Computing number_jaccard + chrf_dist → hybrid_cycle …")
        for d, qc in zip(data, question_cycles):
            d["question_cycle"] = float(qc)
            num_jac  = number_jaccard(d["question"], d["question_Q_prime"])
            chrf_d   = chrf_distance(d["question"], d["question_Q_prime"])
            d["number_jaccard"] = num_jac
            d["chrf_dist"]      = chrf_d
            d["hybrid_cycle"]   = compute_hybrid_cycle(float(qc), num_jac, chrf_d)

        save_jsonl(data, step4_path)
        del embedder
        torch.cuda.empty_cache()

    # ── Steps 5-7: Extract answers, equivalence, correctness ──────────
    results_path = os.path.join(OUTPUT_DIR, "results.jsonl")
    data = _cached(results_path, resume)
    if data is None:
        data = load_jsonl(step4_path)
        print("[Steps 5–7] Extracting & comparing answers …")
        for d in data:
            # answer_match: use \boxed{} from both S and S' (model reliably outputs it)
            boxed_A       = extract_boxed_answer(d["solution_S"])
            boxed_A_prime = extract_boxed_answer(d["solution_S_prime"])
            d["answer_A"]       = boxed_A       if boxed_A       else extract_answer(d["solution_S"])
            d["answer_A_prime"] = boxed_A_prime if boxed_A_prime else extract_answer(d["solution_S_prime"])

            d["answer_match"] = int(
                answers_equivalent(d["answer_A"], d["answer_A_prime"])
            )
            d["correct"] = int(
                answers_equivalent(d["answer_A"], d["ground_truth"])
            )
            d["combined_reward"] = d["answer_match"] - d["hybrid_cycle"]
        save_jsonl(data, results_path)

    # ── Step 9: Metrics — overall + per-dataset ────────────────────────
    sep = "\n" + "=" * 70
    print(sep)
    print(" Step 9 — Discrimination Metrics")
    print("=" * 70)

    datasets_present = sorted(set(d["dataset"] for d in data))

    # Overall
    compute_all_metrics(data, label="ALL datasets combined")

    # Per dataset
    for ds in datasets_present:
        subset = [d for d in data if d["dataset"] == ds]
        compute_all_metrics(subset, label=ds.upper())

    # ── Step 10: Visualizations — per-dataset sub-folders ─────────────
    print(sep)
    print(" Step 10 — Visualizations (per-dataset folders)")
    print("=" * 70)
    generate_all_plots(data, PLOTS_DIR)

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed / 60:.1f} min.  Outputs → {OUTPUT_DIR}")
    print(f"  Plots saved to sub-folders inside {PLOTS_DIR}/")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 1: cycle-consistency")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps whose checkpoint files already exist.",
    )
    args = parser.parse_args()
    main(resume=args.resume)
