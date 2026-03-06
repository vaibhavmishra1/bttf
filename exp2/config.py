"""Experiment 2 configuration — cycle-consistency with three fixes applied.

Fixes vs Exp 1:
  Fix 1 — LLM answer judge  : feed A and A' to an LLM, ask YES/NO instead of regex.
  Fix 2 — Reconstructor prompt: few-shot + QUESTION: prefix + think=False (same as updated exp1).
  Fix 3 — BLEU + embedding    : hybrid_cycle = BLEU_WEIGHT * bleu_cycle
                                               + (1 - BLEU_WEIGHT) * question_cycle
"""

import os

# ── Models ──────────────────────────────────────────────────────────────────
SOLVER_MODEL        = "Qwen/Qwen2.5-3B-Instruct"
RECONSTRUCTOR_MODEL = "Qwen/Qwen3-4B"
EMBEDDING_MODEL     = "Qwen/Qwen3-Embedding-0.6B"
JUDGE_MODEL         = "Qwen/Qwen2.5-3B-Instruct"   # Fix 1: LLM answer equivalence judge

# ── Dataset sizes ───────────────────────────────────────────────────────────
GSM8K_SAMPLES          = 1000
MATH_SAMPLES           = 500
OLYMPIADBENCH_SAMPLES  = 300

# ── Generation settings ─────────────────────────────────────────────────────
MAX_NEW_TOKENS           = 2048
SOLVER_BATCH_SIZE        = 256    # ignored by vLLM; kept for API compat
EMBEDDING_BATCH_SIZE     = 32
RECONSTRUCTOR_CONCURRENT = 20     # unused; kept for API compat

# ── Fix 3: BLEU weight for hybrid cycle distance ─────────────────────────────
# hybrid_cycle = BLEU_WEIGHT * bleu_cycle + (1 - BLEU_WEIGHT) * question_cycle
BLEU_WEIGHT = 0.5

# ── Answer comparison ────────────────────────────────────────────────────────
FLOAT_TOL = 1e-6

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")

# Optional: path to a completed exp1 outputs folder to reuse generation
# checkpoints (steps 0-4).  Set to None to always run from scratch.
EXP1_OUTPUTS = os.path.join(BASE_DIR, "..", "exp1", "outputs")
