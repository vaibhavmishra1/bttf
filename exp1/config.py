"""Experiment 1 configuration — cycle-consistency for LLM reasoning."""

import os

# ── Models ──────────────────────────────────────────────────────────────────
SOLVER_MODEL = "Qwen/Qwen2.5-3B-Instruct"
RECONSTRUCTOR_MODEL = "Qwen/Qwen3-4B"       # Qwen3.5-4B not yet in transformers<5/vLLM 0.16
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# ── Dataset sizes ───────────────────────────────────────────────────────────
GSM8K_SAMPLES          = 1000
MATH_SAMPLES           = 500
OLYMPIADBENCH_SAMPLES  = 300   # hard olympiad competition problems

# ── Generation settings ────────────────────────────────────────────────────
MAX_NEW_TOKENS         = 2048
SOLVER_BATCH_SIZE      = 256   # ignored by vLLM; kept for API compat
EMBEDDING_BATCH_SIZE   = 32
RECONSTRUCTOR_CONCURRENT = 20  # unused; kept for API compat

# ── Answer comparison ──────────────────────────────────────────────────────
FLOAT_TOL = 1e-6

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
