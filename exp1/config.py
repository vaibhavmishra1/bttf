"""Experiment 1 configuration — cycle-consistency for LLM reasoning."""

import os

# ── Models ──────────────────────────────────────────────────────────────────
SOLVER_MODEL = "Qwen/Qwen2.5-3B-Instruct"
RECONSTRUCTOR_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# ── Dataset sizes ───────────────────────────────────────────────────────────
GSM8K_SAMPLES = 1000
MATH_SAMPLES = 500

# ── Generation settings ────────────────────────────────────────────────────
MAX_NEW_TOKENS = 2048
SOLVER_BATCH_SIZE = 8
EMBEDDING_BATCH_SIZE = 32
RECONSTRUCTOR_CONCURRENT = 20

# ── Answer comparison ──────────────────────────────────────────────────────
FLOAT_TOL = 1e-6

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
