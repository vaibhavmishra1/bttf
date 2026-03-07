"""Experiment 2.5 configuration — Per-Trajectory Composite Reward Analysis."""

import os

# ── Models (same as Exp1) ────────────────────────────────────────────────────
SOLVER_MODEL = "Qwen/Qwen2.5-3B-Instruct"
RECONSTRUCTOR_MODEL = "Qwen/Qwen3-8B-Base"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"

# ── Generation settings ──────────────────────────────────────────────────────
MAX_NEW_TOKENS = 2048
EMBEDDING_BATCH_SIZE = 32

# ── Answer comparison (needed by exp1/utils.py) ──────────────────────────────
FLOAT_TOL = 1e-6

# ── Majority Voting / K settings ─────────────────────────────────────────────
K_MAX = 16
K_VALUES = [2, 4, 8, 16]

# ── Composite reward α-sweep ────────────────────────────────────────────────
ALPHA_SWEEP = [i / 20 for i in range(21)]  # 0.0, 0.05, ..., 1.0

# ── Filtered voting thresholds ───────────────────────────────────────────────
FILTER_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP1_DIR = os.path.join(BASE_DIR, "..", "exp1")
EXP2_DIR = os.path.join(BASE_DIR, "..", "exp2")

# Input data from Exp2
EXP2_STEP2 = os.path.join(EXP2_DIR, "outputs", "step2_k_solutions.jsonl")
EXP2_STEP3 = os.path.join(EXP2_DIR, "outputs", "step3_voting_signals.jsonl")

# Output
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
