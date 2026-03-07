"""Experiment 2 configuration — Majority Voting vs Cycle Consistency."""

import os

# ── Models ──────────────────────────────────────────────────────────────────
SOLVER_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ── Dataset sizes (same as exp1) ────────────────────────────────────────────
GSM8K_SAMPLES = 1000
MATH_SAMPLES = 500
OLYMPIADBENCH_SAMPLES = 300

# ── Majority Voting settings ────────────────────────────────────────────────
K_MAX = 16                         # generate all 16 upfront
K_VALUES = [2, 4, 8, 16]          # subsample to these for analysis
TEMPERATURE = 0.7                  # stochastic sampling temperature
TEMPERATURE_ABLATIONS = [0.5, 0.7, 1.0]
TOP_P = 0.95                       # nucleus sampling
MAX_NEW_TOKENS = 2048

# ── Solver batch settings ───────────────────────────────────────────────────
SOLVER_BATCH_SIZE = 256            # ignored by vLLM; kept for compat

# ── Answer comparison ──────────────────────────────────────────────────────
FLOAT_TOL = 1e-6

# ── Fusion ──────────────────────────────────────────────────────────────────
ALPHA_SWEEP = [i / 20 for i in range(21)]  # 0.0, 0.05, ..., 1.0
LOGISTIC_CV_FOLDS = 5

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP1_RESULTS = os.path.join(BASE_DIR, "..", "exp1", "outputs_v2", "results.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# ── Cycle Consistency cost model (solver-equivalent passes) ────────────────
CC_COST = 3.5   # 1 solver + 1 reconstructor (~1.5 solver-equiv) + 1 solver + embed (~0)
