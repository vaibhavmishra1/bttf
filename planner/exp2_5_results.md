# Experiment 2.5 — Per-Trajectory Composite Reward Analysis: Results

> **⚠️ STATUS: PREVIOUS ANALYSIS INVALID — AWAITING RE-RUN**
>
> The previous analysis in this file was based on **Exp2's question-level outputs
> that were mistakenly copied** into `exp2_5/outputs/`. The per-trajectory CC
> pipeline (`exp2_5/run.py`) was **never actually executed**.
>
> All stale data and plots have been cleaned up. Run the real pipeline:
> ```bash
> cd exp2_5 && python run.py
> ```
> Then re-analyse the results to populate this file.

## What Went Wrong

1. `exp2_5/run.py` was written correctly with full per-trajectory CC (reconstruct → re-solve → embed for each of 28,800 trajectories)
2. But it was **never executed**
3. Instead, Exp2's 1,800-row question-level outputs were byte-for-byte copied into `exp2_5/outputs/`
4. Analysis was done via inline terminal scripts against these wrong files
5. Since CC was constant within each question group, it mathematically cancelled in GRPO advantage

## What Has Been Fixed

- ✅ Removed all wrongly-copied Exp2 files from `exp2_5/outputs/`
- ✅ Removed all stale plots from `exp2_5/outputs/plots/`
- ✅ Fixed indentation bug in `run.py` Step 5 (progress logging was outside loop)
- ✅ Verified all imports, data paths, and analysis functions via dry-run tests
- ✅ Confirmed analysis code correctly handles per-trajectory CC variation

## To Generate Real Results

```bash
cd exp2_5 && python run.py
```

Expected pipeline (≈70–95 min on GPU):
- Step 1: Flatten 1,800 × 16 → 28,800 trajectories
- Step 2: Reconstruct Q'_i for each trajectory (28,800 Reconstructor calls)
- Step 3: Re-solve Q'_i → S'_i (28,800 Solver calls)
- Step 4: Embed Q & Q'_i + text CC → hybrid_cycle_i + combined_reward_i (28,800 embeddings)
- Step 5: Compute per-trajectory MV agreement
- Step 6: Compute composite rewards
- Step 7: Run all 7 analyses
- Step 8: Generate all plots
