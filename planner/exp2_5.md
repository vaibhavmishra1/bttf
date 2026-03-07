# Experiment 2.5 — Per-Trajectory Composite Reward Analysis

## 1. Motivation

Exp2 showed that MV and CC are **complementary at the question level** (Jaccard overlap of false positives = 0.40), and that fusion improves AUROC — especially at low K (+7.3 pts at K=2). But the CC signal in Exp2 was computed on only **one** solver trajectory (from Exp1's greedy solution). This limits what we can learn about per-trajectory reward quality.

The ultimate goal is to use a composite reward `r_i = MV_agreement(s_i) + CC_score(s_i)` for GRPO-style RL training. Before building the full training loop, we can **statically evaluate** how good this reward signal is by computing it on the K=16 trajectories we already have and measuring how well it identifies correct solutions.

**This experiment answers**: *"If we had a per-trajectory composite reward, how well would it rank correct vs incorrect solutions — and can it improve answer selection beyond standard majority voting?"*

---

## 2. What We Already Have (from Exp2)

For each of the 1800 questions:
- `question` (Q), `ground_truth`
- `correct` — whether the greedy (Exp1) solution was correct
- `k_solutions` — 16 stochastic solver outputs (in `step2_k_solutions.jsonl`, ~49MB)
- `combined_reward`, `hybrid_cycle`, `answer_match` — CC signals for the **Exp1 greedy solution only**
- `voting_confidence_K{2,4,8,16}`, `majority_answer_K{...}` — MV signals

**What's missing**: CC signals for each of the 16 stochastic solutions.

---

## 3. What We Need to Compute

### 3.1 Reconstruct Q' for Each Trajectory

For each question Q and each of its K=16 solutions s_1, ..., s_16:

```
s_i  →  Reconstructor  →  Q'_i
```

This produces 1800 × 16 = **28,800 reconstructor calls**.

**Model**: Qwen3-8B-Base (same as Exp1), via vLLM.

### 3.2 Compute Per-Trajectory CC Scores

For each (Q, s_i, Q'_i) triple, compute:

```
number_jaccard_i  = number_jaccard(Q, Q'_i)        # Jaccard distance on numeric tokens
chrf_dist_i       = chrf_distance(Q, Q'_i)          # character n-gram distance
```

These two metrics are text-based and CPU-only — very cheap.

**Optional (expensive)**: Embed Q and Q'_i to get `question_cycle_i` (cosine distance). This requires 28,800 additional embedding calls. The full `hybrid_cycle` uses all three:

```
hybrid_cycle_i = 0.4 * question_cycle_i + 0.4 * number_jaccard_i + 0.2 * chrf_dist_i
```

**Decision**: Run the **full** cycle-consistency pipeline on every trajectory — no shortcuts.

All three CC components are computed:
1. `question_cycle_i` — embedding cosine distance (Q, Q'_i)
2. `number_jaccard_i` — Jaccard distance on numeric tokens
3. `chrf_dist_i` — character n-gram distance

Combined into: `hybrid_cycle_i = 0.4 * question_cycle_i + 0.4 * number_jaccard_i + 0.2 * chrf_dist_i`

**We also run the full re-solving step** (Q'_i → Solver → S'_i → answer_match_i):
- Each reconstructed Q'_i is solved again by the same solver
- `answer_match_i = 1(extract_answer(S_i) == extract_answer(S'_i))`
- `combined_reward_i = answer_match_i - hybrid_cycle_i`

This matches the full Exp1 pipeline exactly, applied to every trajectory. It costs more (28,800 reconstructor calls + 28,800 solver calls + 30,600 embedding calls) but gives us the complete picture:
- We can evaluate which CC components matter at the trajectory level
- `combined_reward` was the strongest CC signal in Exp1 — we need it here too
- If we later want to use a cheaper variant for GRPO training, we can ablate downward from the full results

### 3.3 Per-Trajectory Ground Truth

For each solution s_i, extract the answer and check correctness:

```
answer_i = extract_final_answer(s_i)
trajectory_correct_i = answers_equivalent(answer_i, ground_truth)
```

This gives us the label we need to evaluate how well our composite reward identifies correct trajectories.

### 3.4 Per-Trajectory MV Agreement

For each solution s_i (within the K-group for a given question):

```
mv_agreement_i(K) = 1 if answer_i == majority_answer_K else 0
```

We can also compute a softer version:

```
mv_agreement_soft_i(K) = (# of other solutions in group with same answer as s_i) / (K - 1)
```

### 3.5 Composite Rewards

Combine MV and CC into per-trajectory rewards. The primary CC signal is `combined_reward_i = answer_match_i - hybrid_cycle_i` (same as Exp1), which captures both answer-level and question-level cycle consistency.

```
reward_binary_i    = 1(answer_i == majority_answer) + 1(combined_reward_i > 0)
reward_soft_i      = mv_agreement_soft_i + combined_reward_i
reward_weighted_i(α) = α * mv_agreement_soft_i + (1 - α) * combined_reward_i
```

We also test variants using `hybrid_cycle_i` directly (without `answer_match`):
```
reward_hybrid_i(α) = α * mv_agreement_soft_i + (1 - α) * (1 - hybrid_cycle_i)
```

---

## 4. Analyses to Run

### Analysis 1: Per-Trajectory Correctness Prediction (AUROC)

**Goal**: How well does each signal predict whether an individual trajectory is correct?

For all 28,800 trajectories (1800 questions × 16 solutions), compute AUROC:

| Signal | Description | Direction |
|---|---|---|
| `mv_agreement` | Does s_i agree with majority? (binary) | higher = better |
| `mv_agreement_soft` | Fraction of peers agreeing with s_i | higher = better |
| `hybrid_cycle_i` | Full CC distance (embed + text) | lower = better |
| `answer_match_i` | Does S_i and S'_i give same answer? | higher = better |
| `combined_reward_i` | answer_match_i − hybrid_cycle_i | higher = better |
| `reward_soft_i(α)` | α·mv_agreement_soft + (1−α)·combined_reward | higher = better |

Sweep α from 0 to 1 (same as Exp2's sweep). Report per-dataset and overall.

**This is the headline result**: If composite AUROC > MV-agreement AUROC, then the composite reward is a better training signal for GRPO than MV alone.

### Analysis 2: Reward-Weighted Voting

**Goal**: Does weighting votes by CC quality improve majority voting accuracy?

Instead of equal-weight majority voting (standard MV), weight each solution's vote by its CC score:

```
For each candidate answer a:
  weighted_vote(a) = Σ_{i: answer_i == a} weight_i
  where weight_i = 1 - hybrid_cycle_i    (or exp(-hybrid_cycle_i), or 1/(1 + hybrid_cycle_i))

Predicted answer = argmax_a weighted_vote(a)
```

Compare accuracy:
- Standard MV (equal weights)
- CC-weighted MV
- Composite-weighted MV (using reward_soft)

Report for K = 2, 4, 8, 16 and per-dataset.

### Analysis 3: Best-of-K Selection

**Goal**: If you could only pick ONE solution from the K samples, which selection criterion gives the highest accuracy?

Strategies:
1. **Random**: Pick a random solution → expected accuracy = per-trajectory accuracy
2. **MV-aligned**: Pick a solution that agrees with the majority (random among agreeing)
3. **Best-CC**: Pick the solution with the lowest hybrid_cycle
4. **Best-composite**: Pick the solution with the highest composite reward
5. **Oracle**: Pick a correct solution if one exists (upper bound)

Report accuracy for each strategy at K = 2, 4, 8, 16.

### Analysis 4: Filtered Voting

**Goal**: Does removing cycle-inconsistent trajectories before voting improve accuracy?

```
filtered_solutions = [s_i for s_i in solutions if hybrid_cycle_i < threshold]
if len(filtered_solutions) == 0:
    fall back to standard MV
else:
    majority_vote(filtered_solutions)
```

Sweep threshold from 0.1 to 0.9 and measure accuracy. Report per-dataset and per-K.

### Analysis 5: When Majority Is Wrong — CC as Safety Net

**Goal**: Among questions where the majority answer is wrong, do CC scores help identify the correct minority?

For questions where `majority_correct_K{k} == 0`:
- Is there at least one correct trajectory among the K?
- Among correct trajectories, is their CC score better than incorrect trajectories?
- Could a composite reward have rescued the answer?

Report:
- % of wrong-majority questions where a correct solution exists in the K-pool
- AUROC of CC for identifying those correct solutions within wrong-majority groups
- "Rescue rate": % of wrong-majority questions where the best-composite selection would have been correct

### Analysis 6: CC Score Distribution by Trajectory Position

**Goal**: Understand the relationship between CC and solution quality.

For each trajectory (correct vs incorrect):
- Plot distribution of `hybrid_cycle_i` for correct vs incorrect trajectories
- Compute Cohen's d for the separation
- Check if CC is better at distinguishing correct/incorrect trajectories within the same question (within-group) vs across questions (between-group)

### Analysis 7: Simulated GRPO Advantage

**Goal**: Simulate what the GRPO advantage signal would look like.

For each question with K solutions:

```
rewards = [reward_soft_i for i in 1..K]
advantages = (rewards - mean(rewards)) / std(rewards)
```

Analyze:
- Are advantages positive for correct trajectories and negative for incorrect?
- What fraction of correct trajectories receive positive advantage?
- What fraction of incorrect trajectories receive negative advantage?
- Compare using MV-only reward vs composite reward

---

## 5. Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Exp 2.5 Pipeline                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 0: Load step2_k_solutions.jsonl (1800 × 16 solutions)         │
│          Load results.jsonl (MV signals, ground truth)               │
│                                                                      │
│  Step 1: Flatten → 28,800 (question, solution_i) pairs              │
│          Extract answer_i, compute trajectory_correct_i              │
│                                                                      │
│  Step 2: Reconstruct Q'_i for each solution_i       [GPU: Recon]    │
│          s_i → Reconstructor → Q'_i                                  │
│          → 28,800 reconstructor calls via vLLM                       │
│                                                                      │
│  Step 3: Re-solve Q'_i → S'_i                       [GPU: Solver]   │
│          Q'_i → Solver → S'_i                                        │
│          → 28,800 solver calls via vLLM                              │
│          Extract answer_A'_i, compute answer_match_i                 │
│                                                                      │
│  Step 4: Embed Q and Q'_i → question_cycle_i        [GPU: Embedder] │
│          → 1,800 Q embeddings + 28,800 Q'_i embeddings               │
│          Compute text CC: number_jaccard_i, chrf_dist_i              │
│          Combine → hybrid_cycle_i                                    │
│          Compute combined_reward_i = answer_match_i - hybrid_cycle_i │
│                                                                      │
│  Step 5: Compute MV agreement signals per trajectory                 │
│          → mv_agreement_i, mv_agreement_soft_i for each K           │
│                                                                      │
│  Step 6: Compute composite rewards (α-sweep)                        │
│          → reward_soft_i(α) for each trajectory                      │
│                                                                      │
│  Step 7: Run all analyses (1–7)                                      │
│          → metrics, tables, plots                                    │
│                                                                      │
│  Step 8: Generate visualizations and summary                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. Compute Budget

| Step | Operation | Count | Model | Est. Time |
|---|---|---|---|---|
| 1 | Answer extraction + correctness | 28,800 | CPU only | ~1 min |
| 2 | Reconstructor inference (s_i → Q'_i) | 28,800 | Qwen3-8B-Base (vLLM) | ~30–45 min |
| 3 | Re-solver inference (Q'_i → S'_i) | 28,800 | Qwen2.5-3B-Instruct (vLLM) | ~20–30 min |
| 4a | Embedding Q and Q'_i | 1,800 + 28,800 | Qwen3-Embedding-0.6B (HF) | ~10 min |
| 4b | Text CC metrics + hybrid_cycle + combined_reward | 28,800 | CPU only | ~5 min |
| 5–8 | MV signals + Composite + Analysis + Plots | — | CPU only | ~5 min |
| **Total** | | | | **~70–95 min** |

Steps 2 and 3 are the bottlenecks — they require loading large models sequentially:
1. Load Reconstructor (Qwen3-8B-Base, bfloat16, ~16GB VRAM) → run 28,800 prompts → destroy
2. Load Solver (Qwen2.5-3B-Instruct, float16, ~6GB VRAM) → run 28,800 prompts → destroy
3. Load Embedder (Qwen3-Embedding-0.6B, float16, ~1.2GB VRAM) → embed 30,600 texts → destroy

All three models are loaded and destroyed sequentially to avoid GPU OOM.

---

## 7. Checkpointing Strategy

Each step saves to a separate JSONL file in `exp2_5/outputs/`:

| File | Contents |
|---|---|
| `step1_trajectories.jsonl` | 28,800 rows: question_idx, solution_i, answer_i, trajectory_correct_i |
| `step2_reconstructed.jsonl` | + Q'_i for each trajectory |
| `step3_resolved.jsonl` | + S'_i, answer_A'_i, answer_match_i for each trajectory |
| `step4_cc_signals.jsonl` | + question_cycle_i, number_jaccard_i, chrf_dist_i, hybrid_cycle_i, combined_reward_i |
| `step5_mv_signals.jsonl` | + mv_agreement per K, mv_agreement_soft per K |
| `step6_composite.jsonl` | + reward_soft(α) for each α |
| `results.jsonl` | Final merged data with all signals + analysis results |

The `--resume` flag skips steps whose output files already exist.

---

## 8. Expected Outputs

### Tables (printed to console + saved as JSON)

1. **Per-trajectory AUROC table**: MV-agreement vs CC vs Composite, per dataset, per K
2. **Voting accuracy table**: Standard MV vs CC-weighted MV vs Composite-weighted MV, per K, per dataset
3. **Best-of-K accuracy table**: Random vs MV-aligned vs Best-CC vs Best-composite vs Oracle
4. **Wrong-majority rescue table**: Rescue rate per K and dataset
5. **Simulated GRPO advantage table**: % correct-positive, % incorrect-negative

### Plots (saved to `exp2_5/outputs/plots/`)

1. **Trajectory AUROC bar chart**: Grouped bars (MV / CC / Composite) per dataset
2. **α-sweep curve**: AUROC vs α for per-trajectory correctness prediction
3. **CC distribution violin plots**: hybrid_cycle for correct vs incorrect trajectories (per dataset)
4. **Voting accuracy vs K**: Lines for standard MV, CC-weighted MV, composite MV, and oracle
5. **Best-of-K accuracy vs K**: Lines for each selection strategy
6. **Filtered voting accuracy vs threshold**: Accuracy as CC threshold varies
7. **Wrong-majority rescue scatter**: For each wrong-majority question, show (CC-correct, CC-incorrect) scores
8. **Simulated GRPO advantage histogram**: Distribution of advantages for correct vs incorrect trajectories
9. **Summary dashboard**: 2×2 grid with headline results

---

## 9. Hypotheses

| # | Hypothesis | Testable Criterion |
|---|---|---|
| H1 | Per-trajectory CC predicts correctness | AUROC(hybrid_cycle → trajectory_correct) > 0.55 |
| H2 | Composite reward beats MV-agreement | AUROC(composite) > AUROC(mv_agreement) by ≥2 pts |
| H3 | CC-weighted voting beats standard MV | Weighted MV accuracy > Standard MV accuracy at K≥4 |
| H4 | Composite selects better single solution | Best-of-K(composite) > Best-of-K(MV-aligned) |
| H5 | CC can rescue wrong-majority answers | Rescue rate > 5% of wrong-majority questions |
| H6 | Simulated GRPO advantage is correctly signed | >70% correct trajectories get positive advantage |

---

## 10. Key Differences from Exp2

| Dimension | Exp2 | Exp2.5 |
|---|---|---|
| CC applied to | 1 greedy solution | All 16 stochastic solutions |
| Unit of analysis | Question-level | **Trajectory-level** |
| What it evaluates | MV vs CC as evaluation signals | **Composite reward quality for GRPO** |
| MV signal | Voting confidence (scalar) | **Per-trajectory agreement** (binary/soft) |
| Key metric | Question-level AUROC | **Trajectory-level AUROC + voting accuracy** |
| Answers the question | "Do MV and CC complement each other?" | **"Is the composite reward good enough to train with?"** |

---

## 11. Code Structure

```
exp2_5/
├── config.py          # Paths, model names, K values, α-sweep range
├── signals.py         # Per-trajectory CC, MV-agreement, composite reward computation
├── analysis.py        # All 7 analyses (AUROC, weighted voting, best-of-K, etc.)
├── visualize.py       # All plots
├── run.py             # Main pipeline orchestrator (Steps 0–8)
│                        Step 2: Reconstructor (vLLM) — 28,800 calls
│                        Step 3: Re-solver (vLLM) — 28,800 calls
│                        Step 4: Embedder (HF) — 30,600 calls + text metrics
└── outputs/
    ├── step1_trajectories.jsonl
    ├── step2_reconstructed.jsonl
    ├── step3_resolved.jsonl
    ├── step4_cc_signals.jsonl
    ├── step5_mv_signals.jsonl
    ├── step6_composite.jsonl
    ├── results.jsonl
    └── plots/
        ├── all/
        ├── gsm8k/
        ├── math/
        └── olympiadbench/
```

Modules reused from exp1 (via sys.path):
- `utils.py`: `extract_boxed_answer`, `extract_answer`, `answers_equivalent`, `number_jaccard`, `chrf_distance`, `compute_hybrid_cycle`
- `models.py`: `Reconstructor`, `Embedder`, `Solver` classes

Modules reused from exp2 (via sys.path):
- `voting.py`: `extract_final_answer`, `cluster_answers`

---

## 12. Success Criteria

**Strong success** (→ directly motivates GRPO Exp3):
- H2 confirmed: composite AUROC beats MV-agreement by ≥2 pts on trajectory correctness
- H3 confirmed: CC-weighted voting outperforms standard MV
- H5 confirmed: >5% rescue rate on wrong-majority questions

**Moderate success** (→ motivates GRPO with caveats):
- H1 confirmed but H2 marginal: CC adds signal but lift is small
- H3 partially confirmed: improvement only at low K

**Weak/negative** (→ reconsider GRPO approach):
- H1 rejected: per-trajectory CC doesn't predict correctness
- CC is too noisy at the trajectory level to help

In any case, this experiment provides the definitive answer to whether CC is useful as a per-trajectory reward signal — which is essential before investing in a full GRPO training run.
