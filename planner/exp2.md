
# Experiment 2 — Majority Voting vs Cycle Consistency

## Self-Supervised Verification for LLM Reasoning: A Head-to-Head Comparison

---

# 1. MOTIVATION

## 1.1 The Problem

LLMs can solve math problems, but they cannot reliably tell you *when* they are wrong. A model with 56% accuracy produces 44% confident-sounding garbage. Any deployment — education, scientific computing, autonomous agents — requires a correctness signal that doesn't depend on ground truth labels.

Two families of self-supervised verification have emerged:

1. **Self-Consistency / Majority Voting** — Sample multiple solutions to the same question; the answer that appears most often is likely correct.
2. **Cycle Consistency** — Loop the solution back through a reconstructor to recover the question; if the reconstruction is faithful, the solution is likely correct.

Both are inference-time methods. Both require no ground truth. Both cost extra compute. **Nobody has directly compared them on the same model, same data, same evaluation.**

## 1.2 Why This Comparison Matters

| Dimension | Majority Voting | Cycle Consistency |
|---|---|---|
| Signal source | Agreement among K i.i.d. solution attempts | Invertibility of the Q→S→Q' transformation |
| What it detects | Stochastic reasoning errors (different samples disagree) | Structural reasoning errors (solution doesn't faithfully encode the question) |
| Blind spot | **Systematic errors** — if the model always makes the same mistake, all K samples agree on the wrong answer | **Consistently wrong cycles** — if the wrong solution reconstructs a question that yields the same wrong answer |
| Compute cost | K forward passes through the solver | 1 forward (solver) + 1 forward (reconstructor) + 1 forward (solver) + embeddings |
| Prior work | Wang et al., "Self-Consistency Improves Chain of Thought Reasoning" (2023) | Our Experiment 1 (this project) |

The key scientific question is: **Are these blind spots complementary?** Majority voting fails on systematic errors; cycle consistency fails on consistent cycles. If they fail on *different* samples, combining them could beat both.

## 1.3 Research Questions

| # | Question |
|---|---|
| **RQ1** | Which method achieves higher AUROC for predicting solver correctness? |
| **RQ2** | Do they fail on different samples? (Complementarity analysis) |
| **RQ3** | Does a simple combination (e.g., weighted sum) outperform either alone? |
| **RQ4** | How does majority voting's K trade off against cycle consistency's fixed cost? |
| **RQ5** | Does the comparison depend on problem difficulty (GSM8K vs MATH vs OlympiadBench)? |

## 1.4 Hypotheses

| ID | Hypothesis | Rationale |
|---|---|---|
| **H1** | Majority voting will have higher AUROC on GSM8K (easy) | Easy problems have higher stochastic variance → more disagreement signal when wrong |
| **H2** | Cycle consistency will have higher AUROC on OlympiadBench (hard) | Hard problems produce systematic errors → all K samples agree on the wrong answer, defeating voting; but the cycle may detect structural breaks |
| **H3** | Overlap of failure cases will be < 60% | The methods use fundamentally different information (answer agreement vs. question reconstruction) |
| **H4** | A simple linear combination will outperform both by ≥ 3 AUROC points | Complementary failure modes → information gain from fusion |
| **H5** | Majority voting at K=8 will match or exceed cycle consistency | K=8 gives sufficient vote diversity while costing similar total FLOPs |

## 1.5 Theoretical Foundation: How and Why the Two Methods Differ

### The Information-Theoretic View

Majority voting and cycle consistency probe fundamentally different properties of LLM reasoning. Understanding *what* each measures — and therefore *what each is blind to* — is the key to understanding why their combination is non-trivial.

**Majority Voting** measures **stochastic stability**: "If I solve this problem K times with temperature τ > 0, do I converge to the same answer?" It estimates the entropy of P(answer | question, model). When the model is confident and correct, answers cluster tightly. When the model is uncertain or wrong in a non-deterministic way, answers scatter.

**Cycle Consistency** measures **structural invertibility**: "Does the solution S preserve enough information about Q that a second model can reconstruct Q from S?" It tests whether the mapping Q → S is information-preserving. A correct solution must encode the original problem's constraints, numbers, and relationships; a structurally wrong solution may encode a *different* problem.

These are **orthogonal properties**. Stochastic stability says nothing about whether the solution encodes the right problem. Structural invertibility says nothing about whether the model would give the same answer again.

```
                    What each method actually tests
                    ─────────────────────────────────

    MV asks:   "Does P(answer | Q) have low entropy?"
               → YES = model is confident (maybe right, maybe systematically wrong)
               → NO  = model is uncertain (probably wrong)

    CC asks:   "Is the map Q → S → Q' approximately identity?"
               → YES = the solution faithfully encodes Q (maybe right, maybe computationally wrong)
               → NO  = the solution encodes a different problem (probably structurally wrong)
```

### Error Taxonomy

LLM reasoning errors decompose into three structurally distinct categories. MV and CC have different detection profiles for each:

**1. Stochastic Errors** — The model "slips" randomly: arithmetic carry mistake, skipped condition, off-by-one. Different samples at τ > 0 produce *different* wrong answers because the slip is non-deterministic.

```
    MV detection:  ██████████  STRONG — answers scatter, voting_confidence drops
    CC detection:  ████░░░░░░  WEAK   — depends on whether the specific slip breaks reconstruction
```

**2. Systematic / Structural Errors** — The model consistently applies the wrong approach: wrong theorem, misunderstood problem structure, missing case in combinatorics. All K samples converge on the *same wrong approach* with the same wrong answer.

```
    MV detection:  ░░░░░░░░░░  BLIND  — all K samples agree on wrong answer → high confidence → FP
    CC detection:  ██████░░░░  MODERATE — wrong approach often encodes a different problem
                                          (Q' ≠ Q → high hybrid_cycle → detected)
                                          BUT: some structural errors reconstruct a compatible Q'
```

**3. Computational Errors** — The model understands the problem correctly, sets up the right equations, applies the right approach, but makes a *numerical error at a late step* (wrong multiplication, sign error). The solution's narrative still perfectly describes the original problem — only the final number is wrong.

```
    MV detection:  ██████░░░░  MODERATE — arithmetic slips vary across samples, some disagreement
    CC detection:  ░░░░░░░░░░  BLIND    — solution encodes Q correctly → Q' ≈ Q → low hybrid_cycle
                                          AND: same approach → S' may reproduce same computational slip
                                          → answer_match = 1, hybrid_cycle ≈ 0 → false positive
```

Summarised:

| Error Type | Frequency (from Exp1) | MV detection | CC detection | Both miss? |
|---|---|---|---|---|
| Stochastic | Common on easy problems | ✅ Strong | ⚠️ Weak | No — MV catches it |
| Systematic / Structural | Common on hard problems | ❌ Blind | ⚠️ Moderate | Partially — CC catches some |
| Computational (late-step) | ~20% of incorrect (Exp1's "consistently wrong" cohort) | ⚠️ Moderate | ❌ Blind | Partially — MV catches some |

**The critical insight: MV is blind to systematic errors. CC is blind to computational errors. These blind spots are structurally different — they don't overlap perfectly.** This is the theoretical foundation for why fusion should work.

### The "Verification Diversity" Principle

In ensemble learning, combining classifiers works best when they have **low error correlation** (Dietterich, 2000; Krogh & Vedelsby, 1994). The same principle applies to verification signals:

- Two **voting-based** methods at different temperatures (e.g. τ=0.5 and τ=1.0) have **highly correlated errors** — both fail whenever the model systematically converges to the wrong answer, regardless of temperature.
- Two **structurally different** methods (voting + cycle) have **lower error correlation** — MV fails on systematic errors, CC fails on computational errors. These are different subsets of the incorrect samples.

```
    Redundant verification (high error correlation):
    ┌─────────────────────────┐
    │  MV τ=0.5 failures      │
    │  ┌─────────────────────┐│
    │  │  MV τ=1.0 failures  ││   ← ~80% overlap (same systematic errors)
    │  └─────────────────────┘│
    └─────────────────────────┘

    Diverse verification (low error correlation):
    ┌──────────────┐  ┌──────────────┐
    │  MV failures │  │  CC failures │
    │              ├──┤              │   ← <50% overlap (different error types)
    │ (systematic) │  │(computational│
    │              │  │  + consistent│
    │              │  │   cycles)    │
    └──────────────┘  └──────────────┘
```

The fusion doesn't just add information — it **diversifies the verification strategy across error types**. This is analogous to the distinction between bagging and stacking in ML: MV at various K values is "bagging" (same method, more runs), while MV + CC is "stacking" (different methods targeting different weaknesses).

## 1.6 The Novelty Argument

### What Has Been Done (Prior Art)

| Work | What it does | Relationship to this experiment |
|---|---|---|
| Wang et al. (2023), "Self-Consistency" | Majority voting for CoT reasoning | Our MV baseline; we reproduce and extend |
| Zhu et al. (2017), CycleGAN | Cycle consistency for image-to-image translation | Inspiration for CC, but different domain (vision vs. reasoning) |
| Back-translation in MT | Round-trip translation as quality signal | Closest NLP analogue, but not applied to LLM math reasoning |
| Li et al. (2023), "Making LLMs Better Reasoners with Step-Aware Verifier" | Learned verifier (ORM/PRM) for reasoning | Supervised approach; requires labelled data. We are self-supervised |
| Lightman et al. (2023), "Let's Verify Step by Step" | Process Reward Models | Requires human step-level labels. We need zero labels |
| Cobbe et al. (2021), GSM8K + Verifier | Separate verifier model trained on solver outputs | Supervised verifier. Our signals are label-free |

**Gap in the literature:** No prior work has (a) formalised cycle consistency as a self-supervised verification signal for LLM reasoning, (b) directly compared it to majority voting, or (c) analysed whether they are complementary.

### What This Experiment Contributes (Novelty Ladder)

The novelty of showing CC + MV > MV depends entirely on *how deeply we characterise the result*. There is a clear hierarchy:

```
    NOVELTY LEVEL                            VENUE LEVEL
    ─────────────                            ───────────
    Level 1: CC+MV > MV alone               Workshop paper (expected if CC has any signal)
         │
         ▼
    Level 2: + complementarity analysis      Short paper / poster
             (failure overlap < 50%,         (proves they detect different errors)
              Venn diagram of FPs)
         │
         ▼
    Level 3: + compute-equivalent Pareto     Solid contribution
             (MV@K=4 + CC beats MV@K=8      (practical principle for deployment:
              at same or lower cost)          "diversify, don't scale")
         │
         ▼
    Level 4: + error taxonomy explaining     Strong conceptual contribution
             WHY they're complementary        (about the structure of LLM errors,
             (stochastic vs systematic vs     not just an empirical result)
              computational decomposition)
         │
         ▼
    Level 5: + difficulty-dependent regime   Complete story
             analysis (CC contributes most    (principled guidance: when to use
              on hard problems where           each method, not just "combine
              systematic errors dominate)       everything always")
```

**To be a solid contribution, we must achieve at least Levels 2–4.** Level 1 alone is not novel. Level 5 makes the paper complete.

### The Core Novel Claim (One Sentence)

> **LLM reasoning errors have structure — they decompose into stochastic, systematic, and computational components — and combining verification methods that target different components (majority voting for stochastic, cycle consistency for systematic) yields a more compute-efficient correctness signal than scaling either method alone.**

This is novel because:

1. **It's not just "A+B > A"** — it's a statement about the **geometry of LLM errors**.
2. **It reframes verification as a coverage problem** — you're not looking for the "best" single method, you're looking for the *minimum set of methods that covers all error types*.
3. **It provides a practical design principle** — "Don't scale one signal. Diversify your verification strategy." This is actionable: instead of spending 16× compute on K=16 majority votes, spend 4× on MV + 3.5× on CC + 0.5× on fusion = 8× total for better results.
4. **It connects to a theoretical framework** (ensemble diversity / error decorrelation) that has well-understood foundations in classical ML.

### What Would Make It Weak vs. Strong

| If we find... | Then the contribution is... | Because... |
|---|---|---|
| Fusion improves by < 2 AUROC points | Weak / incremental | "Statistically significant but practically irrelevant" |
| Failure overlap > 70% | Weak | Methods are mostly redundant, not complementary |
| MV at K=16 beats fusion | Weak (practically) | "Just sample more" defeats our argument |
| Fusion improves by ≥ 3 AUROC, overlap < 50% | Moderate / solid | Complementarity is real and measurable |
| MV@K=4 + CC beats MV@K=8 at same cost | Strong | Pareto improvement — can't be dismissed as "just add compute" |
| All of the above + difficulty-dependent regime shift | Very strong | Complete narrative: *when*, *why*, and *how much* each contributes |

### The Compute-Efficiency Argument (Key to Novelty)

The strongest version of this paper is NOT about absolute AUROC. A reviewer can always say "just use K=32." The strongest version is about **compute-optimal verification**:

```
    Budget = 8 solver-equivalent forward passes

    Option A:  MV(K=8)                         → AUROC = X
    Option B:  MV(K=4) + Cycle Consistency     → AUROC = X + Δ   (cost ≈ 4 + 3.5 = 7.5)
                                                                    ↑ CHEAPER AND BETTER

    This is a Pareto improvement: better results at lower cost.
    It cannot be countered by "just use more K."
```

If we can demonstrate this, the contribution shifts from "we combined two things and it helped" (boring) to "**there exist verification strategies that dominate pure scaling on a cost-accuracy Pareto frontier**" (interesting). This is the argument that majority voting is a *local optimum* in verification strategy space, and the Pareto frontier includes structurally diverse methods.

### Connection to the Broader "Self-Supervised Verification" Research Agenda

This experiment fits into a larger question that the field is grappling with:

> *How do you verify LLM outputs without ground truth, without human labels, and without a stronger model?*

Current approaches (self-consistency, reward models, constitutional AI) each make different assumptions. Our contribution is:

1. **Empirical:** Cycle consistency works as a self-supervised signal (Exp 1, already shown).
2. **Comparative:** It detects different errors than majority voting (Exp 2, this experiment).
3. **Practical:** Combining them is compute-efficient (Exp 2, the Pareto argument).
4. **Forward-looking:** The best combined signal becomes the reward for self-improvement via GRPO (Exp 3, future work).

The narrative is: *Self-supervised verification is not a single method but a portfolio. Different methods cover different error types. The optimal portfolio depends on problem difficulty and compute budget.*

---

# 2. METHOD OVERVIEW

## 2.1 Shared Components

Both methods share the same solver, datasets, and evaluation:

```
Solver model      :  Qwen/Qwen2.5-3B-Instruct  (frozen, local vLLM)
Datasets          :  GSM8K (1000) · MATH (500) · OlympiadBench (300) = 1800 total
Answer extraction :  \boxed{} → regex fallback → last number
Correctness label :  answers_equivalent(answer_A, ground_truth)
```

Both methods output a scalar **confidence score** per sample. We evaluate these scores as predictors of the binary `correct` label.

## 2.2 What Differs

| | Majority Voting | Cycle Consistency |
|---|---|---|
| Additional models | None (reuses solver) | Reconstructor (Qwen3-8B-Base) + Embedder (Qwen3-Embedding-4B) |
| Temperature | τ > 0 (stochastic sampling) | τ = 0 (greedy) |
| # forward passes | K per question | 3 per question (greedy solve + reconstruct + greedy solve) |
| Output signal | Voting confidence ∈ [0, 1] | combined_reward = answer_match − hybrid_cycle |

---

# 3. PIPELINE DIAGRAMS

## 3.1 Majority Voting Pipeline

```
                         ┌──────────────────────────────────────────────────┐
                         │          MAJORITY VOTING  (per question Q)       │
                         └──────────────────────────────────────────────────┘

                                         Q (question)
                                             │
                      ┌──────────────────────┼──────────────────────┐
                      │                      │                      │
                      ▼                      ▼                      ▼
               ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
               │  Solver(Q)  │       │  Solver(Q)  │  ...  │  Solver(Q)  │
               │  τ > 0      │       │  τ > 0      │       │  τ > 0      │
               │  sample 1   │       │  sample 2   │       │  sample K   │
               └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
                      │                      │                      │
                      ▼                      ▼                      ▼
                    S_1                    S_2          ...        S_K
                      │                      │                      │
                      ▼                      ▼                      ▼
               ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
               │  Extract    │       │  Extract    │       │  Extract    │
               │  Answer     │       │  Answer     │       │  Answer     │
               └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
                      │                      │                      │
                      ▼                      ▼                      ▼
                    A_1                    A_2          ...        A_K
                      │                      │                      │
                      └──────────────────────┼──────────────────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │   MAJORITY VOTE     │
                                  │                     │
                                  │  A_maj = mode(A_i)  │
                                  │  count = #{i: A_i   │
                                  │          == A_maj}   │
                                  └──────────┬──────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              │                             │
                              ▼                             ▼
                 ┌──────────────────────┐     ┌──────────────────────┐
                 │  voting_confidence   │     │  majority_answer     │
                 │  = count / K         │     │  = A_maj             │
                 │                      │     │                      │
                 │  (continuous signal   │     │  (used as the        │
                 │   in [1/K, 1])       │     │   "best guess")      │
                 └──────────────────────┘     └──────────────────────┘
```

**Signals produced per sample:**

| Signal | Formula | Range | Direction |
|---|---|---|---|
| `voting_confidence` | count(A_maj) / K | [1/K, 1] | Higher → more likely correct |
| `majority_answer` | mode(A_1, …, A_K) | string | The consensus answer |
| `majority_correct` | answers_equivalent(A_maj, ground_truth) | {0, 1} | Correctness of the consensus |
| `entropy` | −Σ p_i log(p_i) over answer clusters | [0, log(K)] | Lower → more agreement → more likely correct |
| `unique_ratio` | # distinct answers / K | [1/K, 1] | Lower → more agreement |

## 3.2 Cycle Consistency Pipeline

```
                         ┌──────────────────────────────────────────────────┐
                         │       CYCLE CONSISTENCY  (per question Q)        │
                         └──────────────────────────────────────────────────┘

                                         Q (question)
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │   Solver(Q)     │
                                    │   τ = 0 (greedy)│
                                    └────────┬────────┘
                                             │
                                             ▼
                                          S (solution)
                                             │
                              ┌──────────────┴──────────────┐
                              │                             │
                              ▼                             ▼
                    ┌──────────────────┐          ┌─────────────────┐
                    │  Reconstructor   │          │  Extract        │
                    │  (Qwen3-8B-Base) │          │  Answer → A     │
                    │  S → Q'          │          └─────────────────┘
                    └────────┬─────────┘
                             │
                             ▼
                          Q' (reconstructed question)
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
       ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
       │ Solver(Q')   │  │ Embed(Q, Q')     │  │ number_jaccard   │
       │ τ = 0        │  │ → question_cycle  │  │ + chrf_dist      │
       └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘
              │                   │                      │
              ▼                   └──────────┬───────────┘
           S' (second solution)              │
              │                              ▼
              ▼                   ┌──────────────────────┐
       ┌──────────────┐          │  hybrid_cycle         │
       │ Extract      │          │  = 0.4·embed          │
       │ Answer → A'  │          │  + 0.4·num_jaccard    │
       └──────┬───────┘          │  + 0.2·chrF           │
              │                  └──────────┬────────────┘
              ▼                             │
       ┌──────────────────────┐             │
       │ answer_match         │             │
       │ = equiv(A, A')       │             │
       └──────────┬───────────┘             │
                  │                         │
                  └────────────┬────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  combined_reward     │
                    │  = answer_match      │
                    │    − hybrid_cycle    │
                    └──────────────────────┘
```

**Signals produced per sample:**

| Signal | Formula | Range | Direction |
|---|---|---|---|
| `hybrid_cycle` | 0.4·question_cycle + 0.4·number_jaccard + 0.2·chrf_dist | [0, 1] | Lower → more likely correct |
| `answer_match` | equiv(A, A') | {0, 1} | Higher → more likely correct |
| `combined_reward` | answer_match − hybrid_cycle | [−1, 1] | Higher → more likely correct |

## 3.3 Comparison & Fusion Pipeline

```
                         ┌──────────────────────────────────────────────────┐
                         │           COMPARISON & FUSION LAYER              │
                         └──────────────────────────────────────────────────┘

          Majority Voting                              Cycle Consistency
        ┌───────────────────┐                       ┌───────────────────┐
        │ voting_confidence │                       │ combined_reward   │
        │ entropy           │                       │ hybrid_cycle      │
        │ unique_ratio      │                       │ answer_match      │
        └────────┬──────────┘                       └─────────┬────────┘
                 │                                            │
                 └──────────────────┬─────────────────────────┘
                                    │
                      ┌─────────────┼─────────────┐
                      │             │             │
                      ▼             ▼             ▼
              ┌──────────────┐ ┌─────────────┐ ┌──────────────────────┐
              │  Head-to-    │ │ Failure     │ │ Fusion scores        │
              │  Head AUROC  │ │ Overlap     │ │                      │
              │  comparison  │ │ Analysis    │ │ fused = α·vote_conf  │
              │  per dataset │ │ (Venn       │ │   + (1−α)·comb_rew   │
              └──────────────┘ │  diagram)   │ │                      │
                               └─────────────┘ │ + logistic ensemble  │
                                               └──────────────────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────────┐
                                               │  Final evaluation:   │
                                               │  AUROC, AUPRC,       │
                                               │  selective prediction │
                                               │  per dataset          │
                                               └──────────────────────┘
```

---

# 4. EXPERIMENTAL DESIGN

## 4.1 Data Reuse from Experiment 1

Experiment 1 already produced the cycle consistency signals for 1800 samples. We **reuse** the existing results from `exp1/outputs_v2/results.jsonl` and only generate new data for the majority voting arm.

| Step | Cycle Consistency | Majority Voting |
|---|---|---|
| Dataset loading | Reuse exp1 | Same 1800 questions |
| Greedy solve (S) | Reuse exp1 step 1 | — |
| K stochastic solves | — | **New**: K solves per question |
| Reconstruction (Q') | Reuse exp1 step 2 | — |
| Second solve (S') | Reuse exp1 step 3 | — |
| Embeddings | Reuse exp1 step 4 | — |
| Metrics | Reuse exp1 results | **New**: voting signals |

## 4.2 Majority Voting Configuration

| Parameter | Value | Rationale |
|---|---|---|
| K (# samples) | **16** | Generate once, subsample to K ∈ {2, 4, 8, 16} for cost analysis |
| Temperature | τ = 0.7 | Standard for diversity without degeneration; ablate τ ∈ {0.5, 0.7, 1.0} |
| Top-p | 0.95 | Nucleus sampling |
| max_new_tokens | 2048 | Same as exp1 |
| Solver model | Qwen/Qwen2.5-3B-Instruct | Same as exp1 |

**Key design choice:** Generate all K=16 samples upfront. To evaluate K=4, take the first 4. This ensures all K values use nested subsets of the same samples, making comparisons fair and eliminating stochastic variance between K conditions.

## 4.3 Answer Clustering for Majority Voting

Raw extracted answers need clustering before counting votes, since equivalent answers may appear in different formats (e.g., "18", "18.0", "\\$18", "18 dollars"):

```
Step 1: Extract answer from each S_i using boxed → regex → last-number fallback
Step 2: Group answers into equivalence classes using answers_equivalent()
Step 3: Count cluster sizes → the largest cluster is the majority
Step 4: Compute voting_confidence = |largest cluster| / K
```

**Tie-breaking:** If two clusters tie for majority, use the one whose members appeared earliest (lowest sample index). This is arbitrary but deterministic.

## 4.4 Signals to Compare

### Primary Signals (one from each method):

| Signal | Method | Range | Description |
|---|---|---|---|
| `voting_confidence` | Majority Voting | [1/K, 1] | Fraction of samples agreeing on majority answer |
| `combined_reward` | Cycle Consistency | [−1, 1] | answer_match − hybrid_cycle |

### Secondary Signals (alternative formulations):

| Signal | Method | Formula |
|---|---|---|
| `entropy` | MV | −Σ p_i log(p_i) across answer clusters |
| `unique_ratio` | MV | # distinct answer clusters / K |
| `answer_match` | CC | Binary: A == A' |
| `hybrid_cycle` | CC | 0.4·embed + 0.4·num_jaccard + 0.2·chrF |

### Fusion Signals:

| Signal | Formula | Rationale |
|---|---|---|
| `fused_linear` | α · voting_confidence + (1−α) · combined_reward | Simple weighted combination; sweep α ∈ [0, 1] |
| `fused_product` | voting_confidence × (1 + combined_reward) / 2 | Multiplicative: both must agree |
| `fused_logistic` | Logistic regression on [voting_conf, combined_reward, hybrid_cycle, entropy] | Learned combination; train on 70% of data, evaluate on 30% |

---

# 5. EVALUATION PLAN

## 5.1 Primary Metrics

For every signal, compute:

| Metric | Description | Use |
|---|---|---|
| **AUROC** → `correct` | Area under ROC curve predicting solver correctness | Primary comparison metric |
| **AUPRC** → `correct` | Area under precision-recall curve | Important when classes are imbalanced (e.g. OlympiadBench: 24% correct) |
| **Selective Prediction** | Accuracy when keeping only top-P% most confident samples | Practical deployment metric |
| **Cohen's d** | Effect size of correct vs incorrect groups | Measures separation strength |

## 5.2 Analysis Dimensions

### A — Head-to-Head by Dataset

| | GSM8K (easy) | MATH (medium) | OlympiadBench (hard) | All |
|---|---|---|---|---|
| Voting AUROC | ? | ? | ? | ? |
| Cycle AUROC | 0.675 | 0.676 | 0.668 | 0.679 |
| Winner | ? | ? | ? | ? |

**Cycle consistency numbers are already known from exp1 v2.** This table gets filled in after running majority voting.

### B — Complementarity / Failure Overlap

For each method, define "failure" as: method assigns high confidence but solver is actually wrong (false positive).

```
                    ┌───────────────────────────────────────┐
                    │         ALL INCORRECT SAMPLES          │
                    │              (N ≈ 793)                 │
                    │                                       │
                    │    ┌─────────┐       ┌─────────┐     │
                    │    │  MV     │       │  CC     │     │
                    │    │  false  │       │  false  │     │
                    │    │ positive│       │ positive│     │
                    │    │         ├───┐   │         │     │
                    │    │         │ O │   │         │     │
                    │    │         │ V │   │         │     │
                    │    │         │ E │   │         │     │
                    │    │         │ R │   │         │     │
                    │    │         │ L │   │         │     │
                    │    │         │ A │   │         │     │
                    │    │         │ P │   │         │     │
                    │    │         ├───┘   │         │     │
                    │    └─────────┘       └─────────┘     │
                    │                                       │
                    └───────────────────────────────────────┘

Key question: How large is OVERLAP?
  - Large overlap  → methods are redundant
  - Small overlap  → methods are complementary → fusion will help
```

Quantify with:
- **Overlap coefficient** = |FP_MV ∩ FP_CC| / min(|FP_MV|, |FP_CC|)
- **Jaccard(FP_MV, FP_CC)** = |FP_MV ∩ FP_CC| / |FP_MV ∪ FP_CC|

### C — Cost-Accuracy Tradeoff

```
            Accuracy
              ▲
              │              ╭───── Fusion (MV + CC)
              │         ╭───╯
              │    ╭───╯  ● Cycle Consistency (fixed cost = 3 passes)
              │   ╱
              │  ╱
              │ ╱
              │╱  ● MV K=2   ● K=4   ● K=8   ● K=16
              │
              └────────────────────────────────────────► FLOPs / question
                    2x    3x    4x         8x        16x

Goal: Plot this curve to find the compute-optimal strategy.
```

Approximate cost model:

| Method | Forward passes | Which models | Relative cost (solver = 1.0) |
|---|---|---|---|
| MV K=2 | 2 × solver | Solver only | 2.0 |
| MV K=4 | 4 × solver | Solver only | 4.0 |
| MV K=8 | 8 × solver | Solver only | 8.0 |
| MV K=16 | 16 × solver | Solver only | 16.0 |
| Cycle Consistency | 1 solver + 1 reconstructor + 1 solver + embed | Solver + Reconstructor + Embedder | ~3.5 (depends on model sizes) |
| Fusion (MV K=8 + CC) | 8 + 3.5 | All | 11.5 |

### D — K-Sensitivity Curve

```
            AUROC
              ▲
        0.80  │                          ●───────── K=16
              │                     ●
        0.70  │                ●
              │           ●
        0.60  │      ●
              │
        0.50  │──────────────────── random baseline
              └──────────────────────────────────► K
                  1    2    4    8   12   16

Goal: Find the "elbow" — the K beyond which adding more samples
gives diminishing returns. Compare against cycle consistency's
fixed AUROC (horizontal line).
```

### E — Per-Difficulty Analysis

Plot AUROC as a function of problem difficulty:

```
                  Easy (GSM8K)    Medium (MATH)    Hard (OlympiadBench)
                 ┌──────────┐   ┌──────────┐     ┌──────────┐
                 │ MV wins? │   │ Tie?     │     │ CC wins? │
                 │          │   │          │     │          │
                 │ High     │   │ Moderate │     │ Low      │
                 │ stoch.   │   │ stoch.   │     │ stoch.   │
                 │ variance │   │ variance │     │ variance │
                 └──────────┘   └──────────┘     └──────────┘

Hypothesis: MV's advantage shrinks as difficulty increases, because
harder problems produce more systematic (less stochastic) errors.
```

---

# 6. WHY EACH HYPOTHESIS IS EXPECTED

## H1: MV wins on GSM8K

GSM8K problems are straightforward arithmetic word problems. When Qwen2.5-3B gets them wrong, it's often due to a stochastic slip (carry error, wrong operation) rather than a fundamental misunderstanding. Different samples at τ=0.7 will "slip" differently, so wrong answers scatter while correct answers converge. This is exactly what majority voting exploits.

Cycle consistency on GSM8K had AUROC 0.675 in v2 — already decent, but limited by the "consistently wrong" problem (20.9% of incorrect samples had answer_match=1). MV should do better here because its K independent samples are less likely to all agree on the same wrong answer.

## H2: CC wins on OlympiadBench

OlympiadBench problems require multi-step mathematical reasoning. When the model fails, it typically fails *structurally* — wrong approach, wrong theorem, missing case. This structural failure is **deterministic**: re-sampling at τ=0.7 produces the same wrong approach with slightly different wording. All K samples agree on the wrong answer → voting_confidence is high → false positive.

Cycle consistency catches these through a different channel: the wrong solution encodes a wrong question. Even if S and S' both produce the same wrong answer, Q' may diverge from Q — the hybrid_cycle detects the structural break.

## H3: Low overlap of failure cases

MV's false positives occur when the model **systematically** produces the same wrong answer (all K agree). CC's false positives occur when the wrong solution **consistently reconstructs** a compatible question (answer_match=1 and low hybrid_cycle). These are correlated but not identical: a systematic wrong answer doesn't always reconstruct faithfully, and a faithful reconstruction doesn't require all K samples to agree.

## H4: Fusion beats both

If H3 holds (overlap < 60%), a simple combination captures the union of both methods' true positives while filtering the intersection of their false positives. Even α=0.5 should help, and learned weights (logistic regression) should do better.

## H5: MV at K=8 matches CC

Cycle consistency uses ~3.5 solver-equivalent passes. MV at K=8 uses 8 solver passes — about 2.3× the cost. But MV's signal (vote agreement) is directly about the answer, while CC's signal is indirectly about the question. The direct signal at 2.3× the cost should match or exceed the indirect signal.

---

# 7. IMPLEMENTATION PLAN (for future coding phase)

## 7.1 New Code Required

| File | Purpose |
|---|---|
| `exp2/run.py` | Main pipeline: load exp1 data, generate K samples, compute voting signals, compute fusion, evaluate |
| `exp2/voting.py` | Majority voting logic: answer clustering, vote counting, confidence computation, entropy |
| `exp2/fusion.py` | Fusion strategies: linear combination, product, logistic regression |
| `exp2/config.py` | K values, temperature, fusion weights |
| `exp2/visualize.py` | Comparison plots: AUROC bars, K-sensitivity curve, Venn diagrams, cost-accuracy tradeoff |

## 7.2 Reused from Experiment 1

| Component | Source | What it provides |
|---|---|---|
| Cycle consistency signals | `exp1/outputs_v2/results.jsonl` | `combined_reward`, `hybrid_cycle`, `answer_match`, `correct` for all 1800 samples |
| Solver model | `exp1/models.py` | Same Solver class with vLLM |
| Answer extraction | `exp1/utils.py` | `extract_boxed_answer`, `extract_answer`, `answers_equivalent` |
| Metrics computation | `exp1/metrics.py` | AUROC, AUPRC, Cohen's d |

## 7.3 Data Flow

```
exp1/outputs_v2/results.jsonl  ──────────────────────┐
    (1800 samples with cycle consistency signals)     │
                                                      │
                                                      ▼
                                            ┌──────────────────┐
exp2/run.py  ──  Step 1: Load exp1 data ──► │  Merged Dataset  │
                                            │  (1800 samples)  │
                 Step 2: Generate K=16   ──►│  + K solutions   │
                         stochastic         │  + voting signals│
                         solutions          └────────┬─────────┘
                                                     │
                 Step 3: Extract answers,            │
                         cluster, vote    ──────────►│
                                                     │
                 Step 4: Compute fusion   ──────────►│
                         signals                     │
                                                     ▼
                 Step 5: Evaluate         ──► exp2/outputs/results.jsonl
                         (AUROC, AUPRC,       exp2/outputs/plots/
                          selective pred)
```

---

# 8. EXPECTED OUTPUTS

## 8.1 Tables

1. **Main comparison table**: AUROC/AUPRC for voting_confidence vs combined_reward vs fused, per dataset
2. **K-sensitivity table**: AUROC at K ∈ {2, 4, 8, 16} with cycle consistency as a reference line
3. **Selective prediction table**: Accuracy at coverage 10%, 20%, 50% for each method
4. **Failure overlap matrix**: Counts of shared vs unique false positives/negatives
5. **Cost table**: FLOPs per question vs AUROC per method

## 8.2 Plots

1. **AUROC bar chart**: Side-by-side bars for MV, CC, and Fusion, per dataset
2. **K-sensitivity curve**: AUROC vs K, with CC as a horizontal reference line
3. **Cost-accuracy Pareto curve**: AUROC vs compute cost for all configurations
4. **Venn diagram**: False positive overlap between MV and CC
5. **Distribution plots**: voting_confidence distributions split by correct/incorrect
6. **Scatter plot**: voting_confidence vs combined_reward, colored by correct/incorrect (to visualise complementarity)
7. **ROC curves**: Overlaid ROC curves for all signals on the same axes
8. **Fusion weight sweep**: AUROC vs α for fused_linear

## 8.3 Deliverable

A report in `exp2/outputs/results_report.md` answering RQ1–RQ5 with the data from the tables and plots above.

---

# 9. RISKS AND MITIGATIONS

| Risk | Impact | Mitigation |
|---|---|---|
| **Stochastic sampling produces near-identical solutions** at τ=0.7 for easy problems | MV signal collapses: all K agree, voting_confidence ≈ 1 for both correct and incorrect | Ablate τ ∈ {0.5, 0.7, 1.0}; report entropy as secondary signal |
| **Answer extraction failures inflate unique_ratio** — different formatting of the same answer counted as different | MV appears better than it is | Use `answers_equivalent()` for clustering, not exact string match |
| **K=16 is too expensive for OlympiadBench** (300 × 16 = 4800 long generations) | Timeout / GPU hours | OlympiadBench solutions can be very long. Cap at max_new_tokens=2048; accept partial solutions |
| **Cycle consistency's `combined_reward` depends on answer extraction quality** — biased comparison | CC is handicapped by regex failures | Report both regex-based and LLM-judge-based answer_match for CC (from exp1 v2 analysis) |
| **Logistic fusion overfits on 1800 samples** | Inflated fusion AUROC | Use 5-fold cross-validation; report mean ± std AUROC across folds |

---

# 10. SUCCESS CRITERIA

| Criterion | Threshold | What it would mean |
|---|---|---|
| MV at K=8 outperforms CC by ≥ 5 AUROC points overall | AUROC_MV − AUROC_CC ≥ 0.05 | Majority voting is the clear winner for self-verification at comparable cost |
| CC outperforms MV on OlympiadBench by ≥ 5 AUROC points | AUROC_CC − AUROC_MV ≥ 0.05 | Cycle consistency has unique value on hard problems (supports H2) |
| Fusion outperforms both by ≥ 3 AUROC points | AUROC_fused − max(AUROC_MV, AUROC_CC) ≥ 0.03 | The methods are complementary; combining them is worth the extra cost |
| Failure overlap < 50% | Jaccard(FP_MV, FP_CC) < 0.50 | The methods catch different types of errors → strong motivation for fusion |
| Selective prediction at 10% coverage reaches ≥ 90% accuracy | best_method accuracy@10% ≥ 0.90 | The combined system is practically useful for high-stakes deployment |

---

# 11. CONNECTION TO THE BROADER RESEARCH ARC

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                        RESEARCH ARC                              │
  │                                                                  │
  │  Exp 1  ──►  Does cycle consistency correlate with correctness?  │
  │              Result: YES (AUROC 0.679–0.708)                     │
  │                         │                                        │
  │                         ▼                                        │
  │  Exp 2  ──►  Is cycle consistency better or worse than the       │
  │  (THIS)      simplest alternative (majority voting)?             │
  │              And can they be combined?                            │
  │                         │                                        │
  │                         ▼                                        │
  │  Exp 3  ──►  Use the best self-supervised signal as a reward     │
  │  (FUTURE)    for GRPO training. Does it match ground-truth       │
  │              reward? Does it generalise to OOD domains?          │
  └──────────────────────────────────────────────────────────────────┘
```

Experiment 2 is the **calibration step**: before investing in training (Exp 3), we need to know which self-supervised signal to use as the reward. If majority voting dominates, we use vote confidence as the GRPO reward. If cycle consistency wins on hard problems, we use combined_reward. If fusion wins, we use the fused signal. The answer determines the reward function for Exp 3.

---

*End of Experiment 2 Plan*
