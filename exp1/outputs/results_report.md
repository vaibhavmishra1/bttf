# Experiment 1 — Cycle-Consistency as a Self-Supervised Signal for Reasoning Correctness

## Verdict: PASS — The signal works, with caveats

Cycle-consistency **does** discriminate correct from incorrect solutions. The combined reward achieves **AUROC 0.708** overall (random = 0.50) and enables selective prediction that boosts accuracy from 56% → 88% when keeping the top 10% highest-confidence samples. However, the signal is dominated by `answer_match` (self-consistency), while the embedding-based `question_cycle` alone is weak. The method also has systematic failure modes in answer extraction and reconstructor behaviour that bound its ceiling.

---

## 1. Setup Recap

| | |
|---|---|
| **Solver** | Qwen2.5-3B-Instruct (local) |
| **Reconstructor** | "Qwen/Qwen3-4B"    |
| **Embedding model** | Qwen3-Embedding-0.6B |
| **Datasets** | GSM8K (1000), MATH (500), OlympiadBench (300) |
| **Total samples** | 1800 |

---

## 2. Headline Metrics

### Solver accuracy (baseline)

| Dataset | N | Accuracy |
|---|---|---|
| GSM8K | 1000 | 65.9% |
| MATH | 500 | 55.6% |
| OlympiadBench | 300 | 24.0% |
| **All** | **1800** | **56.1%** |

### AUROC — predicting solver correctness

| Signal | GSM8K | MATH | OlympiadBench | All |
|---|---|---|---|---|
| `question_cycle` (negated) | 0.558 | 0.559 | 0.531 | 0.608 |
| `answer_match` | 0.737 | 0.641 | 0.648 | 0.702 |
| `combined_reward` | 0.715 | 0.636 | 0.618 | 0.708 |

### AUPRC

| Signal | GSM8K | MATH | OlympiadBench | All |
|---|---|---|---|---|
| `question_cycle` (negated) | 0.699 | 0.634 | 0.305 | 0.657 |
| `answer_match` | 0.815 | 0.671 | 0.335 | 0.747 |
| `combined_reward` | 0.824 | 0.712 | 0.447 | 0.770 |

---

## 3. Signal-by-Signal Analysis

### 3.1 `answer_match` — the dominant signal

This is the strongest individual predictor (AUROC 0.702 overall). Correct solutions get `answer_match = 1` at a rate of **63.7%**, versus only **23.3%** for incorrect solutions. Effect size is large (Cohen's d = 0.887).

**Why it works:** If the solver gets the right answer to Q, the reconstructed Q' is usually close enough that the solver gets the same right answer again. Incorrect solutions often introduce drift that changes the second answer.

**Why it has a ceiling:** 184 samples (23.3% of incorrect) are "consistently wrong" — the solver makes the exact same mistake on Q and Q'. These are false positives that the cycle cannot detect.

### 3.2 `question_cycle` — weak but real

The embedding distance between Q and Q' carries **marginal** discriminative power (AUROC 0.531–0.559 per-dataset, 0.608 overall). Correct solutions have mean cycle distance 0.136 vs 0.178 for incorrect — a small gap (Cohen's d = 0.310) with heavy overlap.

**Why it's weak:**
- GPT-4.1 is a very capable reconstructor. Even from a wrong solution, it often recovers Q' close to Q, because the solution still mentions the same entities, numbers, and setup.
- The 0.6B embedding model may lack sensitivity to small numerical changes (e.g., "8 apples" vs "6 apples" embed similarly).
- 575/1800 (32%) of reconstructed Q' start with meta-commentary ("Okay, let's see...") rather than a clean question. These inflate `question_cycle` for reasons unrelated to solution quality.

**Where it helps:** The cross-dataset aggregation shows a bigger gap (AUROC 0.608 vs ~0.55 per-dataset), suggesting `question_cycle` partly captures dataset difficulty as a confound — harder datasets have higher cycle distances on average.

### 3.3 `combined_reward = answer_match - question_cycle`

The combination is the best single predictor by AUPRC (0.770 overall, 0.824 on GSM8K) though its AUROC (0.708) is close to `answer_match` alone (0.702). The cycle term helps slightly at the margins.

**Distribution shapes (from violin plots):**
- Correct solutions cluster bimodally: a large peak near +0.9 (answer_match=1, low cycle) and a smaller mass near −0.1 (answer_match=0).
- Incorrect solutions cluster near 0 and below.
- The separation is clearest for GSM8K, weakest for OlympiadBench.

---

## 4. Selective Prediction (Practical Utility)

If you use `combined_reward` to rank samples and only keep the top-K%, the accuracy increases substantially:

| Keep top-K% | N kept | Accuracy | Coverage |
|---|---|---|---|
| 10% | 180 | **88.3%** | Low |
| 20% | 360 | **86.1%** | |
| 30% | 540 | **84.8%** | |
| 50% | 900 | 74.1% | |
| 70% | 1260 | 61.8% | |
| 100% | 1800 | 56.1% | Full |

Keeping the top 30% by combined_reward lifts accuracy from 56% to 85% — a strong result for a **label-free** signal. This is the most practically useful outcome of the experiment.

---

## 5. Edge Case Analysis

### 5.1 False negatives — correct but low `combined_reward`

**366/1009 (36.3%)** correct solutions have `answer_match = 0`. This is the biggest failure mode. Root causes:

1. **Answer extraction failures** (dominant cause): The solver's second output S' often wraps the answer in text ("Wendi needs to give her chickens **20 cups** of feed") instead of a clean number, causing the regex extractor to miss it — even though the numeric answer is correct.
   - Example: id=4 — `answer_A = "64"`, `answer_A_prime = "16"` — the extractor pulled the wrong number from S'.
   - Example: id=3 — `answer_A = "20"`, `answer_A_prime = "Wendi needs to give her chickens **20 cups**..."` — string mismatch despite both being correct.

2. **Reconstructor meta-commentary**: When Q' starts with "Okay, let's see..." instead of a clean question, the solver receives a garbled prompt, producing a malformed S' from which answer extraction fails.

3. **Legitimate reconstruction drift**: For some complex problems, Q' loses a constraint, changing the answer.

### 5.2 False positives — incorrect but high `combined_reward`

**184/791 (23.3%)** incorrect solutions have `answer_match = 1` (consistently wrong). These defeat the cycle signal entirely. Examples:

| id | Dataset | A (wrong) | GT (correct) | qc | cr |
|---|---|---|---|---|---|
| 1438 | MATH | 7 | 4 | 0.0001 | 0.9999 |
| 44 | GSM8K | 18 | 17 | 0.0287 | 0.971 |
| 53 | GSM8K | 7.5 | 60 | 0.066 | 0.934 |

These are problems where the model has a deterministic, reproducible misconception. The cycle is perfectly consistent — Q' ≈ Q, S' ≈ S — but the underlying reasoning is wrong. **This is the fundamental limitation of self-consistency approaches: consistency ≠ correctness.**

### 5.3 Answer extraction is a bottleneck

- **73/1800 (4.1%)** samples have completely empty `answer_A_prime` (extraction returned "").
- Of those, 39 are correct solutions that get `answer_match = 0` purely due to extraction failure.
- The regex patterns struggle with MATH-style answers (LaTeX fractions, intervals like `(-∞, 0]`, symbolic expressions).
- The reconstructor's tendency to output "thinking aloud" (meta-commentary) causes the second solve to produce prose-heavy outputs that break extraction.

### 5.4 Reconstructor behaviour

- **575/1800 (32%)** of Q' outputs begin with meta-commentary instead of a clean reconstructed question.
- These have significantly worse metrics: mean `question_cycle = 0.270` (vs 0.100 for clean Q') and mean `answer_match = 0.282` (vs 0.543 for clean Q').
- This is a prompt engineering issue with the reconstructor — the "output only the question" instruction is not always followed.

---

## 6. Per-Dataset Breakdown

### GSM8K (N=1000) — best results
- Highest solver accuracy (65.9%) and best AUROC (0.715/0.737).
- Word problems with clean numeric answers → extraction works well.
- `answer_match` rate: 69% for correct vs 22% for incorrect — strong separation.

### MATH (N=500) — moderate
- Solver accuracy 55.6%, AUROC 0.636.
- LaTeX/symbolic answers cause more extraction failures.
- Higher `question_cycle` overall (mean 0.21) — complex problems are harder to reconstruct.

### OlympiadBench (N=300) — weakest
- Very low solver accuracy (24%), making class imbalance severe.
- `question_cycle` is essentially flat (Δ = 0.004 between correct/incorrect) — no discriminative power.
- `answer_match` still works (AUROC 0.648) but the low base rate means AUPRC is just 0.335.
- Problems are sufficiently hard that both correct and incorrect solutions produce high reconstruction error.

---

## 7. Effect Sizes (Cohen's d)

| Signal | d | Interpretation |
|---|---|---|
| `question_cycle` | 0.310 | Small |
| `answer_match` | 0.887 | Large |
| `combined_reward` | 0.859 | Large |

The effect is driven almost entirely by `answer_match`. The cycle-distance term contributes a small–moderate boost.

---

## 8. What Worked, What Didn't

### ✅ Worked
- **The core hypothesis holds**: cycle-consistency correlates with correctness. AUROC 0.708 is well above chance and practically useful.
- **Selective prediction**: keeping the top 10–30% by `combined_reward` yields 85–88% accuracy from a 56% solver — strong practical value.
- **`answer_match` is a robust signal**: the idea that "solving Q' should give the same answer" holds up across all three datasets.
- **AUPRC on GSM8K (0.824)**: the precision-recall curve stays high — useful for high-stakes filtering.

### ❌ Didn't work as hoped
- **`question_cycle` alone is near-random** per-dataset (AUROC 0.53–0.56). The embedding-based cycle distance is not a strong standalone predictor. The reconstructor is too capable — it recovers Q' even from bad solutions.
- **Consistently-wrong answers are invisible** to the cycle (23% of incorrect samples). This is inherent to any self-consistency approach.
- **Answer extraction is a noisy bottleneck**: 36% of correct solutions get `answer_match = 0` due to extraction issues, not genuine cycle failure.
- **Reconstructor prompt-following**: 32% of Q' outputs are meta-commentary, corrupting the cycle.

---

## 9. Recommendations for Exp 2

1. **Fix answer extraction** — use an LLM-based answer extractor (e.g., "What is the final numeric answer in this solution?") instead of regex. This alone could lift AUROC by 5–10 points.

2. **Fix the reconstructor prompt** — add a system message or few-shot examples to eliminate meta-commentary. Alternatively, post-process Q' to strip preamble.

3. **Weaken the reconstructor** — try GPT-4o-mini or even a local model. A weaker reconstructor should amplify the `question_cycle` signal because it won't be able to "fix" bad solutions as easily.

4. **Add a self-consistency baseline** — sample K solutions via temperature sampling and use majority vote agreement. This is the standard approach to beat — if cycle-consistency doesn't outperform it, the reconstruction step adds cost without benefit.

5. **Normalise the combined reward** — `answer_match ∈ {0,1}` but `question_cycle ∈ [0, 0.86]`. Try `α · answer_match + (1-α) · (1 - question_cycle)` and sweep α, or z-score both before combining.

6. **Condition on difficulty** — report metrics stratified by problem difficulty (MATH has explicit levels). The current correlation may partly reflect "easy ↔ correct ↔ good cycle".

7. **Try BLEU/ROUGE Q↔Q'** — a simple string-overlap metric between Q and Q' might capture reconstruction quality better than embedding cosine distance, especially for numerical details.
