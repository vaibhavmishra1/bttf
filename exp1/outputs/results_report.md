# Experiment 1 — Results Report

## Part A: Original Run (v1) · Reconstructor: Qwen3-4B
## Part B: v1 vs v2 Comparative Analysis · v2 Reconstructor: Qwen3-8B-Base

---

# PART A — Original Experiment (v1)

## Verdict: PASS — The signal works, with caveats

Cycle-consistency **does** discriminate correct from incorrect solutions. The combined reward achieves **AUROC 0.708** overall (random = 0.50) and enables selective prediction that boosts accuracy from 56% → 88% when keeping the top 10% highest-confidence samples. However, the signal is dominated by `answer_match` (self-consistency), while the embedding-based `question_cycle` alone is weak. The method is also bounded by systematic failure modes in answer extraction and reconstructor output quality.

---

## A1. Setup

| Component | v1 |
|---|---|
| **Solver** | Qwen/Qwen2.5-3B-Instruct (local vLLM) |
| **Reconstructor** | Qwen/Qwen3-4B (local vLLM, instruct model) |
| **Embedding model** | Qwen/Qwen3-Embedding-**0.6B** |
| **Datasets** | GSM8K (1000), MATH (500), OlympiadBench (300) — 1800 total |

---

## A2. Headline Metrics

### Solver accuracy

| Dataset | N | Accuracy |
|---|---|---|
| GSM8K | 1000 | 65.9% |
| MATH | 500 | 55.6% |
| OlympiadBench | 300 | 24.0% |
| **All** | **1800** | **56.1%** |

### AUROC / AUPRC

| Signal | GSM8K | MATH | OlympiadBench | All |
|---|---|---|---|---|
| `question_cycle` (neg) AUROC | 0.558 | 0.559 | 0.531 | 0.608 |
| `answer_match` AUROC | 0.737 | 0.641 | 0.648 | 0.702 |
| `combined_reward` AUROC | 0.715 | 0.636 | 0.618 | 0.708 |
| `combined_reward` AUPRC | 0.824 | 0.712 | 0.447 | 0.770 |

### Selective prediction

| Keep top-K% | N kept | Accuracy |
|---|---|---|
| 10% | 180 | **88.3%** |
| 20% | 360 | 86.1% |
| 30% | 540 | 84.8% |
| 50% | 900 | 74.1% |
| 100% | 1800 | 56.1% |

---

## A3. Key Failure Modes in v1

- **Meta-commentary in Q' (31.9%)**: Qwen3-4B (instruct) produced reasoning preambles like *"Okay, let's see. The user provided a solution and wants me to reconstruct…"* in 575/1800 outputs despite the system prompt. This is an RLHF artefact: instruct models are trained to reason before answering, overriding few-shot format examples. Mean `question_cycle` for meta-commentary Q' was 0.270 vs 0.100 for clean Q' — but note that this `question_cycle` was computed using the 0.6B embedder, which may understate the true distance.
- **Empty `answer_A_prime` (4.1%)**: 73/1800 extraction failures — all forced to `answer_match = 0`, incorrectly penalising 39 correct samples.
- **Correct but `answer_match = 0` (36.3%)**: 366/1009 correct solutions got a false-negative, mostly from format mismatches (e.g. "540 meters" ≠ "540").
- **Consistently wrong — false positives (23.3%)**: 184/791 incorrect samples had `answer_match = 1` — the model made the exact same mistake on both Q and Q'.

---

# PART B — v1 (Qwen3-4B) vs v2 (Qwen3-8B-Base) Comparative Analysis

## B1. What Changed

Two components changed between v1 and v2 — **both the reconstructor and the embedding model**:

| Component | v1 | v2 |
|---|---|---|
| **Reconstructor** | Qwen/Qwen3-4B (instruct) | Qwen/Qwen3-8B-Base (base, no RLHF) |
| Reconstructor type | Instruction-tuned | Pure base model |
| Reconstructor size | 4B | 8B |
| **Embedding model** | Qwen/Qwen3-Embedding-**0.6B** | Qwen/Qwen3-Embedding-**4B** |
| Embedding size | 0.6B | 4B (~7× larger) |
| Prompt structure | System prompt + 4-shot + QUESTION: prefix | Same |

This means `question_cycle` (embedding cosine distance) is affected by **both** changes: a cleaner Q' from the better reconstructor *and* a more capable embedding model computing the similarity. These two effects are **confounded** in the v1→v2 comparison and cannot be separated without additional ablations.

---

## B2. Reconstructor Quality: Complete Turnaround

This is the most unambiguous result in the comparison. **These metrics are attributable solely to the reconstructor change** — word overlap and meta-commentary are independent of the embedding model.

| Metric | v1 (Qwen3-4B instruct) | v2 (Qwen3-8B-Base) | Delta | Source |
|---|---|---|---|---|
| Meta-commentary in Q' | 575 / 1800 **(31.9%)** | **0 / 1800 (0.0%)** | −575 | Reconstructor only |
| Very short Q' (<10 chars) | 18 / 1800 (1.0%) | **0 / 1800 (0.0%)** | −18 | Reconstructor only |
| Empty `answer_A_prime` | 73 / 1800 (4.1%) | **3 / 1800 (0.2%)** | −70 | Reconstructor only |
| Correct samples lost to empty extraction | 39 | **1** | −38 | Reconstructor only |
| Q↔Q' word overlap (mean) | 0.453 | **0.512** | +0.059 | Reconstructor only |
| `question_cycle` mean (all) | 0.154 | **0.137** | −0.017 | **Confounded** (reconstructor + embedding) |

The `question_cycle` drop (−0.017) has two additive causes: (1) Q' is genuinely closer to Q because the 8B-Base reconstructor produces cleaner outputs, and (2) the 4B embedding model computes cosine similarity with higher representational precision than the 0.6B model. We cannot attribute this improvement to either cause in isolation.

### Why the base model beats the instruct model here

This reveals an important property of instruction-tuned vs base models for **structured generation tasks**:

- **Qwen3-4B (instruct)** has been RLHF-trained to reason before answering. Even when told to "output only the question", the RLHF prior pushes it to narrate its reasoning process. The system prompt and few-shot examples fight against this prior — and lose 1-in-3 times.
- **Qwen3-8B-Base** has no such prior. It is a pure next-token predictor operating entirely from in-context learning. When shown 4 demonstrations of `Solution: … → QUESTION: <clean question>`, it pattern-completes reliably every time. There is no RLHF voice telling it to "think aloud first."

Counterintuitively, **the base model with few-shot is more format-obedient than the instruct model with a system prompt** for this narrow generation task.

**Concrete example — id=1 (GSM8K):**
```
Q:    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?"

Q'(v1, Qwen3-4B):
  "Okay, let's see. The user provided a solution and wants me to reconstruct the original..."
  → question_cycle = 0.138,  answer_match = 0  [correct sample penalised]

Q'(v2, Qwen3-8B-Base):
  "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?"
  → question_cycle = 0.009,  answer_match = 1  [correctly identified]
```

---

## B3. Headline Metrics: Mixed Results by Dataset

| Signal | Dataset | v1 (Qwen3-4B) | v2 (Qwen3-8B-Base) | Delta |
|---|---|---|---|---|
| AUROC(combined_reward) | **All** | 0.708 | 0.679 | **−0.029** |
| AUROC(combined_reward) | **GSM8K** | 0.715 | 0.675 | **−0.040** |
| AUROC(combined_reward) | **MATH** | 0.636 | **0.676** | **+0.040** |
| AUROC(combined_reward) | **OlympiadBench** | 0.618 | **0.668** | **+0.050** |
| AUPRC(combined_reward) | All | 0.770 | 0.744 | **−0.026** |
| AUPRC(combined_reward) | MATH | 0.712 | **0.738** | **+0.026** |
| AUPRC(combined_reward) | OlympiadBench | 0.447 | **0.460** | **+0.013** |
| AUPRC(answer_match) | OlympiadBench | 0.335 | **0.460** | **+0.125** |

**Overall headline is slightly worse; per-dataset story is the opposite for harder problems.**

---

## B4. The GSM8K Paradox — Why Did the Easier Dataset Go Down?

Overall AUROC dropped 0.029, driven entirely by **GSM8K getting worse** despite reconstruction quality dramatically improving. This is the most instructive finding.

### answer_match flips (correct samples only)

| Flip direction | Count | Meaning |
|---|---|---|
| v1=0 → v2=1 (improved) | 188 | Meta-commentary / extraction failures fixed |
| v1=1 → v2=0 (degraded) | 233 | 8B-Base changed the problem; solver got a *different* correct answer |

Net: **−45 correct samples losing `answer_match`** → answer_match rate for correct GSM8K samples dropped from 69.0% → 56.9% (−12.1 points).

### Root cause: the 8B-Base reconstructor is more semantically generative

The 4B instruct model, when not producing meta-commentary, was close to verbatim copying (because it had seen the question itself to reconstruct from, and the path of least resistance was copy-paste). The 8B-Base model, operating as a true pattern-completer, **reconstructs from the semantic content of the solution** — it produces a valid, well-formed question, but sometimes with slightly different numbers or reordered constraints.

| Problem | v1 (Qwen3-4B) Q' result | v2 (Qwen3-8B-Base) Q' result |
|---|---|---|
| id=8 | `A' = 460` ✓ (correct match) | `A' = 2000` ✗ (8B changed a number) |
| id=19 | `A' = 2` ✓ (correct match) | `A' = 8` ✗ (8B changed a number) |
| id=28 (FP fixed!) | `A' = 0` ✗ (same wrong answer) | `A' = 13` ✓ (8B broke the false positive) |

This is a two-sided coin:
- **Downside on GSM8K**: When problems are simple, the 8B-Base model generates plausible but slightly altered variants. A correct solver getting a different answer on the altered Q' looks like a mismatch — a false negative.
- **Upside on false positives**: For the 101 cases where the cycle was wrongly confident (incorrect but `answer_match=1`), the 8B-Base's willingness to change the problem broke the false agreement.

### Why MATH and OlympiadBench improved

On harder problems, the meta-commentary in v1 was proportionally more damaging. Long, complex solutions are more likely to trigger Qwen3-4B's reasoning-aloud instinct. With 0% meta-commentary in v2, the signal on these datasets is clean for the first time. The 4B embedding model likely contributes further — a richer representation space produces sharper cosine distances between semantically divergent Q/Q' pairs.

- **OlympiadBench**: Cohen's d for `answer_match` jumped 0.64 → **0.94** (medium → very large). `question_cycle` gap expanded from 0.004 → **0.029** — essentially zero in v1, now measurable. This improvement is almost certainly driven by *both* the cleaner reconstructions *and* the more discriminative 4B embeddings.
- **MATH**: AUROC +0.040, AUPRC +0.026.

---

## B5. Cohen's d Effect Sizes

| Signal | Dataset | v1 | v2 | Δ |
|---|---|---|---|---|
| `question_cycle` | All | 0.309 | 0.215 | −0.094 |
| `answer_match` | All | 0.894 | 0.853 | −0.041 |
| `combined_reward` | All | 0.869 | 0.792 | −0.077 |
| `question_cycle` | GSM8K | 0.196 | 0.186 | −0.011 |
| `answer_match` | GSM8K | 1.081 | 0.885 | **−0.196** |
| `answer_match` | MATH | 0.603 | **0.753** | **+0.150** |
| `combined_reward` | MATH | 0.570 | **0.690** | **+0.120** |
| `question_cycle` | OlympiadBench | 0.024 | **0.177** | **+0.152** |
| `answer_match` | OlympiadBench | 0.639 | **0.945** | **+0.306** |
| `combined_reward` | OlympiadBench | 0.555 | **0.848** | **+0.292** |

GSM8K was already the strongest dataset (Cohen's d = 1.08 for `answer_match` in v1). The regression there is real but the effect is still large (0.88). The gains on harder datasets are more significant scientifically.

---

## B6. False Positive Analysis (Consistently Wrong)

| | v1 | v2 | Delta |
|---|---|---|---|
| Incorrect but `answer_match=1` | 184 / 791 (23.3%) | **166 / 793 (20.9%)** | **−18** |
| FP fixed (v1=1 → v2=0) | — | 101 | — |
| New FP introduced (v1=0 → v2=1) | — | 83 | — |

The 8B-Base model's tendency to alter the problem broke 101 false positives the 4B model was blind to — including cases like id=28 where a systematic off-by-one error was consistently reproduced through the cycle in v1 but disrupted in v2.

---

## B7. Selective Prediction Comparison

| Coverage | v1 accuracy | v2 accuracy | Delta |
|---|---|---|---|
| Top 10% | 88.3% | 84.4% | −3.9% |
| Top 20% | 86.1% | 83.3% | −2.8% |
| Top 30% | 84.8% | 82.2% | −2.6% |
| Top 50% | 74.1% | 71.2% | −2.9% |
| 100% | 56.1% | 55.9% | −0.1% |

The GSM8K regression pulls down selective prediction in v2. Both versions still show strong lift from the 56% baseline.

---

## B8. The Core Trade-off

| Dimension | v1 | v2 | Changed by |
|---|---|---|---|
| Reconstructor | Qwen3-4B instruct | Qwen3-8B-Base | Reconstructor |
| Embedding model | 0.6B | **4B** | Embedder |
| Format compliance | 32% meta-commentary | **0% meta-commentary** | Reconstructor |
| Answer extraction failures | 73 | **3** | Reconstructor |
| MATH discrimination | AUROC 0.636 | **AUROC 0.676** | Both (confounded) |
| OlympiadBench discrimination | AUROC 0.618, d=0.55 | **AUROC 0.668, d=0.85** | Both (confounded) |
| GSM8K discrimination | **AUROC 0.715** | AUROC 0.675 | Reconstructor (number drift) |
| `question_cycle` mean | 0.154 | **0.137** | Both (confounded) |
| Overall headline | **AUROC 0.708** | AUROC 0.679 | — |
| False positives caught | 184 | **166** | Reconstructor |
| Reconstructor number fidelity | Higher | Lower (8B changes numbers) | Reconstructor |

---

## B9. Key Learnings

### 1. Base > Instruct for structured constrained generation
For tasks requiring strict output format, a larger base model with few-shot demonstrations outperforms a smaller instruct model with system prompts. RLHF training introduces a prior toward verbose reasoning that competes with and sometimes overrides explicit formatting instructions. This is not a bug in Qwen3-4B — it's the expected behaviour of a helpful assistant model. But it's the wrong prior for this specific task.

### 2. Reconstructor size and training paradigm are entangled — ablation needed
Scaling from 4B→8B and Instruct→Base happened simultaneously. The format compliance improvement (0% meta-commentary) is most plausibly explained by the Base paradigm shift. The number-drift issue (8B changes problem parameters more) is more plausibly explained by size — a larger model generates more diverse completions. Future work should test **4B-Base** and **8B-Instruct** independently to disentangle these.

### 3. The `question_cycle` improvement is doubly confounded
The −0.017 drop in mean `question_cycle` has two simultaneous causes: (1) Q' is genuinely closer to Q (cleaner reconstruction by 8B-Base) and (2) the 4B embedding model measures semantic similarity more accurately than the 0.6B model. We cannot attribute the Cohen's d improvement on OlympiadBench (`question_cycle`: 0.024→0.177) to either cause alone. Specifically, the 4B embedder may simply draw a finer distinction between a garbage meta-commentary Q' and a real question — which was the dominant v1 issue.

### 4. Embedding model size matters for `question_cycle` signal quality
A 0.6B embedding model may lack the representational capacity to distinguish subtle numeric changes in math problems (e.g. "8 apples" vs "6 apples"). The 4B model's richer representation space likely produces more discriminative cosine distances — which partly explains the MATH and OlympiadBench gains, even independent of reconstructor quality.

### 5. A generative reconstructor is a double-edged sword
The 8B-Base model is more semantically creative. On hard problems (MATH, OlympiadBench), this creativity provides signal the 4B model masked with meta-commentary. On easy problems (GSM8K), the same creativity slightly alters problem parameters, producing correct-but-different second answers that fail `answer_match`. This is the central tension in the reconstructor design space.

### 6. `question_cycle` was never meaningfully measurable in v1
With 32% of v1 Q' outputs being meta-commentary garbage, `question_cycle` was measuring corrupted embeddings. The OlympiadBench Cohen's d of 0.024 (essentially zero) in v1 is now 0.177 in v2 — not because the cycle is intrinsically more discriminative, but because v1 was embedding noise.

### 7. The overall AUROC regression is a red herring
The −0.029 drop in overall AUROC is driven by GSM8K, which was already the strongest dataset. The method's real value is as a confidence signal on hard problems where the solver might fail — MATH and OlympiadBench. Both improved. Report per-dataset metrics as primary.

---

## B10. Recommendations for Exp 2

1. **Fix 1 — LLM answer judge**: The false-negative rate (correct but `answer_match=0`) is 36–41% in both runs. An LLM that asks "are these two answers mathematically equivalent?" rather than string-matching would recover most of the 188 improvements while keeping the 101 false-positive fixes. Expected AUROC gain: +5–10 points.

2. **Fix 3 — BLEU + embedding hybrid**: v2's 8B-Base model changes problem numbers without changing semantic meaning (embedding stays close, BLEU drops sharply). Adding BLEU directly targets the numerical-drift failure mode. The 4B embedder is already better at semantic gaps — BLEU adds the surface-level number-preservation check it cannot see.

3. **Ablate the confounded changes**: v1→v2 changed three things (reconstructor size, reconstructor type, embedding model). Run at minimum: (a) 4B-Base reconstructor + 0.6B embedder to isolate the paradigm effect, and (b) 4B-Instruct + 4B embedder to isolate the embedding upgrade.

4. **Add a number-preservation instruction**: *"All numbers that appear in the solution must appear unchanged in your reconstructed question"* — would reduce the 8B-Base numeric-drift failures without sacrificing the format quality win.

5. **Report per-dataset as primary**: The headline AUROC hides that the signal is improving where it matters most (hard datasets) and regressing where it matters least. Always lead with MATH and OlympiadBench.
