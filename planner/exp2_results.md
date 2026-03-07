# Experiment 2 — Majority Voting vs Cycle Consistency: Results

## 1. Setup Recap

| Parameter | Value |
|---|---|
| Solver | Qwen2.5-3B-Instruct (vLLM) |
| Reconstructor | Qwen3-8B-Base (from Exp1) |
| Embedding model | Qwen3-Embedding-4B (from Exp1) |
| Datasets | GSM8K (1000), MATH (500), OlympiadBench (300) — 1800 total |
| MV samples (K) | 2, 4, 8, 16 |
| MV temperature | 0.7, top_p=0.95 |
| CC signal source | Exp1 results (one cycle per question) |
| Fusion strategies | Linear α-sweep, Product, Logistic Regression (5-fold CV) |

**Overall accuracy**: 55.9% (1007/1800)
- GSM8K: 65.7% (657/1000)
- MATH: 55.4% (277/500)
- OlympiadBench: 24.3% (73/300)

---

## 2. Head-to-Head AUROC — The Core Comparison

### 2.1 Individual Signals

| Signal | ALL | GSM8K | MATH | OlympiadBench |
|---|---|---|---|---|
| **CC: combined_reward** | 0.692 | 0.677 | 0.691 | 0.702 |
| CC: hybrid_cycle | 0.595 | 0.569 | 0.587 | 0.596 |
| CC: answer_match | 0.697 | 0.700 | 0.695 | 0.697 |
| MV: vote_conf K=2 | 0.679 | 0.634 | 0.689 | 0.749 |
| **MV: vote_conf K=4** | **0.789** | 0.753 | 0.798 | 0.802 |
| **MV: vote_conf K=8** | **0.830** | 0.804 | 0.818 | 0.856 |
| **MV: vote_conf K=16** | **0.852** | 0.836 | 0.837 | 0.864 |

**Takeaway**: MV dominates CC on raw AUROC at K≥4. Even at K=2, MV (0.679) is only marginally below CC (0.692). CC as a standalone verifier is clearly the weaker signal.

### 2.2 Fusion Signals

| Signal | ALL | GSM8K | MATH | OlympiadBench |
|---|---|---|---|---|
| Fusion: linear K=2 | **0.765** | 0.744 | 0.760 | 0.786 |
| Fusion: linear K=4 | **0.821** | 0.805 | 0.813 | 0.812 |
| Fusion: linear K=8 | **0.846** | 0.836 | 0.824 | 0.855 |
| Fusion: linear K=16 | **0.860** | 0.853 | 0.839 | 0.864 |
| Fusion: logistic K=16 | **0.860** | 0.859 | 0.837 | 0.862 |
| Fusion: product K=16 | 0.812 | 0.793 | 0.795 | 0.831 |

**Takeaway**: Linear fusion and logistic fusion perform nearly identically, both consistently beat MV alone. Product fusion is significantly worse (loses information due to scaling).

---

## 3. The Low-K Sweet Spot — Strongest Finding

At **K=2**, where MV and CC are roughly equal in power, fusion provides a massive improvement:

| Dataset | MV (K=2) | CC | Fusion (K=2+CC) | Lift over best |
|---|---|---|---|---|
| **ALL** | 0.679 | 0.692 | **0.765** | **+7.3 pts** |
| GSM8K | 0.634 | 0.677 | **0.744** | **+6.7 pts** |
| MATH | 0.689 | 0.691 | **0.760** | **+6.9 pts** |
| OlympiadBench | 0.749 | 0.702 | **0.786** | **+3.7 pts** |

The α-sweep plot at K=2 shows a clean inverted-U with peak at **α=0.45** (nearly equal weight on MV and CC), confirming both signals contribute approximately equally at this budget.

### Diminishing Returns as K Grows

| K | Best α | Fusion Lift over MV-alone | CC weight in optimal mix |
|---|---|---|---|
| 2 | 0.45 | **+7.3 pts** | 55% |
| 4 | 0.65 | +3.2 pts | 35% |
| 8 | 0.75 | +1.6 pts | 25% |
| 16 | 0.85 | +0.8 pts | 15% |

**Key insight**: The optimal α shifts monotonically from 0.45 (balanced) to 0.85 (MV-dominated) as K grows. This directly supports the error taxonomy: CC detects systematic errors that MV can't catch with few samples, but as MV gets more samples, it increasingly subsumes CC's contribution.

---

## 4. Majority Vote Accuracy (Answer Improvement)

MV also improves the raw accuracy of the solver by selecting the majority answer:

| Dataset | Greedy | K=4 | K=8 | K=16 |
|---|---|---|---|---|
| ALL | 0.559 | 0.635 (+7.6) | 0.677 (+11.8) | 0.690 (+13.1) |
| GSM8K | 0.657 | 0.777 (+12.0) | 0.823 (+16.6) | 0.840 (+18.3) |
| MATH | 0.554 | 0.570 (+1.6) | 0.608 (+5.4) | 0.620 (+6.6) |
| OlympiadBench | 0.243 | 0.270 (+2.7) | 0.307 (+6.3) | 0.307 (+6.3) |

**Note**: K=2 actually hurts accuracy slightly (−2.2 pts overall), because with only 2 samples, ties are common and the wrong answer can win by coin flip. MV requires K≥4 to reliably improve accuracy.

---

## 5. Selective Prediction — Practical Value

When serving only the highest-confidence answers:

| Method | Acc @10% | Acc @20% | Acc @50% |
|---|---|---|---|
| CC: combined_reward | 0.872 | 0.839 | 0.714 |
| MV: vote_conf K=8 | 0.894 | 0.900 | 0.811 |
| MV: vote_conf K=16 | 0.917 | 0.925 | 0.828 |
| **Fusion: linear K=8** | **0.922** | **0.917** | **0.829** |
| **Fusion: linear K=16** | **0.933** | **0.933** | **0.836** |
| Fusion: logistic K=16 | 0.917 | 0.919 | 0.842 |

**Per-dataset highlights** (Fusion K=16):
- **GSM8K @10%**: 99.0% accuracy (near-perfect selective prediction)
- **GSM8K @20%**: 98.0% accuracy
- **OlympiadBench @10%**: 86.7% accuracy (up from 24.3% base)

---

## 6. Failure Overlap Analysis — Error Complementarity

Jaccard overlap of false positives between MV (K=8) and CC:

| Dataset | FP(MV only) | FP(Both) | FP(CC only) | Jaccard | Overlap Coeff |
|---|---|---|---|---|---|
| ALL | 172 | 230 | 167 | **0.404** | 0.579 |
| GSM8K | 105 | 116 | 56 | 0.419 | 0.674 |
| MATH | 63 | 83 | 29 | 0.474 | 0.741 |
| OlympiadBench | 45 | 70 | 44 | 0.440 | 0.614 |

**Takeaway**: The Jaccard index of ~0.40 means the two methods fail on different problems roughly 60% of the time. This is the structural evidence that justifies fusion — they aren't redundant.

### Disagreement Analysis (K=8)

| Condition | N | Accuracy |
|---|---|---|
| Both high (MV≥0.75, CC>0.5) | 529 | **0.885** |
| Both low (MV<0.5, CC<0) | 400 | 0.140 |
| MV high, CC low (MV≥0.75, CC<0) | 361 | 0.704 |
| MV low, CC high (MV<0.5, CC>0.5) | 59 | 0.356 |

When both signals agree, accuracy is very high (88.5%) or very low (14.0%). The interesting zone is disagreement:
- **MV high, CC low** (N=361): 70.4% accurate — MV is usually right here, CC is too conservative
- **MV low, CC high** (N=59): 35.6% accurate — CC is misleading here, these are genuinely hard problems

---

## 7. Compute-Efficiency (Pareto Analysis)

Cost measured in solver-equivalent forward passes (CC ≈ 3.5 passes):

| Config | AUROC | Cost (passes) | AUROC/pass above random |
|---|---|---|---|
| CC alone | 0.692 | 3.5 | 0.055 |
| MV K=2 | 0.679 | 2.0 | **0.089** |
| MV K=4 | 0.789 | 4.0 | 0.072 |
| Fusion K=2+CC | 0.765 | 5.5 | 0.048 |
| MV K=8 | 0.830 | 8.0 | 0.041 |
| Fusion K=4+CC | 0.821 | 7.5 | 0.043 |
| Fusion K=8+CC | 0.846 | 11.5 | 0.030 |
| MV K=16 | 0.852 | 16.0 | 0.022 |
| Fusion K=16+CC | 0.860 | 19.5 | 0.018 |

**Key comparisons at equal cost**:
- Fusion(K=4)+CC at 7.5 passes → 0.821 vs MV(K=8) at 8 passes → 0.830 (**MV slightly wins, −0.9 pts**)
- Fusion(K=8)+CC at 11.5 passes → 0.846 vs MV(K=16) at 16 passes → 0.852 (**MV slightly wins, but Fusion is 28% cheaper**)
- Fusion(K=16)+CC at 19.5 passes → 0.860 vs MV(K=16) at 16 passes → 0.852 (**Fusion wins, +0.8 pts, but costs more**)

**Takeaway**: Fusion doesn't strictly dominate the Pareto frontier. MV alone is remarkably cost-efficient. However, Fusion(K=8)+CC nearly matches MV(K=16) while using 28% fewer compute passes.

---

## 8. Hypothesis Outcomes

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | MV wins on easy problems (GSM8K) | ✅ **Confirmed** | MV(K=8)=0.804 vs CC=0.677 (+12.7 pts) |
| H2 | CC wins on hard problems (OlympiadBench) | ❌ **Rejected** | MV(K=8)=0.856 vs CC=0.702 (+15.4 pts for MV) |
| H3 | Failure overlap < 60% | ✅ **Confirmed** | Jaccard=0.404 |
| H4 | Fusion ≥ +3 AUROC over best single | ⚠️ **Partially** | ✅ at K≤4 (+7.3 pts), ❌ at K≥8 (+1.6 pts) |
| H5 | MV at K=8 matches or exceeds CC | ✅ **Confirmed** | MV(K=8)=0.830 >> CC=0.692 |

---

## 9. Strength Assessment

### What's strong ✅
1. **K=2 fusion result**: +7.3 AUROC points over best individual signal, consistent across all 3 datasets
2. **Error complementarity**: Jaccard=0.40 proves the two methods fail on genuinely different problems
3. **α-shift pattern**: Clean monotonic shift from 0.45→0.85 as K grows — directly supports the theoretical error taxonomy (stochastic vs systematic)
4. **Selective prediction**: Fusion gives best accuracy at every coverage level (93.3% @10% at K=16)

### What's weak ❌
1. **CC alone is clearly inferior**: MV beats CC by 13.8 points at K=8 — even on OlympiadBench
2. **Fusion doesn't dominate the Pareto frontier**: At comparable compute, MV alone is usually slightly better
3. **Lift at high K is negligible**: +0.8 points at K=16 is within noise
4. **H4 failed at K≥8**: The pre-registered hypothesis of ≥3 point improvement was not met at practical operating points

### Overall rating: **Moderate — strong supporting experiment, not standalone contribution**

---

## 10. What These Results Motivate

The most compelling finding is **not** that fusion improves evaluation, but that **the two signals are structurally complementary**:
- MV detects stochastic errors (random slips that other samples don't repeat)
- CC detects some systematic errors (wrong method → Q' drifts from Q)
- Jaccard overlap = 0.40 confirms they catch different mistakes

This motivates the next experiment: **using both signals as per-trajectory rewards in GRPO-style training**.

### Exp3 Concept: Composite Self-Supervised Reward for RL Training

Instead of passively fusing MV and CC scores after generation, use them as the training reward:

```
For each training question Q:
  1. Sample K solutions: s_1, ..., s_K from policy π
  2. For each s_i, compute reward:
       r_i = α · 1(answer_i == majority_answer) + (1-α) · cc_score(s_i)
  3. Compute group-relative advantages (GRPO):
       A_i = (r_i - mean(r)) / std(r)
  4. Update π with policy gradient using advantages A_i
```

**Why this addresses Exp2's limitation**: In Exp2, CC's marginal value diminishes at high K because MV is already strong. But in a training loop, the composite reward directly penalizes systematic errors — even when all K solutions agree on the wrong answer (MV gives r=1 to all, but CC gives low r to solutions with poor cycle consistency). This could make the model *learn* to avoid systematic failures, not just detect them post-hoc.

**The α-shift finding from Exp2 directly informs the reward weighting**: Start with α≈0.45 (balanced) and potentially increase as the model improves (analogous to curriculum learning).
