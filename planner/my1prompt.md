
**"I want to verify whether cycle consistency can serve as a self-supervised reward signal for LLM reasoning. This is inspired by CycleGAN's bidirectional consistency loss applied to LLM question-solving.**

**The core idea:**
```
Q → Solver LLM → Solution S → Reconstructor LLM → Q'
reward = -||embed(Q) - embed(Q')||
```
If the solution is correct, Q' should semantically recover Q. If the solution is wrong or hallucinated, the embedding distance should be large.

**We want to run 3 experiments in order of complexity:**

---

**Experiment 1 — Correlation Check (no training, just inference)**

- Load GSM8K and MATH-500 datasets (use HuggingFace datasets library)
- Use a frozen small LLM (e.g. Qwen2.5-3B-Instruct or similar) as the Solver to generate one solution per question Q
- use openai model as the Reconstructor by calling api: given solution S, prompt it to reconstruct the original question
- Embed both Q and Q' using a sentence embedding model (e.g. `Qwen/Qwen3-Embedding-0.6B`)
- Compute cosine distance between embed(Q) and embed(Q') for each sample
- Since GSM8K and MATH-500 have ground truth labels, also compute whether each solution is correct (string match or sympy-based equivalence check)
- **Output:** AUROC and a violin plot of cycle distances split by correct vs incorrect solutions. We want to know if cycle distance discriminates correct from incorrect solutions.

---

**Experiment 2 — Reward Replacement via GRPO**

- Use the same dataset but now train the Solver using GRPO
- Run two training runs in parallel:
  - **Baseline:** standard binary reward from ground truth verifier (correct/incorrect)
  - **Cycle reward:** replace the verifier reward entirely with `-cosine_distance(embed(Q), embed(Q'))`, normalized across the GRPO group
- Use TRL library's GRPO trainer or implement a lightweight custom GRPO loop
- Backbone model: Qwen2.5-3B-Base or similar small model
- Evaluate both on GSM8K test set after training
- **Output:** learning curves and final accuracy comparison between the two reward strategies

---

**Experiment 3 — OOD Generalization**

- Take the cycle-reward-trained model from Experiment 2
- Evaluate zero-shot on a domain where no verifier exists: use ARC-Challenge or a subset of MMLU-Pro
- Compare against the ground-truth-reward baseline on the same OOD eval
- **Output:** accuracy table showing whether cycle reward generalizes better or comparably to supervised reward

---

**Implementation requirements:**

- Use Python, PyTorch, HuggingFace `transformers` and `datasets`
- For GRPO, use `trl` library if it supports it, otherwise implement advantage normalization manually
- Keep the reconstructor frozen throughout all experiments (do not train it)
- For the embedding model, keep it frozen too
- Log all runs with wandb or at minimum save metrics to a CSV
- Structure the code as separate scripts: `exp1_correlation.py`, `exp2_grpo.py`, `exp3_ood.py` with a shared `utils.py` for embedding, reconstruction prompting, and evaluation helpers

**Reconstructor prompt to use:**
```
Given the following solution to a math problem, reconstruct the original question that was asked. Output only the question, nothing else.

Solution: {S}

Original question:
```

**Key metric to track across all experiments:** cosine distance between embed(Q) and embed(Q') — this is the cycle consistency score.

Start with Experiment 1 only. Show me the AUROC result and violin plot before proceeding to Experiment 2."**

