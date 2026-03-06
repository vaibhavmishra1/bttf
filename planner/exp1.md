
Implement an experiment to test cycle-consistency as a self-supervised signal for LLM reasoning correctness using math datasets.

Goal: determine whether cycle invariance (Q → S → Q’ → S’) correlates with correct solutions.

⸻

Pipeline

Q → Solver → S
S → Reconstructor → Q'
Q' → Solver → S'

Compute two signals:

question_cycle = 1 - cosine(embed(Q), embed(Q'))
answer_match = 1 if final_answer(S) == final_answer(S') else 0
combined_reward = answer_match - question_cycle

Lower question_cycle and higher answer_match should correlate with correct solutions.

⸻

Models

Solver (local HF):

Qwen/Qwen2.5-3B-Instruct

Reconstructor (OpenAI API):

gpt-4.1 or equivalent

Embedding model:

Qwen/Qwen3-Embedding-0.6B

Use cosine similarity.

⸻

Datasets

Load with HuggingFace:

gsm8k
hendrycks/competition_math (sample 500 → MATH-500)

Fields:

Q = question
A = ground truth answer

Sample sizes:

1000 GSM8K
500 MATH


⸻

Step 1 — Solve Question

Prompt:

Solve the following math problem step by step and give the final answer clearly.

Problem:
{Q}

Solution:

Generate:

S


⸻

Step 2 — Reconstruct Question

Prompt:

Given the following solution to a math problem, reconstruct the original question.

The reconstructed question should preserve the same numbers, entities, and relationships.

Output only the question.

Solution:
{S}

Original Question:

Output:

Q'


⸻

Step 3 — Second Solve

Run solver again:

S' = Solver(Q')

Use same solver prompt.

⸻

Step 4 — Embeddings

Embed:

Q
Q'

Compute:

question_cycle = 1 - cosine(embed(Q), embed(Q'))


⸻

Step 5 — Extract Final Answers

Extract final numeric answers from S and S'.

Methods:
	1.	regex patterns:

#### number
Answer: number
The answer is number

	2.	fallback: last number in text.

Return:

A  = answer(S)
A' = answer(S')


⸻

Step 6 — Answer Equivalence

Check equality using:
	1.	exact string match
	2.	float comparison (tol = 1e-6)
	3.	sympy equivalence when possible

Return:

answer_match ∈ {0,1}


⸻

Step 7 — Solver Correctness

Compare A with ground truth answer from dataset.

Use same equivalence logic.

Return:

correct ∈ {0,1}





Step 9 — Metrics

Compute AUROC predicting solver correctness using:

question_cycle
answer_match
combined_reward

Where:

combined_reward = answer_match - question_cycle

Also compute:

AUPRC


⸻

Step 10 — Visualizations

Generate:

Violin plot

Distributions of:

question_cycle
combined_reward

Split by:

correct vs incorrect


⸻

Histogram

question_cycle (correct vs incorrect)


⸻

ROC Curve

combined_reward → correctness



