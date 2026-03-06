"""Model wrappers: Solver (local HF), Reconstructor (OpenAI), Embedder."""

import os
import concurrent.futures

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from openai import OpenAI


# ── Solver ──────────────────────────────────────────────────────────────────

SOLVER_PROMPT = (
    "Solve the following math problem step by step "
    "and give the final answer clearly.\n\n"
    "Problem:\n{question}\n\nSolution:"
)


class Solver:
    """Qwen2.5-3B-Instruct local solver with batched generation."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # Left-pad for batched decoder generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    # ── helpers ─────────────────────────────────────────────────────────
    def _build_prompt(self, question: str) -> str:
        messages = [
            {"role": "user", "content": SOLVER_PROMPT.format(question=question)}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ── public ──────────────────────────────────────────────────────────
    def solve_batch(
        self,
        questions: list[str],
        max_new_tokens: int = 1024,
        batch_size: int = 8,
    ) -> list[str]:
        solutions: list[str] = []
        for i in tqdm(range(0, len(questions), batch_size), desc="Solving"):
            batch_q = questions[i : i + batch_size]
            prompts = [self._build_prompt(q) for q in batch_q]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            for j, output_ids in enumerate(outputs):
                prompt_len = inputs["input_ids"][j].shape[0]
                generated = self.tokenizer.decode(
                    output_ids[prompt_len:], skip_special_tokens=True
                )
                solutions.append(generated.strip())

        return solutions


# ── Reconstructor ───────────────────────────────────────────────────────────

RECONSTRUCTOR_PROMPT = (
    "Given the following solution to a math problem, "
    "reconstruct the original question.\n\n"
    "The reconstructed question should preserve the same numbers, "
    "entities, and relationships.\n\n"
    "Output only the question.\n\n"
    "Solution:\n{solution}\n\nOriginal Question:"
)


class Reconstructor:
    """GPT-4.1 question reconstructor via OpenAI API."""

    def __init__(self, model: str = "gpt-4.1"):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def _reconstruct_one(self, solution: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": RECONSTRUCTOR_PROMPT.format(solution=solution),
                }
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()

    def reconstruct_batch(
        self, solutions: list[str], max_concurrent: int = 20
    ) -> list[str]:
        results: list[str | None] = [None] * len(solutions)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent
        ) as pool:
            future_to_idx = {
                pool.submit(self._reconstruct_one, sol): idx
                for idx, sol in enumerate(solutions)
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="Reconstructing",
            ):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"  [!] index {idx} failed: {exc}")
                    results[idx] = ""

        return results  # type: ignore[return-value]


# ── Embedder ────────────────────────────────────────────────────────────────


class Embedder:
    """Qwen3-Embedding-0.6B embedder (last-token pooling, L2-normalised)."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"  # pad right for embedding

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def _last_token_pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool using the last non-padding token."""
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, seq_lengths]

    def embed(
        self, texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        all_embs: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            embs = self._last_token_pool(
                outputs.last_hidden_state, inputs["attention_mask"]
            )
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().float().numpy())

        return np.concatenate(all_embs, axis=0)
