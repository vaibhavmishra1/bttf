"""Model wrappers: Solver, Reconstructor (Fix 2), Embedder, AnswerJudge (Fix 1).

Changes vs exp1/models.py:
  + AnswerJudge  — Feed extracted A and A' to a local LLM; get YES/NO
                   equivalence judgment. Replaces regex matching for answer_match.
  = Reconstructor — unchanged from exp1 (few-shot + QUESTION: prefix is Fix 2).
  = Solver, Embedder — unchanged.
"""

import gc
import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams


# ── vLLM memory cleanup ──────────────────────────────────────────────────────

def _destroy_vllm(llm: LLM) -> None:
    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("  [vLLM] model destroyed, GPU memory freed.")


# ── Solver ───────────────────────────────────────────────────────────────────

SOLVER_PROMPT = (
    "Solve the following math problem step by step "
    "and give the final answer clearly.\n\n"
    "Problem:\n{question}\n\nSolution:"
)


class Solver:
    """Qwen2.5-3B-Instruct vLLM solver — identical to exp1."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        print(f"  [vLLM] Loading Solver: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._llm = LLM(
            model=model_name,
            dtype="float16",
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            enforce_eager=False,
        )

    def _build_prompt(self, question: str) -> str:
        messages = [
            {"role": "user", "content": SOLVER_PROMPT.format(question=question)}
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def solve_batch(
        self,
        questions: list[str],
        max_new_tokens: int = 2048,
        batch_size: int = 256,
    ) -> list[str]:
        prompts = [self._build_prompt(q) for q in questions]
        params  = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        outputs = self._llm.generate(prompts, params)
        return [o.outputs[0].text.strip() for o in outputs]

    def destroy(self) -> None:
        _destroy_vllm(self._llm)


# ── Reconstructor — Fix 2 ────────────────────────────────────────────────────
# Fix 2: system message + 4-shot examples + mandatory "QUESTION:" prefix
# + think=False to suppress Qwen3 extended-thinking preamble.

_FEW_SHOT = [
    (
        "The total eggs laid per day is 16. Janet eats 3 for breakfast and uses 4 for "
        "muffins, so she uses 7 eggs altogether. Remaining eggs: 16 − 7 = 9. "
        "Selling 9 eggs at $2 each gives 9 × 2 = \\boxed{18} dollars.",
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
        "and bakes muffins for her friends every day with four. She sells the remainder "
        "at the farmers' market daily for $2 per fresh duck egg. How much in dollars "
        "does she make every day at the farmers' market?",
    ),
    (
        "A robe requires 2 bolts of blue fiber. White fiber = half of blue = "
        "1/2 × 2 = 1 bolt. Total bolts = 2 + 1 = \\boxed{3}.",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?",
    ),
    (
        "James runs 3 sprints per session, 60 meters each, so one session = "
        "3 × 60 = 180 meters. He trains 3 sessions per week, so the weekly total is "
        "3 × 180 = \\boxed{540} meters.",
        "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. "
        "How many total meters does he run a week?",
    ),
    (
        "Wendi has 20 chickens, each needing 3 cups/day, so total daily need = 60 cups. "
        "Morning: 15 cups. Afternoon: 25 cups. Already given: 15 + 25 = 40 cups. "
        "Final meal: 60 − 40 = \\boxed{20} cups.",
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. "
        "She gives the chickens their feed in three separate meals. In the morning, she "
        "gives her flock of chickens 15 cups of feed. In the afternoon, she gives her "
        "chickens another 25 cups of feed. How many cups of feed does she need to give "
        "her chickens in the final meal of the day if the size of Wendi's flock is "
        "20 chickens?",
    ),
]

_RECONSTRUCTOR_SYSTEM = (
    "You are an expert at recovering the original math word problem from its solution.\n"
    "Rules:\n"
    "  1. Preserve all numbers, names, units, and relationships from the solution.\n"
    "  2. Output ONLY the reconstructed question — nothing else.\n"
    "  3. Always start your output with the exact token: QUESTION:"
)

_QUESTION_RE = re.compile(r"QUESTION:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _build_reconstructor_messages(solution: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _RECONSTRUCTOR_SYSTEM}]
    for sol_ex, q_ex in _FEW_SHOT:
        messages.append({"role": "user",      "content": f"Solution:\n{sol_ex}"})
        messages.append({"role": "assistant", "content": f"QUESTION: {q_ex}"})
    messages.append({"role": "user", "content": f"Solution:\n{solution}"})
    return messages


class Reconstructor:
    """Qwen3-4B local reconstructor via vLLM — identical to exp1."""

    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        print(f"  [vLLM] Loading Reconstructor: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._llm = LLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            enforce_eager=False,
        )
        self._params = SamplingParams(temperature=0, max_tokens=512)

    def _build_prompt(self, solution: str) -> str:
        messages = _build_reconstructor_messages(solution)
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, think=False
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def reconstruct_batch(
        self, solutions: list[str], max_concurrent: int = 20
    ) -> list[str]:
        print(f"  Reconstructing {len(solutions)} solutions via vLLM …")
        prompts = [self._build_prompt(s) for s in solutions]
        outputs = self._llm.generate(prompts, self._params)

        results: list[str] = []
        n_fallback = 0
        _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
        for o in outputs:
            text = o.outputs[0].text.strip()
            text = _THINK_RE.sub("", text).strip()
            m = _QUESTION_RE.search(text)
            if m:
                question = m.group(1).strip().splitlines()[0].strip()
                results.append(question)
            else:
                lines = [ln.strip() for ln in text.splitlines()
                         if ln.strip() and not ln.strip().startswith("<think")]
                first_line = lines[0] if lines else ""
                if first_line:
                    n_fallback += 1
                results.append(first_line)
        if n_fallback:
            print(f"  [!] {n_fallback}/{len(outputs)} reconstructions used fallback (no QUESTION: prefix)")
        return results

    def destroy(self) -> None:
        _destroy_vllm(self._llm)


# ── Embedder ─────────────────────────────────────────────────────────────────

class Embedder:
    """Qwen3-Embedding-0.6B — identical to exp1."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": 0},
            trust_remote_code=True,
        )
        self.model.eval()

    def _last_token_pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx   = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, seq_lengths]

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_embs: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch  = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(self.model.device)
            with torch.no_grad():
                out = self.model(**inputs)
            embs = self._last_token_pool(out.last_hidden_state, inputs["attention_mask"])
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)


# ── AnswerJudge — Fix 1 ───────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a math grader. Your only job is to decide whether two extracted "
    "math answers are mathematically equivalent.\n"
    "Rules:\n"
    "  1. Ignore units, formatting, and phrasing differences.\n"
    "  2. '18 dollars', '$18', and '18' are equivalent.\n"
    "  3. '540 meters' and '540' are equivalent.\n"
    "  4. Fractions and decimals that equal the same value are equivalent.\n"
    "  5. Respond with ONLY the single word YES or NO."
)

_JUDGE_FEW_SHOT = [
    ("18 dollars",           "18",          "YES"),
    ("540 meters",           "540",          "YES"),
    ("\\frac{1}{2}",         "0.5",          "YES"),
    ("(-\\infty, 0]",        "(-\\infty, 0)","NO"),
    ("7",                    "4",            "NO"),
    ("Shandy drove 230 more","230",          "YES"),
    ("",                     "12",           "NO"),
]

_JUDGE_PROMPT = "Answer 1: {a}\nAnswer 2: {a_prime}\nAre these equivalent? YES or NO:"


def _build_judge_messages(a: str, a_prime: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _JUDGE_SYSTEM}]
    for a_ex, ap_ex, verdict in _JUDGE_FEW_SHOT:
        messages.append({"role": "user",
                         "content": _JUDGE_PROMPT.format(a=a_ex, a_prime=ap_ex)})
        messages.append({"role": "assistant", "content": verdict})
    messages.append({"role": "user",
                     "content": _JUDGE_PROMPT.format(a=a, a_prime=a_prime)})
    return messages


class AnswerJudge:
    """Fix 1: LLM-based answer equivalence judge.

    Feeds extracted answer strings A and A' to a local LLM and asks
    whether they are mathematically equivalent (YES / NO).

    Why this is better than regex:
      - Handles unit suffixes: "18 dollars" vs "18"
      - Handles verbose extractions: "Wendi needs 20 cups" vs "20"
      - Handles LaTeX vs decimal: "\\frac{1}{2}" vs "0.5"
      - Handles empty extractions gracefully (→ NO)
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        print(f"  [vLLM] Loading AnswerJudge: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._llm = LLM(
            model=model_name,
            dtype="float16",
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            enforce_eager=False,
        )
        self._params = SamplingParams(temperature=0, max_tokens=4)

    def _build_prompt(self, a: str, a_prime: str) -> str:
        messages = _build_judge_messages(a, a_prime)
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def judge_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[int]:
        """Return 1 (equivalent) or 0 (not equivalent) for each (A, A') pair."""
        print(f"  Judging {len(pairs)} answer pairs via vLLM …")
        prompts = [self._build_prompt(a, ap) for a, ap in pairs]
        outputs = self._llm.generate(prompts, self._params)

        results: list[int] = []
        n_yes, n_no, n_unclear = 0, 0, 0
        for o in outputs:
            text = o.outputs[0].text.strip().upper()
            if "YES" in text:
                results.append(1); n_yes += 1
            elif "NO" in text:
                results.append(0); n_no += 1
            else:
                results.append(0); n_unclear += 1  # conservative: treat as NO

        print(f"  Judge results: YES={n_yes}  NO={n_no}  unclear→NO={n_unclear}")
        return results

    def destroy(self) -> None:
        _destroy_vllm(self._llm)
