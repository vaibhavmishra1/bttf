"""Dataset loading for GSM8K and MATH."""

import re
import random
from datasets import load_dataset


# ── Ground-truth extraction helpers ────────────────────────────────────────

def _extract_gsm8k_gt(answer_text: str) -> str:
    """GSM8K answers end with '#### <number>'."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip()


def _extract_math_gt(solution_text: str) -> str:
    r"""MATH answers are inside \boxed{...}."""
    # Handle nested braces: match the *last* \boxed{...}
    matches = re.findall(
        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution_text
    )
    if matches:
        return matches[-1].strip()
    return solution_text.strip()


# ── Dataset loaders ────────────────────────────────────────────────────────

def load_gsm8k(n_samples: int = 1000, seed: int = 42) -> list[dict]:
    ds = load_dataset("gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    if len(indices) > n_samples:
        random.seed(seed)
        indices = sorted(random.sample(indices, n_samples))
    ds = ds.select(indices)

    records = []
    for item in ds:
        records.append(
            {
                "dataset": "gsm8k",
                "question": item["question"],
                "ground_truth": _extract_gsm8k_gt(item["answer"]),
            }
        )
    return records


def load_math(n_samples: int = 500, seed: int = 42) -> list[dict]:
    ds = load_dataset("hendrycks/competition_math", split="test")
    indices = list(range(len(ds)))
    if len(indices) > n_samples:
        random.seed(seed)
        indices = sorted(random.sample(indices, n_samples))
    ds = ds.select(indices)

    records = []
    for item in ds:
        records.append(
            {
                "dataset": "math",
                "question": item["problem"],
                "ground_truth": _extract_math_gt(item["solution"]),
            }
        )
    return records


def load_all_data(
    gsm8k_samples: int = 1000, math_samples: int = 500, seed: int = 42
) -> list[dict]:
    """Load both datasets, assign sequential ids."""
    data = load_gsm8k(gsm8k_samples, seed) + load_math(math_samples, seed)
    for i, d in enumerate(data):
        d["id"] = i
    return data
