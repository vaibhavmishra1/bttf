"""Dataset loading for GSM8K, MATH, and OlympiadBench."""

import ast
import re
import random
import pandas
from datasets import load_dataset


# ── Ground-truth extraction helpers ────────────────────────────────────────

def _extract_gsm8k_gt(answer_text: str) -> str:
    """GSM8K answers end with '#### <number>'."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip()


def _extract_math_gt(solution_text: str) -> str:
    r"""MATH answers are inside \boxed{...}."""
    matches = re.findall(
        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution_text
    )
    if matches:
        return matches[-1].strip()
    return solution_text.strip()


def _parse_olympiad_answer(raw) -> str:
    """Parse OlympiadBench final_answer field.

    The field may be a Python list (already parsed by HF) or a list-string
    like ``"['2']"`` or ``"['\\\\frac{1}{2}']"``.  We extract the first element.
    """
    # Already a list (native HF parsing)
    if isinstance(raw, list):
        return str(raw[0]).strip() if raw else ""

    raw = str(raw).strip()
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    except Exception:
        pass
    # Fallback: strip brackets and quotes manually
    raw = re.sub(r"^\[|\]$", "", raw).strip().strip("'\"")
    return raw


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
    df = pandas.read_csv(
        "https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv"
    )
    examples = [row.to_dict() for _, row in df.iterrows()]
    indices = list(range(len(examples)))
    if len(indices) > n_samples:
        random.seed(seed)
        indices = sorted(random.sample(indices, n_samples))
    examples = [examples[i] for i in indices]

    records = []
    for item in examples:
        records.append(
            {
                "dataset": "math",
                "question": item["Question"],
                "ground_truth": _extract_math_gt(item["Answer"]),
            }
        )
    return records


def load_olympiadbench(n_samples: int = 300, seed: int = 42) -> list[dict]:
    """Load OlympiadBench competition problems (numerical single-answer only)."""
    ds = load_dataset("zwhe99/simplerl-OlympiadBench", split="test")

    # Keep only single numerical answers for clean evaluation
    def _fa_ok(item):
        fa = item.get("final_answer", "")
        if isinstance(fa, list):
            return len(fa) > 0 and str(fa[0]).strip() not in ("", "None")
        return str(fa).strip() not in ("", "None", "['']")

    items = [
        item for item in ds
        if item.get("answer_type") == "Numerical"
        and str(item.get("is_multiple_answer", "False")).lower() == "false"
        and _fa_ok(item)
    ]

    indices = list(range(len(items)))
    if len(indices) > n_samples:
        random.seed(seed)
        indices = sorted(random.sample(indices, n_samples))
    items = [items[i] for i in indices]

    records = []
    for item in items:
        # Prepend context if it's not "None"
        ctx = str(item.get("context", "None")).strip()
        question = item["question"].strip()
        if ctx and ctx.lower() != "none":
            question = f"{ctx}\n\n{question}"

        records.append(
            {
                "dataset": "olympiadbench",
                "question": question,
                "ground_truth": _parse_olympiad_answer(item["final_answer"]),
            }
        )
    return records


def load_all_data(
    gsm8k_samples: int = 1000,
    math_samples: int = 500,
    olympiadbench_samples: int = 300,
    seed: int = 42,
) -> list[dict]:
    """Load all three datasets and assign sequential ids."""
    data = (
        load_gsm8k(gsm8k_samples, seed)
        + load_math(math_samples, seed)
        + load_olympiadbench(olympiadbench_samples, seed)
    )
    for i, d in enumerate(data):
        d["id"] = i
    return data
