"""Answer extraction and equivalence checking."""

import re
from config import FLOAT_TOL


# ── Boxed-answer extraction (primary for S vs S') ────────────────────────────

# Matches the last \boxed{...} — handles one level of nested braces
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{...} from a model solution.

    Qwen2.5-3B-Instruct reliably wraps its final answer in \\boxed{}, so this
    is used as the primary extractor when comparing S with S'.
    Falls back to an empty string if no boxed answer is found.
    """
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return ""


# ── Heuristic answer extraction (fallback / ground-truth comparison) ─────────

_ANSWER_PATTERNS = [
    # #### <number>  (GSM8K style)
    r"####\s*([^\n]+)",
    # "The answer is <X>"
    r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*([^\n\.]+)",
    # "Answer: <X>"
    r"[Aa]nswer[:\s]+([^\n]+)",
    # \boxed{<X>}
    r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
]

_NUMBER_RE = re.compile(
    r"-?\d[\d,]*\.?\d*(?:/\d+)?(?:\s*\\?%)?|"  # decimal / fraction / percent
    r"-?\.\d+"  # e.g. .5
)


def extract_answer(text: str) -> str:
    """Extract the final answer from a model solution string.

    Used for comparing the model's answer against the ground-truth label.
    Tries structured patterns first, then falls back to the last number.
    """
    for pattern in _ANSWER_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1].strip().replace(",", "")
            answer = answer.strip(" .$")
            if answer:
                return answer

    # Fallback: last number in text
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").strip()

    return ""


# ── Answer equivalence ──────────────────────────────────────────────────────

def _try_float(s: str) -> float | None:
    """Try to parse *s* as a float, stripping common decorations."""
    s = s.strip().replace(",", "").rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _sympy_equiv(a: str, b: str) -> bool | None:
    """Return True/False if sympy can decide; None if it can't parse."""
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify, nsimplify

        def _parse(expr_str: str):
            try:
                return parse_latex(expr_str)
            except Exception:
                pass
            from sympy import sympify
            return sympify(expr_str)

        ea = _parse(a)
        eb = _parse(b)
        diff = simplify(nsimplify(ea) - nsimplify(eb))
        return diff == 0
    except Exception:
        return None


def answers_equivalent(a: str, b: str, tol: float = FLOAT_TOL) -> bool:
    """Check if two answer strings are equivalent.

    1. Exact string match
    2. Float comparison (with tolerance)
    3. Sympy symbolic equivalence
    """
    a, b = a.strip(), b.strip()

    # 1. exact match
    if a == b:
        return True

    # 2. float comparison
    fa, fb = _try_float(a), _try_float(b)
    if fa is not None and fb is not None:
        if abs(fa - fb) <= tol:
            return True

    # 3. sympy
    sym = _sympy_equiv(a, b)
    if sym is not None:
        return sym

    return False
