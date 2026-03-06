"""Answer extraction, equivalence checking, and Fix 3: BLEU + embedding hybrid.

Changes vs exp1/utils.py:
  + compute_bleu_score(Q, Q')   — sentence-level BLEU with smoothing.
  + compute_bleu_cycle(Q, Q')   — 1 - BLEU; a distance metric (lower = more preserved).
  + compute_hybrid_cycle(qc, bc, w) — weighted blend of embedding + BLEU distances.
  = extract_answer, extract_boxed_answer, answers_equivalent — unchanged from exp1.
"""

import re

from config import FLOAT_TOL, BLEU_WEIGHT


# ── Boxed-answer extraction ───────────────────────────────────────────────────

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{...} from a solution string."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""


# ── Heuristic answer extraction ───────────────────────────────────────────────

_ANSWER_PATTERNS = [
    r"####\s*([^\n]+)",
    r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*([^\n\.]+)",
    r"[Aa]nswer[:\s]+([^\n]+)",
    r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
]

_NUMBER_RE = re.compile(
    r"-?\d[\d,]*\.?\d*(?:/\d+)?(?:\s*\\?%)?|"
    r"-?\.\d+"
)


def extract_answer(text: str) -> str:
    """Extract the final answer via structured patterns then last-number fallback."""
    for pattern in _ANSWER_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1].strip().replace(",", "").strip(" .$")
            if answer:
                return answer
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return ""


# ── Regex answer equivalence (kept for comparison with exp1) ──────────────────

def _try_float(s: str) -> float | None:
    s = s.strip().replace(",", "").rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _sympy_equiv(a: str, b: str) -> bool | None:
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

        ea, eb = _parse(a), _parse(b)
        return simplify(nsimplify(ea) - nsimplify(eb)) == 0
    except Exception:
        return None


def answers_equivalent(a: str, b: str, tol: float = FLOAT_TOL) -> bool:
    """Regex-based equivalence: exact → float → sympy."""
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    fa, fb = _try_float(a), _try_float(b)
    if fa is not None and fb is not None and abs(fa - fb) <= tol:
        return True
    sym = _sympy_equiv(a, b)
    if sym is not None:
        return sym
    return False


# ── Fix 3: BLEU-based cycle distance ─────────────────────────────────────────

def compute_bleu_score(reference: str, hypothesis: str) -> float:
    """Sentence-level BLEU between *reference* and *hypothesis*.

    Uses NLTK sentence_bleu with smoothing (method1) so that short / zero-
    overlap sentences don't collapse to 0. Tokenisation is simple whitespace
    + lowercase, which is sufficient for measuring Q ↔ Q' word overlap.

    Returns a value in [0, 1] where 1 = identical token sequences.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    sf = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=sf)


def compute_bleu_cycle(q: str, q_prime: str) -> float:
    """BLEU-based cycle distance: 1 − BLEU(Q, Q').

    Lower → Q' is more similar to Q (good cycle).
    Higher → Q' diverges from Q   (bad cycle).
    Same direction as question_cycle so they can be weighted directly.
    """
    return 1.0 - compute_bleu_score(q, q_prime)


def compute_hybrid_cycle(
    question_cycle: float,
    bleu_cycle: float,
    weight: float = BLEU_WEIGHT,
) -> float:
    """Weighted blend of embedding and BLEU cycle distances.

    hybrid_cycle = weight * bleu_cycle + (1 - weight) * question_cycle

    Default weight=0.5 gives equal contribution. The embedding captures
    deep semantic similarity; BLEU captures surface n-gram overlap,
    which is more sensitive to exact number preservation.
    """
    return weight * bleu_cycle + (1.0 - weight) * question_cycle


def compute_bleu_cycles_batch(
    questions: list[str], questions_prime: list[str]
) -> list[float]:
    """Compute bleu_cycle for every (Q, Q') pair."""
    return [compute_bleu_cycle(q, qp) for q, qp in zip(questions, questions_prime)]
