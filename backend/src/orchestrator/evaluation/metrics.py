"""pass@k and pass^k from n repeated trials of one task, c of which passed.

pass@k is the chance that at least one of k trials, drawn without replacement
from the n, passed. pass^k is the chance that all k passed (tau-bench's
reliability measure). Both are exact, not sampled.
"""

from math import comb


def _check(n: int, c: int, k: int) -> None:
    if not 0 <= c <= n:
        raise ValueError(f"passes must be between 0 and trials, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and trials, got k={k}, n={n}")


def pass_at_k(n: int, c: int, k: int) -> float:
    _check(n, c, k)
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    _check(n, c, k)
    return comb(c, k) / comb(n, k)
