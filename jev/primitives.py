"""Primitives (v1) + shared math.
Groundings: proper scoring rules Gneiting&Raftery JASA'07 (Brier calibration =
the RLCD analog); Wilson '27 intervals; VI Putnam'94; MaxEnt soft VI
Ziebart'10 / Haarnoja ICML'18 (tau=None -> hard VI exactly)."""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Question:
    name: str
    qtype: str                                    # "choice" | "noul" | "score"
    options: Optional[List[str]] = None
    score_levels: Optional[List[str]] = None
    instructions: str = ""
    cost_matrix: Optional[Dict[str, Dict[str, float]]] = None
    weight: float = 1.0

    def __post_init__(self):
        if self.qtype == "noul":
            self.options = ["no", "yes"]
        if self.qtype == "choice":
            assert 2 <= len(self.options) <= 255  # Jev-style option cap
        if self.qtype == "score":
            assert 2 <= len(self.score_levels) <= 10

    @property
    def n_actions(self) -> int:
        return len(self.options)

    def cost_array(self) -> Optional[np.ndarray]:
        if self.cost_matrix is None:
            return None
        n = len(self.options)
        C = np.full((n, n), 1.0)
        np.fill_diagonal(C, 0.0)
        for i, ti in enumerate(self.options):
            for j, tj in enumerate(self.options):
                if ti in self.cost_matrix and tj in self.cost_matrix[ti]:
                    C[i, j] = self.cost_matrix[ti][tj]
        return C


@dataclass
class MultiChoiceQuestion(Question):
    """k-of-n slate (v10.3): Cesa-Bianchi&Lugosi'12 semi-bandits."""
    k: int = 2


def wilson_interval(success, total, z: float = 1.96):
    s = np.asarray(success, float); n = np.asarray(total, float)
    p = s / np.maximum(n, 1e-9)
    denom = 1.0 + z * z / np.maximum(n, 1e-9)
    center = (p + z * z / (2 * np.maximum(n, 1e-9))) / denom
    half = (z * np.sqrt(p * (1 - p) / np.maximum(n, 1e-9)
                        + z * z / (4 * np.maximum(n, 1e-9) ** 2)) / denom)
    has = n > 0
    return (np.where(has, np.clip(center - half, 0, 1), 0.0),
            np.where(has, np.clip(center + half, 0, 1), 1.0))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 12) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = ((conf >= bins[i]) & (conf <= bins[i + 1])) if i == 0 else \
            ((conf > bins[i]) & (conf <= bins[i + 1]))
        if m.any():
            e += m.mean() * abs(float(conf[m].mean()) - float(acc[m].mean()))
    return float(e)


def macro_f1(pred, truth, n_classes) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (truth == c)).sum())
        fp = int(((pred == c) & (truth != c)).sum())
        fn = int(((pred != c) & (truth == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def _logsumexp(m: np.ndarray, axis: int = 1) -> np.ndarray:
    mx = m.max(axis=axis, keepdims=True)
    return (mx + np.log(np.exp(m - mx).sum(axis=axis, keepdims=True))).squeeze(axis)


def solve_mdp(R, T, term, gamma, tau=None, iters=400, tol=1e-9):
    """Hard VI (tau=None) or entropy-regularized soft VI. Terminals absorbing;
    their value is the one-time entry reward (v3 convention)."""
    K, A, _ = T.shape
    V = np.zeros(K)
    for _ in range(iters):
        Q = R + gamma * np.einsum("kak,k->ka", T, V)
        Vn = np.where(term, R[:, 0],
                      Q.max(1) if tau is None else tau * _logsumexp(Q / tau, 1))
        if np.abs(Vn - V).max() < tol:
            V = Vn; break
        V = Vn
    Q = R + gamma * np.einsum("kak,k->ka", T, V)
    if tau is None:
        Pi = np.eye(A)[Q.argmax(1)].copy()
    else:
        Pi = np.exp((Q - Q.max(1, keepdims=True)) / tau)
        Pi = Pi / Pi.sum(1, keepdims=True)
    Pi[term] = 0.0
    return V, Q, Pi


def dict_to_text(d, nd: int = 2) -> str:
    """Deterministic quantized serialization (v5) — cluster-friendly states."""
    if isinstance(d, str):
        return d
    items = []
    for k in sorted(d):
        v = d[k]
        if isinstance(v, float):
            v = round(v, nd)
        if isinstance(v, (list, tuple)):
            v = " ".join(str(round(x, nd)) if isinstance(x, float) else str(x)
                         for x in v)
        items.append(f"{k}={v}")
    return "; ".join(items)


def apply_mask(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """v5 legality masking (Alshiekh AAAI'18 shielding, practical form)."""
    p = probs * mask
    s = p.sum()
    if s <= 1e-9:
        p, s = mask.astype(float), max(mask.sum(), 1e-9)
    return p / s


def expected_cost_action(probs: np.ndarray, cost: np.ndarray) -> int:
    """Bayes-risk rule argmin_a E[cost] — Chow'70 rejection special case."""
    return int(np.argmin(probs @ cost))
