"""Human alignment + data-quality layer.
- PreferenceReward (v8.1): Bradley-Terry (Christiano NeurIPS'17).
- kl_regularized_improvement (v8.1): EXACT max_pi E[R] - beta*KL(pi||pi_ref)
  via the Donsker-Varadhan dual (Ziegler'19) — a hard drift brake.
- InteractiveCorrector (v8.2): COACH/TAMER (MacGlashan AAAI'17).
- gce_loss (v9.10, Zhang NeurIPS'18); dart_augment (v8.4a, de Haan'19);
  cfda_augment (v8.4b, Pitis ICML'20); GroupDRO (v8.4d, Sagawa ICLR'20);
  LabelModel (v6.15, Ratner VLDB'17 Snorkel)."""
import math, random
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F


class PreferenceReward:
    def __init__(self, K, A, lr=0.1, l2=1e-4):
        self.K, self.A, self.lr, self.l2 = K, A, lr, l2
        self.r = np.zeros((K, A))

    def fit(self, triples, epochs=300):
        P = [(k, ap, am) for k, ap, am in triples
             if ap != am and 0 <= k < self.K and ap < self.A and am < self.A]
        if not P:
            return 0.0
        last = -math.inf
        for _ in range(epochs):
            g = np.zeros_like(self.r); ll = 0.0
            for k, ap, am in P:
                p = 1.0 / (1.0 + math.exp(-(self.r[k, ap] - self.r[k, am])))
                ll += math.log(max(p, 1e-12))
                g[k, ap] += 1 - p; g[k, am] -= 1 - p
            self.r += self.lr * (g / len(P) - self.l2 * self.r)
            if abs(ll - last) < 1e-9:
                break
            last = ll
        return last / len(P)

    def matrix(self):
        return self.r - self.r.mean(1, keepdims=True)  # differences only


def kl_regularized_improvement(pi_ref, R, beta=0.1):
    """pi*(a|s) proportional to pi_ref(a|s) * exp(R(s,a)/beta) — exact brake."""
    logits = np.log(np.clip(pi_ref, 1e-9, None)) + R / beta
    logits -= logits.max(1, keepdims=True)
    P = np.exp(logits)
    return P / P.sum(1, keepdims=True)


class InteractiveCorrector:
    """Human +/- after watching an action -> pseudo-reward h*b (COACH)."""
    def __init__(self, K, A, h=0.5):
        self.sum = np.zeros((K, A)); self.n = np.zeros((K, A)); self.h = h

    def update(self, k, a, b):                    # b in {+1, 0, -1}
        self.sum[k, a] += self.h * b; self.n[k, a] += 1

    def reward_addendum(self, min_n=1):
        return np.where(self.n >= min_n,
                        self.sum / np.maximum(self.n, 1), 0.0)


def gce_loss(logits, target, q=0.7):
    """Generalized cross-entropy (Zhang'18): CE(q->0) <-> MAE(q=1)."""
    p = F.softmax(logits, -1).gather(1, target.unsqueeze(1)).squeeze(1)
    return ((1.0 - p.clamp_min(1e-9) ** q) / q).mean()


def dart_augment(texts, echo_keys=("prev_action", "last_tool", "last_step"),
                 p_drop=0.5, rng=random.Random(0)):
    """Strip action-echo fields w.p. p_drop — breaks BC copycat behavior."""
    def strip(t, key):
        return ";".join(p for p in t.split(";")
                        if not p.strip().startswith(f"{key}="))
    return [strip(t, rng.choice(echo_keys)) if rng.random() < p_drop else t
            for t in texts]


def cfda_augment(states: List[Dict], swappable: Dict[str, List], n=1,
                 rng=random.Random(0)):
    """Swap action-INDEPENDENT fields — locally-factored transitions (Pitis'20)."""
    out = []
    for _ in range(n):
        for s in states:
            s2 = dict(s)
            for k, vals in swappable.items():
                if vals:
                    s2[k] = rng.choice(vals)
            out.append(s2)
    return out


class GroupDRO:
    """Exponentiated-gradient group weights -> worst-cluster minimizer."""
    def __init__(self, n_groups, step=0.25):
        self.eta = np.full(n_groups, 1.0 / n_groups); self.step = step

    def weights(self, group_losses):
        self.eta *= np.exp(self.step * group_losses)
        self.eta /= self.eta.sum()
        return self.eta


class LabelModel:
    """Naive-Bayes teacher-vote combiner (Snorkel); abstain = -1."""
    def __init__(self, n_classes):
        self.C, self.acc = n_classes, None

    def fit(self, gold_votes, gold_y, n_teachers):
        self.acc = np.full((n_teachers, self.C), 0.5)
        for t in range(n_teachers):
            for c in range(self.C):
                sel = np.where(gold_y == c)[0]
                cov = (gold_votes[sel, t] >= 0).sum()
                self.acc[t, c] = ((gold_votes[sel, t] == c).sum() + 1) / (cov + 2)
        return self

    def predict(self, votes):
        logp = np.full(self.C, -math.log(self.C))
        for t, l in enumerate(votes):
            if l < 0:
                continue
            like = np.array([self.acc[t, c] if l == c else 1.0 - self.acc[t, c]
                             for c in range(self.C)])
            logp += np.log(np.clip(like, 1e-6, 1.0))
        p = np.exp(logp - logp.max()); p /= p.sum()
        return int(p.argmax()), p
