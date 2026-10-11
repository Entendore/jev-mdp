"""Query-level nonparametric memory (v9.2).
P = lam * P_knn + (1-lam) * P_net; lam rises when neighbors are close or the
parametric model is weak (kNN-LM interpolation, Khandelwal ICLR'20)."""
from typing import List, Tuple
import numpy as np
import torch


class KNNMemory:
    def __init__(self, engine, tau=0.5, l0=0.3, k=15):
        self.eng, self.tau, self.l0, self.k = engine, tau, l0, k
        self.E = np.zeros((0, engine.dim))
        self.y: List[int] = []
        self.texts: List[str] = []

    @torch.no_grad()
    def add(self, texts, labels):
        q = self.eng._q(self.eng._primary())
        self.E = np.vstack([self.E, self.eng.encoder(texts).cpu().numpy()])
        self.y += [q.options.index(l) for l in labels]
        self.texts += list(texts)

    @torch.no_grad()
    def posterior(self, text, q, p_net) -> Tuple[np.ndarray, float]:
        if len(self.y) < 5:
            return p_net, 0.0
        e = self.eng.encoder([text]).cpu().numpy()[0]
        d2 = ((self.E - e) ** 2).sum(1)
        idx = np.argsort(d2)[: min(self.k, len(self.y))]
        w = np.exp(-d2[idx] / self.tau ** 2)
        p = np.zeros(q.n_actions)
        np.add.at(p, np.array(self.y)[idx], w)
        if p.sum() <= 0:
            return p_net, 0.0
        p /= p.sum()
        near = float(np.exp(-d2[idx].min() / self.tau ** 2))
        lam = float(np.clip(self.l0 + 0.5 * near, 0.0, 0.8))
        out = lam * p + (1 - lam) * p_net
        return out / out.sum(), lam

    def dynamic_exemplars(self, text, k=3):
        """Top-k (neighbor_text, label) pairs -> in-context scorer EXAMPLES."""
        if len(self.y) < k:
            return []
        e = self.eng.encoder([text]).cpu().numpy()[0]
        idx = np.argsort(((self.E - e) ** 2).sum(1))[:k]
        q = self.eng._q(self.eng._primary())
        return [(self.texts[i], q.options[self.y[i]]) for i in idx]
