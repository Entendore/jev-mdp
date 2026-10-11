"""Fleet layer (v9.12): MoE routing over specialist engines (Shazeer ICLR'17,
load-balancing auxiliary) + FrugalGPT cascade (Chen'23) with learned
per-stage confidence thresholds (coordinate ascent on validation)."""
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    def __init__(self, dim: int, n_experts: int, coef: float = 0.01):
        super().__init__()
        self.gate = nn.Linear(dim, n_experts)
        self.coef = coef

    def forward(self, h):
        return F.softmax(self.gate(h), -1)

    def load_balance_loss(self, gates: torch.Tensor) -> torch.Tensor:
        f = gates.mean(0)
        P = gates.mean(0)
        return self.coef * gates.shape[1] * (f * P).sum()

    def fit(self, X: torch.Tensor, y_expert: torch.Tensor, epochs=200, lr=1e-3):
        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        for _ in range(epochs):
            g = self(X)
            loss = F.cross_entropy(self.gate(X), y_expert) \
                   + self.load_balance_loss(g)
            opt.zero_grad(); loss.backward(); opt.step()
        self.eval()
        return self


class EngineFleet:
    """Route each state to its best specialist engine."""
    def __init__(self, specialists: Dict[str, object]):
        self.names = list(specialists)
        self.specialists = specialists
        first = specialists[self.names[0]]
        self.router = MoERouter(first.dim, len(self.names))
        self.qname = first._primary()

    def fit_router(self, val_texts, val_labels, epochs=200):
        """Best specialist per sample = correct + most confident."""
        feats, targets = [], []
        with torch.no_grad():
            embs = self.specialists[self.names[0]].encoder(val_texts)
            for i, (t, lab) in enumerate(zip(val_texts, val_labels)):
                best, bj = -1.0, 0
                for j, name in enumerate(self.names):
                    r = self.specialists[name].decide(t)
                    e = r["outputs"].get(self.qname, {})
                    ok = float(e.get("choice") == lab[self.qname])
                    score = ok * 2 + e.get("confidence", 0.0)
                    if score > best:
                        best, bj = score, j
                feats.append(embs[i])
                targets.append(bj)
        self.router.fit(torch.stack(feats),
                        torch.tensor(targets), epochs=epochs)
        return self

    def route(self, text) -> Tuple[str, Dict]:
        with torch.no_grad():
            emb = self.specialists[self.names[0]].encoder([text])
            g = self.router(emb)[0]
        name = self.names[int(g.argmax())]
        return name, self.specialists[name].decide(text)


class Cascade:
    """FrugalGPT-lite: stages of (engine, cost); thresholds learned by
    coordinate ascent under an accuracy floor."""
    def __init__(self, stages: List[Dict]):
        self.stages = stages                       # {"engine", "cost"}
        self.taus = [0.5] * len(stages)
        self.qname = stages[0]["engine"]._primary()

    def _simulate(self, texts, labels):
        cost = acc = 0.0
        for t, y in zip(texts, labels):
            for st, tau in zip(self.stages, self.taus):
                r = st["engine"].decide(t)
                cost += st["cost"]
                e = r["outputs"].get(self.qname, {})
                if e.get("confidence", 0.0) >= tau:
                    acc += float(e.get("choice") == y[self.qname])
                    break
        return cost / max(len(texts), 1), acc / max(len(texts), 1)

    def fit_thresholds(self, val_texts, val_labels, target_acc=0.9, iters=60):
        for _ in range(iters):
            for i in range(len(self.taus)):
                c0, a0 = self._simulate(val_texts, val_labels)
                self.taus[i] = min(0.99, self.taus[i] + 0.02)
                c1, a1 = self._simulate(val_texts, val_labels)
                if a1 < target_acc or c1 > c0:
                    self.taus[i] -= 0.02
        return {"taus": [round(t, 3) for t in self.taus]}

    def decide(self, text) -> Dict:
        r = None
        for st, tau in zip(self.stages, self.taus):
            r = st["engine"].decide(text)
            e = r["outputs"].get(self.qname, {})
            if e.get("confidence", 0.0) >= tau:
                return {**r, "cascade_stage": self.stages.index(st)}
        return {**r, "cascade_stage": "escalate"}
