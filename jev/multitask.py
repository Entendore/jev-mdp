"""In-context zero-shot decision model (v8.5 capstone).
ONE scorer over (state, option-text) pairs; the schema lives in the prompt ->
new option sets are zero-shot (supervised-pretraining in-context RL,
Lee et al. NeurIPS'23, decision-engine edition). Score-then-choose = Jev's
own >255 two-stage design. Gate: monotone k-curve (acc vs #exemplars)."""
import random
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TextEncoder


class MultiTaskScorer:
    def __init__(self, dim=96, hidden=128, seed=0):
        torch.manual_seed(seed); random.seed(seed)
        self.enc = TextEncoder(dim=dim, hidden=128)
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
        self.T = 1.0

    @staticmethod
    def prompt(schema, instructions, exemplars, state):
        L = [f"TASK: {schema}", "OPTIONS:"]
        L += [f"- {o}: {instructions.get(o, '')}" for o in instructions]
        if exemplars:
            L.append("EXAMPLES:")
            L += [f"Q: {s} -> A: {a}" for s, a in exemplars]
        L.append(f"Q: {state} -> A:")
        return "\n".join(L)

    def _logits(self, prompts, option_lists):
        s = self.enc(prompts)
        A = max(len(o) for o in option_lists)
        E = torch.zeros(len(prompts), A, s.shape[1])
        mask = torch.zeros(len(prompts), A, dtype=torch.bool)
        for i, opts in enumerate(option_lists):
            E[i, :len(opts)] = self.enc(opts)
            mask[i, :len(opts)] = True
        s3 = s.unsqueeze(1).expand(-1, A, -1)
        z = self.net(torch.cat([s3 * E, s3 - E, s3], -1)).squeeze(-1)
        return z.masked_fill(~mask, -1e9)

    def fit(self, tasks, epochs=12, batch=32, lr=1e-3):
        recs = []
        for t in tasks:
            for s_, lab in t["samples"]:
                recs.append((self.prompt(t["schema"], t["instructions"],
                                         t.get("exemplars"), s_),
                             t["options"], t["options"].index(lab)))
        opt = torch.optim.AdamW(list(self.enc.parameters())
                                + list(self.net.parameters()), lr=lr)
        for _ in range(epochs):
            random.shuffle(recs)
            for i in range(0, len(recs), batch):
                b = recs[i:i + batch]
                logits = self._logits([r[0] for r in b], [r[1] for r in b])
                loss = F.cross_entropy(logits, torch.tensor([r[2] for r in b]))
                opt.zero_grad(); loss.backward(); opt.step()
        self.enc.eval(); self.net.eval()
        return self

    def fit_temperature(self, val_tasks):
        L, Y = [], []
        with torch.no_grad():
            for t in val_tasks:
                for s_, lab in t["samples"]:
                    L.append(self._logits(
                        [self.prompt(t["schema"], t["instructions"],
                                     t.get("exemplars"), s_)],
                        [t["options"]])[0])
                    Y.append(t["options"].index(lab))
        L, Y = torch.stack(L), torch.tensor(Y)
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(L / log_t.exp(), Y)
            loss.backward()
            return loss
        opt.step(closure)
        self.T = float(log_t.exp())
        return self

    @torch.no_grad()
    def decide(self, schema, instructions, exemplars, state, options):
        p = F.softmax(self._logits(
            [self.prompt(schema, instructions, exemplars, state)],
            [options])[0] / self.T, -1)
        top = int(p.argmax())
        return {"choice": options[top], "confidence": round(float(p[top]), 4),
                "probabilities": {o: round(float(p[i]), 4)
                                  for i, o in enumerate(options)}}

    @torch.no_grad()
    def evaluate(self, tasks):
        hits = tot = 0
        for t in tasks:
            for s_, lab in t["samples"]:
                r = self.decide(t["schema"], t["instructions"],
                                t.get("exemplars"), s_, t["options"])
                hits += r["choice"] == lab; tot += 1
        return hits / max(tot, 1)

    def save(self, path):
        torch.save({"enc": self.enc.state_dict(),
                    "net": self.net.state_dict(),
                    "T": self.T,
                    "dim": self.enc.emb.embedding_dim}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu")
        m = cls(dim=d["dim"])
        m.enc.load_state_dict(d["enc"]); m.net.load_state_dict(d["net"])
        m.T = d["T"]
        m.enc.eval(); m.net.eval()
        return m
