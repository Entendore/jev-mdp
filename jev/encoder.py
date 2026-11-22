"""Encoders.
- TextEncoder (v1): hashed uni+bigram bag (swap for sentence-transformers).
- BeliefEncoder (v7.9): GRU belief state — Kaelbling'98 POMDP, DRQN AAAI'15.
- RepresentationPretrainer (v9.1): InfoNCE CPC (van den Oord'18) + SPR
  dynamics-consistency (Schwarzer ICLR'21)."""
import random
from typing import List, Optional, Tuple
import zlib
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, dim=128, vocab=1 << 15, max_len=256, hidden=256, dropout=0.1):
        super().__init__()
        self.vocab, self.max_len = vocab, max_len
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.out = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.LayerNorm(hidden), nn.GELU())

    def _ids(self, t: str) -> torch.Tensor:
        toks = t.lower().split()[: self.max_len] or ["<empty>"]
        toks = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        return torch.tensor([zlib.crc32(w.encode()) % self.vocab for w in toks])

    def forward(self, texts: List[str]) -> torch.Tensor:
        dev = self.emb.weight.device
        rows = [self.emb(self._ids(t).to(dev)).mean(0) for t in texts]
        return self.out(torch.stack(rows))


class BeliefEncoder(nn.Module):
    def __init__(self, dim: int, hidden: int = 96):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True)

    def forward(self, emb_seq):                      # (B, L, dim)
        _, h = self.gru(emb_seq)
        return h[-1]                                 # (B, hidden)


class RepresentationPretrainer:
    """(a) InfoNCE on token-dropout views; (b) SPR: dyn(enc(s)) ~ enc(s')."""
    def __init__(self, engine, proj_dim: int = 64, temp: float = 0.1):
        self.eng, self.temp = engine, temp
        d = engine.dim
        self.proj = nn.Sequential(nn.Linear(d, proj_dim), nn.ReLU(),
                                  nn.Linear(proj_dim, proj_dim)).to(engine.device)
        self.dyn = nn.Linear(d, d).to(engine.device)

    @staticmethod
    def _augment(texts: List[str], rng: random.Random) -> List[str]:
        out = []
        for t in texts:
            toks = [w for w in t.split() if rng.random() > 0.15] or [t]
            rng.shuffle(toks)
            out.append(" ".join(toks))
        return out

    def pretrain(self, stream: List[str],
                 pairs: Optional[List[Tuple[str, str]]] = None,
                 steps: int = 800, batch: int = 64, lr: float = 1e-3):
        eng = self.eng
        opt = torch.optim.AdamW(list(eng.encoder.parameters())
                                + list(self.proj.parameters())
                                + list(self.dyn.parameters()), lr=lr)
        rng = random.Random(0)
        for _ in range(steps):
            b = rng.sample(stream, min(batch, len(stream)))
            za = F.normalize(self.proj(eng.encoder(self._augment(b, rng))), dim=1)
            zb = F.normalize(self.proj(eng.encoder(self._augment(b, rng))), dim=1)
            loss = F.cross_entropy(za @ zb.T / self.temp,
                                   torch.arange(len(b), device=eng.device))
            if pairs:
                sa = eng.encoder([p[0] for p in pairs])
                sb = eng.encoder([p[1] for p in pairs])
                loss = loss + F.mse_loss(self.dyn(sa), sb.detach())
            opt.zero_grad(); loss.backward(); opt.step()
        return {"steps": steps, "n_stream": len(stream)}
