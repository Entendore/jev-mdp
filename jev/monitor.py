"""Ops loop (v8.7 + v6.11 + v10.8).
- OveroptimizationMonitor (Gao'23 scaling laws for reward overoptimization):
  track (KL estimate, real-data OPE proxy); HALT when KL > kappa while the
  OPE slope <= 0; restore the best checkpoint.
- LiveOps: reference-vs-recent embedding drift (C2ST, v6.11) + EM label-shift
  correction of the P matrix (Saerens'02) + regime switching hook (v10.8)."""
from typing import List, Optional
import numpy as np
import torch

from .calibration import c2st, em_label_shift


class OveroptimizationMonitor:
    def __init__(self, kappa: float = 0.5, window: int = 5):
        self.kl: List[float] = []
        self.ope: List[float] = []
        self.kappa, self.window = kappa, window
        self.best = -1e18
        self.best_state: Optional[dict] = None

    def probe(self, kl: float, ope: float, state_dict=None):
        self.kl.append(kl); self.ope.append(ope)
        if ope > self.best:
            self.best, self.best_state = ope, state_dict

    def should_stop(self):
        if len(self.kl) < self.window:
            return False, {}
        kl = float(np.mean(self.kl[-self.window:]))
        slope = float(np.polyfit(range(self.window),
                                 self.ope[-self.window:], 1)[0])
        return bool(kl > self.kappa and slope <= 0.0), {
            "kl": round(kl, 4), "ope_slope": round(slope, 5),
            "best_ope": round(self.best, 4)}

    def restore_best(self, engine):
        if self.best_state is not None:
            engine.policy.load_state_dict(self.best_state)
            engine._cache.clear()


class LiveOps:
    """Production drift loop: append served texts; check C2ST vs the
    calibration reference; correct label shift when it fires."""
    def __init__(self, engine, ref_embs: np.ndarray, window: int = 300):
        self.eng = engine
        self.ref = ref_embs
        self.buf: List[str] = []
        self.window = window

    def observe(self, text: str):
        self.buf.append(text)
        if len(self.buf) > self.window:
            self.buf.pop(0)

    def drift_report(self, seed=0) -> dict:
        if len(self.buf) < 50:
            return {"n": len(self.buf), "skip": True}
        with torch.no_grad():
            cur = self.eng.encoder(self.buf).cpu().numpy()
        return c2st(self.ref, cur, seed=seed)

    def correct_label_shift(self, target_prior: np.ndarray) -> dict:
        qn = self.eng._primary()
        self.eng.pm.P[qn] = em_label_shift(self.eng.pm.P[qn], target_prior)
        if self.eng.T is not None:
            self.eng.build_reward()
        self.eng._cache.clear()
        return {"corrected": qn}
