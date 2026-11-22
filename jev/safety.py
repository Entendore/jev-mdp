"""Safety filter (v6.14): Control Barrier Functions (Ames ECC'17; IEEE T-RO'19)
— a FORMAL shield replacing hand-written if-else masks. Discrete condition:
keep action a only if h(x') >= (1-gamma)*h(x) with per-action one-step margin
deltas dh(a). Forward invariance of the safe set follows on the continuous
system; here it is the discrete practical filter."""
from typing import Callable, Dict, List, Tuple
import numpy as np


class CBFFilter:
    def __init__(self, h_fn: Callable[[dict], float],
                 dh_fn: Callable[[dict], Dict[str, float]], gamma: float = 0.4):
        self.h, self.dh, self.gamma = h_fn, dh_fn, gamma

    def filter(self, state, probs: np.ndarray,
               options: List[str]) -> Tuple[str, bool]:
        h = float(self.h(state))
        deltas = self.dh(state)
        legal = [i for i, a in enumerate(options)
                 if h + float(deltas.get(a, 0.0)) >= (1.0 - self.gamma) * h]
        if not legal:
            # emergency: take the action with the LARGEST margin increase
            i = int(np.argmax([deltas.get(a, 0.0) for a in options]))
            return options[i], False
        m = np.zeros(len(options)); m[legal] = 1.0
        p = probs * m
        if p.sum() <= 1e-9:
            return options[int(np.argmax(m))], True
        return options[int(np.argmax(p / p.sum()))], True


def compose_masks(*masks):
    """AND-compose legality masks (API preconditions AND CBF AND velocity)."""
    out = None
    for m in masks:
        if m is None:
            continue
        out = m.copy() if out is None else out * m
    return out
