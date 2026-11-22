"""Interpretability pack (v9.9).
- shapley_fields: permutation Shapley over serialized dict fields
  (Lundberg&Lee NeurIPS'17, permutation form).
- counterfactual_flip: minimal field edit that flips the decision
  (Wachter Mittelstadt Russell'17).
- run_checklist: INV (paraphrase stability) + DIR (marker direction)
  functional tests (Ribeiro ACL'20) — CI-gate these."""
import math, random
from typing import Dict, List
import numpy as np

from .primitives import dict_to_text


def shapley_fields(eng, state: Dict, fields: List[str], n_perm: int = 24,
                   seed: int = 0) -> Dict:
    rng = random.Random(seed)

    def value(active: List[str]) -> float:
        r = eng.decide(dict_to_text({f: state[f] for f in active}))
        conf = r.get("confidence", 0.0)
        return math.log(max(conf, 1e-9)) if r.get("action") not in (
            None, "ESCALATE") else -9.0

    phi = {f: 0.0 for f in fields}
    for _ in range(n_perm):
        order = list(fields); rng.shuffle(order)
        cur: List[str] = []
        for f in order:
            v0 = value(cur)
            cur = cur + [f]
            phi[f] += (value(cur) - v0) / n_perm
    total = value(list(fields))
    return {"attributions": {f: round(v, 4) for f, v in phi.items()},
            "log_conf_total": round(total, 4),
            "base_residual": round(total - sum(phi.values()), 4)}


def counterfactual_flip(eng, state: Dict, candidates: Dict[str, List]) -> Dict:
    base = eng.decide(dict_to_text(state))
    base_action = base.get("action")
    with __import__("torch").no_grad():
        e0 = eng.encoder([dict_to_text(state)]).cpu().numpy()
    best = None
    for f, vals in candidates.items():
        for v in vals:
            s2 = {**state, f: v}
            r = eng.decide(dict_to_text(s2))
            if r.get("action") != base_action and r.get("action") not in (
                    None, "ESCALATE"):
                with __import__("torch").no_grad():
                    d = float(np.linalg.norm(
                        eng.encoder([dict_to_text(s2)]).cpu().numpy() - e0))
                if best is None or d < best["dist"]:
                    best = {"field": f, "value": v,
                            "flips_to": r["action"], "dist": round(d, 4)}
    return {"decision": base_action, "minimal_flip": best}


def run_checklist(eng, tests: List[Dict]) -> Dict:
    """tests: {'type':'inv','text':...,'min_conf':0.9} or
    {'type':'dir','text':...,'marker':'urgent','question':qn,'label':'yes'}"""
    res = {"inv_pass": 0, "inv_total": 0, "dir_pass": 0, "dir_total": 0}
    for t in tests:
        if t["type"] == "inv":
            r0 = eng.decide(t["text"])
            if r0.get("confidence", 0) < t.get("min_conf", 0.9):
                continue
            base = r0.get("action")
            for _ in range(5):
                p = " ".join(w for w in t["text"].split()
                             if random.random() > 0.2) or t["text"]
                res["inv_total"] += 1
                res["inv_pass"] += eng.decide(p).get("action") == base
        elif t["type"] == "dir":
            p0 = eng.decide(t["text"])["outputs"][t["question"]] \
                .get("probabilities", {}).get(t["label"], 0.0)
            p1 = eng.decide(t["text"] + " " + t["marker"]) \
                ["outputs"][t["question"]].get("probabilities", {}) \
                .get(t["label"], 0.0)
            res["dir_total"] += 1
            res["dir_pass"] += p1 >= p0
    rate = ((res["inv_pass"] + res["dir_pass"])
            / max(res["inv_total"] + res["dir_total"], 1))
    res["pass_rate"] = round(rate, 4)
    return res
