"""Config loader — makes config/base.yaml actually drive the engine.
Deep-merges user YAML over defaults; engine_from_config builds JevMDP."""
import copy
from typing import Dict
import yaml

from .primitives import Question
from .engine import JevMDP

DEFAULTS: Dict = {
    "dim": 128, "seed": 0, "k": 24, "gamma": 0.9, "dirichlet_alpha": 0.5,
    "kappa": 20.0, "talpha": 0.5, "n_min": 8, "memory_enabled": True,
    "ood_fpr": 0.05, "belief_enabled": False, "evidential": False,
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(dict(base[k]), v)
        else:
            base[k] = v
    return base


def load_config(path) -> Dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return _deep_merge(copy.deepcopy(DEFAULTS), user.get("engine", user))


def questions_from_config(cfg) -> list:
    return [Question(**q) for q in cfg.get("questions", [])]


def engine_from_config(cfg, questions=None) -> JevMDP:
    qs = questions if questions is not None else questions_from_config(cfg)
    kwargs = {k: cfg[k] for k in DEFAULTS if k in cfg}
    return JevMDP(qs, **kwargs)
