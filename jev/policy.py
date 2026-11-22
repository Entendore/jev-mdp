"""Policy network + heads.
- DecisionPolicy (v2): shared trunk, one head per question, ALL questions in
  ONE forward pass (the MDP analog of Jev's parallel sampler).
- EvidentialHead (v6.6): Sensoy NeurIPS'18; Amini NeurIPS'20.
- UncertaintyWeights (v6.7): Kendall Gal Cipolla CVPR'18.
- fit_temperature (v2): Guo ICML'17."""
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionPolicy(nn.Module):
    def __init__(self, dim, questions, hidden=256, dropout=0.1,
                 evidential: bool = False):
        super().__init__()
        self.questions = questions
        self.evidential = evidential
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout))
        if evidential:
            self.heads = nn.ModuleDict(
                {q.name: EvidentialHead(hidden, q.n_actions) for q in questions})
        else:
            self.heads = nn.ModuleDict(
                {q.name: nn.Linear(hidden, q.n_actions) for q in questions})

    def logits(self, s) -> Dict[str, torch.Tensor]:
        h = self.trunk(s)
        out = {}
        for q in self.questions:
            z = self.heads[q.name](h)
            out[q.name] = z[0] if isinstance(z, tuple) else z
        return out

    def evidence(self, s) -> Dict[str, torch.Tensor]:
        assert self.evidential
        h = self.trunk(s)
        return {q.name: self.heads[q.name](h)[0] for q in self.questions}

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


class EvidentialHead(nn.Module):
    """y|s ~ Dir(alpha), alpha = 1 + Softplus(e(s)). Returns (alpha, E[p], u=K/S)."""
    def __init__(self, dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.n = n_actions
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, n_actions), nn.Softplus())

    def forward(self, h):
        alpha = self.net(h) + 1.0
        S = alpha.sum(-1, keepdim=True)
        return alpha, alpha / S, self.n / S


def _kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    K = alpha.shape[-1]
    S = alpha.sum(-1, keepdim=True)
    return (torch.lgamma(S) - torch.lgamma(alpha).sum(-1, keepdim=True)
            - torch.lgamma(torch.tensor(float(K), device=alpha.device))
            + ((alpha - 1.0) * (torch.digamma(alpha)
                                - torch.digamma(S))).sum(-1, keepdim=True)
            ).squeeze(-1)


def edl_loss(alpha: torch.Tensor, target: torch.Tensor,
             epoch: int = 10, anneal: float = 10.0) -> torch.Tensor:
    S = alpha.sum(-1, keepdim=True)
    p = alpha / S
    y = F.one_hot(target, p.shape[-1]).float()
    a_t = min(1.0, epoch / anneal)
    alpha_t = y + (1.0 - y) * alpha
    return ((((y - p) ** 2).sum(-1)
             + ((p * (1 - p)) / (S + 1)).sum(-1)).mean()
            + a_t * _kl_dirichlet_uniform(alpha_t).mean())


class UncertaintyWeights(nn.Module):
    def __init__(self, names: List[str]):
        super().__init__()
        self.log_var = nn.ParameterDict(
            {n: nn.Parameter(torch.zeros(())) for n in names})

    def penalty(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(0.5 * torch.exp(-2.0 * self.log_var[n]) * losses[n]
                   + self.log_var[n] for n in losses)


def fit_temperature(logits: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Per-head temperature scaling (Guo et al. 2017)."""
    T = {}
    for name, lg in logits.items():
        lg = lg.detach()
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(lg / log_t.exp(), targets[name])
            loss.backward()
            return loss
        opt.step(closure)
        T[name] = float(log_t.exp())
    return T
