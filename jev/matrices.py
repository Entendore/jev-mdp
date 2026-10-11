"""Explicit MDP matrices (v3-v4) + successor features (v7.3).
- StateMatrix: k-means++ (Arthur&Vassilvitskii'07) + absorbing terminals.
- ProbabilisticMatrix: Dirichlet-Jeffreys posterior blended with the neural
  policy (kNN-LM-style trust) + Wilson intervals.
- RewardMatrix (v4): base / potential-shaped (Ng Harada Russell ICML'99) /
  bonus / pessimistic (Wilson lower bounds; Rashidinejad'21, SPIBB Laroche'19).
- SuccessorFeatures (v7.3): Barreto ICML'17 — Q = psi @ w for ANY reward
  r = phi @ w; new cost matrices = one matrix-vector product."""
from typing import Dict, List, Optional
import numpy as np
import torch

from .primitives import wilson_interval, solve_mdp


class StateMatrix:
    def __init__(self, k=24, seed=0):
        self.k, self.seed = k, seed
        self.centroids = None; self.visits = None
        self.exemplars: List[Optional[str]] = []
        self.n_spatial = 0; self.k_total = 0
        self.terminal_names: Dict[str, int] = {}

    def fit(self, emb, texts=None, iters=50):
        X = emb.detach()
        k = max(2, min(self.k, len(X)))
        g = torch.Generator().manual_seed(self.seed)
        cents = [X[torch.randint(len(X), (1,), generator=g)]]
        for _ in range(k - 1):                              # k-means++ seeding
            d2 = torch.cdist(X, torch.vstack(cents)).amin(1) ** 2
            cents.append(X[torch.multinomial(
                (d2 / (d2.sum() + 1e-12)).clamp_min(0), 1, generator=g)])
        C = torch.vstack(cents)
        assign = None
        for _ in range(iters):
            assign = torch.cdist(X, C).argmin(1)
            newC = C.clone()
            for j in range(k):
                m = assign == j
                newC[j] = X[m].mean(0) if m.any() else X[torch.randint(len(X), (1,))]
            done = torch.allclose(C, newC, atol=1e-6)
            C = newC
            if done:
                break
        self.n_spatial = self.k_total = k
        self.centroids = C
        self.visits = np.bincount(assign.numpy(), minlength=k).astype(float)
        self.exemplars = [None] * k
        if texts:
            for j in range(k):
                m = (assign == j).nonzero().squeeze(1)
                if len(m):
                    dd = (X[m] - C[j]).norm(dim=1)
                    self.exemplars[j] = texts[m[dd.argmin()].item()]
        return self

    def add_terminal(self, name):
        emb = torch.randn(1, self.centroids.shape[1],
                          generator=torch.Generator().manual_seed(self.seed))
        self.centroids = torch.vstack([self.centroids, emb])
        self.visits = np.append(self.visits, 0.0)
        self.exemplars.append(f"<terminal:{name}>")
        self.terminal_names[name] = self.k_total
        self.k_total += 1

    def terminal_index(self, name):
        return self.terminal_names.get(name)

    @property
    def terminal_mask(self):
        return np.arange(self.k_total) >= self.n_spatial

    @torch.no_grad()
    def assign(self, emb):
        return torch.cdist(emb, self.centroids[: self.n_spatial]).argmin(1)


class ProbabilisticMatrix:
    def __init__(self, alpha=0.5, kappa=20.0):
        self.alpha, self.kappa = alpha, kappa
        self.counts: Dict[str, np.ndarray] = {}
        self.P: Dict[str, np.ndarray] = {}
        self.lo: Dict[str, np.ndarray] = {}
        self.hi: Dict[str, np.ndarray] = {}

    def fit(self, cl, label_ids, neural, questions, K):
        for q in questions:
            A = q.n_actions
            n = np.zeros((K, A))
            np.add.at(n, (cl, label_ids[q.name]), 1.0)
            N = n.sum(1, keepdims=True)
            emp = (self.alpha + n) / (self.alpha * A + N + 1e-12)
            w = N / (N + self.kappa)
            P = w * emp + (1 - w) * neural[q.name]
            self.counts[q.name] = n
            self.P[q.name] = P / P.sum(1, keepdims=True)
            lo, hi = wilson_interval(n, np.repeat(N, A, axis=1))
            self.lo[q.name], self.hi[q.name] = lo, hi

    def label_dist(self, qname):
        n = self.counts[qname]; A = n.shape[1]
        N = n.sum(1, keepdims=True)
        return (self.alpha + n) / (self.alpha * A + N + 1e-12)

    def pad_to(self, K):
        for name in list(self.P):
            add = K - self.P[name].shape[0]
            if add <= 0:
                continue
            A = self.P[name].shape[1]
            z = np.zeros((add, A)); u = np.full((add, A), 1.0 / A)
            self.counts[name] = np.vstack([self.counts[name], z])
            self.P[name] = np.vstack([self.P[name], u])
            self.lo[name] = np.vstack([self.lo[name], z])
            self.hi[name] = np.vstack([self.hi[name], np.ones((add, A))])


class RewardMatrix:
    """R components (v4). compose('eval'|'safe') — bonuses are TRAIN-only."""
    def __init__(self, gamma=0.9, bonus_beta=0.3, bonus_n0=8.0):
        self.gamma, self.beta, self.n0 = gamma, bonus_beta, bonus_n0
        self.C = self.R_base = self.R_shaped = self.R_bonus = self.R_pess = None
        self.V = self.Q = None; self.terminal_reward = None

    def fit(self, prob_matrix, question, T, terminal_mask, terminal_reward):
        A = question.n_actions
        self.C = (question.cost_array() if question.cost_matrix
                  else 1.0 - np.eye(A))
        lab = prob_matrix.label_dist(question.name)
        self.R_base = -(lab @ self.C)
        counts = prob_matrix.counts[question.name]
        N = counts.sum(1, keepdims=True)
        lo, _ = wilson_interval(counts, np.repeat(N, A, axis=1))
        rs = lo.sum(1, keepdims=True)
        lo = np.where(rs > 1e-9, lo / np.maximum(rs, 1e-9),
                      np.full_like(lo, 1.0 / A))
        self.R_pess = -(lo @ self.C)
        self.R_bonus = self.beta / np.sqrt(counts + self.n0)
        self.terminal_reward = terminal_reward
        for Rv in (self.R_base, self.R_pess):
            Rv[terminal_mask] = terminal_reward[terminal_mask][:, None]
        V0, _, _ = solve_mdp(self.R_base, T, terminal_mask, self.gamma, tau=None)
        self.R_shaped = (self.R_base - V0[:, None]
                         + self.gamma * np.einsum("kak,k->ka", T, V0))
        self.R_shaped[terminal_mask] = terminal_reward[terminal_mask][:, None]
        self.V, self.Q, _ = solve_mdp(self.R_base, T, terminal_mask,
                                      self.gamma, tau=None)
        return self

    def compose(self, kind="eval"):
        if kind == "safe":
            return self.R_pess
        return self.R_shaped


class SuccessorFeatures:
    """psi = phi + gamma * T @ (pi @ psi). q_for(w): Q for ANY r = phi @ w."""
    def __init__(self, T, Pi, phi, gamma=0.9, iters=300):
        Psi = phi.copy()
        for _ in range(iters):
            G = np.einsum("sa,sd->sd", Pi, Psi)
            Psi_new = phi + gamma * np.einsum("kak,sd->kad", T, G)
            if np.abs(Psi_new - Psi).max() < 1e-10:
                Psi = Psi_new; break
            Psi = Psi_new
        self.psi = Psi

    def q_for(self, w):
        return self.psi @ w

    @staticmethod
    def cost_basis(label_dist, A):
        """phi(s,a)[c*A + j] = label_dist(s,c) * 1{j==a} -> w = -C.flatten()."""
        K = label_dist.shape[0]
        phi = np.zeros((K, A, A * A))
        for c in range(A):
            for a in range(A):
                phi[:, a, c * A + a] = label_dist[:, c]
        return phi


def policy_evaluate_flow(R_flow, T, Pi, gamma=0.9):
    """Linear-solve policy evaluation with flow rewards."""
    r_bar = (Pi * R_flow).sum(1)
    P_pi = np.einsum("ka,kak->k", Pi, T)
    V = np.linalg.solve(np.eye(T.shape[0]) - gamma * P_pi, r_bar)
    return R_flow + gamma * np.einsum("kak,k->ka", T, V)
