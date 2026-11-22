"""Advanced planning.
- DistributionalSolver (v7.4): particle VI (Bellemare ICML'17; Dabney AAAI'18)
  -> CVaR risk-averse action selection.
- robust_vi (v7.8): interval/BMDP worst case (Givan ICML'00).
- CEMMPC (v9.7): receding-horizon control (Rubinstein'97; Pinneri CoRL'20;
  TD-MPC Hansen ICLR'22 value bootstrap).
- psrl_sample_T (v6): posterior-sampled dynamics (Osband ICML'13)."""
import numpy as np


class DistributionalSolver:
    def __init__(self, R, T, terminals, gamma=0.9, n_particles=64, seed=0):
        self.R, self.T, self.term = R, T, terminals
        self.gamma, self.N = gamma, n_particles
        self.rng = np.random.default_rng(seed)

    def solve(self, iters=120):
        K, A, _ = self.T.shape
        cdf = np.cumsum(self.T, axis=-1)
        Z = np.zeros((K, A, self.N))
        ar = np.arange(self.N)[None, None, :]
        for _ in range(iters):
            u = self.rng.random((K, A, self.N))
            s2 = (u[:, :, None, :] > cdf[:, :, None, :]).sum(-1).clip(0, K - 1)
            vmax = Z.max(axis=1)
            Z = self.R[:, :, None] + self.gamma * vmax[s2, ar]
            Z[self.term] = self.R[self.term, 0][:, None]
        self.Z = Z
        return Z

    def cvar(self, alpha=0.1):
        q = np.quantile(self.Z, alpha, axis=-1, keepdims=True)
        tail = np.where(self.Z <= q, self.Z, np.nan)
        return np.nanmean(tail, axis=-1)

    def risk_plan(self, alpha=0.1):
        return self.cvar(alpha).argmax(1)


def worst_case_row(p_row, kappa, v_succ):
    """Adversarial T' within an l-inf ball (Givan'00)."""
    lo = np.clip(p_row - kappa, 0.0, None)
    rem = 1.0 - lo.sum()
    if rem <= 0:
        return lo / max(lo.sum(), 1e-9)
    for i in np.argsort(v_succ):
        add = min(rem, 1.0 - lo[i])
        lo[i] += add; rem -= add
        if rem <= 1e-12:
            break
    return lo


def robust_vi(T, R, terminals, kappa=0.05, gamma=0.9, iters=150):
    K, A, _ = T.shape
    V = np.zeros(K)
    for _ in range(iters):
        Twc = np.empty_like(T)
        for k in range(K):
            for a in range(A):
                Twc[k, a] = worst_case_row(T[k, a], kappa, V)
        Q = R + gamma * np.einsum("kak,k->ka", Twc, V)
        Vn = np.where(terminals, R[:, 0], Q.max(1))
        if np.abs(Vn - V).max() < 1e-9:
            V = Vn; break
        V = Vn
    return V, Q


class CEMMPC:
    """Plan action SEQUENCES in the tabular model; execute first; re-plan."""
    def __init__(self, T, R_eval, V_prior, terminal_reward, terminal_mask,
                 horizon=4, pop=64, elites=8, gens=3, seed=0):
        self.T, self.R = T, R_eval
        self.V, self.r_term, self.term = V_prior, terminal_reward, terminal_mask
        self.H, self.pop, self.n_elite, self.gens = horizon, pop, elites, gens
        self.rng = np.random.default_rng(seed)
        self.cdf = np.cumsum(T, axis=-1)
        self.pi_prior = None                       # set per-cluster at act()

    def act(self, k0: int) -> int:
        A = self.T.shape[1]; K = self.T.shape[0]
        theta = (np.tile(self.pi_prior[k0], (self.H, 1))
                 if self.pi_prior is not None
                 else np.full((self.H, A), 1.0 / A))
        for _ in range(self.gens):
            u = self.rng.random((self.pop, self.H, 1))
            seq = (u > np.cumsum(theta[None], axis=1)[:, :, :]).sum(-1).clip(0, A - 1)
            rets = np.zeros(self.pop)
            ks = np.full(self.pop, k0)
            alive = np.ones(self.pop, dtype=bool)
            for h in range(self.H):
                a = seq[:, h]
                r = self.R[ks, a]
                nx = (self.rng.random(self.pop)[:, None]
                      > self.cdf[ks, a]).sum(-1).clip(0, K - 1)
                r = r + self.r_term[nx] * self.term[nx] * alive
                rets += np.where(alive, r, 0.0)
                alive &= ~self.term[nx]
                ks = np.where(self.term[nx], ks, nx)
            rets += np.where(alive, self.V[ks], 0.0)
            elite = np.argsort(rets)[-self.n_elite:]
            theta = seq[elite].mean(0, keepdims=True) * 0.9 + theta * 0.1
            theta = theta / theta.sum(1, keepdims=True)
        return int(np.argmax(theta[0]))


def psrl_sample_T(nT: np.ndarray, rng: np.random.Generator, alpha=1e-3):
    """One posterior sample of the transition tensor (Dirichlet per row)."""
    return rng.dirichlet((nT + alpha).reshape(-1, nT.shape[-1])
                         ).reshape(nT.shape)
