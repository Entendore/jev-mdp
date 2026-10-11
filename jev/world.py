"""World-model audit & adaptation (v10).
- empowerment_map (Klyubin ECAL'05; Blahut-Arimoto on T's rows).
- anderson_goodman_homogeneity / markov_order_test (A&G Ann.Math.Statist.'57).
- posterior_predictive_check (Gelman'96; Cranmer PNAS'20).
- RegimeSwitchingMatrices (v10.8): lambda-forgetting counts + fast-vs-slow
  predictive-LL switch test (Adams&MacKay'07 spirit).
- robust_kappa_from_ensemble (v10.9, Tobin ICLR'18 / Peng ICML'18)."""
import math
import numpy as np


def empowerment_map(T, iters=80):
    K, A, _ = T.shape
    E = np.zeros(K)
    for k in range(K):
        q_a = np.full(A, 1.0 / A)
        for _ in range(iters):
            q_s = q_a @ T[k]
            D = np.array([sum(T[k, a, j] * math.log(
                max(T[k, a, j] / max(q_s[j], 1e-12), 1e-12))
                for j in range(K) if T[k, a, j] > 0) for a in range(A)])
            qa = q_a * np.exp(D)
            q_a = qa / max(qa.sum(), 1e-12)
        q_s = q_a @ T[k]
        cap = sum(q_a[a] * sum(T[k, a, j] * math.log(
            max(T[k, a, j] / max(q_s[j], 1e-12), 1e-12))
            for j in range(K) if T[k, a, j] > 0) for a in range(A))
        E[k] = cap / math.log(2.0)
    return E


def anderson_goodman_homogeneity(episodes_by_epoch, K):
    """LR test of transition homogeneity across 2 time strata."""
    from scipy.stats import chi2
    N = np.zeros((2, K, K))
    for h, eps in enumerate(episodes_by_epoch[:2]):
        for k, a, k2 in eps:
            N[h, k, k2] += 1
    Ni = N.sum(2, keepdims=True)
    pool = N.sum(0, keepdims=True)
    NiP = pool.sum(2, keepdims=True)
    lr = 0.0
    for h in range(2):
        with np.errstate(divide="ignore", invalid="ignore"):
            exp = Ni[h] * (pool[0] / np.maximum(NiP[0], 1e-9)[:, None])
            m = N[h] > 0
            lr += 2.0 * float(np.sum(N[h][m] * np.log(N[h][m] / exp[m])))
    df = K * (K - 1)
    p = float(chi2.sf(lr, df))
    return {"lr": round(lr, 3), "df": df, "p_value": round(p, 4),
            "non_stationary": bool(p < 0.05)}


def markov_order_test(episodes, K):
    """Order-1 vs order-2 chain LR (A&G'57) — reject => partially observed."""
    from scipy.stats import chi2
    c1 = np.zeros((K, K)); c2 = np.zeros((K * K, K))
    for ep in episodes:
        for t in range(len(ep) - 1):
            c1[ep[t], ep[t + 1]] += 1
            if t >= 1:
                c2[ep[t - 1] * K + ep[t], ep[t + 1]] += 1
    ll1 = sum(n * math.log(n / max(c1[i].sum(), 1e-9))
              for i in range(K) for n in c1[i] if n > 0)
    ll2 = sum(n * math.log(n / max(c2[i].sum(), 1e-9))
              for i in range(K * K) for n in c2[i] if n > 0)
    lr, df = 2.0 * (ll2 - ll1), K * (K - 1) ** 2
    p = float(chi2.sf(lr, df))
    return {"lr": round(lr, 3), "df": df, "p_value": round(p, 4),
            "partial_observed": bool(p < 0.05)}


def posterior_predictive_check(T, terminal_mask, real_lens, n_sims=300,
                               seed=0):
    rng = np.random.default_rng(seed)
    K, A, _ = T.shape
    len_sim = []
    for _ in range(n_sims):
        k = int(rng.integers(K)); steps, done = 0, False
        while not done and steps < 100:
            a = int(rng.integers(A))
            k = int(rng.choice(K, p=T[k, a]))
            done = bool(terminal_mask[k]); steps += 1
        len_sim.append(steps)
    return {"real_mean_len": round(float(np.mean(real_lens)), 2),
            "sim_mean_len": round(float(np.mean(len_sim)), 2),
            "model_ok": abs(float(np.mean(len_sim))
                            - float(np.mean(real_lens)))
            < 0.25 * max(float(np.mean(real_lens)), 1.0)}


class RegimeSwitchingMatrices:
    """rho-discounted counts -> rolling T; fast-vs-slow LL gap => switch."""
    def __init__(self, eng, rho_fast=0.97):
        self.eng, self.rho = eng, rho_fast
        self.regime_id = 0

    def observe_episode(self, triples):
        nT = getattr(self.eng, "nT", None)
        if nT is None:
            return
        nT *= self.rho
        K, A = nT.shape[:2]
        for k, a, k2 in triples:
            if 0 <= k < K and 0 <= a < A:
                nT[k, a, k2] += 1.0
        self.eng.T = nT / nT.sum(-1, keepdims=True)
        for t in np.where(self.eng.state_matrix.terminal_mask)[0]:
            self.eng.T[t] = 0.0
            self.eng.T[t, :, t] = 1.0
        if self.eng.reward is not None:
            self.eng.build_reward()

    def regime_check(self, eps_fast=0.05):
        tr = getattr(self.eng, "_episodes", [])[-20:]
        if len(tr) < 10 or self.eng.T is None:
            return False

        def ll(T):
            return sum(math.log(max(T[k, a, k2], 1e-9)) for k, a, k2 in tr)
        gap = (ll(self.eng.T) - ll(self._slow_T())) / max(len(tr), 1)
        if gap < -eps_fast:
            self.regime_id += 1
            return True
        return False

    def _slow_T(self):
        T = np.full_like(self.eng.T, self.eng.talpha)
        for k, a, k2 in getattr(self.eng, "_episodes", []):
            T[k, a, k2] += 1.0
        return T / T.sum(-1, keepdims=True)


def robust_kappa_from_ensemble(T_list, cap=0.10):
    """kappa per entry ~ 2*std across randomized worlds (capped)."""
    stack = np.stack(T_list)
    return float(min(cap, 2.0 * stack.std(0).max()))
