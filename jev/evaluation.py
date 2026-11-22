"""Evaluation & certification.
- ope_report: Direct + Doubly-Robust OPE (Jiang&Li ICML'16; Dudik'11) with
  bootstrap CI + empirical-Bernstein HCOPE LB (Thomas ICML'15; Maurer&Pontil'09).
- ope_ensemble: {DR, FQE (Le ICML'19), MIS (Xie NeurIPS'19)} + median
  (Voloshin ICML'21 ensembling recommendation).
- PolicyVerifier (v9.6): EXACT reachability on the tabular MDP
  (Baier&Katoen'08; PRISM Kwiatkowska'07) — h = (I - Q_pi)^-1 b.
- iqm_ci (v10.11, rliable Agarwal NeurIPS'21) — replaces mean±std folklore.
- fairness_audit (v8.11); pac_bayes_bound (v10.10, on-policy — flagged)."""
import math
import numpy as np


def ope_report(logs, P_b, Q, pi_e, n_boot=2000, delta=0.05, b=1.5, seed=0,
               w_clip=10.0):
    """logs rows: (cluster, action, reward)."""
    L = np.array(logs, dtype=float)
    k, a, r = L[:, 0].astype(int), L[:, 1].astype(int), L[:, 2]
    rho = np.clip(pi_e[k, a] / np.maximum(P_b[k, a], 1e-6), 0, w_clip)
    dr = rho * (r - Q[k, a]) + (pi_e[k] * Q[k]).sum(1)
    dm = (pi_e[k] * Q[k]).sum(1)
    rng = np.random.default_rng(seed)
    boots = np.array([dr[rng.integers(0, len(dr), len(dr))].mean()
                      for _ in range(n_boot)])
    v = np.clip(dr, -b, b); c = math.log(2 / delta); n = len(v)
    lb = (v.mean() - math.sqrt(2 * v.var() * c / n)
          - 7 * b * c / (3 * (n - 1)))
    return {"dm": round(float(dm.mean()), 4), "dr": round(float(dr.mean()), 4),
            "dr_ci95": [round(float(x), 3) for x in
                        np.percentile(boots, [2.5, 97.5])],
            "hcope_lb": round(float(lb), 4), "n": int(n)}


def fqe_value(logs4, K, A, pi_e, gamma=0.9, iters=300, alpha=0.1):
    """Fitted-Q evaluation. logs4 rows: (k, a, k2, r)."""
    Q = np.zeros((K, A))
    for _ in range(iters):
        Qn = Q.copy()
        for k, a, k2, r in logs4:
            nxt = float(pi_e[k2] @ Q[k2]) if 0 <= k2 < K else 0.0
            Qn[k, a] += alpha * (r + gamma * nxt - Qn[k, a])
        if np.abs(Qn - Q).max() < 1e-10:
            break
        Q = Qn
    return float((pi_e * Q).sum(1).mean())


def mis_value(logs4, T, pi_b, pi_e, gamma=0.9, n_boot=1000, seed=0):
    """Marginalized IS: state-visitation ratio from flow equations."""
    def d_of(pi):
        P_pi = np.einsum("ka,kak->k", pi, T)
        d = np.full(len(P_pi), 1.0 / len(P_pi))
        for _ in range(300):
            dn = d + gamma * (P_pi.T @ d)
            if np.abs(dn - d).max() < 1e-11:
                d = dn; break
            d = dn
        return np.clip(d / max(d.sum(), 1e-12), 1e-9, None)
    db, de = d_of(pi_b), d_of(pi_e)
    w = np.clip(de / db, 0.0, 20.0)
    r = np.array([x[3] for x in logs4], dtype=float)
    ks = np.array([x[0] for x in logs4], dtype=int)
    v = w[ks] * r
    rng = np.random.default_rng(seed)
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return {"mis_value": round(float(v.mean()), 4),
            "ci95": [round(float(x), 4) for x in
                     np.percentile(boots, [2.5, 97.5])]}


def ope_ensemble(logs4, P_b, Q, T, pi_e, gamma=0.9, seed=0):
    """{DR, FQE, MIS} + median (Voloshin'21). logs4: (k, a, k2, r)."""
    logs3 = [(k, a, r) for k, a, k2, r in logs4]
    dr = ope_report(logs3, P_b, Q, pi_e, seed=seed)
    K, A = P_b.shape
    fqe = fqe_value(logs4, K, A, pi_e, gamma=gamma, seed=seed)
    mis = mis_value(logs4, T, P_b, pi_e, gamma=gamma, seed=seed)
    med = float(np.median([dr["dr"], fqe, mis["mis_value"]]))
    return {"dr": dr, "fqe": round(fqe, 4), "mis": mis,
            "median": round(med, 4)}


def hcope_lower_bound(values, delta=0.05, b=1.5):
    """Empirical-Bernstein LB on E[values], bounded in [-b, b]."""
    v = np.clip(np.asarray(values, float), -b, b)
    n = len(v); c = math.log(2 / delta)
    return float(v.mean() - math.sqrt(2 * v.var() * c / n)
                 - 7 * b * c / (3 * (n - 1)))


def ope_support_audit(logs, P_b, Q, pi_e, tau=0.05):
    """Share of |DR contribution| from low-support cells — confounding
    early warning (Kallus&Zhou'18 spirit; audit, not bound)."""
    contribs, flagged = [], []
    for c, a, r in logs:
        rho = min(pi_e[c, a] / max(P_b[c, a], 1e-6), 10.0)
        dr = rho * (r - Q[c, a]) + float(pi_e[c] @ Q[c])
        contribs.append(abs(dr)); flagged.append(P_b[c, a] < tau)
    tot = float(np.sum(contribs)) + 1e-12
    share = float(sum(c for c, f in zip(contribs, flagged) if f)) / tot
    return {"low_support_value_share": round(share, 3),
            "trustworthy": bool(np.mean(flagged) < 0.2)}


class PolicyVerifier:
    """Exact reachability on the tabular MDP (absorbing-chain solve)."""
    def __init__(self, state_matrix, T):
        self.sm, self.T = state_matrix, T

    def reach_prob(self, pi_rows, terminal_name):
        K = self.T.shape[0]
        if pi_rows.shape[0] < K:
            pi_rows = np.vstack([pi_rows, np.zeros(
                (K - pi_rows.shape[0], pi_rows.shape[1]))])
        P = np.einsum("ka,kak->k", pi_rows, self.T)
        t = self.sm.terminal_index(terminal_name)
        trans = np.where(~self.sm.terminal_mask)[0]
        h = np.zeros(K)
        h[trans] = np.linalg.solve(np.eye(len(trans)) - P[np.ix_(trans, trans)],
                                   P[trans, t])
        h[t] = 1.0
        return h

    def check(self, pi_rows, spec, start_dist=None):
        sm = self.sm
        d0 = start_dist if start_dist is not None else \
            sm.visits / max(sm.visits[: sm.n_spatial].sum(), 1e-9)
        out = {}
        for name, eps in spec.items():
            val = float(d0 @ self.reach_prob(pi_rows, name))
            out[f"P_reach[{name}]"] = {"value": round(val, 4),
                                       "pass": bool(val <= eps)}
        return out


def iqm_ci(scores, n_boot=2000, seed=0):
    """IQM with stratified bootstrap CI over seeds x tasks (rliable)."""
    scores = np.asarray(scores)
    S = scores.shape[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ss = rng.integers(0, S, S)
        tt = rng.integers(0, scores.shape[1], scores.shape[1])
        m = np.sort(scores[np.ix_(ss, tt)].mean(1))
        boots.append(np.median(m[max(1, S // 4): max(2, 3 * S // 4)]))
    iqm = float(np.mean(np.sort(scores.mean(1))[
        max(1, S // 4): max(2, 3 * S // 4)]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"iqm": round(iqm, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def performance_profile(score_matrix, taus):
    """Fraction of (seed, task) runs scoring >= tau — the tail-honest view."""
    return {round(float(t), 2): float((np.asarray(score_matrix) >= t).mean())
            for t in taus}


def fairness_audit(decide_fn, texts, labels, group_fn, qname, swap_map=None):
    """Per-group accuracy + gap; optional counterfactual token-swap flips."""
    preds = [decide_fn(t).get(qn, {}).get("choice") for t in texts]
    groups = [group_fn(t) for t in texts]
    acc = {g: float(np.mean([preds[i] == labels[i][qn]
                             for i, gg in enumerate(groups) if gg == g]))
           for g in set(groups)}
    out = {"group_accuracy": {g: round(a, 4) for g, a in acc.items()},
           "accuracy_gap": round(max(acc.values()) - min(acc.values()), 4)}
    if swap_map:
        import re
        patt = re.compile("|".join(map(re.escape, swap_map)), re.IGNORECASE)
        flips = sum(decide_fn(patt.sub(
            lambda m: swap_map.get(m.group(0), m.group(0)), t)
        ).get(qn, {}).get("choice") != p
            for t, p in zip(texts, preds))
        out["cf_flip_rate"] = round(flips / max(len(texts), 1), 4)
    return out


def pac_bayes_bound(empirical_loss, n_params, sigma=0.1, sigma0=1.0,
                    n=100, delta=0.05):
    """Maurer'04-form PB bound (ON-POLICY loss of a Gaussian posterior)."""
    kl = n_params / 2 * (sigma ** 2 / sigma0 ** 2
                         + math.log(sigma0 ** 2 / sigma ** 2) - 1)
    bound = empirical_loss + math.sqrt(
        (kl + math.log(2 * math.sqrt(n) / delta)) / (2 * (n - 1)))
    return {"empirical_loss": round(empirical_loss, 4),
            "kl_nats": round(kl, 1), "pac_bayes_upper": round(bound, 4)}
