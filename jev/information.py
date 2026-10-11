"""The information & indices layer (v10).
- evsi_query: expected value of sample information (Raiffa&Schlaifer'61;
  Howard'66) on our own Dirichlet posterior — ask iff EVSI > price.
- sprt_decision: Wald's SPRT (1945) over sequential answers.
- gittins_index: Beta-arm Gittins index (Gittins JRSS-B'79) via the retirement
  formulation; provably optimal for discounted bandits.
- bwk_pacing: Badanidiyuru STOC'13 primal pacing.
- whittle_index_two_state / whittle_router (Whittle'88; Weber&Weiss'90)."""
import math
from typing import Dict
import numpy as np

_GITT: Dict = {}


def evsi_query(eng, k, qname, query_cost, draws=200, seed=0):
    rng = np.random.default_rng(seed)
    lab = eng.pm.label_dist(qname)[k]
    C = eng.C
    v0 = float((lab @ C).min())                      # best expected cost now
    gains = [float(C[int(rng.choice(len(lab), p=lab))].min()) - v0
             for _ in range(draws)]                  # if label were revealed
    evsi = float(np.mean(gains)) - query_cost
    return {"v_now": round(v0, 4),
            "ev_info": round(float(np.mean(gains)), 4),
            "evsi": round(evsi, 4), "recommend_query": bool(evsi > 0)}


def sprt_decision(eng, k, qname, answers, alpha=0.05, beta=0.10):
    q = eng._q(qname)
    P = eng.pm.P[qname][k]
    top2 = np.argsort(-P)[:2]
    a1, a2 = int(top2[0]), int(top2[1])
    llr = 0.0
    upper, lower = math.log((1 - beta) / alpha), math.log(alpha / (1 - beta))
    lab = eng.pm.label_dist(qname)[k]
    for ans in answers:
        i = q.options.index(ans)
        like1 = lab[a1] if i == a1 else 0.25
        like2 = lab[a2] if i == a2 else 0.25
        llr += math.log(max(like1, 1e-6)) - math.log(max(like2, 1e-6))
        if llr >= upper:
            return {"decide": q.options[a1], "stopped_at": len(answers),
                    "rule": "H1"}
        if llr <= lower:
            return {"decide": q.options[a2], "stopped_at": len(answers),
                    "rule": "H2"}
    return {"decide": None, "stopped_at": None, "llr": round(llr, 3),
            "rule": "continue"}


def gittins_index(a, b, gamma=0.95, max_n=24, sweeps=250, bisect=22):
    """Beta(a,b) Gittins index; truncated lattice DP + bisection on the
    retirement level c (V = max(c/(1-gamma), cont))."""
    ai_, bj_ = max(1, int(round(a))), max(1, int(round(b)))
    key = (ai_, bj_, gamma, max_n)
    if key in _GITT:
        return _GITT[key]
    states = [(i, n0 - i) for n0 in range(max_n + 1) for i in range(n0 + 1)]
    ix = {s: n for n, s in enumerate(states)}
    A = np.array([s[0] for s in states], float)
    B = np.array([s[1] for s in states], float)
    m = A / (A + B)
    succ = np.array([ix.get((int(x) + 1, int(y)), ix[(int(x), int(y))])
                     for x, y in zip(A, B)])
    fail = np.array([ix.get((int(x), int(y) + 1), ix[(int(x), int(y))])
                     for x, y in zip(A, B)])
    tgt = ix[(ai_, bj_)]

    def value(c):
        V = m / (1 - gamma)
        for _ in range(sweeps):
            cont = m + gamma * (m * V[succ] + (1 - m) * V[fail])
            Vn = np.maximum(c / (1 - gamma), cont)
            if np.abs(Vn - V).max() < 1e-10:
                V = Vn; break
            V = Vn
        return V

    lo, hi = 0.0, 1.0
    for _ in range(bisect):
        c = 0.5 * (lo + hi)
        V = value(c)
        cont = (m[tgt] + gamma * (m[tgt] * V[succ[tgt]]
                                  + (1 - m[tgt]) * V[fail[tgt]]))
        if cont >= c / (1 - gamma):
            lo = c
        else:
            hi = c
    _GITT[key] = 0.5 * (lo + hi)
    return _GITT[key]


def gittins_action(eng, cl, qname, gamma=0.95):
    q = eng._q(qname); A = q.n_actions
    counts = eng.pm.counts[qname][cl]; N = counts.sum()
    best, bi = -1.0, 0
    for i in range(A):
        g = gittins_index(eng.pm.alpha + counts[i],
                          eng.pm.alpha * (A - 1) + max(N - counts[i], 0.0),
                          gamma=gamma)
        if g > best:
            best, bi = g, i
    return bi


def bwk_pacing(budget, horizon_left):
    """Remaining-budget pacing (Badanidiyuru'13 primal heuristic)."""
    frac = max(0.0, 1.0 - budget.cost_spent / max(budget.max_cost, 1e-9))
    return min(1.0, frac)


def whittle_index_two_state(r0, r1, q01, q11, p01, p11, gamma=0.9, iters=200):
    """State-1 Whittle index for a two-state restless arm."""
    lo, hi = 0.0, max(r1, r0) + 1.0
    for _ in range(40):
        lam = 0.5 * (lo + hi)
        V = np.zeros(2)
        for _ in range(iters):
            act1 = r1 + gamma * (q11 * V[1] + (1 - q11) * V[0])
            pas1 = (r1 - lam) + gamma * (p11 * V[1] + (1 - p11) * V[0])
            act0 = r0 + gamma * (q01 * V[1] + (1 - q01) * V[0])
            pas0 = (r0 - lam) + gamma * (p01 * V[1] + (1 - p01) * V[0])
            Vn = np.array([max(act0, pas0), max(act1, pas1)])
            if np.abs(Vn - V).max() < 1e-10:
                V = Vn; break
            V = Vn
        act1 = r1 + gamma * (q11 * V[1] + (1 - q11) * V[0])
        pas1 = (r1 - lam) + gamma * (p11 * V[1] + (1 - p11) * V[0])
        if act1 >= pas1:
            lo = lam
        else:
            hi = lam
    return 0.5 * (lo + hi)


def whittle_router(queue_states, arms, serve_budget=1):
    idxs = [(whittle_index_two_state(**arm), i)
            for i, arm in enumerate(arms) if queue_states[i] == 1]
    idxs.sort(reverse=True)
    return [i for _, i in idxs[:serve_budget]]
