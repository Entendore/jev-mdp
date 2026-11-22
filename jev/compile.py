"""Policy compilation (v10.1): Bastani Pu Solar-Lezama NeurIPS'18 —
distill the neural policy into a decision tree, CERTIFY fidelity, VERIFY the
tree inside the MDP (v9.6 reachability), serve program-first with the network
as residual. 'Verified' = against the learned T matrix (docs/HONESTY.md)."""
import numpy as np

from .evaluation import PolicyVerifier


def compile_program(eng, texts, depth=5, fidelity_target=0.95, spec=None):
    from sklearn.tree import DecisionTreeClassifier
    spec = spec or {"failed": 0.12}
    cut = int(0.8 * len(texts))
    fit_texts, hold_texts = texts[:cut], texts[cut:]
    with __import__("torch").no_grad():
        Ef = eng.encoder(fit_texts).cpu().numpy()
        Eh = eng.encoder(hold_texts).cpu().numpy()
    yf = np.array([eng._policy_choice(t)[0] for t in fit_texts])
    yh = np.array([eng._policy_choice(t)[0] for t in hold_texts])
    tree = DecisionTreeClassifier(max_depth=depth,
                                  random_state=eng.seed).fit(Ef, yf)
    fidelity = float((tree.predict(Eh) == yh).mean())

    ver = PolicyVerifier(eng.state_matrix, eng.T)
    Ks = eng.state_matrix.n_spatial
    qn = eng._primary(); q = eng._q(qn)
    rows = np.zeros((Ks, q.n_actions))
    for k in range(Ks):
        e = eng.encoder([eng.state_matrix.exemplars[k] or "<empty>"]).cpu().numpy()
        rows[k, q.options.index(tree.predict(e)[0])] = 1.0
    checks = ver.check(rows, spec)

    reach_neural = float(
        (eng.state_matrix.visits[:Ks]
         / max(eng.state_matrix.visits[:Ks].sum(), 1e-9))
        @ ver.reach_prob(eng.pi_target[qn][:Ks], "failed"))
    info = {"leaves": int(tree.get_n_leaves()),
            "fidelity": round(fidelity, 4), "checks": checks,
            "reach_failed_tree": checks["P_reach[failed]"]["value"],
            "reach_failed_neural": round(reach_neural, 4),
            "ok": fidelity >= fidelity_target
                  and all(c["pass"] for c in checks.values())}
    eng.compiled = {"tree": tree, "ok": info["ok"]}
    return info


def serve_compiled(eng, state):
    """Program-first serving; neural fallback on disagreement."""
    r = eng.decide(state)
    if eng.compiled and eng.compiled["ok"]:
        qn = eng._primary()
        tree_a = eng.compiled["tree"].predict(
            eng.encoder([state]).cpu().numpy())[0]
        if r["outputs"].get(qn, {}).get("choice") == tree_a:
            r["source"] = "program"
    return r
