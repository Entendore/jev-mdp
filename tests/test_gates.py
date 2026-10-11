"""Research gates (fast subset). Full registry: docs/RESEARCH_MAP.md.
Statistical gates should be re-run with >=5 seeds as IQM CIs (rliable)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

from jev.primitives import solve_mdp, ece, wilson_interval
from jev.matrices import SuccessorFeatures, policy_evaluate_flow
from jev.evaluation import iqm_ci, hcope_lower_bound, PolicyVerifier
from jev.information import gittins_index
from jev.calibration import AdaptiveConformal


def _tiny_mdp(K=6, A=3, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.dirichlet(np.ones(K), size=(K, A, K))
    term = np.zeros(K, dtype=bool); term[K - 1] = True
    T[K - 1] = 0.0; T[K - 1, :, K - 1] = 1.0
    R = -rng.random((K, A)); R[K - 1] = -1.0
    return T, R, term


def test_value_iteration_absorbing_terminal():
    T, R, term = _tiny_mdp()
    V, Q, Pi = solve_mdp(R, T, term, 0.9, tau=None)
    assert V[term][0] == pytest.approx(R[term][0, 0])
    assert np.allclose(Pi[term], 0.0)
    assert np.allclose(Pi.sum(1)[~term], 1.0)


def test_soft_vi_converges_to_hard_as_tau_shrinks():
    T, R, term = _tiny_mdp()
    Vh, Qh, _ = solve_mdp(R, T, term, 0.9, tau=None)
    _, Qs, _ = solve_mdp(R, T, term, 0.9, tau=0.01)
    assert np.abs(Qh - Qs).max() < 1e-3


def test_sf_identity_for_any_cost_matrix():
    T, R, term = _tiny_mdp()
    A = T.shape[1]
    lab = np.full((T.shape[0], A), 1.0 / A)
    Pi = np.full((T.shape[0], A), 1.0 / A); Pi[term] = 0.0
    phi = SuccessorFeatures.cost_basis(lab, A); phi[term] = 0.0
    sf = SuccessorFeatures(T, Pi, phi, gamma=0.9)
    rng = np.random.default_rng(0)
    C = rng.random((A, A)); np.fill_diagonal(C, 0.0)
    Q_sf = sf.q_for(-C.flatten())
    Q_ref = policy_evaluate_flow(-(lab @ C), T, Pi, 0.9)
    assert np.abs(Q_sf - Q_ref).max() < 1e-6


def test_reachability_exact_on_absorbing_chain():
    T, R, term = _tiny_mdp()
    K, A = T.shape[0], T.shape[1]

    class SM:
        n_spatial = K - 1
        k_total = K
        visits = np.ones(K)
        terminal_mask = term
        def terminal_index(self, name):
            return K - 1
    ver = PolicyVerifier(SM(), T)
    pi = np.full((K, A), 1.0 / A)
    h = ver.reach_prob(pi, "failed")
    assert h[K - 1] == 1.0
    assert np.all((h >= 0) & (h <= 1))


def test_gittins_confident_arm_beats_uniform():
    assert gittins_index(21.0, 3.0) > gittins_index(1.0, 1.0)


def test_hcope_lb_below_mean_and_conservative():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.1, 500)
    lb = hcope_lower_bound(vals, delta=0.05)
    assert 0.0 < lb < vals.mean()


def test_iqm_constant_scores():
    out = iqm_ci(np.full((5, 4), 0.9))
    assert out["iqm"] == pytest.approx(0.9)
    assert out["ci95"][0] <= 0.9 <= out["ci95"][1]


def test_adaptive_conformal_coverage_recovers():
    rng = np.random.default_rng(1)
    aci = AdaptiveConformal(alpha=0.1, gamma=0.05)
    aci.qhat = 0.5                        # deliberately miscalibrated start
    errs = [aci.observe(rng.dirichlet(np.ones(3)), int(rng.integers(3)))
            for _ in range(2000)]
    cov = 1 - np.mean(errs[-500:])
    assert 0.80 <= cov <= 0.98            # long-run coverage ~ 1-alpha


def test_ece_perfect_calibration():
    rng = np.random.default_rng(2)
    P = rng.dirichlet(np.ones(3) * 50, size=2000)
    y = np.array([rng.choice(3, p=P[i]) for i in range(2000)])
    assert ece(P, y) < 0.06


def test_wilson_interval_brackets():
    lo, hi = wilson_interval([7], [10])
    assert 0.35 <= lo[0] <= 0.7 <= hi[0] <= 0.99
