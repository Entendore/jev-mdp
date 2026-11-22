import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from jev.engine import demo_engine, simulate_episodes

eng, (Xte, Yte) = demo_engine(seed=0)
info = eng.compile_program(Xte[:120], fidelity_target=0.95,
                           spec={"failed": 0.12})
print("tree leaves:", info["leaves"], "| fidelity:", info["fidelity"],
      "| certified:", info["ok"])
print("P(reach 'failed')  tree:", info["reach_failed_tree"],
      "| neural:", info["reach_failed_neural"])
print("serve mix:", dict(Counter(eng.decide(s)["source"]
                                 for s in Xte[120:])))

# v10.7 world-model audit
import jev.world as W
eps = simulate_episodes(eng, Xte[:100], Yte[:100], eps=0.3, seed=7)
K = eng.state_matrix.k_total
print("A&G homogeneity:",
      W.anderson_goodman_homogeneity([eps[:50], eps[50:]], K))
lens = [3] * 40 + [4] * 40 + [5] * 20
print("PPC:", W.posterior_predictive_check(eng.T,
      eng.state_matrix.terminal_mask, lens))
