import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from jev.engine import demo_engine, simulate_episodes

eng, (Xte, Yte) = demo_engine(seed=0)
qn = eng._primary()
eps = simulate_episodes(eng, Xte[:100], Yte[:100], eps=0.3, seed=99)
logs3 = [(k, a, float(eng.terminal_reward[t])) for k, a, t in eps]
logs4 = [(k, a, t, float(eng.terminal_reward[t])) for k, a, t in eps]

print("candidate DR+HCOPE:", eng.ope_check(logs3))
print("ensemble {DR,FQE,MIS}:", eng.ope_full(logs4))
beh = eng.ope_check(logs3, challenger_rows=eng.pm.P[qn])
print("behavior (self-check):", beh)
print("PROMOTE" if eng.ope_check(logs3)["hcope_lb"] > beh["dm"]
      else "DO NOT PROMOTE")

from jev.evaluation import iqm_ci                       # v10.11 rliable
accs = []
for s in range(5):
    e, (Xv, Yv) = demo_engine(seed=s)
    accs.append(e.report(Xv, Yv)["department"]["accuracy"])
print("accuracy over 5 seeds:", [round(a, 3) for a in accs])
print("IQM:", iqm_ci(np.array(accs).reshape(5, 1)))
