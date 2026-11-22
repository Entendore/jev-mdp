import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
from jev.engine import demo_engine
from jev.matrices import SuccessorFeatures, policy_evaluate_flow

eng, _ = demo_engine(seed=0)
qn = eng._primary(); A = eng._q(qn).n_actions
term = eng.state_matrix.terminal_mask
lab = eng.pm.label_dist(qn)
Pi = eng.pm.P[qn].copy(); Pi[term] = 0.0
T = eng.T
C_old = eng.C.copy()

phi = SuccessorFeatures.cost_basis(lab, A); phi[term] = 0.0
t0 = time.perf_counter()
sf = SuccessorFeatures(T, Pi, phi, gamma=eng.gamma)
sf_ms = (time.perf_counter() - t0) * 1000

def ref_Q(C):
    R = -(lab @ C); R[term] = 0.0
    return policy_evaluate_flow(R, T, Pi, eng.gamma)

Q_sf = sf.q_for(-C_old.flatten())
print(f"SF vs re-solve (old costs): max|diff| = "
      f"{np.abs(Q_sf - ref_Q(C_old)).max():.2e}  (built in {sf_ms:.1f} ms)")

C_new = C_old.copy(); C_new[0, 1] = C_new[0, 2] = 5.0   # billing errors 5x
t0 = time.perf_counter()
Q_new = sf.q_for(-C_new.flatten())                       # ZERO retraining
ms = (time.perf_counter() - t0) * 1000
print(f"SF vs re-solve (new costs): max|diff| = "
      f"{np.abs(Q_new - ref_Q(C_new)).max():.2e}  (transfer {ms:.2f} ms)")
print("clusters re-routed:",
      int((ref_Q(C_old).argmax(1) != ref_Q(C_new).argmax(1))[~term].sum()))
