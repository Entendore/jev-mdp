import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev.engine import (JevMDP, make_tickets, simulate_episodes,
                        default_ticket_questions)

Xtr, Ytr = make_tickets(240, seed=1)
Xcal, Ycal = make_tickets(70, seed=2)
Xte, Yte = make_tickets(140, seed=3)

eng = JevMDP(default_ticket_questions(), seed=0)
eng.fit_supervised(Xtr, Ytr, val=(Xcal, Ycal))      # v2 BC distillation
eng.fit_rl(Xtr, Ytr)                                # v1 REINFORCE+Brier (RLCD)
eng.calibrate(Xcal, Ycal)                           # v2 temp + OOD thresholds
eng.build_matrices(Xtr, Ytr)                        # v3 S, pi matrices
eng.add_terminals({"resolved": 1.0, "escalated": -0.25, "failed": -1.0})
eng.learn_transitions(simulate_episodes(eng, Xtr, Ytr, eps=0.30))  # v3 T
print("reward:", eng.build_reward())                # v4 base+shaped+pess
print("policy:", eng.fit_policy(Xtr, Ytr))          # v4 soft VI->SPIBB
eng.calibrate(Xcal, Ycal)                           # recalibrate post-distill
print("crc:", eng.fit_crc(Xcal, Ycal))              # v6.2 cost-bounded tau

print("\n== one ticket, every serving mode ==")
s = "i was charged twice. please refund. urgent"
for mode in ["neural", "matrix", "plan", "policy", "thompson", "gittins"]:
    r = eng.decide(s, mode=mode)
    print(f"{mode:<9} -> {r['action']:<10} conf={r['confidence']:.2f}"
          f"  src={r['source']}")

print("\n== all-question outputs (parallel pass) ==")
print({k: v.get("confidence") for k, v in eng.decide(s)["outputs"].items()})

print("\n== save/load roundtrip ==")
eng.save("artifacts/engine.pt")
from jev.engine import JevMDP
e2 = JevMDP.load("artifacts/engine.pt")
a = eng.decide(s); b = e2.decide(s)
print("identical after reload:",
      a["action"] == b["action"]
      and abs(a["confidence"] - b["confidence"]) < 1e-6)

print("\n== test-set report ==")
print(eng.report(Xte, Yte))
