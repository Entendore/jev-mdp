import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev.engine import demo_engine

eng, (Xte, Yte) = demo_engine(seed=0)

for s, y in zip(Xte[:40], Yte[:40]):                # v6.1 adaptive conformal
    eng.decide(s)
    eng.observe_outcome(s, y["department"])
print("ACI realized coverage:", round(eng.aci.realized_coverage, 3),
      "| alpha_t:", round(eng.aci.alpha_t, 3))

eng.learn_correction("need a quote for 50 seats",
                     labels={"department": "sales"})          # v2 correction
eng.learn_correction("charged twice!!",
                     avoid={"department": "sales"})           # v2 unlikelihood
r = eng.decide("charged twice!!")
print("post-correction:", r["action"], round(r["confidence"], 2))

res = eng.learn_preferences([                        # v8.1 BT + KL brake
    ("i want my money back", "billing", "sales"),
    ("refund the order", "billing", "sales"),
    ("give me my money back now", "billing", "sales"),
], kl_beta=0.1)
print("preferences:", res)
print("decision after preferences:",
      eng.decide("i want my money back")["action"])

# v9.9 interpretability (guarded: module absent under --minimal)
try:
    from jev.interpret import shapley_fields, counterfactual_flip
except ImportError:
    shapley_fields = counterfactual_flip = None
if shapley_fields is not None:
    st = {"text": "i want my money back", "tone": "angry"}
    print("shapley:", shapley_fields(eng, st, ["text", "tone"]))
    print("cf-flip:", counterfactual_flip(
        eng, st, {"text": ["pricing question", "app keeps crashing"]}))
else:
    print("(jev.interpret not installed — skipped; regenerate without "
          "--minimal to enable)")
