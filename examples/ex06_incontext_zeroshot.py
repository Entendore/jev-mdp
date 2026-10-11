import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from jev.multitask import MultiTaskScorer

def synth_task(schema, pools, instructions, n=40, seed=0):
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        lab = rng.choice(list(pools))
        samples.append((rng.choice(pools[lab]), lab))
    return {"schema": schema, "instructions": instructions,
            "options": list(pools), "samples": samples}

tasks = [
    synth_task("ticket department",
               {"billing": ["payment failed", "refund my order", "charged twice"],
                "technical": ["app keeps crashing", "500 errors", "integration broken"],
                "sales": ["pricing question", "want to upgrade", "need a quote"]},
               {"billing": "payments, refunds", "technical": "bugs, outages",
                "sales": "pricing, accounts"}, seed=1),
    synth_task("sentiment",
               {"positive": ["love this", "great job team", "works perfectly"],
                "negative": ["this is broken", "terrible support", "very frustrated"],
                "neutral": ["question about billing cycle", "what are your hours"]},
               {"positive": "satisfied", "negative": "dissatisfied",
                "neutral": "no emotion"}, seed=2),
    synth_task("priority",
               {"P0": ["production is down", "all users affected", "data loss"],
                "P1": ["feature broken for some", "workaround exists", "degraded"],
                "P2": ["typo in docs", "cosmetic issue", "nice to have"]},
               {"P0": "drop everything", "P1": "this sprint", "P2": "backlog"}, seed=3),
]
unseen = synth_task("issue channel",
                    {"app": ["ios app crashes on login", "android app freezing", "mobile push broken"],
                     "web": ["website 500 error", "checkout button broken on site", "page not loading"],
                     "api": ["api returns 401", "endpoint timeout", "graphql schema error"]},
                    {"app": "mobile clients", "web": "browser clients",
                     "api": "developer integrations"}, seed=4)

tr = [{**t, "samples": t["samples"][:30]} for t in tasks]
val = [{**t, "samples": t["samples"][30:], "exemplars": t["samples"][:3]}
       for t in tasks]
m = MultiTaskScorer(dim=96, seed=0).fit(tr, epochs=12).fit_temperature(val)
print("train-schema val accuracy:", round(m.evaluate(val), 3))
for k in [0, 1, 3]:                                    # in-context k-curve
    acc = m.evaluate([{**unseen, "exemplars": unseen["samples"][:k]}])
    print(f"UNSEEN schema, k={k} exemplars: acc={acc:.2f}")
m.save("artifacts/scorer.pt")
print("scorer saved -> artifacts/scorer.pt (serves POST /v1/schema/decide)")
print(m.decide(unseen["schema"], unseen["instructions"],
               unseen["samples"][:3], "the ios app crashes on login",
               unseen["options"]))
