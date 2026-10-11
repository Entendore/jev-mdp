import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
import jev.information as info
from jev.engine import demo_engine

eng, (Xte, Yte) = demo_engine(seed=0)
print("CRC auto-act threshold:", eng.crc_info)

cases = ["i was charged twice. please refund. urgent",
         "i was charged twice and the app keeps crashing",
         "recipe for lasagna with extra cheese"]
for s in cases:
    r = eng.decide(s)
    print(f"{r['source']:<20} {str(r['action']):<12} conf={r['confidence']:.2f}")

print("\n-- ambiguous ticket, query channel available at USD 0.02 --")
r = eng.decide("i was charged twice and the app keeps crashing",
               query_channel="ask_user")
print("with query channel:", r["action"], "|", r["source"])
print("EVSI detail:", info.evsi_query(
    eng, r["cluster"] or 0, eng._primary(), 0.02))

s = "i was charged twice. urgent"
print("\nthompson histogram (30 draws):",
      dict(Counter(eng.decide(s, mode="thompson")["action"]
                   for _ in range(30))))
print("gittins:", eng.decide(s, mode="gittins")["action"],
      "| cached indices:", len(info._GITT))
print("\nsource mix over 60 test tickets:")
print(dict(Counter(eng.decide(s)["source"] for s in Xte[:60])))
print("telemetry:", eng.tel_summary())
