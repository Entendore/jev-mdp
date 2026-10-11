"""Integration gates: full pipeline, ladder OOD behavior, save/load
roundtrip, compile gate."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

from jev.engine import demo_engine, JevMDP


@pytest.fixture(scope="module")
def eng():
    return demo_engine(seed=0)


def test_full_pipeline_and_ladder(eng):
    e, (Xte, Yte) = eng
    r = e.decide("i was charged twice. urgent")
    assert r["action"] == "billing"
    off = e.decide("recipe for lasagna with extra cheese")
    assert off["action"] == "ESCALATE"
    assert off["source"] in {"deadman", "low-confidence",
                             "crosscheck-disagree", "evsi-not-worth-it"}
    rep = e.report(Xte, Yte)
    assert rep["department"]["accuracy"] > 0.85


def test_all_question_outputs_parallel(eng):
    e, _ = eng
    r = e.decide("i was charged twice. urgent")
    assert set(r["outputs"]) == {"department", "is_urgent", "frustration"}
    assert "probability" in r["outputs"]["is_urgent"]
    assert "score" in r["outputs"]["frustration"]


def test_save_load_roundtrip(eng, tmp_path):
    e, _ = eng
    p = str(tmp_path / "eng.pt")
    e.save(p)
    e2 = JevMDP.load(p)
    s = "integration broken. urgent"
    a, b = e.decide(s), e2.decide(s)
    assert a["action"] == b["action"]
    assert abs(a["confidence"] - b["confidence"]) < 1e-6


def test_compile_gate(eng):
    e, (Xte, _) = eng
    info = e.compile_program(Xte[:100])
    assert info["fidelity"] >= 0.85       # toy-domain threshold
