"""Domain APIs (v5): uniform DecisionRequest -> DecisionResponse protocol
with budget, telemetry, action masks, deadline holds, deadman defaults.
- WorkflowAPI: parallel tool/done/risk; separate guardrail engine.
- RoboticsAPI: control loop + CBF hook (v6.14, Ames'17)."""
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .primitives import Question, dict_to_text, apply_mask


@dataclass
class DecisionRequest:
    state: Any
    context: Dict[str, Any] = field(default_factory=dict)
    deadline_ms: Optional[float] = None
    mode: str = "neural"


@dataclass
class DecisionResponse:
    action: Any
    outputs: Dict[str, Dict[str, Any]]
    confidence: float
    meta: Dict[str, Any] = field(default_factory=dict)


class DecisionBudget:
    """Rate + cost governance — the USD/hour story with YOUR numbers."""
    def __init__(self, max_calls_per_sec=None, max_cost_usd=None,
                 est_usd_per_1k=0.004):
        self.rate, self.max_cost = max_calls_per_sec, max_cost_usd
        self.per_1k = est_usd_per_1k
        self._stamps: deque = deque()
        self.cost_spent, self.total_calls = 0.0, 0

    def allow(self) -> bool:
        now = time.time()
        if self.rate is not None:
            while self._stamps and now - self._stamps[0] > 1.0:
                self._stamps.popleft()
            if len(self._stamps) >= self.rate:
                return False
        if self.max_cost is not None and self.cost_spent >= self.max_cost:
            return False
        self._stamps.append(now)
        self.cost_spent += self.per_1k / 1000.0
        self.total_calls += 1
        return True

    def usd_per_hour(self):
        if not self._stamps:
            return 0.0
        span = max(time.time() - self._stamps[0], 1e-6)
        return (len(self._stamps) / span) * 3600.0 * self.per_1k / 1000.0


class Telemetry:
    def __init__(self):
        self.events: deque = deque(maxlen=5000)

    def record(self, **kw):
        self.events.append({"t": time.time(), **kw})

    def summary(self):
        if not self.events:
            return {}
        lat = sorted(e["latency_ms"] for e in self.events if "latency_ms" in e)
        n = len(self.events)
        pct = lambda p: lat[min(int(p * len(lat)), len(lat) - 1)] if lat else 0
        from collections import Counter
        return {"decisions": n, "latency_p50_ms": round(pct(0.5), 2),
                "latency_p99_ms": round(pct(0.99), 2),
                "source_mix": {s: round(c / n, 3) for s, c in
                               Counter(e.get("source", "?")
                                       for e in self.events).items()}}


class BaseDecisionAPI(ABC):
    def __init__(self, engine, questions, primary, budget=None, telemetry=None,
                 thresholds=(0.90, 0.70), seed=0):
        self.eng, self.questions, self.primary = engine, questions, primary
        self.budget = budget or DecisionBudget()
        self.tel = telemetry or Telemetry()
        self.hi, self.lo = thresholds
        self.rng = random.Random(seed)
        self.last_action = None

    @abstractmethod
    def serialize(self, state):
        ...
    def mask(self, state):
        return None
    def safe_default(self, state):
        return self.last_action

    def propose(self, req: DecisionRequest) -> DecisionResponse:
        t0 = time.perf_counter()
        if not self.budget.allow():
            return DecisionResponse(self.safe_default(req.state), {}, 0.0,
                                    {"degraded": True, "source": "budget"})
        text = self.serialize(req.state)
        r = self.eng.decide(text, mode=req.mode)
        lat = (time.perf_counter() - t0) * 1000.0
        late = req.deadline_ms is not None and lat > req.deadline_ms
        q = next(x for x in self.questions if x.name == self.primary)
        out = r["outputs"].get(self.primary, {})
        probs = np.array([out.get("probabilities", {}).get(o, 0.0)
                          for o in q.options]) \
            or np.full(q.n_actions, 1 / q.n_actions)
        m = self.mask(req.state)
        if m is not None:
            probs = apply_mask(probs, m)
        ai = int(np.argmax(probs))
        action = q.options[ai]
        self.last_action = action
        self.tel.record(latency_ms=lat, source=r.get("source", req.mode),
                        degraded=late)
        return DecisionResponse(action, r.get("outputs", {}), float(probs[ai]),
                                {"latency_ms": round(lat, 2),
                                 "degraded": late,
                                 "source": r.get("source", req.mode)})

    def propose_verified(self, req, verifier=None) -> DecisionResponse:
        """Ladder: hi -> act | mid -> matrix cross-check | lo -> verifier."""
        r = self.propose(req)
        if r.confidence >= self.hi:
            return r
        if r.confidence >= self.lo:
            r2 = self.propose(DecisionRequest(req.state, req.context,
                                              req.deadline_ms, "matrix"))
            if r2.outputs and r2.action == r.action:
                r.meta["verified_by"] = "matrix-cross-check"
                return r
            r.meta["escalated"] = "matrix-disagreement"
        if verifier is not None and verifier(req.state) < self.lo:
            r.action, r.confidence = self.safe_default(req.state), 0.0
        return r


class WorkflowAPI(BaseDecisionAPI):
    def __init__(self, engine, tools, preconditions=None, budget=None,
                 telemetry=None, thresholds=(0.90, 0.70),
                 independent_guardrail=None):
        qs = [Question("tool", "choice", options=tools + ["finish"],
                       instructions="Next action"),
              Question("done", "noul"),
              Question("risk", "choice", options=["safe", "risky", "blocked"],
                       weight=0.5)]
        super().__init__(engine, qs, "tool", budget, telemetry, thresholds)
        self.preconds = preconditions or {}
        self.guardrail = independent_guardrail

    def serialize(self, state):
        return dict_to_text(state)

    def mask(self, state):
        q = self.questions[0]
        m = np.ones(len(q.options))
        for i, t in enumerate(q.options):
            fn = self.preconds.get(t)
            if fn is not None and not fn(state):
                m[i] = 0.0
        return m

    def run(self, env, task, max_steps=12, mode="neural"):
        trace, obs = [], env.reset(task)
        for step in range(max_steps):
            resp = self.propose_verified(
                DecisionRequest(obs, deadline_ms=800, mode=mode))
            tool = resp.action
            if tool == "finish" or resp.outputs.get("done", {}).get(
                    "probability", 0) > 0.9:
                trace.append({"step": step, "tool": "finish", "status": "done"})
                break
            if tool == "blocked":
                obs = env.blocked("unsafe")
                trace.append({"step": step, "tool": tool,
                              "status": "guard-blocked"})
                continue
            result = env.execute(tool)
            trace.append({"step": step, "tool": tool, "status": "ok",
                          "conf": round(resp.confidence, 3)})
            obs = result
        return {"trace": trace, "telemetry": self.tel.summary(),
                "budget_usd": round(self.budget.cost_spent, 4)}


class RoboticsAPI(BaseDecisionAPI):
    def __init__(self, engine, actions, mask_fn=None, control_hz=50.0,
                 safe_action="stop", budget=None, telemetry=None, cbf=None,
                 seed=0):
        qs = [Question("action", "choice", options=actions),
              Question("magnitude", "score",
                       score_levels=["off", "low", "med", "high", "max"],
                       weight=0.3)]
        super().__init__(engine, qs, "action", budget, telemetry,
                         (0.9, 0.7), seed)
        self._mask_fn, self.control_hz = mask_fn, control_hz
        self.safe_action = safe_action
        self.cbf = cbf                                     # v6.14 filter

    def serialize(self, state):
        return dict_to_text(state)

    def mask(self, state):
        return self._mask_fn(state) if self._mask_fn else None

    def safe_default(self, state):
        return self.safe_action

    def shield(self, state, action):
        if self.cbf is not None:
            q = self.questions[0]
            probs = np.array(list(self.eng.decide(self.serialize(state))
                                  ["outputs"]["action"]["probabilities"]
                                  .values()))
            return self.cbf.filter(state, probs, q.options)[0]
        m = self.mask(state)
        if m is None:
            return action
        q = self.questions[0]
        i = q.options.index(action)
        return action if m[i] > 0 else q.options[int(np.argmax(m))]

    def run(self, sim, seconds=5.0, mode="neural"):
        period = 1.0 / self.control_hz
        deadline_ms = period * 1000 * 0.6
        frames, held = int(seconds * self.control_hz), 0
        traj = []
        for _ in range(frames):
            obs = sim.sense()
            resp = self.propose(DecisionRequest(obs, deadline_ms=deadline_ms,
                                                mode=mode))
            action = resp.action
            if resp.meta["degraded"]:
                action, held = self.last_action, held + 1
            action = self.shield(obs, action)
            mag = resp.outputs.get("magnitude", {}).get("score", 0.0)
            sim.apply(action, float(mag))
            traj.append((self.serialize(obs),
                         self.questions[0].options.index(action),
                         sim.reward_delta()))
            if sim.terminated():
                break
        return {"frames": len(traj), "held_cycles": held,
                "total_reward": round(sum(r for _, _, r in traj), 3),
                "telemetry": self.tel.summary()}
