"""JevMDP — the composer. Wires every subsystem from the research series:
v1 MDP core · v2 calibration · v3 matrices · v4 soft-VI/SPIBB policy ·
v5 ladder/budget/telemetry · v6 ACI/CRC/OOD/drift · v9 kNN memory ·
v10 EVSI ladder + Gittins serving + program-first compilation + persistence.

Lifecycle:
  fit_supervised -> fit_rl -> calibrate -> build_matrices -> add_terminals ->
  learn_transitions -> build_reward -> fit_policy -> calibrate -> fit_crc ->
  [compile_program] -> save -> serve (decide) -> observe_outcome / learn_*
"""
import copy, math, random, time
from collections import OrderedDict, deque
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .primitives import Question, MultiChoiceQuestion, solve_mdp, ece
from .encoder import TextEncoder, BeliefEncoder
from .policy import DecisionPolicy, fit_temperature
from .matrices import StateMatrix, ProbabilisticMatrix, RewardMatrix
from .calibration import (AdaptiveConformal, DriftDetector,
                          ConformalRiskController)
from .memory import KNNMemory
from .training import fit_supervised, fit_rl, distill, learn_feedback_der
from .alignment import PreferenceReward, kl_regularized_improvement
from .evaluation import ope_report, ope_ensemble
from .information import evsi_query, gittins_action
from .compile import compile_program
from .world import RegimeSwitchingMatrices


class JevMDP:
    def __init__(self, questions, dim=128, seed=0, k=24, gamma=0.9,
                 dirichlet_alpha=0.5, kappa=20.0, talpha=0.5, n_min=8,
                 memory_enabled=True, ood_fpr=0.05, belief_enabled=False,
                 evidential=False):
        self.questions = questions
        self.seed = seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.dim, self.k, self.gamma = dim, k, gamma
        self.alpha_d, self.kappa, self.talpha = dirichlet_alpha, kappa, talpha
        self.n_min, self.ood_fpr = n_min, ood_fpr
        self.gittins_gamma, self.crosscheck_lo = 0.95, 0.55
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = TextEncoder(dim=dim).to(self.device)
        self.belief = (BeliefEncoder(dim).to(self.device)
                       if belief_enabled else None)
        self.policy = DecisionPolicy(dim, questions,
                                     evidential=evidential).to(self.device)
        self.temp: Dict[str, float] = {}

        self.state_matrix: Optional[StateMatrix] = None
        self.pm: Optional[ProbabilisticMatrix] = None
        self.reward: Optional[RewardMatrix] = None
        self.T = self.nT = self.C = None
        self.terminal_reward = None
        self.pi_soft = None; self.pi_target: Dict[str, np.ndarray] = {}
        self._episodes: List[Tuple[int, int, int]] = []
        self.outcome_rewards: Dict[str, float] = {}
        self.regimes = RegimeSwitchingMatrices(self)

        self.aci = AdaptiveConformal()
        self.qhat: Dict[str, float] = {}
        self.ood_energy_thr = -math.inf
        self.ood_dist_thr = math.inf
        self.crc_tau, self.crc_info = 0.0, {}
        self.memory = KNNMemory(self) if memory_enabled else None
        self.compiled = None
        self._cache: "OrderedDict" = OrderedDict()
        self._pending: List[Dict] = []
        self._corr_buf = deque(maxlen=64)
        self.tel = {"n": 0, "sources": {}, "lat": []}
        self._fit_texts = None; self._fit_labels = None
        self._ref_embs: Optional[np.ndarray] = None

    # ------------------------------------------------------------ helpers --
    def _primary(self):
        return next(q.name for q in self.questions if q.qtype == "choice")

    def _q(self, name):
        return next(q for q in self.questions if q.name == name)

    def _label_index(self, q, raw):
        if q.qtype == "noul":
            return 1 if str(raw).lower() in ("yes", "true", "1") else 0
        if q.qtype == "score":
            return int(raw) if not isinstance(raw, str) else q.score_levels.index(raw)
        return q.options.index(raw)

    def _params(self):
        return list(self.encoder.parameters()) + list(self.policy.parameters())

    def _freeze_eval(self):
        self.policy.eval(); self.encoder.eval()

    def _record(self, source, lat):
        self.tel["n"] += 1
        self.tel["sources"][source] = self.tel["sources"].get(source, 0) + 1
        self.tel["lat"].append(round(lat, 2))

    @torch.no_grad()
    def _val_loss(self, vtexts, vlabels):
        h = self.encoder(vtexts)
        return sum(float(q.weight * F.cross_entropy(
            self.policy.heads[q.name](h),
            torch.tensor([self._label_index(q, l[q.name]) for l in vlabels])))
            for q in self.questions)

    # ------------------------------------------------------- Phase 1-2 -----
    def fit_supervised(self, texts, labels, val=None, **kw):
        return fit_supervised(self, texts, labels, val=val, **kw)

    def fit_rl(self, texts, labels, **kw):
        return fit_rl(self, texts, labels, **kw)

    def learn_correction(self, text, labels=None, avoid=None, **kw):
        return learn_feedback_der(self, text, labels, avoid, **kw)

    # --------------------------------------------------- Phase 3: calib ----
    def calibrate(self, val_texts, val_labels):
        self.policy.eval()
        with torch.no_grad():
            h = self.encoder(val_texts)
            logits = self.policy.logits(h)
        targets = {q.name: torch.tensor([self._label_index(q, l[q.name])
                                         for l in val_labels])
                   for q in self.questions}
        self.temp = fit_temperature(logits, targets)
        energies = torch.stack([torch.logsumexp(logits[q.name], -1)
                                for q in self.questions]).sum(0).cpu().numpy()
        self.ood_energy_thr = float(np.quantile(energies, self.ood_fpr))
        if self.state_matrix is not None:
            d = torch.cdist(h, self.state_matrix.centroids[
                : self.state_matrix.n_spatial]).min(1).values.cpu().numpy()
            self.ood_dist_thr = float(np.quantile(d, 0.95))
        rep = {}
        for q in self.questions:
            if q.qtype != "choice":
                continue
            P = F.softmax(logits[q.name] / self.temp[q.name], -1).cpu().numpy()
            t = targets[q.name].cpu().numpy()
            rep[q.name] = {"ece": round(ece(P, t), 4)}
            scores = 1.0 - P[np.arange(len(t)), t]
            kth = math.ceil((len(t) + 1) * 0.9)
            self.qhat[q.name] = float(np.sort(scores)[min(kth, len(t)) - 1])
            if q.name == self._primary():
                self.aci.warm(scores)
                rep[q.name]["conformal_qhat"] = round(self.qhat[q.name], 3)
        confs = F.softmax(logits[self._primary()]
                          / self.temp[self._primary()], -1) \
            .max(1).values.cpu().numpy()
        self.drift = DriftDetector(); self.drift.fit(confs)
        return rep

    # ------------------------------------------------ Phase 4: matrices ----
    def build_matrices(self, texts, labels):
        self.policy.eval()
        with torch.no_grad():
            emb = self.encoder(texts)
            logits = self.policy.logits(emb)
        self._ref_embs = emb.cpu().numpy()          # reference for LiveOps
        self.state_matrix = StateMatrix(k=self.k, seed=self.seed).fit(emb, texts)
        cl = self.state_matrix.assign(emb).cpu().numpy()
        neural = {q.name: F.softmax(logits[q.name] / self.temp.get(q.name, 1.0),
                                    -1).cpu().numpy() for q in self.questions}
        label_ids = {q.name: np.array([self._label_index(q, l[q.name])
                                       for l in labels]) for q in self.questions}
        self.pm = ProbabilisticMatrix(self.alpha_d, self.kappa)
        self.pm.fit(cl, label_ids, neural, self.questions,
                    self.state_matrix.k_total)
        return {"k": self.state_matrix.n_spatial}

    def add_terminals(self, rewards: Dict[str, float]):
        for name in rewards:
            self.state_matrix.add_terminal(name)
        K = self.state_matrix.k_total
        self.terminal_reward = np.zeros(K)
        for name, r in rewards.items():
            self.terminal_reward[self.state_matrix.terminal_index(name)] = r
        self.pm.pad_to(K)
        self.outcome_rewards = dict(rewards)

    def learn_transitions(self, episodes):
        self._episodes = list(getattr(self, "_episodes", [])) + list(episodes)
        qn = self._primary()
        K = self.state_matrix.k_total
        A = self._q(qn).n_actions
        nT = np.full((K, A, K), self.talpha)
        for k, a, k2 in episodes:
            if 0 <= k < K and 0 <= a < A:
                nT[k, a, k2] += 1.0
        self.nT = nT
        self.T = nT / nT.sum(-1, keepdims=True)
        for t in np.where(self.state_matrix.terminal_mask)[0]:
            self.T[t] = 0.0
            self.T[t, :, t] = 1.0

    def build_reward(self):
        qn = self._primary(); q = self._q(qn)
        self.reward = RewardMatrix(gamma=self.gamma).fit(
            self.pm, q, self.T, self.state_matrix.terminal_mask,
            self.terminal_reward)
        self.C = self.reward.C
        self.V, self.Q = self.reward.V, self.reward.Q
        return {"R_base_range": [round(float(self.reward.R_base.min()), 3),
                                 round(float(self.reward.R_base.max()), 3)],
                "max_base_pess_gap": round(float(
                    np.abs(self.reward.R_base - self.reward.R_pess).max()), 3)}

    # ------------------------------------------------ Phase 5: policy ------
    def fit_policy(self, texts=None, labels=None, rounds=3, tau0=1.0,
                   tau_decay=0.5, tau_min=0.15, distill_epochs=8, anchor=0.2,
                   n_min=None):
        texts = texts or self._fit_texts
        labels = labels or self._fit_labels
        self._fit_texts, self._fit_labels = list(texts), list(labels)
        qn = self._primary()
        term = self.state_matrix.terminal_mask
        counts = self.pm.counts[qn]
        w = np.clip(counts.sum(1, keepdims=True) / (n_min or self.n_min), 0, 1)
        beh = self.pm.P[qn].copy(); beh[term] = 0.0
        hist, tau = [], tau0
        for r in range(rounds):
            _, _, Pi = solve_mdp(self.reward.compose("eval"), self.T, term,
                                 self.gamma, tau=tau)
            self.pi_soft = Pi
            tgt = w * Pi + (1 - w) * beh                 # SPIBB blend (v4)
            row = tgt.sum(1, keepdims=True)
            self.pi_target = {qn: np.where(row > 1e-9,
                                           tgt / np.maximum(row, 1e-9), beh)}
            distill(self, texts, labels, epochs=distill_epochs, anchor=anchor)
            H = -(Pi * np.log(Pi + 1e-12)).sum(1)[~term]
            hist.append({"round": r, "tau": round(tau, 3),
                         "mean_entropy": round(float(H.mean()), 3)})
            tau = max(tau * tau_decay, tau_min)
        return hist

    # ----------------------------------------------- Phase 6: CRC threshold
    def fit_crc(self, val_texts, val_labels, alpha_R=0.1):
        qn = self._primary(); q = self._q(qn)
        with torch.no_grad():
            h = self.policy.trunk(self.encoder(val_texts))
            P = F.softmax(self.policy.heads[qn](h) / self.temp[qn],
                          -1).cpu().numpy()
        t = np.array([self._label_index(q, l[qn]) for l in val_labels])
        Cn = self.C / max(self.C.max(), 1e-9)
        crc = ConformalRiskController(alpha_R=alpha_R, B=1.0)
        info = crc.fit(P, t, Cn[t, P.argmax(1)])
        self.crc_tau, self.crc_info = crc.tau, info
        return info

    # --------------------------------------------------------- SERVING -----
    @torch.no_grad()
    def _policy_choice(self, state):
        qn = self._primary(); q = self._q(qn)
        h = self.encoder([state])
        p = F.softmax(self.policy.heads[qn](self.policy.trunk(h))
                      / self.temp.get(qn, 1.0), -1)[0].cpu().numpy()
        return q.options[int(p.argmax())], p

    def _conf_set(self, p, qn):
        return [i for i, v in enumerate(p) if v >= 1 - self.qhat.get(qn, 1.0)]

    @torch.no_grad()
    def decide(self, state, mode="neural", mask=None, query_channel=None,
               query_cost=0.02, history=None):
        """THE DECISION LADDER (mode='neural'):
        OOD -> deadman | conf >= CRC tau -> auto-act (cost-bounded)
        | query channel + EVSI > 0 -> buy information
        | mid confidence -> matrix cross-check | else escalate.
        Modes: neural | matrix | plan | policy | thompson | gittins.
        Returns typed outputs for EVERY question (parallel-pass analog)."""
        t0 = time.perf_counter()
        qn = self._primary(); q = self._q(qn)
        ck = (mode, state, query_channel)
        if ck in self._cache:
            self._cache.move_to_end(ck)
            return copy.deepcopy(self._cache[ck])

        h = self.encoder([state])
        if self.belief is not None and history:
            h = h + self.belief(self.encoder(history[-8:]))
        logits = self.policy.logits(h)

        # 1 -- OOD gate: energy (v2/Liu'20) U centroid distance (practical)
        if self.state_matrix is not None:
            energy = float(sum(torch.logsumexp(logits[x.name], -1)[0]
                               for x in self.questions))
            d = float(torch.cdist(h, self.state_matrix.centroids[
                : self.state_matrix.n_spatial]).min())
            if energy < self.ood_energy_thr or d > self.ood_dist_thr:
                self._record("deadman", (time.perf_counter() - t0) * 1000)
                return {"action": "ESCALATE", "source": "deadman",
                        "confidence": 0.0, "mode": mode, "cluster": None,
                        "outputs": {}, "latency_ms": self.tel["lat"][-1]}

        # 2 -- probabilities: neural (+temp) -> kNN blend -> matrix row
        probs = {x.name: F.softmax(logits[x.name] / self.temp.get(x.name, 1.0),
                                   -1)[0] for x in self.questions}
        mem_lam = 0.0
        if self.memory is not None:
            p_mem, mem_lam = self.memory.posterior(state, q,
                                                   probs[qn].cpu().numpy())
            probs[qn] = torch.tensor(p_mem, dtype=torch.float32)
        cl = int(self.state_matrix.assign(h)[0]) if self.state_matrix else None
        if mode == "matrix" and self.pm is not None:
            probs[qn] = torch.tensor(self.pm.P[qn][cl], dtype=torch.float32)

        p = probs[qn].cpu().numpy()
        if mask is not None:
            pm_ = p * mask
            p = pm_ / max(pm_.sum(), 1e-9)

        # 3 -- action selection per mode
        if mode == "policy":
            ai = int(np.random.choice(len(p), p=p))
        elif mode == "thompson" and self.pm is not None:
            theta = np.random.dirichlet(self.alpha_d
                                        + self.pm.counts[qn][cl] + 1e-9)
            ai = int(np.argmin(theta @ self.C)) if self.C is not None \
                else int(np.argmax(theta))
        elif mode == "plan" and self.Q is not None:
            ai = int(self.Q[cl].argmax())
        elif mode == "gittins" and self.pm is not None:
            ai = gittins_action(self, cl, qn, self.gittins_gamma)
        else:
            ai = int(np.argmax(p))
        action, conf, source = q.options[ai], float(p[ai]), mode

        # 4 -- THE LADDER (neural mode only)
        if mode == "neural":
            if conf >= self.crc_tau:
                source = "crc-act"
            elif query_channel is not None and self.pm is not None:
                ev = evsi_query(self, cl, qn, query_cost)
                if ev["recommend_query"]:
                    action, source = f"QUERY:{query_channel}", "evsi"
                else:
                    action, source = "ESCALATE", "evsi-not-worth-it"
            elif conf >= self.crosscheck_lo and self.pm is not None:
                if int(np.argmax(self.pm.P[qn][cl])) == ai:
                    source = "crosscheck-ok"
                else:
                    action, source = "ESCALATE", "crosscheck-disagree"
            else:
                action, source = "ESCALATE", "low-confidence"
            if (self.compiled and self.compiled["ok"]
                    and action in q.options):
                tree_a = self.compiled["tree"].predict(
                    self.encoder([state]).cpu().numpy())[0]
                if tree_a == action:
                    source = "program"

        # 5 -- typed entries for EVERY question (parallel-pass analog)
        outputs: Dict[str, Any] = {}
        for x in self.questions:
            px = p if x.name == qn else probs[x.name].cpu().numpy()
            e: Dict[str, Any] = {
                "probabilities": {o: round(float(px[i]), 4)
                                  for i, o in enumerate(x.options)},
                "confidence": round(float(px.max()), 4)}
            if x.qtype == "noul":
                e = {"probability": float(px[1]),
                     "confidence": round(max(px[0], px[1]), 4)}
            elif x.qtype == "score":
                e["score"] = float((px * np.arange(len(px))).sum())
            if x.qtype == "choice" and x.name == qn:
                e["conformal_set"] = [x.options[i]
                                      for i in self._conf_set(p, qn)]
                if self.pm is not None and cl is not None:
                    e["wilson_95"] = {
                        o: [round(float(self.pm.lo[qn][cl, i]), 3),
                            round(float(self.pm.hi[qn][cl, i]), 3)]
                        for i, o in enumerate(x.options)}
            outputs[x.name] = e

        self._pending.append({"state": state, "probs": p, "qname": qn})
        if len(self._pending) > 512:
            self._pending.pop(0)
        lat = (time.perf_counter() - t0) * 1000
        self._record(source, lat)
        out = {"action": action, "source": source, "confidence": round(conf, 4),
               "mode": mode, "cluster": cl,
               "memory_lambda": round(mem_lam, 3),
               "outputs": outputs, "latency_ms": round(lat, 2)}
        self._cache[ck] = out
        if len(self._cache) > 4096:
            self._cache.popitem(last=False)
        return copy.deepcopy(out)

    # ------------------------------------------------------ FEEDBACK LOOP --
    def observe_outcome(self, state, true_label):
        """Ground truth -> adaptive conformal (v6.1) + kNN memory (v9.2)."""
        pend = next((x for x in self._pending if x["state"] == state), None)
        if pend is not None:
            q = self._q(pend["qname"])
            if true_label in q.options:
                self.aci.observe(pend["probs"], q.options.index(true_label))
            self._pending.remove(pend)
        if self.memory is not None:
            self.memory.add([state], [true_label])

    def learn_preferences(self, pairs, kl_beta=0.1, blend_w=0.5,
                          distill_epochs=4):
        """Bradley-Terry -> blend into R -> EXACT KL-braked improvement (v8.1)."""
        qn = self._primary(); q = self._q(qn)
        K = self.state_matrix.k_total
        triples = []
        with torch.no_grad():
            for text, better, worse in pairs:
                k = int(self.state_matrix.assign(self.encoder([text]))[0])
                triples.append((k, q.options.index(better),
                                q.options.index(worse)))
        pref = PreferenceReward(K, q.n_actions)
        ll = pref.fit(triples)
        term = self.state_matrix.terminal_mask
        R = self.reward.compose("eval").copy()
        R[~term] = ((1 - blend_w) * R[~term] + blend_w * pref.matrix()[~term])
        pi_ref = self.pm.P[qn].copy(); pi_ref[term] = 0.0
        self.pi_target = {qn: kl_regularized_improvement(pi_ref, R, kl_beta)}
        distill(self, self._fit_texts, self._fit_labels,
                epochs=distill_epochs)
        move = float(np.abs(self.pi_target[qn][~term]
                            - self.pm.P[qn][~term]).sum(1).mean())
        return {"pref_loglik": round(ll, 4), "L1_move": round(move, 4),
                "kl_beta": kl_beta}

    # ------------------------------------------- CERTIFICATION / REPORT ----
    def compile_program(self, texts, **kw):
        return compile_program(self, texts, **kw)

    def ope_check(self, logs3, challenger_rows=None):
        """DR + HCOPE on (k, a, reward) logs (v6.3 + v9.5)."""
        qn = self._primary()
        pi_e = (challenger_rows if challenger_rows is not None
                else self.pi_target[qn])
        return ope_report(logs3, self.pm.P[qn], self.Q, pi_e)

    def ope_full(self, logs4, challenger_rows=None, gamma=None):
        """{DR, FQE, MIS} + median (v8.6). logs4 rows: (k, a, k2, reward)."""
        qn = self._primary()
        pi_e = (challenger_rows if challenger_rows is not None
                else self.pi_target[qn])
        return ope_ensemble(logs4, self.pm.P[qn], self.Q, self.T, pi_e,
                            gamma=gamma or self.gamma)

    def report(self, texts, labels):
        self.policy.eval()
        out = {}
        with torch.no_grad():
            logits = self.policy.logits(self.encoder(texts))
        for q in self.questions:
            P = F.softmax(logits[q.name] / self.temp.get(q.name, 1.0),
                          -1).cpu().numpy()
            t = np.array([self._label_index(q, l[q.name]) for l in labels])
            d = {"accuracy": round(float((P.argmax(1) == t).mean()), 4),
                 "ece": round(ece(P, t), 4)}
            if q.qtype == "choice":
                sets = [self._conf_set(p, q.name) for p in P]
                d["coverage"] = round(float(np.mean(
                    [t[i] in s for i, s in enumerate(sets)])), 4)
                d["avg_set_size"] = round(float(np.mean(
                    [len(s) for s in sets])), 2)
            out[q.name] = d
        return out

    def tel_summary(self):
        lat = sorted(self.tel["lat"])
        return {"decisions": self.tel["n"],
                "latency_p50_ms": lat[len(lat) // 2] if lat else 0,
                "latency_p99_ms": lat[min(len(lat) - 1,
                                          int(0.99 * len(lat)))] if lat else 0,
                "source_mix": dict(self.tel["sources"])}

    # ------------------------------------------------------ PERSISTENCE ----
    def save(self, path):
        """Full persistence: nets, calibration, matrices, compiled tree,
        kNN memory."""
        payload = {
            "init": {"questions": [asdict(q) for q in self.questions],
                     "dim": self.dim, "seed": self.seed, "k": self.k,
                     "gamma": self.gamma,
                     "dirichlet_alpha": self.alpha_d, "kappa": self.kappa,
                     "talpha": self.talpha, "n_min": self.n_min,
                     "memory_enabled": self.memory is not None,
                     "ood_fpr": self.ood_fpr,
                     "belief_enabled": self.belief is not None,
                     "evidential": getattr(self.policy, "evidential", False)},
            "encoder": self.encoder.state_dict(),
            "policy": self.policy.state_dict(),
            "temp": dict(self.temp), "qhat": dict(self.qhat),
            "ood_energy_thr": self.ood_energy_thr,
            "ood_dist_thr": self.ood_dist_thr,
            "crc_tau": self.crc_tau,
            "sm": {"centroids": self.state_matrix.centroids,
                   "visits": self.state_matrix.visits,
                   "exemplars": list(self.state_matrix.exemplars),
                   "terminal_names": dict(self.state_matrix.terminal_names),
                   "n_spatial": self.state_matrix.n_spatial},
            "pm": {q.name: {"counts": self.pm.counts[q.name],
                            "P": self.pm.P[q.name],
                            "lo": self.pm.lo[q.name],
                            "hi": self.pm.hi[q.name]}
                   for q in self.questions},
            "T": self.T, "nT": self.nT, "C": self.C,
            "terminal_reward": self.terminal_reward,
            "pi_target": self.pi_target,
            "compiled_tree": (self.compiled["tree"] if self.compiled else None),
            "memory": ({"E": self.memory.E, "y": list(self.memory.y),
                        "texts": list(self.memory.texts)}
                       if self.memory is not None else None),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu")
        init = dict(payload["init"]); qd = init.pop("questions")
        qs = []
        for d in qd:
            d = dict(d)
            if "k" in d:
                qs.append(MultiChoiceQuestion(**d))
            else:
                qs.append(Question(**d))
        eng = cls(qs, **init)
        eng.encoder.load_state_dict(payload["encoder"])
        eng.policy.load_state_dict(payload["policy"])
        eng.temp = payload["temp"]; eng.qhat = payload["qhat"]
        eng.ood_energy_thr = payload["ood_energy_thr"]
        eng.ood_dist_thr = payload["ood_dist_thr"]
        eng.crc_tau = payload["crc_tau"]
        sm = eng.state_matrix = StateMatrix(k=init["k"], seed=init["seed"])
        s = payload["sm"]
        sm.centroids = s["centroids"]; sm.visits = s["visits"]
        sm.exemplars = list(s["exemplars"])
        sm.terminal_names = dict(s["terminal_names"])
        sm.n_spatial = int(s["n_spatial"])
        sm.k_total = sm.centroids.shape[0]
        eng.pm = ProbabilisticMatrix(init["dirichlet_alpha"], init["kappa"])
        for name, d in payload["pm"].items():
            eng.pm.counts[name] = d["counts"]; eng.pm.P[name] = d["P"]
            eng.pm.lo[name] = d["lo"]; eng.pm.hi[name] = d["hi"]
        eng.T = payload["T"]; eng.nT = payload["nT"]
        eng.C = payload["C"]; eng.terminal_reward = payload["terminal_reward"]
        eng.pi_target = payload["pi_target"]
        if payload.get("compiled_tree") is not None:
            eng.compiled = {"tree": payload["compiled_tree"], "ok": True}
        if payload.get("memory") is not None and eng.memory is not None:
            eng.memory.E = payload["memory"]["E"]
            eng.memory.y = list(payload["memory"]["y"])
            eng.memory.texts = list(payload["memory"]["texts"])
        if eng.T is not None:
            eng.build_reward()
        eng._freeze_eval()
        return eng


# ----------------------------------------------------------- demo factory --
_TICKET_POOLS = {
    "billing":   ["payment failed", "invoice is wrong", "refund my order",
                  "i was charged twice"],
    "technical": ["app keeps crashing", "integration broken",
                  "getting 500 errors", "export bug"],
    "sales":     ["pricing question", "want to upgrade",
                  "opening new account", "need a quote"],
}
_HOT = ["asap", "urgent", "this is down", "immediately"]
_COOL = ["no rush", "whenever you can", "small thing", "fyi"]


def make_tickets(n, seed=0):
    rng = random.Random(seed)
    X, Y = [], []
    for _ in range(n):
        d = rng.choice(list(_TICKET_POOLS))
        u = rng.random() < 0.5
        X.append(rng.choice(_TICKET_POOLS[d]) + ". "
                 + rng.choice(_HOT if u else _COOL))
        Y.append({"department": d, "is_urgent": "yes" if u else "no",
                  "frustration": rng.choice([3, 4] if u else [0, 1])})
    return X, Y


def default_ticket_questions():
    return [Question("department", "choice",
                     options=["billing", "technical", "sales"],
                     cost_matrix={"billing":   {"sales": 3.0, "technical": 3.0},
                                  "technical": {"sales": 1.0},
                                  "sales":     {"technical": 1.0}}),
            Question("is_urgent", "noul", weight=0.5),
            Question("frustration", "score",
                     score_levels=["calm", "annoyed", "frustrated",
                                   "angry", "furious"], weight=0.5)]


def simulate_episodes(eng, texts, labels, eps=0.3, seed=0,
                      p_ok=(0.92, 0.08), p_bad=(0.35, 0.45, 0.20)):
    """Explore the MDP (correct action w.p. 1-eps); outcomes simulate a
    routing pipeline — correct routes mostly resolve."""
    rng = random.Random(seed)
    qn = eng._primary(); q = eng._q(qn)
    with torch.no_grad():
        cl = eng.state_matrix.assign(eng.encoder(texts)).cpu().numpy()
    out = []
    for y, c in zip(labels, cl):
        a_true = q.options.index(y[qn])
        a = a_true if rng.random() > eps else rng.randrange(q.n_actions)
        end = (rng.choices(["resolved", "escalated"], list(p_ok))[0]
               if a == a_true else
               rng.choices(["resolved", "escalated", "failed"],
                           list(p_bad))[0])
        out.append((int(c), a, eng.state_matrix.terminal_index(end)))
    return out


def demo_engine(seed=0):
    Xtr, Ytr = make_tickets(240, seed=1)
    Xcal, Ycal = make_tickets(70, seed=2)
    Xte, Yte = make_tickets(140, seed=3)
    eng = JevMDP(default_ticket_questions(), seed=seed)
    eng.fit_supervised(Xtr, Ytr, val=(Xcal, Ycal))
    eng.fit_rl(Xtr, Ytr)
    eng.calibrate(Xcal, Ycal)
    eng.build_matrices(Xtr, Ytr)
    eng.add_terminals({"resolved": 1.0, "escalated": -0.25, "failed": -1.0})
    eng.learn_transitions(simulate_episodes(eng, Xtr, Ytr, eps=0.30, seed=seed))
    eng.build_reward()
    eng.fit_policy(Xtr, Ytr)
    eng.calibrate(Xcal, Ycal)
    eng.fit_crc(Xcal, Ycal)
    return eng, (Xte, Yte)
