"""Training loops (engine-facing).
- fit_supervised (v2): BC distillation; label smoothing; cosine LR; early stop.
- fit_rl (v1): REINFORCE + Brier shaping = the RLCD analog (Gneiting&Raftery'07
  — strictly proper scoring yields honest calibrated probabilities).
- distill (v4): SPIBB targets -> neural policy (Laroche NeurIPS'19).
- learn_feedback_der (v2 + v6.10): corrections + unlikelihood (Welleck'19)
  with DER++ logit replay (Buzzega ECCV'20).
- PrioritizedReplay (v7.11, Schaul ICLR'16); her_relabel (v7.10,
  Andrychowicz NeurIPS'17); reinforce_update (self-play, v5)."""
import copy, math, random
from collections import deque
from typing import List
import numpy as np
import torch
import torch.nn.functional as F


def fit_supervised(eng, texts, labels, val=None, epochs=40, lr=1e-3, batch=32,
                   smoothing=0.05, patience=6, verbose=False):
    eng._fit_texts, eng._fit_labels = list(texts), list(labels)
    opt = torch.optim.AdamW(eng._params(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, wait = math.inf, None, 0
    for ep in range(epochs):
        eng.policy.train()
        order = torch.randperm(len(texts)).tolist()
        for i in range(0, len(texts), batch):
            b = order[i:i + batch]
            h = eng.encoder([texts[j] for j in b])
            loss = 0.0
            for q in eng.questions:
                t = torch.tensor([eng._label_index(q, labels[j][q.name])
                                  for j in b])
                loss = loss + q.weight * F.cross_entropy(
                    eng.policy.heads[q.name](h), t, label_smoothing=smoothing)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if val is not None:
            vl = eng._val_loss(val[0], val[1])
            if verbose:
                print(f"  ep {ep:02d} val {vl:.4f}")
            if vl < best - 1e-4:
                best, wait = vl, 0
                best_state = copy.deepcopy(eng.policy.state_dict())
            elif (wait := wait + 1) >= patience:
                break
    if best_state is not None:
        eng.policy.load_state_dict(best_state)
    eng._freeze_eval()
    if eng.memory is not None:
        eng.memory.add(eng._fit_texts,
                       [l[eng._primary()] for l in eng._fit_labels])
    return eng


def fit_rl(eng, texts, labels, epochs=6, lr=3e-4, batch=32, lam=0.5, ent=0.02):
    """REINFORCE on r = 1{a=a*} - lam*Brier + entropy bonus."""
    opt = torch.optim.AdamW(eng._params(), lr=lr)
    for _ in range(epochs):
        order = torch.randperm(len(texts)).tolist()
        for i in range(0, len(texts), batch):
            b = order[i:i + batch]
            h = eng.encoder([texts[j] for j in b])
            loss = 0.0
            for q in eng.questions:
                logp = F.log_softmax(eng.policy.heads[q.name](h), -1)
                p = logp.exp()
                tgt = torch.tensor([eng._label_index(q, labels[j][q.name])
                                    for j in b])
                a = torch.multinomial(p.detach(), 1).squeeze(1)
                with torch.no_grad():
                    y = torch.zeros_like(p)
                    y.scatter_(1, tgt.unsqueeze(1), 1.0)
                    brier = ((p - y) ** 2).sum(-1)
                    adv = (((a == tgt).float() - lam * brier)
                           - ((a == tgt).float() - lam * brier).mean())
                loss = loss + q.weight * (
                    -(adv * logp.gather(1, a.unsqueeze(1)).squeeze(1)).mean()
                    + lam * brier.mean()
                    - ent * (-(p * logp).sum(-1)).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    eng._freeze_eval()
    return eng


def distill(eng, texts, labels, epochs=8, lr=5e-4, anchor=0.2, batch=32):
    """pi_theta <- anchor*onehot(label) + (1-anchor)*SPIBB_target[cluster]."""
    qn = eng._primary()
    with torch.no_grad():
        cl = eng.state_matrix.assign(eng.encoder(texts)).cpu().numpy()
    for p_ in eng.encoder.parameters():
        p_.requires_grad = False
    opt = torch.optim.AdamW(eng.policy.parameters(), lr=lr)
    n = len(texts)
    for _ in range(epochs):
        order = torch.randperm(n).tolist()
        for i in range(0, n, batch):
            b = order[i:i + batch]
            h = eng.policy.trunk(eng.encoder([texts[j] for j in b]))
            loss = 0.0
            for q in eng.questions:
                logp = F.log_softmax(eng.policy.heads[q.name](h), -1)
                tgt = torch.zeros(len(b), q.n_actions)
                for bi, j in enumerate(b):
                    oh = np.zeros(q.n_actions)
                    oh[eng._label_index(q, labels[j][q.name])] = 1.0
                    if q.name == qn and eng.pi_target:
                        t_ = anchor * oh + (1 - anchor) * eng.pi_target[qn][cl[j]]
                    else:
                        t_ = oh
                    tgt[bi] = torch.tensor(t_ / t_.sum())
                loss = loss + q.weight * (-(tgt * logp).sum(1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    for p_ in eng.encoder.parameters():
        p_.requires_grad = True
    eng._freeze_eval(); eng._cache.clear()


def learn_feedback_der(eng, text, labels=None, avoid=None, steps=4, lr=1e-4):
    """Correction + unlikelihood, with DER++ logit-matching replay."""
    if not hasattr(eng, "_der_buf"):
        eng._der_buf = deque(maxlen=128)
    with torch.no_grad():
        h0 = eng.encoder([text])
        snap = {q.name: eng.policy.heads[q.name](eng.policy.trunk(h0))[0].cpu()
                for q in eng.questions}
    eng._der_buf.append((text, snap, labels or {}, avoid or {}))
    eng._corr_buf.append((text, labels or {}, avoid or {}))
    opt = torch.optim.AdamW(eng.policy.parameters(), lr=lr)
    eng.policy.train()
    for _ in range(steps):
        pool = list(eng._der_buf)
        batch = random.sample(pool, min(len(pool), 8))
        h = eng.policy.trunk(eng.encoder([b[0] for b in batch]))
        loss = 0.0
        for q in eng.questions:
            lg = eng.policy.heads[q.name](h)
            for bi, (_, snap_, lab, avo) in enumerate(batch):
                if q.name in lab:
                    t = torch.tensor([eng._label_index(q, lab[q.name])])
                    loss = loss + F.cross_entropy(lg[bi:bi + 1], t)
                if q.name in avo:
                    a = eng._label_index(q, avo[q.name])
                    pb = F.softmax(lg[bi:bi + 1], -1)[0, a]
                    loss = loss + F.binary_cross_entropy(
                        pb.clamp(1e-4, 1 - 1e-4), torch.zeros_like(pb))
                loss = loss + 0.5 * F.mse_loss(
                    lg[bi:bi + 1][0], snap_[q.name].to(lg.device))
        opt.zero_grad(); loss.backward(); opt.step()
    eng._freeze_eval(); eng._cache.clear()


class PrioritizedReplay:
    """p_i ~ (|delta_i|+eps)^alpha; IS weights annealed beta 0.4->1 (Schaul'16)."""
    def __init__(self, capacity=100_000, alpha=0.6, beta0=0.4, eps=1e-4):
        self.buf: List = []
        self.prio = np.zeros(capacity)
        self.alpha, self.beta0, self.eps = alpha, beta0, eps
        self.pos, self.max_prio, self.beta = 0, 1.0, beta0

    def push(self, item, priority=None):
        if len(self.buf) < len(self.prio):
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
        p = self.max_prio if priority is None else priority
        self.prio[self.pos] = (p + self.eps) ** self.alpha
        self.pos = (self.pos + 1) % len(self.prio)

    def sample(self, n: int):
        ps = self.prio[: len(self.buf)]
        probs = ps / ps.sum()
        idx = np.random.choice(len(self.buf), n, p=probs)
        self.beta = min(1.0, self.beta0 + self.pos / max(len(self.prio), 1)
                        * (1.0 - self.beta0))
        w = (len(self.buf) * probs[idx]) ** (-self.beta)
        return [self.buf[i] for i in idx], idx, w / w.max()

    def update_prios(self, idx, deltas):
        for i, d in zip(idx, np.abs(np.asarray(deltas))):
            self.prio[i] = (d + self.eps) ** self.alpha
            self.max_prio = max(self.max_prio, d)


def her_relabel_workflow(episodes, achieved_extractor, rng=random.Random(0)):
    """Failed traces -> successes for the goal they DID achieve (HER)."""
    extra = []
    for ep in episodes:
        if ep.get("goal_met"):
            continue
        eff = achieved_extractor(ep)
        if not eff:
            continue
        tr = ep["trace"]
        for i, step in enumerate(tr):
            extra.append({**step, "goal": eff, "reward": 1.0,
                          "done": "yes" if i == len(tr) - 1 else "no"})
    return extra


def reinforce_update(eng, episodes, qname=None, lr=1e-3, entropy=0.01,
                     epochs=1, batch=128):
    """REINFORCE on outcome-weighted data: L = -E[z log pi] - b*H."""
    if not episodes:
        return {}
    qn = qname or eng._primary()
    texts = [e[0] for e in episodes]
    acts = torch.tensor([e[1] for e in episodes])
    zs = torch.tensor([e[2] for e in episodes], dtype=torch.float32)
    for p_ in eng.encoder.parameters():
        p_.requires_grad = False
    opt = torch.optim.AdamW(eng.policy.parameters(), lr=lr)
    eng.policy.train()
    for _ in range(epochs):
        idx = torch.randperm(len(texts))
        for i in range(0, len(texts), batch):
            b = idx[i:i + batch]
            h = eng.policy.trunk(eng.encoder([texts[j] for j in b]))
            logp = F.log_softmax(eng.policy.heads[qn](h), -1)
            chosen = logp[torch.arange(len(b)), acts[b]]
            loss = -(zs[b] * chosen).mean() \
                   - entropy * (logp.exp() * logp).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    eng.policy.eval()
    for p_ in eng.encoder.parameters():
        p_.requires_grad = True
    eng._cache.clear()
    return {"n": len(episodes), "mean_z": float(zs.mean())}
