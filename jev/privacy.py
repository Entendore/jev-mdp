"""Privacy (v9.11): DP-SGD (Abadi CCS'16 — pedagogical; production should use
Opacus + RDP accounting), FedAvg (McMahan AISTATS'17), unlearning (practical;
NOT Guo ICML'20-certified — pair with an influence audit, Koh&Liang'17)."""
import copy
from collections import deque
import torch
import torch.nn.functional as F


def dpsgd_fit(eng, texts, labels, epochs=5, C=1.0, sigma=0.8, batch=8,
              lr=1e-3):
    opt = torch.optim.AdamW(eng._params(), lr=lr)
    params = eng._params()
    for _ in range(epochs):
        order = torch.randperm(len(texts)).tolist()
        for i in range(0, len(texts), batch):
            b = order[i:i + batch]
            for p in params:
                p.grad = torch.zeros_like(p)
            for j in b:                                   # per-sample pass
                h = eng.policy.trunk(eng.encoder([texts[j]]))
                loss = sum(F.cross_entropy(
                    eng.policy.heads[q.name](h),
                    torch.tensor([eng._label_index(q, labels[j][q.name])]))
                    for q in eng.questions)
                gs = torch.autograd.grad(loss, params, allow_unused=True)
                for p, g in zip(params, gs):
                    if g is not None:
                        p.grad += g / max(g.norm() / C, 1.0)
            for p in params:
                p.grad += torch.randn_like(p) * sigma * C
                p.grad /= len(b)
            opt.step(); opt.zero_grad()
    return eng


def fedavg(engines, weights=None):
    ws = weights or [1] * len(engines)
    avg = copy.deepcopy(engines[0])
    sds = [e.policy.state_dict() for e in engines]
    sd = avg.policy.state_dict()
    for k in sds[0]:
        acc = sum(w * sds[i][k].float() for i, w in enumerate(ws)) / sum(ws)
        sd[k].copy_(acc.to(sd[k].dtype))
    return avg


def unlearn(eng, remove_texts, anchor_texts, anchor_labels, steps=2, lr=1e-4):
    keep = set(remove_texts)
    eng._corr_buf = deque([x for x in eng._corr_buf if x[0] not in keep],
                          maxlen=eng._corr_buf.maxlen)
    if hasattr(eng, "_der_buf"):
        eng._der_buf = deque([x for x in eng._der_buf if x[0] not in keep],
                             maxlen=eng._der_buf.maxlen)
    if eng.memory is not None and len(eng.memory.texts):
        keep_idx = [i for i, t in enumerate(eng.memory.texts) if t not in keep]
        eng.memory.E = eng.memory.E[keep_idx]
        eng.memory.y = [eng.memory.y[i] for i in keep_idx]
        eng.memory.texts = [eng.memory.texts[i] for i in keep_idx]
    eng._cache.clear()
    eng.fit_supervised(anchor_texts, anchor_labels, epochs=steps, lr=lr)
    return eng
