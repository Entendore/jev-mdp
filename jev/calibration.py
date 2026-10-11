"""Calibration & drift suite.
- SplitConformal (v2, Vovk'05), AdaptiveConformal (v6.1, Gibbs&Candes'21),
  APSConformal (v9.13, Romano NeurIPS'20), weighted quantile (Tibshirani'19).
- ConformalRiskController (v6.2, Angelopoulos'22): auto-act threshold with a
  distribution-free bound on auto-acted COST.
- OODDetector (v2, energy — Liu NeurIPS'20), DriftDetector (PSI),
  c2st (v6.11, Lopez-Paz ICLR'17), bbse/em label shift (Lipton'18, Saerens'02)."""
import math
from collections import deque
from typing import Dict
import numpy as np


class SplitConformal:
    def __init__(self, alpha=0.1):
        self.alpha, self.qhat = alpha, {}

    def fit(self, probs: Dict[str, np.ndarray], labels: Dict[str, np.ndarray]):
        for name, P in probs.items():
            y, n = labels[name], len(labels[name])
            scores = 1.0 - P[np.arange(n), y]
            k = math.ceil((n + 1) * (1 - self.alpha))
            self.qhat[name] = float(np.sort(scores)[min(k, n) - 1])

    def set(self, p, name):
        thr = 1.0 - self.qhat.get(name, 1.0)
        return [i for i, v in enumerate(p) if v >= thr]


class AdaptiveConformal:
    """alpha_t <- alpha_t + gamma*(alpha - err); refresh online quantile."""
    def __init__(self, alpha=0.1, gamma=0.02, window=1500):
        self.alpha0, self.gamma = alpha, gamma
        self.alpha_t = alpha
        self.W = deque(maxlen=window)
        self.qhat = 1.0
        self.errs = deque(maxlen=window)

    def warm(self, scores):
        k = math.ceil((len(scores) + 1) * (1 - self.alpha0))
        self.qhat = float(np.sort(scores)[min(k, len(scores)) - 1])

    def set(self, p):
        return [i for i, v in enumerate(p) if v >= 1 - self.qhat]

    def observe(self, p, true_idx):
        err = 0.0 if true_idx in self.set(p) else 1.0
        self.alpha_t = float(np.clip(
            self.alpha_t + self.gamma * (self.alpha0 - err), 0.01, 0.99))
        self.W.append(1.0 - p[true_idx])
        self.errs.append(err)
        if len(self.W) >= 50:
            self.qhat = float(np.quantile(np.array(self.W), 1 - self.alpha_t))
        return err

    @property
    def realized_coverage(self):
        return 1.0 - float(np.mean(self.errs)) if self.errs else None


class APSConformal:
    """Adaptive Prediction Sets: score = cum-prob mass until true label."""
    def __init__(self, alpha=0.1):
        self.alpha, self.qhat = alpha, {}

    def _aps_score(self, p, y):
        order = np.argsort(-p)
        s = 0.0
        for i in order:
            s += p[i]
            if i == y:
                return s - p[i] * np.random.random()
        return 1.0

    def fit(self, probs: Dict[str, np.ndarray], labels: Dict[str, np.ndarray]):
        for name, P in probs.items():
            sc = np.array([self._aps_score(P[i], labels[name][i])
                           for i in range(len(P))])
            k = math.ceil((len(P) + 1) * (1 - self.alpha))
            self.qhat[name] = float(np.sort(sc)[min(k, len(P)) - 1])

    def set(self, p, name):
        order = np.argsort(-p)
        out, s = [], 0.0
        for i in order:
            s += p[i]
            out.append(int(i))
            if s - p[i] * np.random.random() >= self.qhat.get(name, 1.0):
                break
        return out


class ConformalRiskController:
    """tau = inf{tau : (sum cost[conf>=tau] + B)/(n+1) <= alpha_R}."""
    def __init__(self, alpha_R=0.1, B=1.0):
        self.alpha_R, self.B, self.tau = alpha_R, B, 1.0

    def fit(self, probs, labels, cost):
        conf, pred = probs.max(1), probs.argmax(1)
        n = len(labels)

        def valid(tau):
            return (cost[conf >= tau].sum() + self.B) / (n + 1) <= self.alpha_R
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if valid(mid):
                hi = mid
            else:
                lo = mid
        self.tau = hi
        return {"tau": round(self.tau, 4),
                "act_rate": round(float((conf >= self.tau).mean()), 3)}


class OODDetector:
    """Energy score logsumexp over heads (Liu'20), ID-quantile threshold."""
    def __init__(self, fpr=0.05):
        self.fpr, self.threshold = fpr, -float("inf")

    def fit(self, id_scores):
        self.threshold = float(np.quantile(id_scores, self.fpr))

    def is_id(self, scores):
        return scores >= self.threshold


class DriftDetector:
    """PSI over the confidence histogram."""
    def __init__(self, n_bins=10, psi_alert=0.25, window=2000):
        self.bins = np.linspace(0, 1, n_bins + 1)
        self.psi_alert = psi_alert
        self.window: deque = deque(maxlen=window)
        self.ref = None

    def fit(self, confidences):
        hist, _ = np.histogram(confidences, self.bins)
        self.ref = (hist + 1.0) / (hist.sum() + len(hist))

    def update(self, conf):
        self.window.append(conf)

    def psi(self):
        if self.ref is None or len(self.window) < 50:
            return None
        hist, _ = np.histogram(np.array(self.window), self.bins)
        cur = (hist + 1.0) / (hist.sum() + len(hist))
        return float(np.sum((cur - self.ref) * np.log(cur / self.ref)))

    @property
    def drifted(self):
        p = self.psi()
        return p is not None and p > self.psi_alert


def c2st(ref_emb, cur_emb, seed=0):
    """Classifier two-sample test: AUC ~0.5 no drift; >0.65 shift."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    X = np.vstack([ref_emb, cur_emb])
    y = np.r_[np.zeros(len(ref_emb)), np.ones(len(cur_emb))]
    a, b, c, d = train_test_split(X, y, stratify=y, random_state=seed)
    auc = roc_auc_score(d, LogisticRegression(max_iter=1000)
                        .fit(a, c).predict_proba(b)[:, 1])
    return {"auc": round(float(auc), 3), "drift": bool(auc > 0.65)}


def bbse_prior(confusion, cur_counts):
    """Lipton'18: solve C^T mu = current prediction counts."""
    C = confusion / confusion.sum(1, keepdims=True)
    mu = np.linalg.lstsq(C.T, cur_counts, rcond=None)[0]
    return np.clip(mu / mu.sum(), 1e-4, None)


def em_label_shift(P_rows, target_prior, iters=50):
    """Saerens'02 EM reweighting of posteriors to the new regime."""
    A = P_rows.shape[1]
    w = np.ones(A)
    for _ in range(iters):
        q = P_rows * w
        q /= q.sum(1, keepdims=True)
        w = target_prior / np.maximum(q.mean(0), 1e-9)
        w = w / w.sum() * A
    q = P_rows * w
    return q / q.sum(1, keepdims=True)


def weighted_conformal_qhat(cal_scores, cal_w, test_w_mean, alpha=0.1):
    """Tibshirani'19: weighted empirical quantile under covariate shift."""
    order = np.argsort(cal_scores)
    total = cal_w[order].sum() + test_w_mean
    cum = np.cumsum(cal_w[order]) / total
    k = int(np.searchsorted(cum, 1 - alpha))
    return float(cal_scores[order][min(k, len(order) - 1)])
