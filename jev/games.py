"""Games & self-play (v5) with research upgrades: canonical states, MCTS with
engine priors + root Dirichlet noise (v6.13, Silver'17), SelfPlay + Elo,
PSRO (v8.3, Lanctot'17), GoExplore (v8.10, Ecoffet'21), PLR (v7.14)."""
import copy, math, random
from typing import Dict, List, Tuple
import numpy as np

from .training import reinforce_update


class TicTacToe:
    n_actions = 9

    def reset(self):
        self.board = [" "] * 9
        self._cur = 0
        return {"board": list(self.board), "player": 0, "done": False,
                "winner": None}

    def _win(self, b, p):
        s = "XO"[p]
        return any(b[i] == b[j] == b[k] == s for i, j, k in
                   ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),
                    (0,4,8),(2,4,6)))

    def legal_actions(self, obs):
        return [i for i, c in enumerate(obs["board"]) if c == " "]

    def canonical_text(self, obs):
        p, o = "XO"[obs["player"]], "XO"[1 - obs["player"]]
        cells = "".join(("M" if c == p else "O" if c == o else ".")
                        for c in obs["board"])
        return f"{cells[:3]}/{cells[3:6]}/{cells[6:]}|turn=M"

    def step(self, a):
        mover = self._cur
        self.board[a] = "XO"[mover]
        done = self._win(self.board, mover) or " " not in self.board
        winner = mover if self._win(self.board, mover) else None
        if not done:
            self._cur = 1 - self._cur
        return ({"board": list(self.board), "player": self._cur,
                 "done": done, "winner": winner}, 0.0, done,
                {"mover": mover, "action": a})

    def outcome_for(self, obs, player):
        if obs["winner"] is None:
            return 0.0
        return 1.0 if obs["winner"] == player else -1.0


class GridWorld:
    """Doom-lite demo env."""
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.actions = ["stay", "up", "down", "left", "right"]
        self.reset()

    def reset(self):
        self.px, self.py, self.hp = 0, 0, 100
        self.coins = [(6, 6), (2, 7), (7, 1)]
        self.enemies = [(5, 2), (3, 5)]
        self.t, self._r = 0, 0.0
        return self.sense()

    def sense(self):
        return {"px": self.px, "py": self.py, "hp": self.hp,
                "coins_left": len(self.coins),
                "enemies": list(self.enemies),
                "nearest_coin": min(self.coins,
                                    key=lambda c: abs(c[0]-self.px)
                                    + abs(c[1]-self.py))}

    def _mv(self, pos, a):
        x, y = pos
        return (max(0, min(7, x + (a == "right") - (a == "left"))),
                max(0, min(7, y + (a == "down") - (a == "up"))))

    def apply(self, action, engage=0.0):
        r = 0.0
        self.px, self.py = self._mv((self.px, self.py), action)
        if engage > 0.5 and any(abs(self.px-x)+abs(self.py-y) <= 1
                                for x, y in self.enemies):
            self.enemies = [e for e in self.enemies
                            if abs(self.px-e[0])+abs(self.py-e[1]) > 1]
            r += 5.0
        self.enemies = [self._mv(e, self.rng.choice(self.actions))
                        for e in self.enemies]
        if (self.px, self.py) in self.coins:
            self.coins.remove((self.px, self.py)); r += 10.0
        for x, y in self.enemies:
            if (x, y) == (self.px, self.py):
                self.hp -= 20; r -= 8.0
                self.px, self.py = max(0, self.px-1), max(0, self.py-1)
        self.t += 1
        self._r = r

    def reward_delta(self):
        return self._r

    def terminated(self):
        return self.hp <= 0 or not self.coins or self.t >= 300


class MCTS:
    """Engine-prior PUCT with optional root Dirichlet noise (v6.13)."""
    def __init__(self, env, eng, qname, c_puct=1.4, seed=0, root_noise=None):
        self.env, self.eng, self.qn, self.c = env, eng, qname, c_puct
        self.rng = random.Random(seed)
        self.root_noise = root_noise            # {"eps": .25, "alpha": .3}

    def _prior(self, obs):
        r = self.eng.decide(self.env.canonical_text(obs), mode="neural")
        p = np.array(list(r["outputs"][self.qn]["probabilities"].values()))
        legal = np.zeros(self.env.n_actions)
        idx = self.env.legal_actions(obs)
        legal[idx] = 1.0
        if self.root_noise:
            noise = np.array([self.rng.gammavariate(
                self.root_noise["alpha"], 1.0) for _ in idx])
            noise /= max(noise.sum(), 1e-9)
            p[idx] = ((1 - self.root_noise["eps"]) * p[idx]
                      + self.root_noise["eps"] * noise)
        p = p * legal
        return p / max(p.sum(), 1e-9) if p.sum() > 0 else legal / legal.sum()

    def run(self, obs, sims=64) -> Tuple[int, np.ndarray]:
        N: Dict = {}; W: Dict = {}
        for _ in range(sims):
            board, cur = list(obs["board"]), obs["player"]
            path = []
            done, winner, v = False, None, 0.0
            sign = 1.0
            for _depth in range(64):
                txt = self.env.canonical_text({"board": board, "player": cur})
                legal = [i for i, c in enumerate(board) if c == " "]
                if not legal:
                    v, done = 0.0, True
                    break
                prior = self._prior({"board": board, "player": cur})
                sq = math.sqrt(sum(N.get((txt, a), 0) for a in legal)) + 1e-9
                best, best_a = -1e18, legal[0]
                for a in legal:
                    q = W.get((txt, a), 0.0) / max(N.get((txt, a), 1), 1)
                    u = self.c * prior[a] * sq / (1 + N.get((txt, a), 0))
                    if q + u > best:
                        best, best_a = q + u, a
                path.append((txt, best_a))
                self.env.board = list(board); self.env._cur = cur
                o2, _, d2, info = self.env.step(best_a)
                board, cur = o2["board"], o2["player"]
                done, winner = d2, o2["winner"]
                if done:
                    wp = info["mover"]
                    v = ((1.0 if winner == wp else -1.0)
                         if winner is not None else 0.0)
                    break
            for txt, a in reversed(path):
                N[(txt, a)] = N.get((txt, a), 0) + 1
                W[(txt, a)] = W.get((txt, a), 0.0) + sign * v
                sign = -sign
        root_txt = self.env.canonical_text(obs)
        visits = np.array([N.get((root_txt, a), 0)
                           for a in range(self.env.n_actions)], float)
        return int(visits.argmax()), visits / max(visits.sum(), 1e-9)


class SelfPlayAPI:
    def __init__(self, engine, env, eps=0.15, mcts_sims=0, seed=0):
        self.eng, self.env, self.eps = engine, env, eps
        self.mcts_sims, self.rng = mcts_sims, random.Random(seed)
        self.qn = engine._primary()

    def _probs(self, obs):
        r = self.eng.decide(self.env.canonical_text(obs), mode="neural")
        p = np.array(list(r["outputs"][self.qn]["probabilities"].values()))
        m = np.zeros(self.env.n_actions)
        m[self.env.legal_actions(obs)] = 1.0
        p = np.maximum(p * m, 1e-9)
        return p / p.sum()

    def selfplay_game(self):
        obs = self.env.reset()
        data = []
        while not obs["done"]:
            mover = obs["player"]
            legal = self.env.legal_actions(obs)
            if self.mcts_sims > 0:
                mcts = MCTS(self.env, self.eng, self.qn,
                            seed=self.rng.randint(0, 9))
                a, _ = mcts.run(obs, self.mcts_sims)
            else:
                a = (self.rng.choice(legal) if self.rng.random() < self.eps
                     else int(np.argmax(self._probs(obs))))
            data.append((self.env.canonical_text(obs), a, 0.0, mover))
            obs, _, _, _ = self.env.step(a)
        z = {p: self.env.outcome_for(obs, p) for p in (0, 1)}
        return [(t, a, z[m]) for t, a, _, m in data]

    def train(self, games=60, lr=1e-3, verbose=True):
        buf = []
        for g in range(games):
            buf += self.selfplay_game()
            if (g + 1) % 10 == 0:
                s = reinforce_update(self.eng, buf[-512:], lr=lr)
                if verbose:
                    print(f"  game {g+1}: buffer={len(buf)} "
                          f"mean_z={s['mean_z']:+.2f}")
        return reinforce_update(self.eng, buf[-512:], lr=lr)

    def snapshot(self):
        clone = copy.copy(self)
        clone.eng = copy.deepcopy(self.eng)
        return clone

    def arena(self, challenger, games=60, K=32.0):
        elo_a = getattr(self, "_elo", 1500)
        elo_b = getattr(challenger, "_elo", 1500)
        for g in range(games):
            first_is_self = g % 2 == 0
            obs = self.env.reset()
            while not obs["done"]:
                cur = self if (obs["player"] == 0) == first_is_self \
                    else challenger
                p = cur._probs(obs)
                a = int(np.argmax(p)) if self.rng.random() > 0.05 \
                    else self.rng.choice(self.env.legal_actions(obs))
                obs, _, _, _ = self.env.step(a)
            sa = 0.5 if obs["winner"] is None else \
                (1.0 if ((obs["winner"] == 0) == first_is_self) else 0.0)
            ea = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
            elo_a += K * (sa - ea)
            elo_b += K * ((1 - sa) - (1 - ea))
        self._elo, challenger._elo = round(elo_a), round(elo_b)
        return self._elo, challenger._elo


class PSRO:
    """Population + payoff matrix + fictitious-play meta-solver (v8.3)."""
    def __init__(self, env, act_fn, seed=0):
        self.env, self.act_fn = env, act_fn
        self.pop: List = []
        self.M = np.zeros((0, 0))
        self.rng = np.random.default_rng(seed)

    def add(self, policy, games=30):
        n = len(self.pop)
        M2 = np.zeros((n + 1, n + 1)); M2[:n, :n] = self.M
        for j in range(n):
            w = self._winrate(policy, self.pop[j], games)
            M2[n, j], M2[j, n] = w, 1.0 - w
        self.pop.append(policy); self.M = M2

    def _winrate(self, a, b, games):
        wins = 0
        for g in range(games):
            obs = self.env.reset()
            first_is_a = g % 2 == 0
            while not obs["done"]:
                p = a if (obs["player"] == 0) == first_is_a else b
                obs, _, _, _ = self.env.step(self.act_fn(p, obs))
            wins += (obs["winner"] == 0) == first_is_a
        return wins / max(games, 1)

    def meta_mixture(self, iters=300):
        n = self.M.shape[0]
        counts = np.ones(n); avg = counts / n
        for _ in range(iters):
            opp = int(self.rng.choice(n, p=avg)) if n > 1 else 0
            br = np.zeros(n); br[int(np.argmax(self.M[:, opp]))] = 1.0
            counts += br; avg = counts / counts.sum()
        return avg

    def exploitability(self, sigma):
        return float(np.max(self.M @ sigma) - sigma @ self.M @ sigma)


class PLRCurriculum:
    """Sample training clusters by learning-potential x staleness (Jiang'21)."""
    def __init__(self, K, seed=0):
        self.t, self.last = 0, np.zeros(K)
        self.rng = np.random.default_rng(seed)

    def sample(self, eng, n):
        Ks = eng.state_matrix.n_spatial
        Q = eng.Q[:Ks]; V = eng.V[:Ks]
        potential = np.clip(Q.max(1) - V, 0, None) + 1e-3
        stale = np.log1p(self.t - self.last[:Ks])
        p = potential * stale
        p /= p.sum()
        idx = self.rng.choice(Ks, n, p=p)
        self.last[idx] = self.t
        self.t += n
        return idx


class GoExploreArchive:
    """Best (score, prefix) per quantized cell (Ecoffet Nature'21)."""
    def __init__(self, rng=None):
        self.cells: Dict = {}
        self.t, self.rng = 0, rng or random.Random(0)

    def update(self, key, score, prefix):
        c = self.cells.get(key)
        if c is None or score > c["score"]:
            self.cells[key] = {"score": score, "prefix": list(prefix),
                               "t": self.t}

    def select(self):
        if not self.cells:
            return [], 0.0
        keys = list(self.cells)
        best = max(self.cells[k]["score"] for k in keys)
        w = [math.exp(0.5 * (self.cells[k]["score"] - best))
             / (1 + self.t - self.cells[k]["t"]) for k in keys]
        k = keys[self.rng.choices(range(len(keys)), weights=w)[0]]
        self.t += 1
        return self.cells[k]["prefix"], self.cells[k]["score"]
