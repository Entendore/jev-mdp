"""`jev` command-line interface (entry point: pyproject [project.scripts]).
Subcommands: demo · train · decide · serve · verify · compile · train-scorer."""
import argparse
import json
import os
import sys


def _load_labels(path):
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            texts.append(d["text"])
            labels.append(d["labels"])
    return texts, labels


def _load_tasks(path):
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def _train_pipeline(args):
    from .engine import (JevMDP, default_ticket_questions, make_tickets,
                         simulate_episodes)
    if args.demo:
        texts, labels = make_tickets(args.n, seed=1)
    else:
        assert args.labels, "--labels required unless --demo"
        texts, labels = _load_labels(args.labels)
    eng = JevMDP(default_ticket_questions(), seed=args.seed)
    n_cal = max(20, int(0.2 * len(texts)))
    Xcal, Ycal = texts[-n_cal:], labels[-n_cal:]
    Xtr, Ytr = texts[:-n_cal], labels[:-n_cal]
    eng.fit_supervised(Xtr, Ytr, val=(Xcal, Ycal))
    eng.fit_rl(Xtr, Ytr)
    eng.calibrate(Xcal, Ycal)
    eng.build_matrices(Xtr, Ytr)
    eng.add_terminals({"resolved": 1.0, "escalated": -0.25, "failed": -1.0})
    eng.learn_transitions(simulate_episodes(eng, Xtr, Ytr, eps=0.30))
    eng.build_reward()
    eng.fit_policy(Xtr, Ytr)
    eng.calibrate(Xcal, Ycal)
    eng.fit_crc(Xcal, Ycal)
    return eng


def cmd_demo(args):
    from .engine import demo_engine
    eng, (Xte, Yte) = demo_engine(seed=args.seed)
    s = "i was charged twice. please refund. urgent"
    for mode in ["neural", "matrix", "plan", "policy", "thompson", "gittins"]:
        r = eng.decide(s, mode=mode)
        print(f"{mode:<9} -> {r['action']:<10} conf={r['confidence']:.2f}  "
              f"src={r['source']}")
    print("\nreport:", eng.report(Xte, Yte))
    return 0


def cmd_train(args):
    eng = _train_pipeline(args)
    eng.save(args.out)
    print(f"engine saved -> {args.out}")
    return 0


def cmd_decide(args):
    from .engine import JevMDP
    eng = JevMDP.load(args.engine)
    r = eng.decide(args.state, mode=args.mode,
                   query_channel=args.query_channel)
    print(json.dumps(r, indent=2, default=str))
    return 0


def cmd_serve(args):
    from .engine import JevMDP
    if args.engine:
        eng = JevMDP.load(args.engine)
    else:
        print("[serve] building demo engine (~10 s)... "
              "pass --engine PATH to load a trained one")
        from .engine import demo_engine
        eng, _ = demo_engine(seed=args.seed)
    scorer = None
    if args.scorer:
        from .multitask import MultiTaskScorer
        scorer = MultiTaskScorer.load(args.scorer)
    api_key = args.api_key or os.environ.get("JEVMDP_API_KEY")
    from .server import run
    run(eng, host=args.host, port=args.port, scorer=scorer, api_key=api_key)
    return 0


def cmd_verify(args):
    from .engine import JevMDP
    from .evaluation import PolicyVerifier
    eng = JevMDP.load(args.engine)
    spec = {kv.split("=")[0].strip(): float(kv.split("=")[1])
            for kv in args.spec.split(",")}
    if eng.T is None:
        print("engine has no transition matrix — run episodes first")
        return 1
    ver = PolicyVerifier(eng.state_matrix, eng.T)
    qn = eng._primary()
    rows = eng.pi_target.get(qn) or eng.pm.P[qn]
    results = ver.check(rows, spec)
    for name, res in results.items():
        print(f"{name:<28} value={res['value']:<8} "
              f"{'PASS' if res['pass'] else 'FAIL'} (<= {spec[name]})")
    return 0 if all(r["pass"] for r in results.values()) else 1


def cmd_compile(args):
    from .engine import JevMDP
    eng = JevMDP.load(args.engine)
    if args.demo:
        from .engine import make_tickets
        texts, _ = make_tickets(200, seed=5)
    else:
        texts, _ = _load_labels(args.labels)
    info = eng.compile_program(texts, fidelity_target=args.fidelity)
    print(json.dumps(info, indent=2, default=str))
    if info["ok"]:
        eng.save(args.engine)
        print(f"program certified and persisted into {args.engine}")
        return 0
    print("program NOT certified — neural net remains primary")
    return 1


def cmd_train_scorer(args):
    from .multitask import MultiTaskScorer
    tasks = _load_tasks(args.tasks)
    cut = max(1, int(0.8 * len(tasks)))
    m = MultiTaskScorer(dim=args.dim, seed=args.seed)
    m.fit(tasks[:cut], epochs=args.epochs)
    m.fit_temperature(tasks[cut:])
    m.save(args.out)
    print(f"scorer saved -> {args.out} "
          f"(val acc {m.evaluate(tasks[cut:]):.3f})")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="jev",
                                description="Jev-MDP decision engine CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the end-to-end demo pipeline")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_demo)

    t = sub.add_parser("train", help="train + save an engine")
    t.add_argument("--labels", help="JSONL: {'text':..., 'labels':{...}}")
    t.add_argument("--demo", action="store_true", help="use demo ticket data")
    t.add_argument("--n", type=int, default=300)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="artifacts/engine.pt")
    t.set_defaults(fn=cmd_train)

    c = sub.add_parser("decide", help="one decision from a saved engine")
    c.add_argument("--engine", required=True)
    c.add_argument("--state", required=True)
    c.add_argument("--mode", default="neural")
    c.add_argument("--query-channel", default=None)
    c.set_defaults(fn=cmd_decide)

    s = sub.add_parser("serve", help="HTTP server (Jev contract)")
    s.add_argument("--engine", default=None)
    s.add_argument("--demo-engine", action="store_true")
    s.add_argument("--scorer", default=None,
                   help="in-context scorer for /v1/schema/decide")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--api-key", default=None)
    s.set_defaults(fn=cmd_serve)

    v = sub.add_parser("verify", help="reachability spec check")
    v.add_argument("--engine", required=True)
    v.add_argument("--spec", default="failed=0.12",
                   help="e.g. 'failed=0.01,escalated=0.3'")
    v.set_defaults(fn=cmd_verify)

    cp = sub.add_parser("compile", help="extract + verify a decision tree")
    cp.add_argument("--engine", required=True)
    cp.add_argument("--labels", default=None)
    cp.add_argument("--demo", action="store_true")
    cp.add_argument("--fidelity", type=float, default=0.95)
    cp.set_defaults(fn=cmd_compile)

    ts = sub.add_parser("train-scorer",
                        help="fit the in-context zero-shot scorer")
    ts.add_argument("--tasks", required=True,
                    help="JSONL of task dicts {schema,instructions,options,"
                         "samples,exemplars}")
    ts.add_argument("--out", default="artifacts/scorer.pt")
    ts.add_argument("--dim", type=int, default=96)
    ts.add_argument("--epochs", type=int, default=12)
    ts.add_argument("--seed", type=int, default=0)
    ts.set_defaults(fn=cmd_train_scorer)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
