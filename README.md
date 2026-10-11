# Jev-MDP

Jev's product contract — *unstructured state in, typed probabilistic decisions
out, schema-safe, calibrated, orders of magnitude cheaper* — rebuilt as a Markov
decision process you fully own. Groundings: `docs/RESEARCH_MAP.md`;
approximations: `docs/HONESTY.md`; gates are IQM CIs (rliable).

## Quickstart
    make install                       # venv + pip install -e ".[server,dev]"
    .venv/bin/jev demo                 # full pipeline in ~10 s
    .venv/bin/python -m pytest tests -q

## Lifecycle
    fit_supervised -> fit_rl -> calibrate -> build_matrices -> add_terminals ->
    learn_transitions -> build_reward -> fit_policy -> calibrate -> fit_crc ->
    compile_program? -> save -> serve -> observe_outcome

## Decision ladder (engine.decide, mode="neural")
    OOD -> deadman | conf >= CRC tau -> auto-act (cost-bounded)
    | query channel + EVSI > 0 -> buy information
    | mid confidence -> matrix cross-check | else escalate
    Modes: neural | matrix | plan | policy | thompson | gittins

## Serving
    .venv/bin/jev serve --demo-engine --port 8000
    curl -s localhost:8000/v1/decide -H 'Content-Type: application/json' \
         -d '{"state":"i was charged twice. urgent"}'
    # zero-shot schema (in-context scorer — Jev's >255 design):
    curl -s localhost:8000/v1/schema/decide -H 'Content-Type: application/json' \
         -d '{"state":"ios app crashes","schema":"issue channel",
              "instructions":{"app":"mobile","web":"browser","api":"devs"},
              "options":["app","web","api"]}'

## CLI
    jev demo | train | decide | serve | verify | compile | train-scorer
