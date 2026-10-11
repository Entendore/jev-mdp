# Honest Limitations (ships with the repo)

1. **"Verified"/"certified" means verified against the learned T matrix.**
   Reachability, compiled trees, and robust VI certify consistency with the
   world model; T's fidelity to reality is a data question audited via the
   OPE self-check and episode counts — not proven.
2. **The in-context scorer is the boldest extrapolation.** Whether it
   inherits LLM-style scaling is unproven here; per-task engines are the
   fallback.
3. **Gittins/BwK/Whittle use grid/bisected/heuristic implementations** of
   provably-optimal theory; verify indexability per deployment.
4. **Reward learning is hackable**; the KL brake bounds drift, not deception.
   The overoptimization monitor (jev/monitor.py) detects, does not prevent.
5. **Confounded OPE is audited, not bounded** (Kallus-Zhou LP unimplemented).
6. **Privacy is approximate**: DP-SGD is pedagogical (use Opacus+RDP in
   prod); unlearning is not Guo-'20-certified.
7. **All "faster/cheaper" claims are absent by design** — this project
   reports its own IQM-measured numbers or nothing.
8. **Shapley fields** use the permutation estimator over serialized dict
   fields — exact for the sampled permutations, not exhaustive.
9. **kNN memory and the compiled tree are persisted in engine.save(); the
   replay/DER buffers are not** — reload re-learns from feedback.
10. **Gates must be re-run with >=5 seeds as IQM CIs** before any number is
    quoted externally (v10.11 applies retroactively).
