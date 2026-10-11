"""HTTP serving — the Jev contract over the wire (v5 protocol, real API).
Endpoints:
  GET  /health              liveness + decision counter
  POST /v1/decide           full engine ladder: {state, mode?, query_channel?}
  POST /v1/schema/decide    ZERO-SHOT schemas via the in-context scorer
                            (v8.5) — arbitrary option lists, Jev's >255
                            two-stage score-then-choose equivalent
  GET  /metrics             telemetry + ACI realized coverage + drift PSI
Optional X-API-Key auth (cli --api-key or JEVMDP_API_KEY)."""
from typing import Dict, List, Optional


def create_app(engine, scorer=None, api_key: Optional[str] = None):
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from pydantic import BaseModel
    except ImportError as e:
        raise SystemExit("server extra required: pip install 'jev-mdp[server]'") from e

    app = FastAPI(title="Jev-MDP", version="10.1.0",
                  description="Unstructured state in, typed probabilistic "
                              "decisions out — an MDP you own.")

    def auth(x_api_key: Optional[str] = Header(default=None)):
        if api_key and x_api_key != api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    class DecideIn(BaseModel):
        state: str
        mode: str = "neural"
        query_channel: Optional[str] = None
        query_cost: float = 0.02

    class SchemaIn(BaseModel):
        state: str
        schema: str
        instructions: Dict[str, str]
        options: List[str]
        k: int = 3

    @app.get("/health", dependencies=[Depends(auth)])
    def health():
        return {"status": "ok", "decisions": engine.tel["n"]}

    @app.post("/v1/decide", dependencies=[Depends(auth)])
    def decide(body: DecideIn):
        return engine.decide(body.state, mode=body.mode,
                             query_channel=body.query_channel,
                             query_cost=body.query_cost)

    @app.post("/v1/schema/decide", dependencies=[Depends(auth)])
    def schema_decide(body: SchemaIn):
        if scorer is None:
            raise HTTPException(
                status_code=501,
                detail="no in-context scorer loaded — start server with "
                       "--scorer artifacts/scorer.pt")
        exemplars = []
        if engine.memory is not None and len(engine.memory.texts):
            exemplars = engine.memory.dynamic_exemplars(body.state, k=body.k)
        return scorer.decide(body.schema, body.instructions, exemplars,
                             body.state, body.options)

    @app.get("/metrics", dependencies=[Depends(auth)])
    def metrics():
        out = {"telemetry": engine.tel_summary()}
        cov = engine.aci.realized_coverage
        if cov is not None:
            out["aci_realized_coverage"] = round(cov, 4)
            out["aci_alpha_t"] = round(engine.aci.alpha_t, 4)
        drift = getattr(engine, "drift", None)
        if drift is not None:
            out["drift_psi"] = drift.psi()
            out["drifted"] = drift.drifted
        return out

    return app


def run(engine, host="127.0.0.1", port=8000, scorer=None, api_key=None):
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit("pip install 'jev-mdp[server]'") from e
    app = create_app(engine, scorer=scorer, api_key=api_key)
    uvicorn.run(app, host=host, port=port, log_level="info")
