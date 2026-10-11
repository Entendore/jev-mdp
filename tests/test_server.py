"""HTTP server contract test — skipped if fastapi/httpx not installed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from jev.engine import demo_engine
from jev.server import create_app


@pytest.fixture(scope="module")
def client():
    eng, _ = demo_engine(seed=0)
    from fastapi.testclient import TestClient
    return TestClient(create_app(eng, api_key="secret"))


def test_health_auth(client):
    assert client.get("/health").status_code == 401
    r = client.get("/health", headers={"X-API-Key": "secret"})
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_decide_endpoint(client):
    r = client.post("/v1/decide", headers={"X-API-Key": "secret"},
                    json={"state": "i was charged twice. urgent"})
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in {"billing", "technical", "sales", "ESCALATE"}
    assert "department" in body["outputs"]


def test_metrics(client):
    r = client.get("/metrics", headers={"X-API-Key": "secret"})
    assert r.status_code == 200 and "telemetry" in r.json()
