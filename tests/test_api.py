"""FastAPI service tests (skipped when FastAPI or the model file is missing)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from braillelm.braille_g1 import to_braille  # noqa: E402
from braillelm.paths import MODEL_ONNX  # noqa: E402

pytestmark = pytest.mark.skipif(not MODEL_ONNX.exists(), reason="model file not present")


@pytest.fixture(scope="module")
def client():
    from braillelm.serving.api import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_translate(client):
    r = client.post("/v1/translate", json={"braille": to_braille("Don't panic, it's 9 am.")})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "Don't panic, it's 9 am."
    assert len(body["chars"]) == len(body["text"])


def test_translate_rejects_bad_input(client):
    assert client.post("/v1/translate", json={"braille": "hello"}).status_code == 422
    assert client.post("/v1/translate", json={"braille": "⠓", "mode": "guess"}).status_code == 422


def test_demo_endpoints(client):
    assert "BrailleLM" in client.get("/").text
    assert "metrics" in client.get("/api/meta").json()
    r = client.post("/api/translate", json={"print": "Hello world.", "noise": 0})
    assert r.status_code == 200 and r.json()["faithful"]["text"] == "Hello world."
    assert client.post("/api/translate", json={"print": "100% @home"}).status_code == 400
