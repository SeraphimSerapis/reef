import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from reef.server import make_handler

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "reef-tiny"


class FakeReef:
    name, device = "fake", "cpu"

    def systemone(self, body):
        if "state" not in body or "questions" not in body:
            raise ValueError("request needs 'state' and 'questions'")
        return {"model": self.name, "answers": {k: {"noul": 0.5} for k in body["questions"]}}


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeReef()))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_healthz(server):
    with urllib.request.urlopen(server + "/healthz") as r:
        assert json.loads(r.read()) == {"ok": True, "model": "fake", "device": "cpu"}


def test_models(server):
    with urllib.request.urlopen(server + "/v1/models/?x=1") as r:
        assert json.loads(r.read())["data"][0]["id"] == "fake"


def test_systemone(server):
    out = _post(server + "/v1/systemone", {"state": "x", "questions": {"q": {"type": "noul"}}})
    assert out["answers"] == {"q": {"noul": 0.5}}


def test_bad_request(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(server + "/v1/systemone", {"state": "x"})
    assert e.value.code == 400


@pytest.mark.skipif(not (ARTIFACT / "reef_int8.tflite").exists(), reason="no exported artifact")
def test_runtime_cpu_end_to_end(questions):
    from reef import Reef

    reef = Reef(ARTIFACT, device="cpu")
    out = reef.decide({"subject": "URGENT: refund", "body": "charged twice, please fix"}, questions)
    assert set(out["answers"]) == set(questions)
    for a in out["answers"].values():
        assert sum(a["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
        assert 0.0 <= a["confidence"] <= 1.0
    assert out["answers"]["team"]["choice"] in {"billing", "tech_support"}
    assert 0.0 <= out["answers"]["urgency"]["score"] <= 4.0
    assert out["usage"]["device"] == "cpu" and out["usage"]["windows"] == 1
