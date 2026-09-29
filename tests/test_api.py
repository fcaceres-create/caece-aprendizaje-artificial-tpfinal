import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from bank_captacion import config as cfg


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["model_name"] == "lightgbm" and "threshold" in info


def test_predict_ok(client, sample_client):
    r = client.post("/predict", json=sample_client)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert 1 <= body["decile"] <= 10
    assert len(body["top_factors"]) == 3
    assert body["recommendation"] in ("llamar", "no llamar")
    assert body["contact_recommended"] == (body["probability"] >= body["threshold"])


def test_predict_rejects_duration(client, sample_client):
    r = client.post("/predict", json={**sample_client, "duration": 300})
    assert r.status_code == 422
    assert "duration" in r.text


@pytest.mark.parametrize("field,value", [("job", "astronaut"), ("age", 5), ("month", "13"), ("day", 40)])
def test_predict_validation(client, sample_client, field, value):
    assert client.post("/predict", json={**sample_client, field: value}).status_code == 422


def test_predict_missing_field(client, sample_client):
    body = dict(sample_client)
    body.pop("poutcome")
    assert client.post("/predict", json=body).status_code == 422


def test_predict_inconsistent_pdays(client, sample_client):
    assert client.post("/predict", json={**sample_client, "pdays": -1, "previous": 2}).status_code == 422


def test_batch_ranks_csv(client):
    df = pd.read_csv(cfg.TEST_FILE, sep=";").head(200)
    csv = df.to_csv(index=False, sep=";").encode()
    r = client.post("/predict/batch", files={"file": ("clientes.csv", csv, "text/csv")})
    assert r.status_code == 200
    assert set(r.headers["X-Ignored-Columns"].split(",")) == {"duration", "y"}
    out = pd.read_csv(io.StringIO(r.text))
    assert len(out) == 200 and out["rank"].tolist() == list(range(1, 201))
    assert out["probability"].is_monotonic_decreasing
    assert "duration" not in out.columns


def test_batch_rejects_bad_csv(client):
    bad = pd.DataFrame({"age": [30], "job": ["x"]}).to_csv(index=False).encode()
    r = client.post("/predict/batch", files={"file": ("bad.csv", bad, "text/csv")})
    assert r.status_code == 422
