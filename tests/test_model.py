import json

import numpy as np
import pandas as pd

from bank_captacion import config as cfg
from bank_captacion.inference import assign_decile, load_model, score_frame, top_contributions


def test_model_loads_and_predicts_probabilities(train_test):
    _, test = train_test
    model, meta = load_model()
    p = model.predict_proba(test.drop(columns=[cfg.TARGET]))[:, 1]
    assert p.shape == (len(test),)
    assert ((p >= 0) & (p <= 1)).all()
    assert meta["model_name"] == "lightgbm"


def test_threshold_consistent_with_training_outputs():
    meta = json.loads(cfg.METADATA_FILE.read_text(encoding="utf-8"))
    thr = json.loads((cfg.TABLES_DIR / "threshold_choice.json").read_text(encoding="utf-8"))
    assert meta["threshold"] == thr["profit_threshold"]
    assert 0 < meta["threshold"] < 1


def test_holdout_metrics_are_reasonable():
    m = pd.read_csv(cfg.TABLES_DIR / "test_metrics.csv").set_index("model")
    assert 0.72 <= m.loc["final", "roc_auc"] <= 0.85          # rango de referencia sin duration
    assert m.loc["final", "pr_auc"] > 3 * m.loc["dummy", "pr_auc"]
    assert m.loc["final", "pr_auc"] > m.loc["logreg", "pr_auc"]
    assert m.loc["with_duration", "roc_auc"] > m.loc["final", "roc_auc"]


def test_deciles_monotonic_and_bounded():
    model, meta = load_model()
    cuts = meta["decile_cuts_oof"]
    assert assign_decile([1.0], cuts)[0] == 1 and assign_decile([0.0], cuts)[0] == 10
    d = assign_decile(np.linspace(0, 1, 101), cuts)
    assert (np.diff(d) <= 0).all()


def test_contributions_and_batch_scoring(train_test):
    _, test = train_test
    model, meta = load_model()
    X = test.drop(columns=[cfg.TARGET]).head(20)
    contrib = top_contributions(model, X, k=3)
    assert len(contrib) == 20 and all(len(c) == 3 for c in contrib)
    # TreeSHAP: suma de contribuciones + base = log-odds predicho.
    Z = model[:-1].transform(X)
    raw = model[-1].predict(Z, pred_contrib=True)
    logit = np.log(model.predict_proba(X)[:, 1] / (1 - model.predict_proba(X)[:, 1]))
    assert np.allclose(raw.sum(1), logit, atol=1e-6)
    scored = score_frame(model, meta, X)
    assert scored["probability"].is_monotonic_decreasing
    assert set(scored["recommendation"]) <= {"llamar", "no llamar"}
