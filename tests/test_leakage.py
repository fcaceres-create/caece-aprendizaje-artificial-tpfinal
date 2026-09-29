"""Controles de fuga de información: duration fuera del modelo y hold-out independiente."""

import json

import joblib
import numpy as np

from bank_captacion import config as cfg
from bank_captacion.data import find_overlap
from bank_captacion.features import FeatureEngineer


def test_final_model_does_not_use_duration():
    model = joblib.load(cfg.MODEL_FILE)
    fe = model.named_steps["features"]
    assert fe.use_duration is False
    assert "duration" not in fe.feature_names_out_
    assert "duration" not in model.named_steps["clf"].feature_name_


def test_duration_is_dropped_even_if_present(raw_rows):
    Z = FeatureEngineer(use_duration=False).fit(raw_rows).transform(raw_rows)
    assert "duration" not in Z.columns


def test_predictions_invariant_to_duration(train_test):
    _, test = train_test
    model = joblib.load(cfg.MODEL_FILE)
    X = test.drop(columns=[cfg.TARGET]).head(300)
    p1 = model.predict_proba(X)[:, 1]
    p2 = model.predict_proba(X.assign(duration=X["duration"] * 10 + 999))[:, 1]
    assert np.allclose(p1, p2)


def test_metadata_declares_pre_contact():
    meta = json.loads(cfg.METADATA_FILE.read_text(encoding="utf-8"))
    assert "duration" not in meta["raw_input_columns"]
    assert "duration" not in meta["engineered_features"]


def test_no_overlap_between_clean_train_and_test(train_test):
    train, test = train_test
    assert int(find_overlap(train, test).sum()) == 0
    assert len(train) == 40_690 and len(test) == 4_521
