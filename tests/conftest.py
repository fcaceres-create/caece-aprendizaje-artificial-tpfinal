import pandas as pd
import pytest

from bank_captacion import config as cfg
from bank_captacion.data import load_clean


@pytest.fixture(scope="session")
def train_test():
    return load_clean()


@pytest.fixture
def sample_client() -> dict:
    return {"age": 35, "job": "management", "marital": "single", "education": "tertiary", "default": "no",
            "balance": 1500, "housing": "no", "loan": "no", "contact": "cellular", "day": 15, "month": "oct",
            "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"}


@pytest.fixture
def raw_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"age": 30, "job": "student", "marital": "single", "education": "unknown", "default": "no", "balance": -50,
         "housing": "no", "loan": "no", "contact": "unknown", "day": 31, "month": "may", "duration": 100,
         "campaign": 80, "pdays": -1, "previous": 0, "poutcome": "unknown"},
        {"age": 70, "job": "retired", "marital": "married", "education": "primary", "default": "yes", "balance": 0,
         "housing": "yes", "loan": "yes", "contact": "cellular", "day": 1, "month": "dec", "duration": 900,
         "campaign": 1, "pdays": 180, "previous": 3, "poutcome": "success"},
    ])


def pytest_configure(config):
    cfg.ensure_dirs()
