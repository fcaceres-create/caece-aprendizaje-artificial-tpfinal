import numpy as np
import pandas as pd
import pytest

from bank_captacion.features import (BASE_NUMERIC, CATEGORICAL, FeatureEngineer, make_preprocessor)


def test_derived_features(raw_rows):
    Z = FeatureEngineer().fit(raw_rows).transform(raw_rows)
    assert list(Z.columns) == BASE_NUMERIC + CATEGORICAL
    # pdays = -1 → NaN + bandera 0; pdays = 180 → valor + bandera 1
    assert np.isnan(Z.loc[0, "pdays_clean"]) and Z.loc[0, "previously_contacted"] == 0
    assert Z.loc[1, "pdays_clean"] == 180 and Z.loc[1, "previously_contacted"] == 1
    # saldo: log con signo y banderas
    assert Z.loc[0, "balance_log"] < 0 and Z.loc[0, "balance_negative"] == 1
    assert Z.loc[1, "balance_log"] == 0 and Z.loc[1, "balance_zero"] == 1
    # otras derivadas
    assert Z.loc[1, "n_credit_products"] == 3
    assert Z.loc[1, "prev_campaign_success"] == 1 and Z.loc[0, "contact_known"] == 0
    assert Z.loc[0, "age_group"] == "25-34" and Z.loc[1, "age_group"] == "65+"
    assert Z.loc[0, "season"] == "spring" and Z.loc[1, "season"] == "winter"
    assert np.isclose(Z.loc[0, "day_sin"] ** 2 + Z.loc[0, "day_cos"] ** 2, 1.0)


def test_winsor_cap_learned_only_in_fit(train_test, raw_rows):
    train, _ = train_test
    fe = FeatureEngineer().fit(train)
    assert fe.campaign_cap_ == pytest.approx(train["campaign"].quantile(0.99))
    Z = fe.transform(raw_rows)  # campaign = 80 en la fila 0 → recortado al cap de train
    assert Z.loc[0, "campaign_w"] == fe.campaign_cap_


def test_unknown_kept_as_category(raw_rows):
    Z = FeatureEngineer().fit(raw_rows).transform(raw_rows)
    assert Z.loc[0, "education"] == "unknown" and Z.loc[0, "contact"] == "unknown"


@pytest.mark.parametrize("kind", ["linear", "tree_onehot", "naive_bayes"])
def test_preprocessors_unique_names_and_no_nan(train_test, kind):
    train, _ = train_test
    Z = FeatureEngineer().fit(train).transform(train)
    prep = make_preprocessor(kind).fit(Z)
    out = prep.transform(Z.head(500))
    names = list(prep.get_feature_names_out())
    assert len(names) == len(set(names)), "nombres de variables duplicados"
    assert not np.isnan(np.asarray(out, dtype=float)).any()


def test_unseen_category_does_not_break(train_test, raw_rows):
    train, _ = train_test
    fe = FeatureEngineer().fit(train)
    Z = fe.transform(train)
    prep = make_preprocessor("tree_native").fit(Z)
    bad = raw_rows.copy()
    bad.loc[0, "job"] = "astronaut"
    out = prep.transform(fe.transform(bad))
    assert pd.isna(out.loc[0, "job"])
