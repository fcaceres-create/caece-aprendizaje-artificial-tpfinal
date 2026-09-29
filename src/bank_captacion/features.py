"""Ingeniería de variables y preprocesamiento (Fase 3).

Todo vive dentro de un ``Pipeline`` para que se ajuste solo con los datos de entrenamiento
de cada fold (sin fuga de información) y para que el modelo final reciba datos crudos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, KBinsDiscretizer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

from . import config as cfg

AGE_BINS = [0, 25, 35, 45, 55, 65, np.inf]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
SEASON = {"dec": "winter", "jan": "winter", "feb": "winter",
          "mar": "spring", "apr": "spring", "may": "spring",
          "jun": "summer", "jul": "summer", "aug": "summer",
          "sep": "autumn", "oct": "autumn", "nov": "autumn"}

BASE_NUMERIC = [
    "age", "balance_log", "balance_negative", "balance_zero", "campaign_w",
    "pdays_clean", "previously_contacted", "previous_w", "day_sin", "day_cos",
    "contact_known", "prev_campaign_success", "n_credit_products",
]
CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome", "age_group", "season",
]
# Numéricas continuas (se discretizan para Naive Bayes); el resto son banderas 0/1 o conteos chicos.
CONTINUOUS = ["age", "balance_log", "campaign_w", "pdays_clean", "previous_w", "day_sin", "day_cos"]

FEATURE_SETS = {"pre_contact": False, "with_duration": True}

# Catálogo documental de cada variable (se exporta a reports/tables/feature_catalog.csv).
FEATURE_CATALOG = [
    ("age", "age", "Sin cambios", "Edad numérica.", False),
    ("age_group", "age", "Bins <25, 25-34, 35-44, 45-54, 55-64, 65+",
     "Efecto no lineal: conversión alta en jóvenes y mayores de 60.", False),
    ("balance_log", "balance", "sign(x)·log(1+|x|)",
     "Saldo muy asimétrico (asimetría 8,5) con negativos: reduce la influencia de extremos sin eliminarlos.", False),
    ("balance_negative", "balance", "1 si balance < 0", "8,4 % de clientes en descubierto.", False),
    ("balance_zero", "balance", "1 si balance = 0", "7,8 % de clientes sin saldo.", False),
    ("campaign_w", "campaign", "Winsorizado al p99 del fold de entrenamiento",
     "Cola larga (máx. 63); el p99 se aprende en fit para no usar información del fold de validación.", True),
    ("previous_w", "previous", "Winsorizado al p99 del fold de entrenamiento",
     "Cola larga; mismo criterio que campaign.", True),
    ("pdays_clean", "pdays", "NaN cuando pdays = -1",
     "El -1 no es un número de días; el faltante se imputa luego según la familia de modelo.", False),
    ("previously_contacted", "pdays", "1 si pdays != -1",
     "Conserva explícitamente la información 'nunca contactado' (23,1 % vs 9,2 % de conversión).", False),
    ("day_sin / day_cos", "day", "sin/cos(2π·day/31)",
     "El día del mes es cíclico: el 31 está cerca del 1.", False),
    ("contact_known", "contact", "1 si contact != unknown",
     "contact=unknown corresponde a un período sin registro del canal (4,0 % de conversión).", False),
    ("prev_campaign_success", "poutcome", "1 si poutcome = success",
     "Señal más fuerte disponible antes de llamar (64,8 % de conversión).", False),
    ("season", "month", "Estación del año (hemisferio norte)",
     "Agrupa meses de bajo volumen para estabilizar su estimación.", False),
    ("n_credit_products", "housing, loan, default", "Suma de yes en housing + loan + default",
     "Tener préstamos reduce la conversión; resume la carga crediticia.", False),
    ("job, marital, education, default, housing, loan, contact, month, poutcome", "originales",
     "Categóricas sin cambios ('unknown' se mantiene como categoría)",
     "unknown no es un faltante aleatorio (ver EDA).", False),
    ("duration", "duration", "Solo en el conjunto with_duration",
     "Leakage: se conoce después de la llamada. Excluida del modelo entregable.", False),
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Crea las variables derivadas a partir de los datos crudos.

    Parámetros
    ----------
    use_duration : bool
        Si es False (modelo pre-contacto), ``duration`` se descarta aunque venga en los datos.
    cap_quantile : float
        Cuantil para winsorizar ``campaign`` y ``previous`` (se aprende en ``fit``).
    """

    def __init__(self, use_duration: bool = False, cap_quantile: float = 0.99):
        self.use_duration = use_duration
        self.cap_quantile = cap_quantile

    def fit(self, X: pd.DataFrame, y=None):
        self.campaign_cap_ = float(X["campaign"].quantile(self.cap_quantile))
        self.previous_cap_ = float(X["previous"].quantile(self.cap_quantile))
        self.feature_names_out_ = self.numeric_features() + CATEGORICAL
        return self

    def numeric_features(self) -> list[str]:
        return BASE_NUMERIC + (["duration"] if self.use_duration else [])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        out = pd.DataFrame(index=X.index)
        out["age"] = X["age"].astype(float)
        bal = X["balance"].astype(float)
        out["balance_log"] = np.sign(bal) * np.log1p(np.abs(bal))
        out["balance_negative"] = (bal < 0).astype(int)
        out["balance_zero"] = (bal == 0).astype(int)
        out["campaign_w"] = X["campaign"].clip(upper=self.campaign_cap_).astype(float)
        out["pdays_clean"] = X["pdays"].where(X["pdays"] != -1, np.nan).astype(float)
        out["previously_contacted"] = (X["pdays"] != -1).astype(int)
        out["previous_w"] = X["previous"].clip(upper=self.previous_cap_).astype(float)
        angle = 2 * np.pi * X["day"].astype(float) / 31.0
        out["day_sin"] = np.sin(angle)
        out["day_cos"] = np.cos(angle)
        out["contact_known"] = (X["contact"] != "unknown").astype(int)
        out["prev_campaign_success"] = (X["poutcome"] == "success").astype(int)
        out["n_credit_products"] = sum((X[c] == "yes").astype(int) for c in ["housing", "loan", "default"])
        if self.use_duration:
            out["duration"] = X["duration"].astype(float)
        for col in cfg.CATEGORICAL_RAW:
            out[col] = X[col].astype(str)
        out["age_group"] = pd.cut(X["age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
        out["season"] = X["month"].map(SEASON).astype(str)
        return out[self.numeric_features() + CATEGORICAL]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Convierte las categóricas a dtype ``category`` con niveles aprendidos en fit.

    Se usa para las categorías nativas de LightGBM/XGBoost. Un nivel no visto en fit
    queda como NaN (el modelo lo trata como faltante).
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns or CATEGORICAL
        self.categories_ = {c: sorted(X[c].astype(str).unique()) for c in cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, cats in self.categories_.items():
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None


def _non_negative_int(a):
    return np.maximum(np.asarray(a, dtype=float), 0).astype(int)


def _numeric(use_duration: bool) -> list[str]:
    return BASE_NUMERIC + (["duration"] if use_duration else [])


def make_preprocessor(kind: str, use_duration: bool = False):
    """Devuelve el preprocesador según la familia de modelo.

    kind:
      - ``linear``: imputación por mediana + escalado + one-hot (regresión logística, KNN, MLP, SVM).
      - ``tree_onehot``: imputación con -1 (sin escalado) + one-hot (árboles, RF, boosting).
      - ``tree_native``: categóricas como dtype category (LightGBM/XGBoost nativo), NaN sin imputar.
      - ``naive_bayes``: numéricas continuas discretizadas en quintiles + categóricas ordinales.
    """
    num = _numeric(use_duration)
    if kind == "linear":
        return ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ], verbose_feature_names_out=False)
    if kind == "tree_onehot":
        return ColumnTransformer([
            ("num", SimpleImputer(strategy="constant", fill_value=-1), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ], verbose_feature_names_out=False)
    if kind == "tree_native":
        return CategoryCaster(CATEGORICAL)
    if kind == "naive_bayes":
        cont = [c for c in CONTINUOUS if c in num] + (["duration"] if use_duration else [])
        discrete = [c for c in num if c not in cont]
        to_int = FunctionTransformer(_non_negative_int, feature_names_out="one-to-one")
        return ColumnTransformer([
            ("cont", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                               ("bins", KBinsDiscretizer(n_bins=5, encode="ordinal",
                                                         strategy="quantile",
                                                         quantile_method="averaged_inverted_cdf")),
                               ("int", to_int)]), cont),
            ("disc", to_int, discrete),
            # CategoricalNB no admite índices negativos: una categoría no vista se asigna al nivel 0.
            ("cat", Pipeline([("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                              ("int", to_int)]), CATEGORICAL),
        ], verbose_feature_names_out=False)
    raise ValueError(f"Tipo de preprocesador desconocido: {kind}")


def build_pipeline(estimator, kind: str, use_duration: bool = False, sampler=None) -> ImbPipeline:
    """Pipeline completo: datos crudos → variables derivadas → preprocesamiento → (remuestreo) → modelo."""
    steps = [("features", FeatureEngineer(use_duration=use_duration)),
             ("prep", make_preprocessor(kind, use_duration))]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


def feature_catalog() -> pd.DataFrame:
    df = pd.DataFrame(FEATURE_CATALOG, columns=["feature", "source", "transformation", "justification",
                                                "learned_in_fit"])
    df.insert(0, "in_pre_contact_model", df["feature"] != "duration")
    return df


def write_feature_catalog() -> pd.DataFrame:
    cfg.ensure_dirs()
    df = feature_catalog()
    df.to_csv(cfg.TABLES_DIR / "feature_catalog.csv", index=False)
    return df


if __name__ == "__main__":
    from .data import load_clean, split_xy
    write_feature_catalog()
    train, _ = load_clean()
    X, y = split_xy(train)
    fe = FeatureEngineer().fit(X)
    Z = fe.transform(X)
    print(Z.head().T)
    print("caps:", fe.campaign_cap_, fe.previous_cap_)
    for kind in ["linear", "tree_onehot", "tree_native", "naive_bayes"]:
        p = make_preprocessor(kind).fit(Z)
        print(kind, np.asarray(p.transform(Z.head())).shape if kind != "tree_native" else p.transform(Z).dtypes.value_counts().to_dict())
