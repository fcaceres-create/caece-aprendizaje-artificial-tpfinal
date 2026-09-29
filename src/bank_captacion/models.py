"""Catálogo de modelos, estrategias de desbalance y espacios de búsqueda de hiperparámetros."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config as cfg
from .features import build_pipeline

N_THREADS = 6  # hilos por modelo; los folds se ejecutan en paralelo


class SubsampledClassifier(ClassifierMixin, BaseEstimator):
    """Entrena el estimador sobre una submuestra estratificada (para SVM RBF, costo O(n²))."""

    def __init__(self, estimator=None, n_samples: int = cfg.SVM_TRAIN_SUBSAMPLE, random_state: int = cfg.SEED):
        self.estimator = estimator
        self.n_samples = n_samples
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        if len(y) > self.n_samples:
            idx, _ = train_test_split(np.arange(len(y)), train_size=self.n_samples,
                                      stratify=y, random_state=self.random_state)
            X, y = X[idx], y[idx]
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    def predict(self, X):
        return self.estimator_.predict(X)


@dataclass
class ModelSpec:
    name: str
    label: str            # nombre en español para tablas y figuras
    family: str
    kind: str             # preprocesador (ver features.make_preprocessor)
    factory: Callable[[bool], object]  # recibe weighted: bool
    supports_weights: bool
    interpretability: int  # 1 (caja negra) a 5 (totalmente interpretable)
    rationale: str
    smote_kind: str | None = None  # preprocesador a usar con SMOTE (necesita todo numérico)
    extra: dict = field(default_factory=dict)


def _lgbm(weighted: bool, **params):
    base = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=cfg.SEED, n_jobs=N_THREADS, verbose=-1)
    base.update(params)
    return LGBMClassifier(class_weight="balanced" if weighted else None, **base)


def _xgb(weighted: bool, **params):
    # scale_pos_weight se ajusta en cada fold con la proporción neg/pos del fold de entrenamiento.
    base = dict(n_estimators=500, learning_rate=0.05, max_depth=5, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                eval_metric="aucpr", random_state=cfg.SEED, n_jobs=N_THREADS,
                enable_categorical=True, max_cat_to_onehot=1)
    base.update(params)
    return XGBClassifier(scale_pos_weight=1.0, **base)


def _rf(weighted: bool, **params):
    base = dict(n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                random_state=cfg.SEED, n_jobs=N_THREADS)
    base.update(params)
    return RandomForestClassifier(class_weight="balanced_subsample" if weighted else None, **base)


def _logreg(weighted: bool, **params):
    base = dict(C=1.0, max_iter=3000, solver="lbfgs")
    base.update(params)
    return LogisticRegression(class_weight="balanced" if weighted else None, **base)


def _tree(weighted: bool, **params):
    base = dict(max_depth=6, min_samples_leaf=50, random_state=cfg.SEED)
    base.update(params)
    return DecisionTreeClassifier(class_weight="balanced" if weighted else None, **base)


def _knn(weighted: bool, **params):
    base = dict(n_neighbors=51, n_jobs=N_THREADS)
    base.update(params)
    return KNeighborsClassifier(**base)


def _nb(weighted: bool, **params):
    base = dict(alpha=1.0)
    base.update(params)
    return CategoricalNB(**base)


def _mlp(weighted: bool, **params):
    base = dict(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                early_stopping=True, max_iter=300, random_state=cfg.SEED)
    base.update(params)
    return MLPClassifier(**base)


def _svm(weighted: bool, **params):
    base = dict(kernel="rbf", C=1.0, gamma="scale")
    base.update(params)
    return SubsampledClassifier(SVC(class_weight="balanced" if weighted else None, **base))


def _dummy(weighted: bool, **params):
    return DummyClassifier(strategy="prior")


MODEL_SPECS: dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("dummy", "Dummy (prior)", "Referencia", "linear",
              _dummy, False, 5,
              "Piso de comparación: predice la prevalencia para todos. PR-AUC ≈ tasa base, ROC-AUC = 0,5."),
    ModelSpec("logreg", "Regresión logística", "Lineal", "linear", _logreg, True, 5,
              "Baseline interpretable: coeficientes = efecto de cada variable en log-odds. "
              "Supone efectos aditivos y lineales en el logit.", smote_kind="linear"),
    ModelSpec("tree", "Árbol de decisión (prof. 6)", "Árboles", "tree_onehot", _tree, True, 4,
              "Reglas explicables al negocio; profundidad acotada para limitar sobreajuste. "
              "Captura interacciones pero es inestable.", smote_kind="tree_onehot"),
    ModelSpec("knn", "K vecinos más cercanos (k=51)", "Basado en instancias", "linear",
              _knn, False, 2,
              "No paramétrico, basado en similitud; requiere escalado. Sufre con alta dimensión y "
              "variables one-hot. k grande para suavizar las probabilidades."),
    ModelSpec("naive_bayes", "Naive Bayes categórico", "Probabilístico", "naive_bayes",
              _nb, False, 4,
              "Generativo, supone independencia condicional. Se usa sobre variables discretizadas "
              "(CategoricalNB), representación adecuada para datos mayormente categóricos."),
    ModelSpec("random_forest", "Random Forest", "Ensamble (bagging)", "tree_onehot", _rf, True, 2,
              "Bagging de árboles: reduce varianza, robusto a outliers y escalas.", smote_kind="tree_onehot"),
    ModelSpec("lightgbm", "LightGBM", "Ensamble (boosting)", "tree_onehot", _lgbm, True, 2,
              "Gradient boosting por hojas, rápido; estado del arte en datos tabulares. "
              "Admite categorías nativas.", smote_kind="tree_onehot"),
    ModelSpec("xgboost", "XGBoost", "Ensamble (boosting)", "tree_onehot", _xgb, True, 2,
              "Gradient boosting por niveles con regularización; desbalance vía scale_pos_weight.",
              smote_kind="tree_onehot"),
    ModelSpec("mlp", "Red neuronal (MLP 64-32)", "Redes neuronales", "linear",
              _mlp, False, 1,
              "Representante de redes neuronales; aproximador universal pero sin ventaja esperable "
              "en datos tabulares chicos.", smote_kind="linear"),
    ModelSpec("svm_rbf", "SVM RBF (submuestra 10k)", "Márgenes / kernels", "linear",
              _svm, True, 1,
              "Kernel RBF: frontera no lineal de máximo margen. Costo O(n²) → entrenado sobre una "
              "submuestra estratificada de 10.000 filas; sin probabilidades calibradas (Brier no aplica)."),
]}


def make_model(name: str, imbalance: str = "weights", use_duration: bool = False,
               kind: str | None = None, **params):
    """Construye el pipeline completo de un modelo del catálogo.

    imbalance: ``none`` (sin tratamiento), ``weights`` (pesos de clase) o ``smote``.
    """
    spec = MODEL_SPECS[name]
    weighted = imbalance == "weights" and spec.supports_weights
    est = spec.factory(weighted, **params)
    sampler = None
    if imbalance == "smote":
        sampler = SMOTE(random_state=cfg.SEED, k_neighbors=5)
        kind = kind or spec.smote_kind or spec.kind
        if kind == "tree_native":
            raise ValueError("SMOTE requiere variables numéricas: usar kind='tree_onehot'.")
    kind = kind or spec.kind
    if name == "xgboost" and kind == "tree_native":
        est.set_params(enable_categorical=True)
    return build_pipeline(est, kind, use_duration=use_duration, sampler=sampler)


def needs_fold_pos_weight(name: str, imbalance: str) -> bool:
    return name == "xgboost" and imbalance == "weights"


# --- Espacios de búsqueda (Optuna) -----------------------------------------------------

def suggest_params(name: str, trial) -> dict:
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 96, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.1, 0.8),
        }
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if name == "mlp":
        return {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(32,), (64,), (64, 32), (128, 64)]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    raise ValueError(f"No hay espacio de búsqueda definido para {name}")
