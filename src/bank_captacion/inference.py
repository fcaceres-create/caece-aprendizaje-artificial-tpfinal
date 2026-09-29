"""Inferencia liviana (sin SHAP ni matplotlib): carga del modelo, deciles y contribuciones.

La usan la API y los tests. Las contribuciones se calculan con TreeSHAP nativo de LightGBM
(``pred_contrib=True``), equivalente a los valores SHAP en escala log-odds.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as cfg

FEATURE_LABELS = {
    "age": "Edad", "balance_log": "Saldo (log)", "balance_negative": "Saldo negativo",
    "balance_zero": "Saldo cero", "campaign_w": "Contactos en campaña", "pdays_clean": "Días desde contacto previo",
    "previously_contacted": "Contactado antes", "previous_w": "Contactos previos", "day_sin": "Día (seno)",
    "day_cos": "Día (coseno)", "contact_known": "Canal conocido", "prev_campaign_success": "Éxito previo",
    "n_credit_products": "Nº de préstamos", "job": "Ocupación", "marital": "Estado civil",
    "education": "Educación", "default": "Impagos", "housing": "Hipoteca", "loan": "Préstamo personal",
    "contact": "Tipo de contacto", "month": "Mes", "poutcome": "Resultado previo", "age_group": "Grupo de edad",
    "season": "Estación", "duration": "Duración",
}

INPUT_COLUMNS = [c for c in cfg.EXPECTED_COLUMNS if c not in (cfg.TARGET, cfg.LEAKAGE_COLUMN)]


def load_model(model_path: Path = cfg.MODEL_FILE, metadata_path: Path = cfg.METADATA_FILE):
    model = joblib.load(model_path)
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return model, meta


def transform_for_model(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Aplica todos los pasos del pipeline salvo el clasificador (y salvo remuestreos)."""
    Z = X
    for _, step in pipe.steps[:-1]:
        if hasattr(step, "fit_resample"):
            continue
        Z = step.transform(Z)
    return Z


def assign_decile(proba, cuts: list[float]) -> np.ndarray:
    """Decil 1 = 10 % de mayor probabilidad, según la distribución OOF de train."""
    k = np.searchsorted(np.asarray(cuts), np.asarray(proba, dtype=float), side="right")
    return 10 - k


def _fmt_value(v):
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return str(v)


def top_contributions(pipe, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """Top-k contribuciones SHAP por fila (mayor valor absoluto)."""
    Z = transform_for_model(pipe, X)
    contrib = pipe[-1].predict(Z, pred_contrib=True)  # (n, n_features + 1); última columna = base
    names = list(Z.columns)
    out = []
    for i in range(len(Z)):
        vals = contrib[i, :-1]
        idx = np.argsort(-np.abs(vals))[:k]
        out.append([{"feature": names[j], "label": FEATURE_LABELS.get(names[j], names[j]),
                     "value": _fmt_value(Z.iloc[i, j]), "shap": round(float(vals[j]), 4),
                     "effect": "aumenta" if vals[j] > 0 else "reduce"} for j in idx])
    return out


def score_frame(pipe, meta: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad, decil, ranking y recomendación para un lote de clientes."""
    p = pipe.predict_proba(X[INPUT_COLUMNS])[:, 1]
    out = X.copy()
    out["probability"] = np.round(p, 5)
    out["decile"] = assign_decile(p, meta["decile_cuts_oof"])
    out["recommendation"] = np.where(p >= meta["threshold"], "llamar", "no llamar")
    out = out.sort_values("probability", ascending=False, kind="mergesort")
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
