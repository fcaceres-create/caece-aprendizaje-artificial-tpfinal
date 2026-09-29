"""Métricas de ranking y de negocio: lift, gain, deciles, beneficio esperado y sensibilidad V/C."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

from . import config as cfg


def _order(scores: np.ndarray) -> np.ndarray:
    """Índices de mayor a menor puntaje (orden estable para empates reproducibles)."""
    return np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")


def top_k_stats(y: np.ndarray, scores: np.ndarray, frac: float) -> dict:
    """Precisión, gain (recall) y lift al contactar la fracción ``frac`` mejor rankeada."""
    y = np.asarray(y)
    k = int(np.ceil(frac * len(y)))
    top = y[_order(scores)[:k]]
    base = y.mean()
    precision = top.mean()
    return {"precision": float(precision), "gain": float(top.sum() / y.sum()),
            "lift": float(precision / base)}


def ranking_metrics(y, scores, proba: bool = True) -> dict:
    """Métricas usadas en CV y en el hold-out."""
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    out = {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s)}
    for f in cfg.TOP_FRACTIONS:
        st = top_k_stats(y, s, f)
        pct = int(round(100 * f))
        out[f"lift@{pct}"] = st["lift"]
        out[f"gain@{pct}"] = st["gain"]
    out["brier"] = brier_score_loss(y, s) if proba else np.nan
    return out


def classification_metrics(y, proba, threshold: float) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {"threshold": threshold, "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "contacted_pct": 100 * pred.mean(),
            "profit": expected_profit(y, pred)}


def expected_profit(y, contacted, cost: float = cfg.COST_PER_CONTACT,
                    value: float = cfg.VALUE_PER_CONVERSION) -> float:
    """Beneficio = V·TP − C·(TP+FP), con TP = conversiones entre los contactados."""
    y = np.asarray(y)
    c = np.asarray(contacted).astype(bool)
    tp = int(y[c].sum())
    return float(value * tp - cost * c.sum())


def decile_table(y, scores) -> pd.DataFrame:
    """Tabla de deciles (decil 1 = 10 % con mayor puntaje), con gain y lift acumulados."""
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    order = _order(s)
    groups = np.array_split(order, 10)
    base = y.mean()
    total_pos = y.sum()
    rows, cum_n, cum_pos = [], 0, 0
    for i, idx in enumerate(groups, start=1):
        n, pos = len(idx), int(y[idx].sum())
        cum_n += n
        cum_pos += pos
        rows.append({"decile": i, "n": n, "conversions": pos, "conversion_rate": pos / n,
                     "lift": (pos / n) / base, "cum_n": cum_n, "cum_pct_contacted": 100 * cum_n / len(y),
                     "cum_conversions": cum_pos, "cum_gain_pct": 100 * cum_pos / total_pos,
                     "cum_lift": (cum_pos / cum_n) / base,
                     "score_min": float(s[idx].min()), "score_max": float(s[idx].max())})
    return pd.DataFrame(rows)


def profit_by_fraction(y, scores, cost: float = cfg.COST_PER_CONTACT,
                       value: float = cfg.VALUE_PER_CONVERSION, n_points: int = 101) -> pd.DataFrame:
    """Beneficio esperado al contactar el top x % (x de 0 a 100)."""
    y = np.asarray(y)
    cum_pos = np.concatenate([[0], np.cumsum(y[_order(scores)])])
    rows = []
    for frac in np.linspace(0, 1, n_points):
        k = int(round(frac * len(y)))
        rows.append({"pct_contacted": 100 * frac, "contacts": k, "conversions": int(cum_pos[k]),
                     "gain_pct": 100 * cum_pos[k] / y.sum(),
                     "profit": value * cum_pos[k] - cost * k})
    return pd.DataFrame(rows)


def threshold_curve(y, proba, cost: float = cfg.COST_PER_CONTACT,
                    value: float = cfg.VALUE_PER_CONVERSION) -> pd.DataFrame:
    """Beneficio, precision, recall y F1 para una grilla de umbrales."""
    y = np.asarray(y)
    p = np.asarray(proba, dtype=float)
    grid = np.unique(np.round(np.concatenate([np.linspace(0.005, 0.95, 190),
                                              np.quantile(p, np.linspace(0.01, 0.99, 99))]), 4))
    rows = []
    for t in grid:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"threshold": float(t), "contacted": tp + fp, "contacted_pct": 100 * (tp + fp) / len(y),
                     "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
                     "profit": value * tp - cost * (tp + fp)})
    return pd.DataFrame(rows)


def sensitivity_table(y, proba, ratios=cfg.VC_RATIOS, cost: float = cfg.COST_PER_CONTACT) -> pd.DataFrame:
    """Para cada ratio V/C: umbral óptimo, % contactado y beneficio vs contactar a todos."""
    y = np.asarray(y)
    rows = []
    for r in ratios:
        value = r * cost
        tc = threshold_curve(y, proba, cost, value)
        best = tc.loc[tc["profit"].idxmax()]
        all_profit = value * y.sum() - cost * len(y)
        rows.append({"vc_ratio": r, "value_V": value, "cost_C": cost,
                     "theoretical_threshold": cost / value,
                     "best_threshold": best["threshold"], "contacted_pct": best["contacted_pct"],
                     "recall": best["recall"], "precision": best["precision"],
                     "profit_model": best["profit"], "profit_contact_all": all_profit,
                     "profit_contact_none": 0.0,
                     "uplift_vs_all": best["profit"] - all_profit})
    return pd.DataFrame(rows)
