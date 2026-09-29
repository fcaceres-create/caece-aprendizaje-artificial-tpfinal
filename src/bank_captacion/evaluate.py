"""Evaluación final en el hold-out (Fase 5).

Reglas: el modelo, sus hiperparámetros y el umbral quedaron fijados en la Fase 4 usando solo
train. Acá se evalúa **una sola vez** sobre ``banca_test.csv``; ninguna decisión se toma
después de ver estos resultados. Los umbrales de los baselines también se eligen con sus
propias predicciones out-of-fold de train, nunca con test.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from . import config as cfg
from .business import (classification_metrics, decile_table, expected_profit, profit_by_fraction,
                       ranking_metrics, threshold_curve, top_k_stats)
from .data import file_sha256, load_clean, split_xy
from .features import AGE_BINS, AGE_LABELS
from .train import cv_evaluate, get_folds

sns.set_theme(style="whitegrid")
LABELS = {"final": "LightGBM (final, pre-contacto)", "logreg": "Regresión logística (baseline)",
          "dummy": "Dummy (prior)", "with_duration": "LightGBM + duration (leakage)"}
COLORS = {"final": "#2c7fb8", "logreg": "#31a354", "dummy": "#999999", "with_duration": "#d95f0e"}


def _savefig(fig, name):
    fig.savefig(cfg.FIGURES_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _oof_profit_threshold(name: str, imbalance: str, kind: str, X, y) -> float:
    """Umbral de máximo beneficio a partir de predicciones OOF de train (para baselines)."""
    _, oof = cv_evaluate(name, X, y, get_folds(X, y), imbalance=imbalance, kind=kind)
    tc = threshold_curve(y, oof)
    return float(tc.loc[tc["profit"].idxmax(), "threshold"])


def run_evaluation() -> dict:
    cfg.ensure_dirs()
    warnings.filterwarnings("ignore")
    meta = json.loads(cfg.METADATA_FILE.read_text(encoding="utf-8"))
    train, test = load_clean()
    Xtr, ytr = split_xy(train)
    Xte, yte = split_xy(test)
    usage = {"holdout_file": cfg.TEST_FILE.name, "rows": int(len(Xte)),
             "evaluated_utc": datetime.now(timezone.utc).isoformat(),
             "model_file_sha256": file_sha256(cfg.MODEL_FILE),
             "note": "Evaluación única del modelo ya seleccionado; modelo, hiperparámetros y umbral "
                     "se fijaron con train (CV). Ningún resultado de este archivo se usó para decidir."}
    (cfg.TABLES_DIR / "holdout_usage.json").write_text(json.dumps(usage, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")

    models = {"final": joblib.load(cfg.MODEL_FILE), "logreg": joblib.load(cfg.BASELINE_LR_FILE),
              "dummy": DummyClassifier(strategy="prior").fit(Xtr, ytr),
              "with_duration": joblib.load(cfg.MODEL_WITH_DURATION_FILE)}
    thresholds = {"final": meta["threshold"],
                  "logreg": _oof_profit_threshold("logreg", "weights", "linear", Xtr, ytr),
                  "dummy": cfg.COST_PER_CONTACT / cfg.VALUE_PER_CONVERSION,
                  "with_duration": meta["threshold"]}
    scores = {k: m.predict_proba(Xte)[:, 1] for k, m in models.items()}

    # --- 1. Métricas ---------------------------------------------------------------
    rows = []
    for k, s in scores.items():
        r = {"model": k, "label": LABELS[k]}
        r.update(ranking_metrics(yte, s))
        cm = classification_metrics(yte, s, thresholds[k])
        r.update({f"{m}_at_thr": v for m, v in cm.items()})
        rows.append(r)
    metrics = pd.DataFrame(rows)
    metrics.round(4).to_csv(cfg.TABLES_DIR / "test_metrics.csv", index=False)

    # Comparación CV vs test del modelo final (brecha de generalización).
    cvm = meta["cv_metrics_mean"]
    fin = metrics.set_index("model").loc["final"]
    pd.DataFrame([{"metric": m, "cv_mean": cvm[m], "cv_std": meta["cv_metrics_std"][m], "test": fin[m]}
                  for m in ["roc_auc", "pr_auc", "lift@10", "lift@20", "lift@30", "gain@20", "brier"]]) \
        .round(4).to_csv(cfg.TABLES_DIR / "test_vs_cv.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k in ["final", "logreg", "with_duration", "dummy"]:
        fpr, tpr, _ = roc_curve(yte, scores[k])
        axes[0].plot(fpr, tpr, color=COLORS[k], ls="--" if k == "with_duration" else "-",
                     label=f"{LABELS[k]} (AUC={roc_auc_score(yte, scores[k]):.3f})")
        p, r, _ = precision_recall_curve(yte, scores[k])
        axes[1].plot(r, p, color=COLORS[k], ls="--" if k == "with_duration" else "-",
                     label=f"{LABELS[k]} (AP={average_precision_score(yte, scores[k]):.3f})")
    axes[0].plot([0, 1], [0, 1], ":", color="k", lw=0.8)
    axes[0].set(title="Curva ROC (test)", xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    axes[1].axhline(yte.mean(), ls=":", color="k", lw=0.8)
    axes[1].set(title="Curva Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8, loc="lower right" if ax is axes[0] else "upper right")
    _savefig(fig, "test_roc_pr_curves.png")

    # --- 2. Matriz de confusión --------------------------------------------------------------------------
    thr = thresholds["final"]
    pred = (scores["final"] >= thr).astype(int)
    cm = confusion_matrix(yte, pred)
    pd.DataFrame(cm, index=["real: no convierte", "real: convierte"],
                 columns=["predicho: no llamar", "predicho: llamar"]).to_csv(cfg.TABLES_DIR / "test_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No llamar", "Llamar"], yticklabels=["No convierte", "Convierte"])
    ax.set(xlabel="Decisión del modelo", ylabel="Resultado real",
           title=f"Matriz de confusión en test (umbral {thr:.3f})")
    _savefig(fig, "test_confusion_matrix.png")

    # --- 3. Deciles, gain y lift --------------------------------------------------------------------
    dec = decile_table(yte, scores["final"])
    dec.round(4).to_csv(cfg.TABLES_DIR / "test_deciles.csv", index=False)
    dec_lr = decile_table(yte, scores["logreg"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.concatenate([[0], dec["cum_pct_contacted"]])
    axes[0].plot(x, np.concatenate([[0], dec["cum_gain_pct"]]), "o-", color=COLORS["final"], label=LABELS["final"])
    axes[0].plot(x, np.concatenate([[0], dec_lr["cum_gain_pct"]]), "s-", color=COLORS["logreg"], label=LABELS["logreg"])
    axes[0].plot([0, 100], [0, 100], ":", color="k", label="Al azar")
    axes[0].set(title="Curva de ganancia acumulada (test)", xlabel="% de clientes contactados (ordenados por score)",
                ylabel="% de conversiones capturadas")
    axes[0].legend()
    axes[1].bar(dec["decile"] - 0.2, dec["lift"], width=0.4, color=COLORS["final"], label=LABELS["final"])
    axes[1].bar(dec_lr["decile"] + 0.2, dec_lr["lift"], width=0.4, color=COLORS["logreg"], label=LABELS["logreg"])
    axes[1].axhline(1, ls=":", color="k")
    axes[1].set(title="Lift por decil (test)", xlabel="Decil (1 = mayor score)", ylabel="Lift vs tasa base")
    axes[1].set_xticks(range(1, 11))
    axes[1].legend()
    _savefig(fig, "test_gain_lift.png")

    # --- 4. Negocio ---------------------------------------------------------------------------------
    base_profit_all = expected_profit(yte, np.ones(len(yte)))
    biz = []
    for frac in cfg.TOP_FRACTIONS:
        st = top_k_stats(yte, scores["final"], frac)
        k = int(np.ceil(frac * len(yte)))
        contacted = np.zeros(len(yte), bool)
        contacted[np.argsort(-scores["final"], kind="mergesort")[:k]] = True
        biz.append({"strategy": f"Top {int(100 * frac)} % del ranking", "contacted": k,
                    "contacted_pct": 100 * frac, "conversions": int(yte[contacted].sum()),
                    "gain_pct": 100 * st["gain"], "precision_pct": 100 * st["precision"], "lift": st["lift"],
                    "profit": expected_profit(yte, contacted)})
    biz.append({"strategy": f"Umbral operativo ({thr:.3f})", "contacted": int(pred.sum()),
                "contacted_pct": 100 * pred.mean(), "conversions": int(yte[pred == 1].sum()),
                "gain_pct": 100 * yte[pred == 1].sum() / yte.sum(),
                "precision_pct": 100 * yte[pred == 1].mean(),
                "lift": yte[pred == 1].mean() / yte.mean(), "profit": expected_profit(yte, pred)})
    biz.append({"strategy": "Contactar a todos", "contacted": len(yte), "contacted_pct": 100.0,
                "conversions": int(yte.sum()), "gain_pct": 100.0, "precision_pct": 100 * yte.mean(),
                "lift": 1.0, "profit": base_profit_all})
    biz.append({"strategy": "No contactar a nadie", "contacted": 0, "contacted_pct": 0.0, "conversions": 0,
                "gain_pct": 0.0, "precision_pct": np.nan, "lift": np.nan, "profit": 0.0})
    biz = pd.DataFrame(biz)
    biz["profit_vs_all"] = biz["profit"] - base_profit_all
    biz.round(2).to_csv(cfg.TABLES_DIR / "test_business_summary.csv", index=False)

    # Sensibilidad: umbrales elegidos en OOF (train) para cada ratio, aplicados a test.
    sens_oof = pd.read_csv(cfg.TABLES_DIR / "sensitivity_oof.csv")
    sens = []
    for _, r in sens_oof.iterrows():
        v = r["value_V"]
        c = scores["final"] >= r["best_threshold"]
        prof = expected_profit(yte, c, cfg.COST_PER_CONTACT, v)
        all_p = expected_profit(yte, np.ones(len(yte)), cfg.COST_PER_CONTACT, v)
        sens.append({"vc_ratio": r["vc_ratio"], "threshold_from_oof": r["best_threshold"],
                     "contacted_pct": 100 * c.mean(), "gain_pct": 100 * yte[c].sum() / yte.sum(),
                     "profit_model": prof, "profit_contact_all": all_p, "profit_vs_all": prof - all_p,
                     "profit_vs_all_pct": 100 * (prof - all_p) / abs(all_p) if all_p else np.nan})
    sens = pd.DataFrame(sens)
    sens.round(3).to_csv(cfg.TABLES_DIR / "test_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, len(cfg.VC_RATIOS), figsize=(18, 4.2))
    for ax, (_, r) in zip(axes, sens.iterrows()):
        v = r["vc_ratio"] * cfg.COST_PER_CONTACT
        pf = profit_by_fraction(yte, scores["final"], cfg.COST_PER_CONTACT, v)
        pl = profit_by_fraction(yte, scores["logreg"], cfg.COST_PER_CONTACT, v)
        ax.plot(pf["pct_contacted"], pf["profit"], color=COLORS["final"], label="LightGBM")
        ax.plot(pl["pct_contacted"], pl["profit"], color=COLORS["logreg"], label="Reg. logística", alpha=0.8)
        ax.plot([0, 100], [0, pf["profit"].iloc[-1]], ":", color="k", label="Al azar")
        ax.axvline(r["contacted_pct"], color="r", ls="--", lw=1, label="Umbral operativo (OOF)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set(title=f"V/C = {int(r['vc_ratio'])}", xlabel="% contactado", ylabel="Beneficio esperado")
    axes[0].legend(fontsize=7)
    fig.suptitle("Beneficio esperado según % de clientes contactados (test) — sensibilidad al ratio V/C")
    fig.tight_layout()
    _savefig(fig, "test_profit_curves.png")

    # --- 5. Calibración ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for k in ["final", "logreg"]:
        pt, pp = calibration_curve(yte, scores[k], n_bins=10, strategy="quantile")
        axes[0].plot(pp, pt, "o-", color=COLORS[k],
                     label=f"{LABELS[k]} (Brier={metrics.set_index('model').loc[k, 'brier']:.4f})")
    axes[0].plot([0, 1], [0, 1], ":", color="k", label="Calibración perfecta")
    axes[0].set(title="Diagrama de confiabilidad (test, deciles)", xlabel="Probabilidad predicha media",
                ylabel="Tasa de conversión observada")
    axes[0].legend(fontsize=8)
    axes[1].hist(scores["final"][yte == 0], bins=40, alpha=0.6, color="#8c9bab", density=True, label="No convierte")
    axes[1].hist(scores["final"][yte == 1], bins=40, alpha=0.6, color="#d9534f", density=True, label="Convierte")
    axes[1].axvline(thr, color="k", ls="--", label=f"Umbral {thr:.3f}")
    axes[1].set(title="Distribución de probabilidades del modelo final (test)", xlabel="Probabilidad predicha",
                ylabel="Densidad")
    axes[1].legend()
    _savefig(fig, "test_calibration.png")

    # --- 6. Leakage ------------------------------------------------------------------------------------
    leak = metrics[metrics.model.isin(["final", "with_duration"])][
        ["model", "label", "roc_auc", "pr_auc", "lift@10", "lift@20", "gain@20", "brier"]]
    leak.round(4).to_csv(cfg.TABLES_DIR / "test_leakage.csv", index=False)

    # --- 7. Segmentos y errores ------------------------------------------------------------------------
    seg = segment_metrics(Xte, yte, scores["final"], thr)
    error_analysis(Xte, yte, scores["final"], thr)

    # --- 8. Razonabilidad -------------------------------------------------------------------------------
    reasonableness(metrics)

    out = {"test_metrics": metrics.set_index("model")[["roc_auc", "pr_auc", "lift@20", "gain@20", "brier"]]
           .round(4).to_dict("index"), "thresholds": thresholds,
           "business": biz.round(2).to_dict("records"), "worst_segments": seg.head(3).to_dict("records")}
    (cfg.TABLES_DIR / "evaluation_summary.json").write_text(json.dumps(out, indent=2, default=float,
                                                                       ensure_ascii=False), encoding="utf-8")
    return out


def segment_metrics(X, y, s, thr) -> pd.DataFrame:
    d = X.assign(y=y.values, score=s, pred=(s >= thr).astype(int))
    d["age_group"] = pd.cut(d["age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    rows = []
    for col in ["job", "age_group", "contact"]:
        for lvl, g in d.groupby(col):
            both = g.y.nunique() == 2
            rows.append({"segment_var": col, "segment": lvl, "n": len(g), "conversions": int(g.y.sum()),
                         "conversion_rate": g.y.mean(),
                         "roc_auc": roc_auc_score(g.y, g.score) if both and g.y.sum() >= 5 else np.nan,
                         "pr_auc": average_precision_score(g.y, g.score) if both and g.y.sum() >= 5 else np.nan,
                         "recall_at_thr": g.loc[g.y == 1, "pred"].mean() if g.y.sum() else np.nan,
                         "precision_at_thr": g.loc[g.pred == 1, "y"].mean() if g.pred.sum() else np.nan,
                         "mean_score": g.score.mean(), "reliable": bool(g.y.sum() >= 20)})
    seg = pd.DataFrame(rows).sort_values(["segment_var", "roc_auc"])
    seg.round(4).to_csv(cfg.TABLES_DIR / "segment_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [3, 1.6, 1]})
    overall = roc_auc_score(y, s)
    for ax, col in zip(axes, ["job", "age_group", "contact"]):
        g = seg[seg.segment_var == col].sort_values("roc_auc")
        colors = ["#337ab7" if r else "#c6d4e1" for r in g["reliable"]]
        ax.barh(g["segment"].astype(str), g["roc_auc"], color=colors)
        for i, (v, n) in enumerate(zip(g["roc_auc"], g["conversions"])):
            ax.text(v, i, f" {v:.2f} (pos={n})", va="center", fontsize=8)
        ax.axvline(overall, color="r", ls="--", lw=1)
        ax.set_xlim(0.5, 1.0)
        ax.set_title({"job": "Ocupación", "age_group": "Grupo de edad", "contact": "Tipo de contacto"}[col])
        ax.set_xlabel("ROC-AUC en test")
    fig.suptitle(f"Desempeño por segmento (línea roja = ROC-AUC global {overall:.3f}; "
                 "celeste claro = menos de 20 conversiones, estimación poco confiable)")
    fig.tight_layout()
    _savefig(fig, "segment_metrics.png")
    return seg[seg.reliable].sort_values("roc_auc")


def error_analysis(X, y, s, thr) -> None:
    pred = (s >= thr).astype(int)
    group = np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 1), (y == 1) & (pred == 0),
                       (y == 0) & (pred == 0)], ["TP", "FP", "FN", "TN"], default="")
    d = X.assign(grupo=group, score=s)
    num = ["age", "balance", "campaign", "pdays", "previous", "day", "score"]
    prof = d.groupby("grupo")[num].mean().round(3)
    prof.insert(0, "n", d.groupby("grupo").size())
    prof["pct_previously_contacted"] = d.groupby("grupo")["pdays"].apply(lambda p: 100 * (p != -1).mean()).round(1)
    prof.reindex(["TP", "FN", "FP", "TN"]).to_csv(cfg.TABLES_DIR / "error_profiles.csv")
    rows = []
    for col in ["job", "contact", "poutcome", "month", "housing", "loan", "education", "marital"]:
        ct = pd.crosstab(d[col], d["grupo"], normalize="columns") * 100
        for lvl in ct.index:
            rows.append({"variable": col, "category": lvl,
                         **{f"pct_{g}": round(float(ct.loc[lvl, g]), 1) for g in ct.columns}})
    pd.DataFrame(rows).to_csv(cfg.TABLES_DIR / "error_categorical_profiles.csv", index=False)


def reasonableness(metrics: pd.DataFrame) -> None:
    fin = metrics.set_index("model").loc["final"]
    dur = metrics.set_index("model").loc["with_duration"]
    rows = [
        {"check": "ROC-AUC sin duration vs referencia 0,75–0,80", "value": round(fin["roc_auc"], 4),
         "reference": "0,75–0,80 (rango de referencia del enunciado para este dataset sin duration)",
         "ok": bool(0.72 <= fin["roc_auc"] <= 0.83)},
        {"check": "ROC-AUC con duration (leakage) en rango típico alto", "value": round(dur["roc_auc"], 4),
         "reference": "> 0,90 esperado: duration por sí sola tiene ROC-AUC ≈ 0,81 en train",
         "ok": bool(dur["roc_auc"] > 0.88)},
        {"check": "PR-AUC muy por encima del azar (prevalencia)", "value": round(fin["pr_auc"], 4),
         "reference": f"piso = prevalencia en test ({metrics.set_index('model').loc['dummy', 'pr_auc']:.3f})",
         "ok": bool(fin["pr_auc"] > 2 * metrics.set_index("model").loc["dummy", "pr_auc"])},
        {"check": "Modelo final supera a la regresión logística", "value": round(fin["pr_auc"], 4),
         "reference": f"PR-AUC logística = {metrics.set_index('model').loc['logreg', 'pr_auc']:.4f}",
         "ok": bool(fin["pr_auc"] > metrics.set_index("model").loc["logreg", "pr_auc"])},
    ]
    meta = json.loads(cfg.METADATA_FILE.read_text(encoding="utf-8"))
    cv = meta["cv_metrics_mean"]["pr_auc"]
    sd = meta["cv_metrics_std"]["pr_auc"]
    rows.append({"check": "PR-AUC en test dentro de ±2 desvíos de la CV", "value": round(fin["pr_auc"], 4),
                 "reference": f"CV = {cv:.4f} ± {sd:.4f}", "ok": bool(abs(fin["pr_auc"] - cv) <= 2 * sd)})
    pd.DataFrame(rows).to_csv(cfg.TABLES_DIR / "reasonableness.csv", index=False)


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2, default=float, ensure_ascii=False))
