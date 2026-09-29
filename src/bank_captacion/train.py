"""Modelado (Fase 4): validación cruzada, comparación, desbalance, tuning, selección y umbral.

Protocolo único: ``StratifiedKFold(5, shuffle=True, random_state=42)`` sobre el train limpio.
Todos los modelos usan exactamente la misma partición. El hold-out no se toca en este módulo.
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from . import config as cfg
from .business import ranking_metrics, sensitivity_table, threshold_curve
from .data import load_clean, split_xy
from .features import FeatureEngineer
from .models import MODEL_SPECS, make_model, needs_fold_pos_weight, suggest_params

STATE_FILE = cfg.TABLES_DIR / "training_state.json"
METRIC_COLS = ["roc_auc", "pr_auc", "lift@10", "lift@20", "lift@30", "gain@10", "gain@20",
               "gain@30", "brier", "fit_time_s", "predict_ms_per_1k"]
# Si dos estrategias de desbalance difieren en PR-AUC menos que esto, se prefiere la que no
# distorsiona las probabilidades (none > weights > smote).
PR_AUC_TOLERANCE = 0.003

sns.set_theme(style="whitegrid")


def get_folds(X, y) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    return list(skf.split(X, y))


def _scores(pipe, X) -> tuple[np.ndarray, bool]:
    if hasattr(pipe, "predict_proba") and hasattr(pipe[-1], "predict_proba"):
        return pipe.predict_proba(X)[:, 1], True
    return pipe.decision_function(X), False


def _fit_fold(name, imbalance, kind, params, use_duration, X, y, tr, va, fold):
    warnings.filterwarnings("ignore")
    pipe = make_model(name, imbalance=imbalance, use_duration=use_duration, kind=kind, **(params or {}))
    ytr = y.iloc[tr]
    if needs_fold_pos_weight(name, imbalance):
        pipe.set_params(clf__scale_pos_weight=float((ytr == 0).sum() / (ytr == 1).sum()))
    t0 = time.perf_counter()
    pipe.fit(X.iloc[tr], ytr)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    s, is_proba = _scores(pipe, X.iloc[va])
    pred_ms = 1000 * (time.perf_counter() - t0) / len(va) * 1000
    m = ranking_metrics(y.iloc[va], s, proba=is_proba)
    m.update(fold=fold, fit_time_s=fit_time, predict_ms_per_1k=pred_ms)
    return m, va, s


def cv_evaluate(name, X, y, folds, imbalance="weights", kind=None, params=None,
                use_duration=False, n_jobs=cfg.N_FOLDS) -> tuple[pd.DataFrame, np.ndarray]:
    """Evalúa un modelo con la partición fija; devuelve métricas por fold y scores out-of-fold."""
    res = Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold)(name, imbalance, kind, params, use_duration, X, y, tr, va, i)
        for i, (tr, va) in enumerate(folds))
    oof = np.zeros(len(y))
    rows = []
    for m, va, s in res:
        oof[va] = s
        rows.append(m)
    df = pd.DataFrame(rows)
    df.insert(0, "model", name)
    df.insert(1, "imbalance", imbalance)
    df.insert(2, "kind", kind or MODEL_SPECS[name].kind)
    return df, oof


def summarize(folds_df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    agg = folds_df.groupby(by, sort=False)[METRIC_COLS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    for m in ["roc_auc", "pr_auc", "lift@20", "brier"]:
        agg[f"{m}_fmt"] = agg.apply(
            lambda r: "n/a" if pd.isna(r[f"{m}_mean"]) else f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}", axis=1)
    return agg


def _load_state() -> dict:
    return json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


# --- 1. Comparación de algoritmos ---------------------------------------------------------------

def compare_models(X, y, folds) -> pd.DataFrame:
    all_folds = []
    for name in MODEL_SPECS:
        t0 = time.perf_counter()
        df, _ = cv_evaluate(name, X, y, folds, imbalance="weights")
        all_folds.append(df)
        print(f"  {name:14s} PR-AUC={df.pr_auc.mean():.4f} ROC-AUC={df.roc_auc.mean():.4f} "
              f"({time.perf_counter() - t0:.1f}s)")
    folds_df = pd.concat(all_folds, ignore_index=True)
    folds_df.to_csv(cfg.TABLES_DIR / "cv_results_folds.csv", index=False)
    summ = summarize(folds_df, ["model"])
    summ.insert(1, "label", summ["model"].map(lambda m: MODEL_SPECS[m].label))
    summ.insert(2, "family", summ["model"].map(lambda m: MODEL_SPECS[m].family))
    summ.insert(3, "class_weights", summ["model"].map(lambda m: MODEL_SPECS[m].supports_weights))
    summ = summ.sort_values("pr_auc_mean", ascending=False)
    summ.round(4).to_csv(cfg.TABLES_DIR / "cv_results.csv", index=False)

    order = summ["model"].tolist()
    labels = [MODEL_SPECS[m].label for m in order]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, metric, title in [(axes[0], "pr_auc", "PR-AUC (average precision)"),
                              (axes[1], "roc_auc", "ROC-AUC"), (axes[2], "lift@20", "Lift en el top 20 %")]:
        sns.boxplot(data=folds_df, y="model", x=metric, order=order, ax=ax, color="#9ecae1")
        sns.stripplot(data=folds_df, y="model", x=metric, order=order, ax=ax, color="k", size=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.set_ylabel("")
        ax.set_title(title)
    axes[0].axvline(y.mean(), ls="--", color="r", lw=1)
    fig.suptitle("Comparación de algoritmos — validación cruzada estratificada 5 folds (modelo pre-contacto)")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "model_cv_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return summ


# --- 2. Codificación de categóricas en boosting ---------------------------------------------------

def compare_encoding(X, y, folds) -> dict:
    rows = []
    for name in ["lightgbm", "xgboost"]:
        for kind in ["tree_onehot", "tree_native"]:
            df, _ = cv_evaluate(name, X, y, folds, imbalance="weights", kind=kind)
            rows.append(df)
            print(f"  {name:10s} {kind:12s} PR-AUC={df.pr_auc.mean():.4f}")
    folds_df = pd.concat(rows, ignore_index=True)
    summ = summarize(folds_df, ["model", "kind"])
    summ.round(4).to_csv(cfg.TABLES_DIR / "encoding_comparison.csv", index=False)
    best = {}
    for name, g in summ.groupby("model"):
        g = g.sort_values("pr_auc_mean", ascending=False)
        # Se prefiere nativo si empata (menos columnas, inferencia más simple).
        top = g.iloc[0]
        native = g[g.kind == "tree_native"].iloc[0]
        best[name] = "tree_native" if top["pr_auc_mean"] - native["pr_auc_mean"] < PR_AUC_TOLERANCE else top["kind"]
    return best


# --- 3. Estrategias de desbalance --------------------------------------------------------------------

def compare_imbalance(X, y, folds, models: list[str], kinds: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    for name in models:
        spec = MODEL_SPECS[name]
        for strat in ["none", "weights", "smote"]:
            if strat == "weights" and not spec.supports_weights:
                continue
            kind = None if strat == "smote" else kinds.get(name)
            df, _ = cv_evaluate(name, X, y, folds, imbalance=strat, kind=kind)
            rows.append(df)
            print(f"  {name:14s} {strat:8s} PR-AUC={df.pr_auc.mean():.4f} Brier={df.brier.mean():.4f}")
    folds_df = pd.concat(rows, ignore_index=True)
    folds_df.to_csv(cfg.TABLES_DIR / "imbalance_results_folds.csv", index=False)
    summ = summarize(folds_df, ["model", "imbalance", "kind"])
    summ.round(4).to_csv(cfg.TABLES_DIR / "imbalance_results.csv", index=False)

    choice = {}
    preference = {"none": 0, "weights": 1, "smote": 2}
    for name, g in summ.groupby("model", sort=False):
        best_pr = g["pr_auc_mean"].max()
        ok = g[g["pr_auc_mean"] >= best_pr - PR_AUC_TOLERANCE].copy()
        ok["pref"] = ok["imbalance"].map(preference)
        pick = ok.sort_values(["pref", "pr_auc_std"]).iloc[0]
        choice[name] = {"imbalance": pick["imbalance"], "kind": pick["kind"],
                        "pr_auc_mean": float(pick["pr_auc_mean"])}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = {"none": "Sin tratamiento", "weights": "Pesos de clase", "smote": "SMOTE"}
    sns.barplot(data=folds_df.assign(estrategia=folds_df["imbalance"].map(names)), x="model", y="pr_auc",
                hue="estrategia", ax=ax, errorbar="sd",
                palette={"Sin tratamiento": "#8c9bab", "Pesos de clase": "#337ab7", "SMOTE": "#f0ad4e"})
    ax.set_ylim(folds_df.pr_auc.min() - 0.05, folds_df.pr_auc.max() + 0.03)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_SPECS[m].label for m in models])
    ax.set_title("Estrategias de desbalance: PR-AUC en CV (media ± desvío)")
    ax.set_xlabel("")
    ax.set_ylabel("PR-AUC")
    ax.legend(title="Estrategia")
    fig.savefig(cfg.FIGURES_DIR / "imbalance_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return summ, choice


# --- 4. Tuning con Optuna ------------------------------------------------------------------------------

def tune(name: str, X, y, folds, imbalance: str, kind: str) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=cfg.SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"tune_{name}")
    default_df, _ = cv_evaluate(name, X, y, folds, imbalance=imbalance, kind=kind)

    def objective(trial):
        params = suggest_params(name, trial)
        df, _ = cv_evaluate(name, X, y, folds, imbalance=imbalance, kind=kind, params=params)
        trial.set_user_attr("pr_auc_std", float(df.pr_auc.std()))
        trial.set_user_attr("roc_auc", float(df.roc_auc.mean()))
        trial.set_user_attr("lift@20", float(df["lift@20"].mean()))
        return float(df.pr_auc.mean())

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=cfg.OPTUNA_TRIALS, timeout=cfg.OPTUNA_TIMEOUT_S)
    elapsed = time.perf_counter() - t0
    trials = study.trials_dataframe()
    trials.to_csv(cfg.TABLES_DIR / f"tuning_trials_{name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    vals = trials["value"].to_numpy()
    ax.plot(trials["number"], vals, "o", alpha=0.5, label="PR-AUC del trial")
    ax.plot(trials["number"], np.maximum.accumulate(np.nan_to_num(vals, nan=-1)), "r-", lw=2, label="Mejor hasta el momento")
    ax.axhline(default_df.pr_auc.mean(), color="gray", ls="--", label="Hiperparámetros por defecto")
    ax.set_xlabel("Trial")
    ax.set_ylabel("PR-AUC (CV 5 folds)")
    ax.set_title(f"Convergencia de la búsqueda Optuna (TPE) — {MODEL_SPECS[name].label}")
    ax.legend()
    fig.savefig(cfg.FIGURES_DIR / f"tuning_convergence_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    best = study.best_trial
    return {"model": name, "imbalance": imbalance, "kind": kind, "n_trials": len(study.trials),
            "elapsed_s": round(elapsed, 1), "timeout_s": cfg.OPTUNA_TIMEOUT_S,
            "default_pr_auc": float(default_df.pr_auc.mean()), "best_pr_auc": float(best.value),
            "best_pr_auc_std": best.user_attrs["pr_auc_std"],
            "improvement": float(best.value - default_df.pr_auc.mean()),
            "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.params.items()}}


# --- 5. Selección final ----------------------------------------------------------------------------------

DECISION_WEIGHTS = {"pr_auc": 0.35, "lift@20": 0.25, "stability": 0.15, "interpretability": 0.15,
                    "inference_speed": 0.10}


def decision_matrix(candidates: list[dict], X, y, folds) -> tuple[pd.DataFrame, dict, dict]:
    rows, oofs = [], {}
    for c in candidates:
        df, oof = cv_evaluate(c["model"], X, y, folds, imbalance=c["imbalance"], kind=c["kind"],
                              params=c.get("params"))
        key = c["key"]
        oofs[key] = oof
        rows.append({"candidate": key, "model": c["model"], "label": MODEL_SPECS[c["model"]].label,
                     "imbalance": c["imbalance"], "kind": c["kind"], "tuned": bool(c.get("params")),
                     "pr_auc": df.pr_auc.mean(), "pr_auc_std": df.pr_auc.std(),
                     "roc_auc": df.roc_auc.mean(), "lift@20": df["lift@20"].mean(),
                     "brier": df.brier.mean(), "predict_ms_per_1k": df.predict_ms_per_1k.mean(),
                     "interpretability": MODEL_SPECS[c["model"]].interpretability})
    dm = pd.DataFrame(rows)

    def norm(s, higher_better=True):
        rng = s.max() - s.min()
        n = (s - s.min()) / rng if rng > 0 else pd.Series(1.0, index=s.index)
        return n if higher_better else 1 - n

    dm["score_pr_auc"] = norm(dm["pr_auc"])
    dm["score_lift@20"] = norm(dm["lift@20"])
    dm["score_stability"] = norm(dm["pr_auc_std"], higher_better=False)
    dm["score_interpretability"] = (dm["interpretability"] - 1) / 4
    dm["score_inference_speed"] = norm(np.log(dm["predict_ms_per_1k"]), higher_better=False)
    dm["weighted_score"] = sum(w * dm[f"score_{k}"] for k, w in DECISION_WEIGHTS.items())
    dm = dm.sort_values("weighted_score", ascending=False)
    dm.round(4).to_csv(cfg.TABLES_DIR / "decision_matrix.csv", index=False)
    winner = dm.iloc[0]
    chosen = next(c for c in candidates if c["key"] == winner["candidate"])
    return dm, chosen, oofs


# --- 6. Umbral operativo --------------------------------------------------------------------------------

def choose_threshold(y, oof: np.ndarray) -> dict:
    tc = threshold_curve(y, oof)
    tc.round(5).to_csv(cfg.TABLES_DIR / "threshold_analysis.csv", index=False)
    best_profit = tc.loc[tc["profit"].idxmax()]
    best_f1 = tc.loc[tc["f1"].idxmax()]
    sens = sensitivity_table(y, oof)
    sens.round(4).to_csv(cfg.TABLES_DIR / "sensitivity_oof.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(tc["threshold"], tc["profit"], color="#2c7fb8", lw=2, label="Beneficio esperado (OOF)")
    ax1.set_xlabel("Umbral de probabilidad")
    ax1.set_ylabel("Beneficio esperado (V=20, C=1)")
    ax2 = ax1.twinx()
    ax2.plot(tc["threshold"], tc["f1"], color="#d95f0e", lw=1.5, label="F1")
    ax2.plot(tc["threshold"], tc["recall"], color="#31a354", lw=1, ls=":", label="Recall")
    ax2.plot(tc["threshold"], tc["precision"], color="#756bb1", lw=1, ls=":", label="Precision")
    ax2.set_ylabel("F1 / Recall / Precision")
    ax2.grid(False)
    ax1.axvline(best_profit["threshold"], color="#2c7fb8", ls="--")
    ax1.axvline(best_f1["threshold"], color="#d95f0e", ls="--")
    ax1.axvline(cfg.COST_PER_CONTACT / cfg.VALUE_PER_CONVERSION, color="gray", ls=":")
    ax1.set_xlim(0, 0.8)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    ax1.set_title(f"Elección del umbral con predicciones out-of-fold\n"
                  f"máx. beneficio: {best_profit['threshold']:.3f} · máx. F1: {best_f1['threshold']:.3f} · "
                  f"teórico C/V: {cfg.COST_PER_CONTACT / cfg.VALUE_PER_CONVERSION:.3f}")
    fig.savefig(cfg.FIGURES_DIR / "threshold_selection.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    out = {"profit_threshold": float(best_profit["threshold"]),
           "profit_at_threshold": float(best_profit["profit"]),
           "contacted_pct_at_threshold": float(best_profit["contacted_pct"]),
           "precision_at_threshold": float(best_profit["precision"]),
           "recall_at_threshold": float(best_profit["recall"]),
           "f1_at_threshold": float(best_profit["f1"]),
           "f1_threshold": float(best_f1["threshold"]), "f1_max": float(best_f1["f1"]),
           "profit_at_f1_threshold": float(best_f1["profit"]),
           "contacted_pct_at_f1_threshold": float(best_f1["contacted_pct"]),
           "theoretical_threshold": cfg.COST_PER_CONTACT / cfg.VALUE_PER_CONVERSION,
           "profit_contact_all": float(cfg.VALUE_PER_CONVERSION * y.sum() - cfg.COST_PER_CONTACT * len(y))}
    (cfg.TABLES_DIR / "threshold_choice.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# --- 7. Entrenamiento final y persistencia -----------------------------------------------------------

def fit_full(name, X, y, imbalance, kind, params=None, use_duration=False):
    warnings.filterwarnings("ignore")
    pipe = make_model(name, imbalance=imbalance, use_duration=use_duration, kind=kind, **(params or {}))
    if needs_fold_pos_weight(name, imbalance):
        pipe.set_params(clf__scale_pos_weight=float((y == 0).sum() / (y == 1).sum()))
    return pipe.fit(X, y)


def run_training() -> dict:
    """Ejecuta toda la Fase 4 y guarda el modelo final."""
    cfg.ensure_dirs()
    warnings.filterwarnings("ignore")
    train, _ = load_clean()
    X, y = split_xy(train)
    folds = get_folds(X, y)
    pd.DataFrame([{"fold": i, "n_train": len(tr), "n_valid": len(va),
                   "pos_rate_train": y.iloc[tr].mean(), "pos_rate_valid": y.iloc[va].mean()}
                  for i, (tr, va) in enumerate(folds)]).round(4).to_csv(cfg.TABLES_DIR / "cv_folds.csv", index=False)
    state = {"started": datetime.now(timezone.utc).isoformat()}

    print("[1/7] Comparación de algoritmos")
    summ = compare_models(X, y, folds)
    ranked = [m for m in summ["model"] if m != "dummy"]
    state["cv_ranking"] = ranked

    print("[2/7] Codificación de categóricas en boosting (one-hot vs nativa)")
    kinds = compare_encoding(X, y, folds)
    state["encoding_choice"] = kinds

    print("[3/7] Estrategias de desbalance sobre los 3 mejores")
    top3 = ranked[:3]
    _, imb_choice = compare_imbalance(X, y, folds, top3, kinds)
    state["imbalance_models"] = top3
    state["imbalance_choice"] = imb_choice

    print("[4/7] Tuning con Optuna de los 2 mejores")
    top2 = sorted(imb_choice, key=lambda m: imb_choice[m]["pr_auc_mean"], reverse=True)[:2]
    tuning = []
    for name in top2:
        c = imb_choice[name]
        res = tune(name, X, y, folds, c["imbalance"], c["kind"])
        print(f"  {name}: {res['default_pr_auc']:.4f} → {res['best_pr_auc']:.4f} "
              f"({res['n_trials']} trials, {res['elapsed_s']:.0f}s)")
        tuning.append(res)
    pd.DataFrame([{**{k: v for k, v in t.items() if k != "best_params"},
                   "best_params": json.dumps(t["best_params"])} for t in tuning]) \
        .round(5).to_csv(cfg.TABLES_DIR / "tuning_summary.csv", index=False)
    state["tuning"] = tuning

    print("[5/7] Matriz de decisión")
    candidates = [{"key": f"{t['model']}_tuned", "model": t["model"], "imbalance": t["imbalance"],
                   "kind": t["kind"], "params": t["best_params"]} for t in tuning]
    candidates += [{"key": f"{name}_default", "model": name, "imbalance": imb_choice[name]["imbalance"],
                    "kind": imb_choice[name]["kind"]} for name in top2]
    if "logreg" not in top2:
        candidates.append({"key": "logreg_default", "model": "logreg", "imbalance": "weights", "kind": "linear"})
    dm, chosen, oofs = decision_matrix(candidates, X, y, folds)
    print(dm[["candidate", "pr_auc", "lift@20", "pr_auc_std", "weighted_score"]].round(4).to_string(index=False))
    state["final"] = chosen

    print("[6/7] Umbral operativo (predicciones out-of-fold)")
    oof = oofs[chosen["key"]]
    pd.DataFrame({"y": y, "oof_score": oof}).to_csv(cfg.DATA_PROCESSED / "oof_final.csv", index=False)
    thr = choose_threshold(y, oof)
    state["threshold"] = thr
    print(f"  umbral beneficio={thr['profit_threshold']:.3f}  umbral F1={thr['f1_threshold']:.3f}")

    print("[7/7] Entrenamiento final sobre todo el train y variantes de comparación")
    final = fit_full(chosen["model"], X, y, chosen["imbalance"], chosen["kind"], chosen.get("params"))
    joblib.dump(final, cfg.MODEL_FILE)
    # Variante con duration (solo análisis de leakage): misma configuración + duration.
    dur_df, _ = cv_evaluate(chosen["model"], X, y, folds, imbalance=chosen["imbalance"], kind=chosen["kind"],
                            params=chosen.get("params"), use_duration=True)
    pre_df, _ = cv_evaluate(chosen["model"], X, y, folds, imbalance=chosen["imbalance"], kind=chosen["kind"],
                            params=chosen.get("params"))
    leak = summarize(pd.concat([pre_df.assign(feature_set="pre_contact"),
                                dur_df.assign(feature_set="with_duration")]), ["feature_set"])
    leak.round(4).to_csv(cfg.TABLES_DIR / "leakage_cv.csv", index=False)
    joblib.dump(fit_full(chosen["model"], X, y, chosen["imbalance"], chosen["kind"], chosen.get("params"),
                         use_duration=True), cfg.MODEL_WITH_DURATION_FILE)
    joblib.dump(fit_full("logreg", X, y, "weights", "linear"), cfg.BASELINE_LR_FILE)

    # Cortes de deciles según la distribución OOF (para asignar decil a clientes nuevos en la API).
    decile_cuts = np.quantile(oof, np.linspace(0.1, 0.9, 9)).tolist()
    cv_final = pre_df[METRIC_COLS].mean().to_dict()
    cv_final_std = pre_df[METRIC_COLS].std().to_dict()
    from .data import file_sha256
    import sklearn, lightgbm, xgboost
    metadata = {
        "model_name": chosen["model"], "model_label": MODEL_SPECS[chosen["model"]].label,
        "imbalance_strategy": chosen["imbalance"], "preprocessing": chosen["kind"],
        "hyperparameters": chosen.get("params") or "default",
        "feature_set": "pre_contact (sin duration)",
        "raw_input_columns": [c for c in cfg.EXPECTED_COLUMNS if c not in (cfg.TARGET, cfg.LEAKAGE_COLUMN)],
        "engineered_features": FeatureEngineer().fit(X).feature_names_out_,
        "threshold": thr["profit_threshold"], "threshold_criterion": "máximo beneficio esperado OOF (V=20, C=1)",
        "threshold_f1": thr["f1_threshold"],
        "business_params": {"cost_per_contact": cfg.COST_PER_CONTACT, "value_per_conversion": cfg.VALUE_PER_CONVERSION},
        "decile_cuts_oof": decile_cuts,
        "cv_metrics_mean": cv_final, "cv_metrics_std": cv_final_std,
        "train_rows": int(len(X)), "train_positive_rate": float(y.mean()),
        "data_sha256": {"train": file_sha256(cfg.TRAIN_FILE), "test": file_sha256(cfg.TEST_FILE)},
        "seed": cfg.SEED, "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                     "xgboost": xgboost.__version__},
    }
    cfg.METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    state["finished"] = datetime.now(timezone.utc).isoformat()
    _save_state(state)
    return state


if __name__ == "__main__":
    run_training()
