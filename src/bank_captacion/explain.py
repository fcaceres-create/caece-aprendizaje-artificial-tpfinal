"""Explicabilidad (Fase 5): SHAP global y local, coeficientes de la logística y árbol de reglas.

Los valores SHAP de LightGBM están en escala de **log-odds** (antes de la sigmoide): un valor
positivo aumenta la probabilidad de conversión y uno negativo la reduce.
"""

from __future__ import annotations

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from . import config as cfg
from .data import load_clean, split_xy
from .features import CATEGORICAL, build_pipeline
from .inference import FEATURE_LABELS, _fmt_value, transform_for_model

sns.set_theme(style="whitegrid")


def _savefig(fig, name):
    fig.savefig(cfg.FIGURES_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def shap_analysis(pipe, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict:
    Z = transform_for_model(pipe, X)
    explainer = shap.TreeExplainer(pipe[-1])
    sv = explainer.shap_values(Z)
    if isinstance(sv, list):
        sv = sv[1]
    base = explainer.expected_value
    base = float(base[1] if np.ndim(base) else base)
    names = list(Z.columns)
    Zcodes = Z.copy()
    for c in CATEGORICAL:
        Zcodes[c] = Zcodes[c].cat.codes.replace(-1, np.nan)
    labels = [FEATURE_LABELS.get(n, n) for n in names]

    imp = pd.DataFrame({"feature": names, "label": labels, "mean_abs_shap": np.abs(sv).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False)
    imp["share_pct"] = 100 * imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp.round(5).to_csv(cfg.TABLES_DIR / "shap_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(sv, Zcodes.values, feature_names=labels, show=False, max_display=15)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.title("SHAP global — impacto de cada variable en el log-odds de conversión (test)\n"
              "(color = valor de la variable; categóricas codificadas por orden alfabético)", fontsize=10)
    _savefig(fig, "shap_summary.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    top = imp.head(15).iloc[::-1]
    ax.barh(top["label"], top["mean_abs_shap"], color="#2c7fb8")
    ax.set(title="Importancia global (media de |SHAP|, test)", xlabel="Media |SHAP| (log-odds)")
    _savefig(fig, "shap_importance.png")

    # Dependencia de las 5 variables más importantes.
    top5 = imp["feature"].head(5).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, f in zip(axes.ravel(), top5):
        j = names.index(f)
        if f in CATEGORICAL:
            d = pd.DataFrame({"cat": Z[f].astype(str), "shap": sv[:, j]})
            order = d.groupby("cat")["shap"].median().sort_values().index
            sns.boxplot(data=d, x="cat", y="shap", order=order, ax=ax, color="#9ecae1", fliersize=1)
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel(FEATURE_LABELS.get(f, f))
        else:
            ax.scatter(Z[f], sv[:, j], s=6, alpha=0.4, c=y.values, cmap="coolwarm")
            ax.set_xlabel(FEATURE_LABELS.get(f, f))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("SHAP (log-odds)")
        ax.set_title(f"Dependencia: {FEATURE_LABELS.get(f, f)}")
    axes.ravel()[-1].axis("off")
    fig.suptitle("SHAP — efecto de las 5 variables principales (puntos: rojo = convirtió, azul = no)")
    fig.tight_layout()
    _savefig(fig, "shap_dependence.png")

    # Casos locales: TP típico, FP con mayor score y FN con menor score.
    p = pipe.predict_proba(X)[:, 1]
    pred = p >= threshold
    yv = y.values
    tp_idx = np.flatnonzero(pred & (yv == 1))
    fp_idx = np.flatnonzero(pred & (yv == 0))
    fn_idx = np.flatnonzero(~pred & (yv == 1))
    cases = {
        "tp": tp_idx[np.argsort(p[tp_idx])[len(tp_idx) // 2]],   # TP de probabilidad mediana
        "fp": fp_idx[np.argmax(p[fp_idx])],                        # FP con mayor probabilidad
        "fn": fn_idx[np.argmin(p[fn_idx])],                        # FN con menor probabilidad
    }
    titles = {"tp": "Verdadero positivo", "fp": "Falso positivo", "fn": "Falso negativo"}
    md = ["# Explicaciones locales (SHAP)\n",
          f"Valor base (log-odds medio del modelo): {base:.3f} → probabilidad {1 / (1 + np.exp(-base)):.3f}. "
          f"Umbral operativo: {threshold:.3f}.\n"]
    local_rows = []
    for key, i in cases.items():
        expl = shap.Explanation(values=sv[i], base_values=base,
                                data=np.array([_fmt_value(v) for v in Z.iloc[i]], dtype=object),
                                feature_names=labels)
        plt.figure()
        shap.plots.waterfall(expl, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(9, 6)
        plt.title(f"{titles[key]} — prob. predicha {p[i]:.3f}, resultado real: "
                  f"{'convirtió' if yv[i] == 1 else 'no convirtió'}", fontsize=10)
        _savefig(fig, f"shap_local_{key}.png")
        order = np.argsort(-np.abs(sv[i]))
        ups = [(labels[j], _fmt_value(Z.iloc[i, j]), sv[i, j]) for j in order if sv[i, j] > 0][:3]
        downs = [(labels[j], _fmt_value(Z.iloc[i, j]), sv[i, j]) for j in order if sv[i, j] < 0][:3]
        raw = X.iloc[i]
        md.append(f"## {titles[key]} (fila {int(X.index[i])} del test)\n")
        md.append(f"- Cliente: {raw['age']} años, ocupación `{raw['job']}`, estado civil `{raw['marital']}`, "
                  f"educación `{raw['education']}`, saldo {raw['balance']}, hipoteca `{raw['housing']}`, "
                  f"préstamo `{raw['loan']}`, contacto `{raw['contact']}`, mes `{raw['month']}`, "
                  f"contactos en campaña {raw['campaign']}, resultado previo `{raw['poutcome']}`.")
        md.append(f"- Probabilidad predicha: **{p[i]:.3f}** (umbral {threshold:.3f}) → "
                  f"{'llamar' if pred[i] else 'no llamar'}. Resultado real: "
                  f"**{'convirtió' if yv[i] == 1 else 'no convirtió'}**.")
        md.append("- Factores que más **aumentan** la probabilidad: " +
                  (", ".join(f"{n} = {v} (+{s:.2f})" for n, v, s in ups) or "ninguno relevante") + ".")
        md.append("- Factores que más la **reducen**: " +
                  (", ".join(f"{n} = {v} ({s:.2f})" for n, v, s in downs) or "ninguno relevante") + ".\n")
        local_rows.append({"case": key, "test_row": int(X.index[i]), "proba": p[i], "y": int(yv[i])})
    (cfg.TABLES_DIR / "shap_local_cases.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(local_rows).round(4).to_csv(cfg.TABLES_DIR / "shap_local_cases.csv", index=False)
    return {"top5": top5, "base_value": base, "importance": imp.head(10)[["feature", "share_pct"]]
            .round(2).to_dict("records")}


def logreg_coefficients(lr_pipe) -> pd.DataFrame:
    prep = lr_pipe.named_steps["prep"]
    names = prep.get_feature_names_out()
    coefs = lr_pipe.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"feature": names, "coef": coefs, "odds_ratio": np.exp(coefs),
                       "abs_coef": np.abs(coefs)}).sort_values("abs_coef", ascending=False)
    df.round(4).to_csv(cfg.TABLES_DIR / "logreg_coefficients.csv", index=False)
    top = df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["coef"], color=["#d9534f" if c < 0 else "#31a354" for c in top["coef"]])
    ax.axvline(0, color="k", lw=0.7)
    ax.set(title="Regresión logística: 20 coeficientes de mayor magnitud\n"
                 "(numéricas estandarizadas; categóricas one-hot; verde = aumenta la conversión)",
           xlabel="Coeficiente (log-odds)")
    _savefig(fig, "logreg_coefficients.png")
    return df


def rules_tree(X: pd.DataFrame, y: pd.Series, Xte: pd.DataFrame, yte: pd.Series) -> dict:
    """Árbol de profundidad 3 entrenado con train: reglas de negocio legibles."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    pipe = build_pipeline(DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=cfg.SEED),
                          "tree_onehot").fit(X, y)
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    tree = pipe.named_steps["clf"]
    txt = export_text(tree, feature_names=names, show_weights=True)
    (cfg.TABLES_DIR / "rules_tree_depth3.txt").write_text(txt, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(tree, feature_names=names, class_names=["No convierte", "Convierte"], filled=True,
              proportion=True, rounded=True, fontsize=9, ax=ax, impurity=False)
    ax.set_title("Árbol de reglas de negocio (profundidad 3, entrenado en train) — "
                 "'value' = proporción [no convierte, convierte]")
    _savefig(fig, "rules_tree_depth3.png")

    # Tabla de hojas con la regla completa.
    t = tree.tree_
    base = y.mean()
    leaves = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            n = int(t.n_node_samples[node])
            rate = float(t.value[node][0][1] / t.value[node][0].sum())
            leaves.append({"rule": " Y ".join(conds), "n_train": n, "share_pct": 100 * n / len(y),
                           "conversion_rate": rate, "lift": rate / base})
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [_cond(f, thr, True)])
        walk(t.children_right[node], conds + [_cond(f, thr, False)])

    walk(0, [])
    lv = pd.DataFrame(leaves).sort_values("conversion_rate", ascending=False)
    lv.round(4).to_csv(cfg.TABLES_DIR / "rules_tree_leaves.csv", index=False)
    s = pipe.predict_proba(Xte)[:, 1]
    return {"test_roc_auc": float(roc_auc_score(yte, s)), "test_pr_auc": float(average_precision_score(yte, s))}


def _cond(f: str, thr: float, left: bool) -> str:
    # Variables one-hot (0/1): "x <= 0.5" significa "no es esa categoría".
    for c in CATEGORICAL:
        if f.startswith(c + "_") and abs(thr - 0.5) < 1e-9:
            lvl = f[len(c) + 1:]
            if c in ("default", "housing", "loan"):
                other = "yes" if lvl == "no" else "no"
                return f"{c} = {other}" if left else f"{c} = {lvl}"
            return f"{c} != {lvl}" if left else f"{c} = {lvl}"
    if abs(thr - 0.5) < 1e-9 and f in {"previously_contacted", "prev_campaign_success", "contact_known",
                                         "balance_negative", "balance_zero"}:
        return f"{f} = 0" if left else f"{f} = 1"
    return f"{f} <= {thr:.3f}" if left else f"{f} > {thr:.3f}"


def run_explain() -> dict:
    cfg.ensure_dirs()
    warnings.filterwarnings("ignore")
    meta = json.loads(cfg.METADATA_FILE.read_text(encoding="utf-8"))
    train, test = load_clean()
    Xtr, ytr = split_xy(train)
    Xte, yte = split_xy(test)
    final = joblib.load(cfg.MODEL_FILE)
    out = {"shap": shap_analysis(final, Xte, yte, meta["threshold"])}
    lr = joblib.load(cfg.BASELINE_LR_FILE)
    coefs = logreg_coefficients(lr)
    out["logreg_top"] = coefs.head(10)[["feature", "coef", "odds_ratio"]].round(3).to_dict("records")
    out["rules_tree"] = rules_tree(Xtr, ytr, Xte, yte)
    (cfg.TABLES_DIR / "explain_summary.json").write_text(json.dumps(out, indent=2, default=float,
                                                                    ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(run_explain(), indent=2, default=float, ensure_ascii=False))
