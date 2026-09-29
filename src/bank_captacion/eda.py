"""Análisis exploratorio (Fase 2). Genera figuras en reports/figures/eda_*.png y tablas.

El EDA se hace **solo sobre train limpio**: el hold-out no se mira para no sesgar
decisiones de modelado.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from . import config as cfg
from .data import load_clean, split_xy

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {0: "#8c9bab", 1: "#d9534f"}
CLASS_LABEL = {0: "No convierte", 1: "Convierte"}

VAR_LABELS = {
    "age": "Edad", "job": "Ocupación", "marital": "Estado civil", "education": "Educación",
    "default": "Préstamos impagos", "balance": "Saldo promedio anual", "housing": "Préstamo hipotecario",
    "loan": "Préstamo personal", "contact": "Tipo de contacto", "day": "Día del mes",
    "month": "Mes", "duration": "Duración de la llamada (s)", "campaign": "Contactos en la campaña",
    "pdays": "Días desde contacto previo", "previous": "Contactos previos",
    "poutcome": "Resultado campaña previa",
}
MONTH_ORDER = cfg.CATEGORY_LEVELS["month"]


def _save(fig, name: str) -> str:
    path = cfg.FIGURES_DIR / f"eda_{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(cfg.ROOT))


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér con corrección de sesgo (Bergsma, 2013)."""
    table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


# --- Bloques del análisis -----------------------------------------------------------

def target_balance(df: pd.DataFrame) -> dict:
    counts = df[cfg.TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["No convierte", "Convierte"], [counts["no"], counts["yes"]], color=[PALETTE[0], PALETTE[1]])
    for i, v in enumerate([counts["no"], counts["yes"]]):
        ax.text(i, v, f"{v:,}\n({100 * v / len(df):.1f} %)".replace(",", "."), ha="center", va="bottom")
    ax.set_title("Distribución de la variable objetivo (train)")
    ax.set_ylabel("Clientes")
    ax.set_ylim(0, counts.max() * 1.18)
    _save(fig, "target_balance")
    return {"n_train": len(df), "n_pos": int(counts["yes"]), "pos_rate": float(counts["yes"] / len(df))}


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos descriptivos por clase de las variables numéricas."""
    rows = []
    for col in cfg.NUMERIC_RAW:
        for cls, g in df.groupby(cfg.TARGET):
            s = g[col]
            rows.append({"variable": col, "y": cls, "n": len(s), "mean": s.mean(), "std": s.std(),
                         "min": s.min(), "p25": s.quantile(.25), "median": s.median(),
                         "p75": s.quantile(.75), "p99": s.quantile(.99), "max": s.max()})
    out = pd.DataFrame(rows).round(2)
    out.to_csv(cfg.TABLES_DIR / "eda_numeric_by_class.csv", index=False)
    return out


def numeric_distributions(df: pd.DataFrame) -> None:
    d = df.assign(clase=df[cfg.TARGET].map({"no": 0, "yes": 1}))
    cols = ["age", "balance", "campaign", "pdays", "previous", "day"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), cols):
        s = d[col]
        lo, hi = s.quantile(0.005), s.quantile(0.99)
        sub = d[(s >= lo) & (s <= hi)]
        if col == "pdays":
            sub = sub[sub.pdays != -1]
        for cls in (0, 1):
            vals = sub.loc[sub.clase == cls, col]
            ax.hist(vals, bins=40, density=True, alpha=0.55, color=PALETTE[cls], label=CLASS_LABEL[cls])
        title = VAR_LABELS[col] + (" (solo contactados antes)" if col == "pdays" else "")
        ax.set_title(title + "\n(recortado al p0,5–p99 para visualizar)")
        ax.set_ylabel("Densidad")
        ax.legend(fontsize=8)
    fig.suptitle("Distribución de variables numéricas por clase (train)", fontsize=14)
    fig.tight_layout()
    _save(fig, "numeric_distributions")


def conversion_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de conversión y volumen por categoría, para cada variable categórica."""
    base = (df[cfg.TARGET] == "yes").mean()
    rows = []
    for col in cfg.CATEGORICAL_RAW:
        g = df.groupby(col)[cfg.TARGET].agg(n="size", conversions=lambda s: (s == "yes").sum())
        g["conversion_rate"] = g["conversions"] / g["n"]
        g["share"] = g["n"] / len(df)
        g["lift_vs_base"] = g["conversion_rate"] / base
        g = g.reset_index().rename(columns={col: "category"})
        g.insert(0, "variable", col)
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out.round(4).to_csv(cfg.TABLES_DIR / "eda_conversion_by_category.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    for ax, col in zip(axes.ravel(), cfg.CATEGORICAL_RAW):
        sub = out[out.variable == col].copy()
        if col == "month":
            sub["category"] = pd.Categorical(sub["category"], MONTH_ORDER, ordered=True)
            sub = sub.sort_values("category")
        else:
            sub = sub.sort_values("conversion_rate", ascending=False)
        colors = ["#d9534f" if c == "unknown" else "#337ab7" for c in sub["category"].astype(str)]
        ax.bar(sub["category"].astype(str), 100 * sub["conversion_rate"], color=colors)
        ax.axhline(100 * base, ls="--", color="k", lw=1, label=f"Tasa base {100 * base:.1f} %")
        for i, (r, n) in enumerate(zip(sub["conversion_rate"], sub["n"])):
            ax.text(i, 100 * r, f"n={n}", ha="center", va="bottom", fontsize=7, rotation=90 if len(sub) > 6 else 0)
        ax.set_title(f"{VAR_LABELS[col]}")
        ax.set_ylabel("Tasa de conversión (%)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    fig.suptitle("Tasa de conversión por categoría (rojo = 'unknown')", fontsize=15)
    fig.tight_layout()
    _save(fig, "conversion_by_category")
    return out


def unknown_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """¿Es 'unknown' aleatorio? Conversión con/sin unknown y concentración temporal."""
    rows = []
    for col in ["job", "education", "contact", "poutcome"]:
        is_unk = df[col] == "unknown"
        for flag, g in df.groupby(is_unk):
            rows.append({"variable": col, "is_unknown": bool(flag), "n": len(g),
                         "conversion_rate": round(float((g[cfg.TARGET] == "yes").mean()), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.TABLES_DIR / "eda_unknown_analysis.csv", index=False)

    # contact = unknown según la posición en el archivo (proxy temporal: el archivo está ordenado).
    pos_bins = pd.cut(np.arange(len(df)) / len(df), bins=10, labels=[f"{10 * i}-{10 * (i + 1)} %" for i in range(10)])
    by_pos = df.assign(tramo=pos_bins).groupby("tramo", observed=True).agg(
        pct_contact_unknown=("contact", lambda s: 100 * (s == "unknown").mean()),
        pct_poutcome_unknown=("poutcome", lambda s: 100 * (s == "unknown").mean()),
        conversion_rate=(cfg.TARGET, lambda s: 100 * (s == "yes").mean()),
    ).round(2)
    by_pos.to_csv(cfg.TABLES_DIR / "eda_unknown_by_file_position.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    by_pos.plot(ax=ax, marker="o")
    ax.set_title("'unknown' y tasa de conversión según la posición en el archivo\n(el archivo está ordenado cronológicamente: 0 % = mayo 2008, 100 % = noviembre 2010)")
    ax.set_xlabel("Tramo del archivo de train (orden cronológico)")
    ax.set_ylabel("%")
    ax.legend(["% contact = unknown", "% poutcome = unknown", "Tasa de conversión"])
    _save(fig, "unknown_by_file_position")
    return out


def pdays_analysis(df: pd.DataFrame) -> dict:
    """Consistencia entre pdays=-1, previous=0 y poutcome=unknown."""
    never = df.pdays == -1
    tab = pd.crosstab([never.rename("pdays_eq_-1"), (df.previous == 0).rename("previous_eq_0")],
                      df.poutcome, margins=True)
    tab.to_csv(cfg.TABLES_DIR / "eda_pdays_previous_poutcome.csv")
    conv = df.groupby(never.map({True: "Nunca contactado (pdays=-1)", False: "Contactado antes"}))[cfg.TARGET] \
        .apply(lambda s: (s == "yes").mean())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(conv.index, 100 * conv.values, color=["#337ab7", "#8c9bab"])
    axes[0].set_title("Conversión según contacto en campaña previa")
    axes[0].set_ylabel("Tasa de conversión (%)")
    prev = df[~never]
    rate_pout = prev.groupby("poutcome")[cfg.TARGET].apply(lambda s: 100 * (s == "yes").mean())
    axes[1].bar(rate_pout.index, rate_pout.values, color="#337ab7")
    axes[1].set_title("Conversión según resultado previo (solo contactados antes)")
    axes[1].set_ylabel("Tasa de conversión (%)")
    _save(fig, "pdays_poutcome")

    return {
        "pct_pdays_minus1": float(100 * never.mean()),
        "pdays_minus1_and_previous0": int((never & (df.previous == 0)).sum()),
        "pdays_minus1_total": int(never.sum()),
        "pdays_minus1_poutcome_unknown": int((never & (df.poutcome == "unknown")).sum()),
        "contacted_before_poutcome_unknown": int((~never & (df.poutcome == "unknown")).sum()),
        "conv_never_contacted": float(conv["Nunca contactado (pdays=-1)"]),
        "conv_contacted_before": float(conv["Contactado antes"]),
        "conv_poutcome_success": float(rate_pout.get("success", np.nan) / 100),
    }


def month_analysis(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("month")[cfg.TARGET].agg(n="size", conversions=lambda s: (s == "yes").sum())
    g = g.reindex(MONTH_ORDER)
    g["conversion_rate"] = g["conversions"] / g["n"]
    g["share_pct"] = 100 * g["n"] / g["n"].sum()
    g.round(4).to_csv(cfg.TABLES_DIR / "eda_month_volume_rate.csv")

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.bar(g.index, g["n"], color="#c6d4e1", label="Contactos (volumen)")
    ax1.set_ylabel("Cantidad de contactos")
    ax2 = ax1.twinx()
    ax2.plot(g.index, 100 * g["conversion_rate"], color="#d9534f", marker="o", lw=2, label="Tasa de conversión")
    ax2.set_ylabel("Tasa de conversión (%)")
    ax2.grid(False)
    ax1.set_title("Volumen de contactos vs tasa de conversión por mes (sin año)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    _save(fig, "month_volume_rate")

    corr = stats.spearmanr(g["n"], g["conversion_rate"])
    g.attrs["spearman_volume_rate"] = (float(corr.statistic), float(corr.pvalue))
    return g


def outliers_analysis(df: pd.DataFrame) -> dict:
    y = (df[cfg.TARGET] == "yes").astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    sns.boxplot(x=df[cfg.TARGET], y=df["balance"], ax=axes[0], hue=df[cfg.TARGET],
                palette={"no": PALETTE[0], "yes": PALETTE[1]}, legend=False)
    axes[0].set_title("Saldo promedio anual por clase")
    axes[0].set_xlabel("¿Convierte?")
    axes[0].set_ylabel("Saldo")
    sns.boxplot(x=df[cfg.TARGET], y=df["campaign"], ax=axes[1], hue=df[cfg.TARGET],
                palette={"no": PALETTE[0], "yes": PALETTE[1]}, legend=False)
    axes[1].set_title("Contactos en esta campaña por clase")
    axes[1].set_xlabel("¿Convierte?")
    axes[1].set_ylabel("Contactos")
    camp = df["campaign"].clip(upper=11).replace(11, 11)
    rate = y.groupby(camp).agg(["mean", "size"])
    labels = [str(i) for i in rate.index[:-1]] + ["11+"]
    axes[2].bar(labels, 100 * rate["mean"], color="#337ab7")
    for i, n in enumerate(rate["size"]):
        axes[2].text(i, 100 * rate["mean"].iloc[i], f"n={n}", ha="center", va="bottom", fontsize=7, rotation=90)
    axes[2].set_title("Conversión según cantidad de contactos")
    axes[2].set_xlabel("Contactos en la campaña")
    axes[2].set_ylabel("Tasa de conversión (%)")
    _save(fig, "outliers_balance_campaign")

    iqr = df.balance.quantile(.75) - df.balance.quantile(.25)
    return {
        "balance_p99": float(df.balance.quantile(.99)),
        "balance_max": int(df.balance.max()),
        "balance_min": int(df.balance.min()),
        "balance_pct_negative": float(100 * (df.balance < 0).mean()),
        "balance_pct_zero": float(100 * (df.balance == 0).mean()),
        "balance_pct_outliers_iqr": float(100 * ((df.balance > df.balance.quantile(.75) + 1.5 * iqr) |
                                                 (df.balance < df.balance.quantile(.25) - 1.5 * iqr)).mean()),
        "balance_skew": float(df.balance.skew()),
        "campaign_p99": float(df.campaign.quantile(.99)),
        "campaign_max": int(df.campaign.max()),
        "campaign_pct_above_p99": float(100 * (df.campaign > df.campaign.quantile(.99)).mean()),
        "conv_campaign_1": float(y[df.campaign == 1].mean()),
        "conv_campaign_ge_5": float(y[df.campaign >= 5].mean()),
    }


def duration_analysis(df: pd.DataFrame) -> dict:
    """Evidencia de por qué duration es leakage."""
    y = (df[cfg.TARGET] == "yes").astype(int)
    dec = pd.qcut(df.duration, 10, duplicates="drop")
    rate = y.groupby(dec, observed=True).mean()
    rate.rename("conversion_rate").to_frame().to_csv(cfg.TABLES_DIR / "eda_duration_deciles.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for cls in (0, 1):
        vals = df.loc[y == cls, "duration"].clip(upper=df.duration.quantile(.99))
        axes[0].hist(vals, bins=50, density=True, alpha=0.55, color=PALETTE[cls], label=CLASS_LABEL[cls])
    axes[0].set_title("Duración de la llamada por clase")
    axes[0].set_xlabel("Segundos (recortado al p99)")
    axes[0].set_ylabel("Densidad")
    axes[0].legend()
    axes[1].bar(range(len(rate)), 100 * rate.values, color="#d9534f")
    axes[1].set_xticks(range(len(rate)))
    axes[1].set_xticklabels([f"{int(i.left)}-{int(i.right)}" for i in rate.index], rotation=45)
    axes[1].set_title("Tasa de conversión por decil de duración")
    axes[1].set_xlabel("Duración (s)")
    axes[1].set_ylabel("Tasa de conversión (%)")
    _save(fig, "duration_leakage")

    from sklearn.metrics import roc_auc_score
    return {
        "duration_zero_n": int((df.duration == 0).sum()),
        "duration_zero_conv": float(y[df.duration == 0].mean()) if (df.duration == 0).any() else None,
        "duration_median_no": float(df.loc[y == 0, "duration"].median()),
        "duration_median_yes": float(df.loc[y == 1, "duration"].median()),
        "duration_auc_alone": float(roc_auc_score(y, df.duration)),
        "conv_duration_top_decile": float(rate.iloc[-1]),
        "conv_duration_bottom_decile": float(rate.iloc[0]),
        "pct_short_calls_lt_60s": float(100 * (df.duration < 60).mean()),
        "conv_short_calls_lt_60s": float(y[df.duration < 60].mean()),
    }


def feature_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Cramér's V (categóricas y numéricas discretizadas en deciles) + mutual information."""
    X, y = split_xy(df)
    rows = []
    for col in X.columns:
        if col in cfg.CATEGORICAL_RAW:
            v = cramers_v(X[col], y)
        else:
            v = cramers_v(pd.qcut(X[col], 10, duplicates="drop").astype(str), y)
        rows.append({"variable": col, "type": "categórica" if col in cfg.CATEGORICAL_RAW else "numérica",
                     "cramers_v": v})
    Xmi = X.copy()
    for col in cfg.CATEGORICAL_RAW:
        Xmi[col] = Xmi[col].astype("category").cat.codes
    discrete = [c in cfg.CATEGORICAL_RAW for c in Xmi.columns]
    mi = mutual_info_classif(Xmi, y, discrete_features=discrete, random_state=cfg.SEED)
    out = pd.DataFrame(rows)
    out["mutual_info"] = mi
    out["is_leakage"] = out["variable"] == cfg.LEAKAGE_COLUMN
    out = out.sort_values("mutual_info", ascending=False).round(4)
    out.to_csv(cfg.TABLES_DIR / "eda_feature_ranking.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, title in [(axes[0], "mutual_info", "Información mutua con el target"),
                              (axes[1], "cramers_v", "V de Cramér con el target")]:
        s = out.sort_values(metric)
        colors = ["#d9534f" if v == cfg.LEAKAGE_COLUMN else "#337ab7" for v in s["variable"]]
        ax.barh([VAR_LABELS[v] for v in s["variable"]], s[metric], color=colors)
        ax.set_title(title + "\n(rojo = leakage, no se usa)")
        ax.set_xlabel(metric.replace("_", " "))
    fig.tight_layout()
    _save(fig, "feature_ranking")
    return out


def numeric_correlation(df: pd.DataFrame) -> None:
    corr = df[cfg.NUMERIC_RAW].corr(method="spearman")
    corr.round(3).to_csv(cfg.TABLES_DIR / "eda_numeric_correlation.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                xticklabels=[VAR_LABELS[c] for c in corr.columns],
                yticklabels=[VAR_LABELS[c] for c in corr.columns])
    ax.set_title("Correlación de Spearman entre variables numéricas")
    _save(fig, "numeric_correlation")


def run_eda() -> dict:
    """Ejecuta todo el EDA y devuelve los números clave (también se guardan en JSON)."""
    cfg.ensure_dirs()
    train, _ = load_clean()
    key = {"target": target_balance(train)}
    numeric_summary(train)
    numeric_distributions(train)
    conversion_by_category(train)
    unknown_analysis(train)
    key["pdays"] = pdays_analysis(train)
    m = month_analysis(train)
    key["month"] = {
        "spearman_volume_vs_rate": m.attrs["spearman_volume_rate"],
        "top_rate_months": m.sort_values("conversion_rate", ascending=False).head(4)[["n", "conversion_rate"]].round(4).to_dict("index"),
        "may_share_pct": float(m.loc["may", "share_pct"]),
        "may_rate": float(m.loc["may", "conversion_rate"]),
    }
    key["outliers"] = outliers_analysis(train)
    key["duration"] = duration_analysis(train)
    fr = feature_ranking(train)
    key["feature_ranking_top"] = fr.head(8)[["variable", "mutual_info", "cramers_v"]].to_dict("records")
    numeric_correlation(train)
    (cfg.TABLES_DIR / "eda_key_numbers.json").write_text(json.dumps(key, indent=2, default=float), encoding="utf-8")
    return key


if __name__ == "__main__":
    print(json.dumps(run_eda(), indent=2, default=float))
