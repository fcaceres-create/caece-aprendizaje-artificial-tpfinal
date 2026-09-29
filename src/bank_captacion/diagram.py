"""Diagramas del flujo del proyecto y del pipeline del modelo final (PNG con matplotlib).

El Mermaid equivalente está en docs/diagrama_pipeline.md; como no hay Mermaid CLI instalado,
los PNG se dibujan con matplotlib para no agregar dependencias de sistema.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config as cfg

C_DATA, C_PREP, C_MODEL, C_EVAL, C_APP = "#dbe9f6", "#e5f5e0", "#fdebd0", "#f2e1f5", "#fde0dd"


def _box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#444", lw=1.1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, p1, p2, text=None, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, color="#333", lw=1.2,
                                 shrinkA=2, shrinkB=2))
    if text:
        ax.text((p1[0] + p2[0]) / 2 + 0.05, (p1[1] + p2[1]) / 2, text, fontsize=7.5, color="#333",
                ha="left", va="center", style="italic")


def _elbow(ax, p1, p2, x_mid):
    """Flecha ortogonal: sale horizontal, sube/baja por x_mid y entra horizontal."""
    ax.plot([p1[0], x_mid, x_mid], [p1[1], p1[1], p2[1]], color="#333", lw=1.2)
    _arrow(ax, (x_mid, p2[1]), p2)


def project_flow(meta: dict, n: dict) -> None:
    fig, ax = plt.subplots(figsize=(16, 9.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.5)
    ax.axis("off")
    W, H = 3.2, 1.1
    cols = [2.0, 6.0, 10.0, 14.0]
    ys = [8.2, 6.6, 5.0, 3.4, 1.8]
    heads = ["1-3 · Datos y preparación", "4 · Modelado (solo train)", "5 · Evaluación (hold-out)", "6 · Despliegue"]
    for x, h in zip(cols, heads):
        ax.text(x, 9.2, h, ha="center", fontsize=11, weight="bold")
    def num(v):
        return f"{v:,}".replace(",", ".")

    col1 = [(f"banca_train.csv ({num(n['train_raw'])} filas)\nbanca_test.csv ({num(n['test'])} filas)", C_DATA),
            (f"Quitar solapamiento train/test\n−{num(n['overlap'])} filas de train\n"
             f"train limpio = {num(n['train_clean'])}", C_PREP),
            ("EDA (solo train)\nunknown, pdays, month,\noutliers, duration = leakage", C_PREP),
            ("Ingeniería de variables\n(FeatureEngineer)\n24 variables pre-contacto", C_PREP)]
    col2 = [("CV estratificada 5 folds\n10 algoritmos, misma partición", C_MODEL),
            ("Codificación (one-hot vs nativa)\nDesbalance: none / pesos / SMOTE", C_MODEL),
            ("Tuning Optuna (TPE, 60 trials)\nsobre los 2 mejores", C_MODEL),
            (f"Matriz de decisión\n→ {meta['model_label']} tuneado", C_MODEL),
            (f"Umbral con predicciones OOF\nmáx. beneficio (V=20, C=1) = {meta['threshold']:.3f}", C_MODEL)]
    col3 = [("Hold-out: uso único\nROC, PR, deciles, lift,\ncalibración, beneficio", C_EVAL),
            ("Leakage: pre-contacto\nvs con duration", C_EVAL),
            ("Explicabilidad: SHAP global/local,\nlogística, árbol de reglas", C_EVAL),
            ("Errores, segmentos\ny razonabilidad", C_EVAL)]
    col4 = [("model_final.joblib\n+ metadata.json", C_APP), ("API FastAPI\n/predict, /predict/batch", C_APP),
            ("UI Streamlit\ncliente / campaña", C_APP)]
    for x, col in zip(cols, [col1, col2, col3, col4]):
        for i, (t, c) in enumerate(col):
            _box(ax, x, ys[i], W, H, t, c, 8.8)
            if i:
                _arrow(ax, (x, ys[i - 1] - H / 2), (x, ys[i] + H / 2))
    # Conexiones entre columnas por los pasillos.
    _elbow(ax, (cols[0] + W / 2, ys[3]), (cols[1] - W / 2, ys[0]), 4.0)
    _elbow(ax, (cols[1] + W / 2, ys[4]), (cols[2] - W / 2, ys[0]), 8.0)
    _elbow(ax, (cols[2] + W / 2, ys[3]), (cols[3] - W / 2, ys[0]), 12.0)
    ax.text(8.15, 1.0, "El test se lee recién aquí,\ncon el modelo y el umbral ya fijados.", fontsize=8,
            style="italic", ha="left")
    ax.set_title("Flujo completo del proyecto (CRISP-DM)", fontsize=14, pad=18)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#444") for c in (C_DATA, C_PREP, C_MODEL, C_EVAL, C_APP)]
    ax.legend(handles, ["Datos", "Comprensión y preparación", "Modelado (solo train)", "Evaluación (hold-out)",
                        "Despliegue"], loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=5, fontsize=9,
              frameon=False)
    fig.savefig(cfg.FIGURES_DIR / "diagrama_flujo_proyecto.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def model_pipeline(meta: dict) -> None:
    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    steps = [
        ("Cliente (datos crudos)\n15 columnas, sin duration", C_DATA),
        ("FeatureEngineer\nderivadas, winsorización p99,\nlog saldo, banderas, cíclicas", C_PREP),
        ("CategoryCaster\n11 categóricas → category\n(niveles aprendidos en fit)", C_PREP),
        (f"LightGBMClassifier\n{meta['hyperparameters']['n_estimators']} árboles, "
         f"{meta['hyperparameters']['num_leaves']} hojas\nlr {meta['hyperparameters']['learning_rate']:.4f}", C_MODEL),
        (f"Probabilidad → umbral {meta['threshold']:.3f}\n→ llamar / no llamar\n+ decil + top-3 SHAP", C_APP),
    ]
    xs = [1.5, 4.5, 7.5, 10.5, 13.5]
    for (t, c), x in zip(steps, xs):
        _box(ax, x, 1.8, 2.7, 1.5, t, c, 8.5)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, (a + 1.35, 1.8), (b - 1.35, 1.8))
    ax.text(7.5, 0.3, "Todo el pipeline se serializa en models/model_final.joblib: la API recibe datos crudos y "
                      "aplica exactamente las mismas transformaciones que en el entrenamiento.",
            ha="center", fontsize=9, style="italic")
    ax.set_title("Pipeline del modelo final (imblearn.Pipeline)", fontsize=13)
    fig.savefig(cfg.FIGURES_DIR / "diagrama_pipeline_modelo.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_diagrams() -> None:
    cfg.ensure_dirs()
    meta = json.loads(cfg.METADATA_FILE.read_text(encoding="utf-8"))
    info = json.loads((cfg.TABLES_DIR / "data_info.json").read_text(encoding="utf-8"))
    n = {"train_raw": info["train_raw_rows"], "train_clean": info["train_clean_rows"],
         "test": info["test_rows"], "overlap": info["overlap_rows_removed"]}
    project_flow(meta, n)
    model_pipeline(meta)


if __name__ == "__main__":
    render_diagrams()
