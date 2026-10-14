"""Genera los datos estáticos del frontend web (app/web/data).

- ``summary.json``: métricas del hold-out y del modelo, leídas de reports/tables (nada escrito a mano).
- ``clientes_ejemplo.csv``: 400 clientes del test (dataset público UCI) para probar la pestaña Campaña.
  Incluye ``y`` para poder mostrar cuántas conversiones reales captura el ranking.
"""

from __future__ import annotations

import json

import pandas as pd

from . import config as cfg

WEB_DATA = cfg.ROOT / "app" / "web" / "data"
SAMPLE_ROWS = 400


def build_web_assets() -> str:
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    tm = pd.read_csv(cfg.TABLES_DIR / "test_metrics.csv").set_index("model")
    dec = pd.read_csv(cfg.TABLES_DIR / "test_deciles.csv")
    meta = json.loads(cfg.METADATA_FILE.read_text(encoding="utf-8"))
    info = json.loads((cfg.TABLES_DIR / "data_info.json").read_text(encoding="utf-8"))

    def row(m):
        r = tm.loc[m]
        return {"label": r["label"], "roc_auc": round(float(r["roc_auc"]), 3), "pr_auc": round(float(r["pr_auc"]), 3),
                "lift10": round(float(r["lift@10"]), 2), "gain20": round(float(r["gain@20"]), 3),
                "brier": round(float(r["brier"]), 3)}

    summary = {
        "model_label": meta["model_label"], "threshold": meta["threshold"],
        "business_params": meta["business_params"],
        "train_rows": info["train_clean_rows"], "test_rows": info["test_rows"],
        "overlap_removed": info["overlap_rows_removed"],
        "test": {m: row(m) for m in ["final", "logreg", "dummy", "with_duration"]},
        "gain_by_decile": [round(float(g), 1) for g in dec["cum_gain_pct"]],
        "cv": {"roc_auc": round(meta["cv_metrics_mean"]["roc_auc"], 3), "pr_auc": round(meta["cv_metrics_mean"]["pr_auc"], 3)},
    }
    (WEB_DATA / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    test = pd.read_csv(cfg.TEST_CLEAN_FILE)
    sample = test.sample(n=SAMPLE_ROWS, random_state=cfg.SEED).drop(columns=[cfg.LEAKAGE_COLUMN])
    sample.to_csv(WEB_DATA / "clientes_ejemplo.csv", index=False)
    return str(WEB_DATA.relative_to(cfg.ROOT))


if __name__ == "__main__":
    print(build_web_assets())
