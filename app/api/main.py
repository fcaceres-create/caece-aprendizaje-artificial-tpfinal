"""API REST del modelo de captación (FastAPI).

Ejecutar:  python -m uvicorn app.api.main:app --port 8000
Documentación interactiva: http://localhost:8000/docs
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from bank_captacion import config as cfg
from bank_captacion.inference import (INPUT_COLUMNS, assign_decile, load_model, score_frame,
                                      top_contributions)

from .schemas import ClientIn, HealthOut, PredictionOut

MODEL_PATH = Path(os.getenv("MODEL_PATH", cfg.MODEL_FILE))
METADATA_PATH = Path(os.getenv("METADATA_PATH", cfg.METADATA_FILE))
MAX_BATCH_ROWS = 200_000
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["model"], STATE["meta"] = load_model(MODEL_PATH, METADATA_PATH)
    yield
    STATE.clear()


app = FastAPI(
    title="API de captación de clientes",
    description="Modelo pre-contacto (LightGBM) que estima la probabilidad de que un cliente contactado "
                "por la campaña se convierta. TP Final Aprendizaje Artificial — CAECE.",
    version="1.0.0",
    lifespan=lifespan,
)


def _explain_text(prob: float, thr: float, decile: int, factors: list[dict]) -> str:
    ups = [f"{f['label']} = {f['value']}" for f in factors if f["effect"] == "aumenta"]
    downs = [f"{f['label']} = {f['value']}" for f in factors if f["effect"] == "reduce"]
    parts = [f"Probabilidad estimada de conversión: {100 * prob:.1f} % (decil {decile}); "
             f"{'supera' if prob >= thr else 'no supera'} el umbral operativo de {100 * thr:.1f} %."]
    if ups:
        parts.append("La aumentan: " + ", ".join(ups) + ".")
    if downs:
        parts.append("La reducen: " + ", ".join(downs) + ".")
    return " ".join(parts)


@app.get("/health", response_model=HealthOut, tags=["estado"])
def health():
    loaded = "model" in STATE
    return HealthOut(status="ok" if loaded else "sin modelo", model_loaded=loaded,
                     model_name=STATE.get("meta", {}).get("model_label"))


@app.get("/model/info", tags=["estado"])
def model_info():
    """Metadata del modelo: hiperparámetros, métricas de CV, umbral, hash de los datos."""
    return STATE["meta"]


@app.post("/predict", response_model=PredictionOut, tags=["predicción"])
def predict(client: ClientIn):
    """Probabilidad, recomendación, decil y las tres variables con mayor contribución SHAP."""
    model, meta = STATE["model"], STATE["meta"]
    X = pd.DataFrame([client.model_dump()])[INPUT_COLUMNS]
    prob = float(model.predict_proba(X)[0, 1])
    thr = float(meta["threshold"])
    decile = int(assign_decile([prob], meta["decile_cuts_oof"])[0])
    factors = top_contributions(model, X, k=3)[0]
    return PredictionOut(probability=round(prob, 5), contact_recommended=prob >= thr,
                         recommendation="llamar" if prob >= thr else "no llamar", threshold=thr,
                         decile=decile, top_factors=factors,
                         explanation=_explain_text(prob, thr, decile, factors))


def _validate_batch(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        return [f"Faltan columnas: {missing}"]
    for col, levels in cfg.CATEGORY_LEVELS.items():
        bad = sorted(set(df[col].astype(str)) - set(levels))
        if bad:
            errors.append(f"Valores no válidos en '{col}': {bad[:5]}")
    for col in ["age", "balance", "day", "campaign", "pdays", "previous"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"La columna '{col}' debe ser numérica")
    if not errors:
        checks = {"age": (18, 100), "day": (1, 31), "campaign": (1, 100), "pdays": (-1, 999), "previous": (0, 300)}
        for col, (lo, hi) in checks.items():
            n = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n:
                errors.append(f"'{col}' fuera de rango [{lo}, {hi}] en {n} filas")
    return errors


@app.post("/predict/batch", tags=["predicción"])
async def predict_batch(file: UploadFile = File(..., description="CSV con una fila por cliente (separador ',' o ';')")):
    """Recibe un CSV y devuelve el mismo CSV rankeado con probabilidad, decil y recomendación.

    Las columnas `duration` e `y` (si vienen) se ignoran: el modelo es pre-contacto. Se informan en el
    encabezado `X-Ignored-Columns`.
    """
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
        sep = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {exc}") from exc
    if len(df) == 0 or len(df) > MAX_BATCH_ROWS:
        raise HTTPException(status_code=400, detail=f"El CSV debe tener entre 1 y {MAX_BATCH_ROWS} filas")
    errors = _validate_batch(df)
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    ignored = [c for c in (cfg.LEAKAGE_COLUMN, cfg.TARGET) if c in df.columns]
    scored = score_frame(STATE["model"], STATE["meta"], df.drop(columns=[cfg.LEAKAGE_COLUMN], errors="ignore"))
    buf = io.StringIO()
    scored.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=ranking_clientes.csv",
                                      "X-Ignored-Columns": ",".join(ignored)})
