"""Configuración central del proyecto: rutas, semilla, columnas y parámetros de negocio.

Todas las rutas se construyen con pathlib para que el proyecto funcione igual en
Windows y en macOS.
"""

from __future__ import annotations

import sys
from pathlib import Path

# La consola de Windows usa cp1252 por defecto: se fuerza UTF-8 para los mensajes en español.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# --- Rutas -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
DOCS_DIR = ROOT / "docs"
NOTEBOOKS_DIR = ROOT / "notebooks"

TRAIN_FILE = DATA_RAW / "banca_train.csv"
TEST_FILE = DATA_RAW / "banca_test.csv"
TRAIN_CLEAN_FILE = DATA_PROCESSED / "train_clean.csv"
TEST_CLEAN_FILE = DATA_PROCESSED / "test_clean.csv"

MODEL_FILE = MODELS_DIR / "model_final.joblib"
MODEL_WITH_DURATION_FILE = MODELS_DIR / "model_with_duration.joblib"
BASELINE_LR_FILE = MODELS_DIR / "baseline_logreg.joblib"
METADATA_FILE = MODELS_DIR / "metadata.json"

# --- Formato de los CSV originales (verificado en la Fase 0) -------------------
CSV_SEP = ";"
CSV_ENCODING = "ascii"

# --- Reproducibilidad -------------------------------------------------------
SEED = 42
N_FOLDS = 5

# --- Esquema ----------------------------------------------------------------
TARGET = "y"
TARGET_MAP = {"no": 0, "yes": 1}

NUMERIC_RAW = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_RAW = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]
EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome", "y",
]
# Variable que solo se conoce después de la llamada: fuga de información.
LEAKAGE_COLUMN = "duration"

# Niveles válidos observados en los datos (se usan para validar la API).
CATEGORY_LEVELS = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["failure", "other", "success", "unknown"],
}

# --- Parámetros de negocio (supuestos, ver docs/01_negocio.md) ---------------
COST_PER_CONTACT = 1.0      # C: costo de contactar a un cliente
VALUE_PER_CONVERSION = 20.0  # V: valor esperado de una conversión
VC_RATIOS = [5, 10, 20, 50]  # sensibilidad del ratio V/C
TOP_FRACTIONS = [0.10, 0.20, 0.30]

# --- Presupuesto de cómputo ---------------------------------------------------
OPTUNA_TRIALS = 60
OPTUNA_TIMEOUT_S = 900       # 15 minutos por búsqueda como máximo
SVM_TRAIN_SUBSAMPLE = 10_000  # SVM RBF entrenado sobre submuestra estratificada


def ensure_dirs() -> None:
    """Crea las carpetas de salida si no existen."""
    for d in (DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, TABLES_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)
