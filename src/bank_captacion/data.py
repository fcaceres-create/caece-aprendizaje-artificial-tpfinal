"""Carga, validación de esquema, perfil de datos y limpieza del solapamiento train/test.

Cada transformación que se aplica a los datos queda registrada en
``reports/tables/data_preparation_steps.csv`` (filas antes/después y motivo), para
que se pueda explicar exactamente qué se hizo con el dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg


# --- Carga y validación ----------------------------------------------------------

def file_sha256(path: Path) -> str:
    """Hash SHA-256 de un archivo (para trazar qué datos se usaron)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    """Lee un CSV original con el separador y encoding verificados."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Copiar banca_train.csv y banca_test.csv en data/raw/."
        )
    df = pd.read_csv(path, sep=cfg.CSV_SEP, encoding=cfg.CSV_ENCODING)
    validate_schema(df, path.name)
    return df


def validate_schema(df: pd.DataFrame, name: str = "df") -> None:
    """Verifica columnas, tipos, niveles categóricos y codificación del target."""
    if list(df.columns) != cfg.EXPECTED_COLUMNS:
        raise ValueError(f"{name}: columnas inesperadas {list(df.columns)}")
    for col in cfg.NUMERIC_RAW:
        if not pd.api.types.is_integer_dtype(df[col]):
            raise ValueError(f"{name}: la columna {col} no es entera ({df[col].dtype})")
    for col, levels in cfg.CATEGORY_LEVELS.items():
        extra = set(df[col].unique()) - set(levels)
        if extra:
            raise ValueError(f"{name}: niveles no esperados en {col}: {extra}")
    extra_y = set(df[cfg.TARGET].unique()) - set(cfg.TARGET_MAP)
    if extra_y:
        raise ValueError(f"{name}: codificación del target inesperada: {extra_y}")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features y target (target codificado como 0/1)."""
    y = df[cfg.TARGET].map(cfg.TARGET_MAP).astype(int)
    X = df.drop(columns=[cfg.TARGET])
    return X, y


# --- Perfil -------------------------------------------------------------------------

def profile(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Perfil por columna: tipo, nulos, cardinalidad, % unknown, rango."""
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "dataset": dataset,
            "column": col,
            "dtype": str(s.dtype),
            "n_rows": len(s),
            "n_null": int(s.isna().sum()),
            "n_unique": int(s.nunique()),
            "pct_unknown": round(100 * float((s == "unknown").mean()), 2) if not is_num else 0.0,
            "min": s.min() if is_num else None,
            "max": s.max() if is_num else None,
            "mean": round(float(s.mean()), 3) if is_num else None,
            "top_value": s.mode().iloc[0],
        })
    return pd.DataFrame(rows)


def find_overlap(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Devuelve una máscara booleana sobre train: True si la fila aparece idéntica en test."""
    keys = test.drop_duplicates().assign(_in_test=True)
    merged = train.merge(keys, how="left", on=list(train.columns))
    return merged["_in_test"].fillna(False).astype(bool).set_axis(train.index)


def split_randomness_checks(train_raw: pd.DataFrame, test: pd.DataFrame,
                            overlap_mask: pd.Series) -> pd.DataFrame:
    """Evidencia de si el test es una muestra aleatoria o un corte temporal.

    bank-full está ordenado cronológicamente (may-2008 a nov-2010). Si el test fuera un
    corte temporal, sus filas estarían concentradas al final del archivo de train; si
    es una muestra aleatoria, sus posiciones deberían ser uniformes.
    """
    positions = np.flatnonzero(overlap_mask.to_numpy()) / (len(train_raw) - 1)
    ks = stats.kstest(positions, "uniform")
    rows = [{
        "check": "posiciones_test_en_train_vs_uniforme (KS)",
        "statistic": round(float(ks.statistic), 4),
        "p_value": round(float(ks.pvalue), 4),
        "detail": f"posición media relativa = {positions.mean():.3f} (0,5 si es uniforme)",
    }]
    train_clean = train_raw.loc[~overlap_mask]
    for col in ["month", "contact", "poutcome", "y"]:
        levels = sorted(set(train_clean[col]) | set(test[col]))
        obs = np.array([
            [int((train_clean[col] == lv).sum()) for lv in levels],
            [int((test[col] == lv).sum()) for lv in levels],
        ])
        chi2, p, _, _ = stats.chi2_contingency(obs)
        rows.append({
            "check": f"distribucion_{col}_train_vs_test (chi2)",
            "statistic": round(float(chi2), 3),
            "p_value": round(float(p), 4),
            "detail": f"{len(levels)} niveles",
        })
    return pd.DataFrame(rows)


# --- Orquestación de la Fase 0 ------------------------------------------------------------

def prepare_datasets() -> dict:
    """Carga los CSV, los perfila, elimina el solapamiento y guarda los datos limpios."""
    cfg.ensure_dirs()
    train_raw = load_raw(cfg.TRAIN_FILE)
    test_raw = load_raw(cfg.TEST_FILE)
    steps = []

    def log(step: str, dataset: str, before: int, after: int, reason: str) -> None:
        steps.append({"order": len(steps) + 1, "step": step, "dataset": dataset,
                      "rows_before": before, "rows_after": after,
                      "rows_removed": before - after, "reason": reason})

    log("load", "train", len(train_raw), len(train_raw),
        "banca_train.csv leído con sep=';' y encoding ASCII; esquema validado (17 columnas).")
    log("load", "test", len(test_raw), len(test_raw),
        "banca_test.csv leído con sep=';' y encoding ASCII; esquema validado (17 columnas).")

    dup_train = int(train_raw.duplicated().sum())
    dup_test = int(test_raw.duplicated().sum())
    log("check_exact_duplicates", "train", len(train_raw), len(train_raw),
        f"Duplicados exactos dentro de train: {dup_train}. No se elimina nada.")
    log("check_exact_duplicates", "test", len(test_raw), len(test_raw),
        f"Duplicados exactos dentro de test: {dup_test}. No se elimina nada.")
    log("check_nulls", "train+test", len(train_raw) + len(test_raw), len(train_raw) + len(test_raw),
        f"Nulos: train={int(train_raw.isna().sum().sum())}, test={int(test_raw.isna().sum().sum())}. "
        "Los faltantes vienen codificados como la categoría 'unknown' y pdays=-1.")

    overlap = find_overlap(train_raw, test_raw)
    n_overlap = int(overlap.sum())
    train_clean = train_raw.loc[~overlap].reset_index(drop=True)
    log("remove_train_test_overlap", "train", len(train_raw), len(train_clean),
        f"{n_overlap} de {len(test_raw)} filas de test ({100 * n_overlap / len(test_raw):.1f} %) "
        "aparecen idénticas en train (bank.csv es una muestra de bank-full). Se quitan de train "
        "para que el hold-out sea independiente.")

    # Verificación post-limpieza: ya no debe haber solapamiento.
    assert int(find_overlap(train_clean, test_raw).sum()) == 0

    train_clean.to_csv(cfg.TRAIN_CLEAN_FILE, index=False)
    test_raw.to_csv(cfg.TEST_CLEAN_FILE, index=False)
    log("save_processed", "train", len(train_clean), len(train_clean),
        "Guardado en data/processed/train_clean.csv (sin transformar variables).")
    log("save_processed", "test", len(test_raw), len(test_raw),
        "Guardado en data/processed/test_clean.csv (idéntico al original).")

    prof = pd.concat([
        profile(train_raw, "train_raw"),
        profile(train_clean, "train_clean"),
        profile(test_raw, "test"),
    ], ignore_index=True)
    prof.to_csv(cfg.TABLES_DIR / "data_profile.csv", index=False)

    summary = pd.DataFrame([
        {"dataset": name, "rows": len(d), "cols": d.shape[1],
         "positives": int((d[cfg.TARGET] == "yes").sum()),
         "positive_rate_pct": round(100 * float((d[cfg.TARGET] == "yes").mean()), 2),
         "exact_duplicates": int(d.duplicated().sum()),
         "nulls": int(d.isna().sum().sum())}
        for name, d in [("train_raw", train_raw), ("train_clean", train_clean), ("test", test_raw)]
    ])
    summary["test_rows_found_in_train"] = [n_overlap, 0, None]
    summary.to_csv(cfg.TABLES_DIR / "data_summary.csv", index=False)

    checks = split_randomness_checks(train_raw, test_raw, overlap)
    checks.to_csv(cfg.TABLES_DIR / "split_checks.csv", index=False)

    pd.DataFrame(steps).to_csv(cfg.TABLES_DIR / "data_preparation_steps.csv", index=False)

    info = {
        "train_raw_rows": len(train_raw),
        "train_clean_rows": len(train_clean),
        "test_rows": len(test_raw),
        "overlap_rows_removed": n_overlap,
        "train_sha256": file_sha256(cfg.TRAIN_FILE),
        "test_sha256": file_sha256(cfg.TEST_FILE),
    }
    (cfg.TABLES_DIR / "data_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (train_clean, test) ya validados. Genera los archivos si no existen."""
    if not cfg.TRAIN_CLEAN_FILE.exists() or not cfg.TEST_CLEAN_FILE.exists():
        prepare_datasets()
    train = pd.read_csv(cfg.TRAIN_CLEAN_FILE)
    test = pd.read_csv(cfg.TEST_CLEAN_FILE)
    validate_schema(train, "train_clean")
    validate_schema(test, "test_clean")
    return train, test


if __name__ == "__main__":
    print(json.dumps(prepare_datasets(), indent=2))
