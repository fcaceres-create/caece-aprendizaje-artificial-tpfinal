"""Orquestador: reconstruye todo el trabajo desde los CSV originales hasta el informe.

Uso:
    python -m bank_captacion.pipeline                 # todas las etapas
    python -m bank_captacion.pipeline --stage train   # una etapa
    python -m bank_captacion.pipeline --skip-notebooks
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings

from . import config as cfg

STAGES = ["data", "eda", "features", "train", "evaluate", "explain", "notebooks", "report"]


def stage_data():
    from .data import prepare_datasets
    print(prepare_datasets())


def stage_eda():
    from .eda import run_eda
    run_eda()


def stage_features():
    from .features import write_feature_catalog
    write_feature_catalog()


def stage_train():
    from .train import run_training
    run_training()


def stage_evaluate():
    from .evaluate import run_evaluation
    run_evaluation()


def stage_explain():
    from .explain import run_explain
    run_explain()


def stage_notebooks():
    from .notebooks_builder import write_notebooks
    write_notebooks()
    for nb in ["01_eda.ipynb", "02_modelado.ipynb", "03_evaluacion_y_negocio.ipynb"]:
        print(f"  ejecutando {nb}")
        subprocess.run([sys.executable, "-m", "papermill", nb, nb, "--cwd", ".", "-k", "python3",
                        "--log-level", "ERROR"], cwd=cfg.NOTEBOOKS_DIR, check=True)


def stage_report():
    from .diagram import render_diagrams
    from .export_report import build_report
    from .slides import build_slides_html
    render_diagrams()
    build_report()
    print("  presentación:", build_slides_html())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pipeline completo del TP (CRISP-DM)")
    parser.add_argument("--stage", choices=STAGES, help="Ejecutar solo una etapa")
    parser.add_argument("--skip-notebooks", action="store_true")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    cfg.ensure_dirs()
    stages = [args.stage] if args.stage else STAGES
    if args.skip_notebooks and "notebooks" in stages:
        stages.remove("notebooks")
    t_all = time.perf_counter()
    for st in stages:
        t0 = time.perf_counter()
        print(f"=== Etapa: {st}")
        globals()[f"stage_{st}"]()
        print(f"=== {st} OK ({time.perf_counter() - t0:.1f}s)")
    print(f"Pipeline completo en {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
