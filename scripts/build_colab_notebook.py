"""Genera notebooks/TP_final_colab.ipynb (notebook autocontenido para Google Colab). Ejecutar desde la raíz del repo."""
import nbformat as nbf

REPO = "fcaceres-create/caece-aprendizaje-artificial-tpfinal"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(f"""# Predicción de captación de clientes — TP Final

**Aprendizaje Artificial** · Maestría en Gestión y Desarrollo de Inteligencia Artificial · Universidad CAECE
**Autor:** Fernando Caceres · **Docentes:** Juan Azcurra · Paul Pablo Hernán

<a href="https://colab.research.google.com/github/{REPO}/blob/main/notebooks/TP_final_colab.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"></a>

Este notebook reproduce el trabajo de punta a punta en Google Colab, siguiendo CRISP-DM:

1. **Instalación** del código del repositorio y sus dependencias.
2. **Datos:** descarga del dataset *Bank Marketing* desde UCI y verificación byte a byte contra los archivos de la consigna.
3. **Preparación:** detección y eliminación del solapamiento entre train y test.
4. **Exploración** de los datos.
5. **Modelado:** modo rápido, que usa el modelo ya entrenado, o modo completo, que reentrena todo.
6. **Evaluación** única sobre el hold-out y análisis de negocio.
7. **Explicabilidad** con SHAP.
8. **Uso del modelo** y llamada a la API publicada.

**Enlaces**
- App web: https://captacion-clientes.onrender.com
- Presentación: https://fcaceres-create.github.io/caece-aprendizaje-artificial-tpfinal/presentacion/
- Código: https://github.com/{REPO}

> **Cómo ejecutarlo:** *Entorno de ejecución → Ejecutar todo*. La primera celda instala las librerías y
> **reinicia el entorno automáticamente** (es necesario para que tome las versiones correctas). Cuando
> termine el reinicio, volver a hacer *Ejecutar todo*: la segunda vez la instalación se saltea.""")

code(f"""# 1. Instalación (en Colab clona el repo, instala versiones fijas y reinicia el entorno una sola vez)
import os, sys, subprocess
from pathlib import Path

EN_COLAB = "google.colab" in sys.modules
REPO_DIR = Path("/content/caece-aprendizaje-artificial-tpfinal") if EN_COLAB else Path.cwd().resolve().parent

if EN_COLAB:
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/{REPO}.git", str(REPO_DIR)], check=True)
    marker = Path("/content/.tp_instalado")
    if not marker.exists():
        print("Instalando dependencias (2-3 minutos)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "pandas==3.0.6", "numpy==2.5.3", "scikit-learn==1.9.1", "imbalanced-learn==0.14.2",
                        "lightgbm==4.7.0", "xgboost==3.4.1", "shap==0.52.0", "optuna==5.0.0",
                        "matplotlib==3.11.2", "seaborn==0.13.2", "joblib==1.6.0", "requests"], check=True)
        marker.touch()
        print("Listo. Reiniciando el entorno: cuando termine, volver a 'Ejecutar todo'.")
        os.kill(os.getpid(), 9)

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR / "src"))
import pandas as pd, numpy as np, sklearn, lightgbm
print("Repositorio:", REPO_DIR)
print("pandas", pd.__version__, "· numpy", np.__version__, "· scikit-learn", sklearn.__version__, "· lightgbm", lightgbm.__version__)""")

md("""## 2. Datos: descarga desde UCI y verificación

Los archivos de la consigna (`banca_train.csv`, `banca_test.csv`) no se versionan en el repositorio. Son el
dataset público *Bank Marketing* (Moro, Cortez y Rita, 2014). Se descargan desde UCI y se comprueba con
SHA-256 que son **exactamente** los mismos archivos:

- `banca_train.csv` = `bank-full.csv`
- `banca_test.csv` = `bank.csv`""")

code("""import hashlib, io, json, urllib.request, zipfile
from bank_captacion import config as cfg

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
esperado = json.loads((cfg.TABLES_DIR / "data_info.json").read_text())   # hashes registrados en el proyecto
cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)

if not (cfg.TRAIN_FILE.exists() and cfg.TEST_FILE.exists()):
    zbytes = urllib.request.urlopen(URL, timeout=120).read()
    interno = zipfile.ZipFile(io.BytesIO(zipfile.ZipFile(io.BytesIO(zbytes)).read("bank.zip")))
    cfg.TRAIN_FILE.write_bytes(interno.read("bank-full.csv"))
    cfg.TEST_FILE.write_bytes(interno.read("bank.csv"))

for nombre, ruta, clave in [("banca_train.csv", cfg.TRAIN_FILE, "train_sha256"), ("banca_test.csv", cfg.TEST_FILE, "test_sha256")]:
    h = hashlib.sha256(ruta.read_bytes()).hexdigest()
    print(f"{nombre:16s} {h[:16]}…  {'✓ idéntico al de la consigna' if h == esperado[clave] else '✗ DISTINTO'}")""")

md("""## 3. Preparación: el hallazgo crítico

Al comparar los dos archivos, **las 4.521 filas del test aparecen idénticas dentro del train**: `bank.csv`
es una muestra de `bank-full.csv`. Si no se corrige, la evaluación final mide memoria y no generalización.
Por eso se quitan esas filas del train, que queda en 40.690.""")

code("""from IPython.display import Image, display
from bank_captacion.data import prepare_datasets

info = prepare_datasets()
display(pd.read_csv(cfg.TABLES_DIR / "data_summary.csv"))
display(pd.read_csv(cfg.TABLES_DIR / "data_preparation_steps.csv")[["order", "step", "dataset", "rows_before", "rows_after", "reason"]])
print("¿Partición aleatoria o temporal? (p > 0,05 = sin diferencias)")
display(pd.read_csv(cfg.TABLES_DIR / "split_checks.csv"))""")

md("""## 4. Exploración de los datos (solo train)""")

code("""from bank_captacion.eda import run_eda
claves = run_eda()
for f in ["eda_conversion_by_category.png", "eda_unknown_by_file_position.png", "eda_month_volume_rate.png", "eda_duration_leakage.png"]:
    display(Image(filename=str(cfg.FIGURES_DIR / f), width=950))""")

md("""**Lectura:**

- `unknown` no es un faltante al azar: por ejemplo, `contact = unknown` corresponde a un período sin
  registro del canal. Por eso se mantiene como categoría propia.
- El mes refleja el volumen de la operación del banco: a más llamadas, menor tasa de conversión.
- `duration` es **leakage**: solo se conoce después de llamar, así que se excluye del modelo.""")

md("""## 5. Modelado

- **`MODO = "rapido"`** (por defecto): usa el modelo final ya entrenado del repositorio y muestra la
  comparación de algoritmos registrada.
- **`MODO = "completo"`**: reentrena todo desde cero (10 algoritmos, 3 estrategias de desbalance, Optuna).
  En el CPU de Colab puede tardar entre 20 y 40 minutos. Los resultados son los mismos porque la semilla
  está fija; solo cambian los tiempos medidos.""")

code("""MODO = "rapido"   # cambiar a "completo" para reentrenar desde cero

if MODO == "completo":
    from bank_captacion.train import run_training
    run_training()

cv = pd.read_csv(cfg.TABLES_DIR / "cv_results.csv")
display(cv[["label", "family", "pr_auc_fmt", "roc_auc_fmt", "lift@20_fmt", "brier_fmt"]])
display(Image(filename=str(cfg.FIGURES_DIR / "model_cv_comparison.png"), width=1000))
display(pd.read_csv(cfg.TABLES_DIR / "decision_matrix.csv")[["candidate", "pr_auc", "pr_auc_std", "lift@20", "weighted_score"]])
print(json.dumps(json.loads((cfg.TABLES_DIR / "threshold_choice.json").read_text()), indent=2))""")

md("""## 5 bis. Los 10 modelos en vivo

Cada modelo se entrena y evalúa **ahora**, con la misma validación cruzada (5 particiones de los 40.690
clientes). Para cada uno se muestra:

- **Familia**, **cómo funciona** y **por qué se incluyó** en la comparación.
- El **diagrama de su pipeline**: preparación de datos → modelo. Se puede hacer clic para desplegar los detalles.
- Su resultado: **PR-AUC** (qué tan bien encuentra a los que convierten) y **ROC-AUC**.

Para probar un solo modelo a pedido: `probar_modelo("knn")`. Los nombres posibles son `lightgbm`, `xgboost`,
`random_forest`, `mlp`, `logreg`, `knn`, `naive_bayes`, `svm_rbf`, `tree` y `dummy`.""")

code("""import time, warnings
from IPython.display import HTML
from sklearn import set_config
from bank_captacion.data import load_clean, split_xy
from bank_captacion.models import MODEL_SPECS, make_model
from bank_captacion.train import cv_evaluate, get_folds

warnings.filterwarnings("ignore")
set_config(display="diagram")
_train, _ = load_clean()
X_tr, y_tr = split_xy(_train)
FOLDS = get_folds(X_tr, y_tr)

POR_QUE = {
    "lightgbm": "Representa el boosting, el estado del arte en datos de tablas. Admite categorías nativas.",
    "xgboost": "Segunda implementación de boosting, para confirmar que el resultado no depende de una librería.",
    "random_forest": "Representa los ensambles por votación (bagging): robusto y difícil de sobreajustar.",
    "mlp": "Representa a las redes neuronales vistas en la materia.",
    "logreg": "Es el baseline clásico, simple e interpretable: todo modelo complejo debe superarlo.",
    "knn": "Representa los métodos basados en similitud entre casos.",
    "naive_bayes": "Representa los modelos probabilísticos: simple, rápido y buena referencia.",
    "svm_rbf": "Representa los métodos de márgenes con kernels (fronteras no lineales).",
    "tree": "Genera reglas que se le pueden explicar al negocio.",
    "dummy": "Sirve como piso: muestra cuánto aporta realmente cada modelo.",
}
RESULTADOS = {}

def probar_modelo(nombre):
    spec = MODEL_SPECS[nombre]
    display(HTML(f"<h3 style='margin-bottom:4px'>{spec.label}</h3>"
                 f"<p style='margin-top:0'><b>Familia:</b> {spec.family}<br><b>Cómo funciona:</b> {spec.rationale}"
                 f"<br><b>Por qué lo incluí:</b> {POR_QUE[nombre]}</p>"))
    display(make_model(nombre))                       # diagrama del pipeline: preparación → modelo
    t0 = time.time()
    folds_df, _ = cv_evaluate(nombre, X_tr, y_tr, FOLDS, imbalance="weights")
    r = {"Modelo": spec.label, "Familia": spec.family,
         "PR-AUC": round(folds_df.pr_auc.mean(), 3), "± variación": round(folds_df.pr_auc.std(), 3),
         "ROC-AUC": round(folds_df.roc_auc.mean(), 3), "Segundos": round(time.time() - t0, 1)}
    RESULTADOS[nombre] = r
    display(HTML(f"<p style='font-size:15px'>➜ <b>PR-AUC = {r['PR-AUC']:.3f}</b> ± {r['± variación']:.3f} · "
                 f"ROC-AUC = {r['ROC-AUC']:.3f} · {r['Segundos']} s</p><hr>"))
    return r

print("Listo. Usar probar_modelo('nombre') o ejecutar la celda siguiente para los 10.")""")

code("""# Los 10 modelos, uno por uno (en Colab tarda unos minutos; el SVM es el más lento)
for nombre in ["dummy", "logreg", "tree", "knn", "naive_bayes", "random_forest", "lightgbm", "xgboost", "mlp", "svm_rbf"]:
    probar_modelo(nombre)

tabla = pd.DataFrame(RESULTADOS.values()).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
tabla.index = tabla.index + 1
display(HTML("<h3>Ranking final (mismo resultado que la slide de comparación)</h3>"))
display(tabla)""")

md("""## 6. Evaluación única sobre el hold-out (4.521 clientes)

El modelo, sus hiperparámetros y el umbral se fijaron antes de mirar el test.""")

code("""from bank_captacion.evaluate import run_evaluation
run_evaluation()
display(pd.read_csv(cfg.TABLES_DIR / "test_metrics.csv")[["label", "roc_auc", "pr_auc", "lift@10", "lift@20", "gain@20", "brier"]])
display(pd.read_csv(cfg.TABLES_DIR / "test_business_summary.csv"))
for f in ["test_gain_lift.png", "test_profit_curves.png"]:
    display(Image(filename=str(cfg.FIGURES_DIR / f), width=1000))
display(pd.read_csv(cfg.TABLES_DIR / "reasonableness.csv"))""")

md("""## 7. Explicabilidad (SHAP, regresión logística y árbol de reglas)""")

code("""from bank_captacion.explain import run_explain
run_explain()
display(Image(filename=str(cfg.FIGURES_DIR / "shap_summary.png"), width=800))
display(pd.read_csv(cfg.TABLES_DIR / "rules_tree_leaves.csv"))""")

md("""## 8. Usar el modelo

Primero localmente, con el pipeline guardado: recibe datos crudos y devuelve la probabilidad y las variables
que más pesaron. Después, contra la **API publicada en la web**.""")

code("""from bank_captacion.inference import load_model, score_frame, top_contributions

modelo, meta = load_model()
cliente = pd.DataFrame([{"age": 64, "job": "retired", "marital": "married", "education": "primary", "default": "no",
                         "balance": 43, "housing": "no", "loan": "no", "contact": "cellular", "day": 10, "month": "mar",
                         "campaign": 1, "pdays": 90, "previous": 2, "poutcome": "success"}])
p = modelo.predict_proba(cliente)[0, 1]
print(f"Probabilidad de conversión: {p:.1%} → {'LLAMAR' if p >= meta['threshold'] else 'no llamar'} (umbral {meta['threshold']:.0%})")
for f in top_contributions(modelo, cliente)[0]:
    print(f"  {f['label']:22s} = {str(f['value']):10s} {f['effect']} la probabilidad ({f['shap']:+.2f})")""")

code("""import requests
API = "https://captacion-clientes.onrender.com"
try:
    r = requests.post(f"{API}/predict", json=cliente.iloc[0].to_dict(), timeout=120)  # la primera vez puede tardar ~1 min (se despierta)
    print("Respuesta de la API publicada:", json.dumps(r.json(), indent=2, ensure_ascii=False)[:600])
except Exception as e:
    print("La API no respondió a tiempo (plan gratuito dormido). Abrir", API, "y reintentar.", e)""")

md(f"""## Material complementario

- **Informe completo:** [`reports/informe_final.pdf`](https://github.com/{REPO}/blob/main/reports/informe_final.pdf)
- **Registro de decisiones:** [`docs/decision_log.md`](https://github.com/{REPO}/blob/main/docs/decision_log.md)
- **Preparación de los datos, paso a paso:** [`docs/03_preparacion_datos.md`](https://github.com/{REPO}/blob/main/docs/03_preparacion_datos.md)
- **Notebooks detallados:** `notebooks/01_eda.ipynb`, `02_modelado.ipynb`, `03_evaluacion_y_negocio.ipynb`""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["colab"] = {"provenance": [], "name": "TP_final_colab.ipynb"}
nbf.write(nb, "notebooks/TP_final_colab.ipynb")
print("ok")
