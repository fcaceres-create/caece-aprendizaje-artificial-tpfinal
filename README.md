# TP Final — Captación de clientes

**Aprendizaje Artificial · Maestría en Gestión y Desarrollo de IA · Universidad CAECE**

El proyecto construye un modelo que ordena a los clientes potenciales de una campaña de marketing
telefónico según su probabilidad de convertirse en clientes del banco. Con ese orden se decide a
quién llamar cuando hay un presupuesto limitado.

El trabajo sigue CRISP-DM de punta a punta y agrega una API (FastAPI) y una interfaz web
(Streamlit) que consumen el modelo.

**Integrantes:** [COMPLETAR] · **Docente:** [COMPLETAR]

## Resultados clave

| | Validación cruzada (train, 5 folds) | Hold-out (`banca_test.csv`) |
|---|---|---|
| Modelo final | LightGBM pre-contacto (sin `duration`) | |
| ROC-AUC | 0,806 ± 0,010 | **0,784** |
| PR-AUC (average precision) | 0,466 ± 0,020 | **0,435** (azar = 0,115) |
| Lift en el top 10 % | 4,40 | **4,21** |
| % de conversiones en el top 20 % | 63,1 % | **59,7 %** |
| Umbral operativo (máx. beneficio, V/C = 20) | 0,050 | contacta al 61 %, captura el 86,6 % |
| Baseline: regresión logística (PR-AUC) | 0,412 | 0,349 |
| Leakage: el mismo modelo con `duration` (ROC-AUC) | 0,937 | 0,933 (no utilizable) |

Todos los números salen de `reports/tables/`. El informe completo está en
[reports/informe_final.md](reports/informe_final.md), también en `.docx` y `.pdf`.

### Hallazgo crítico en los datos

Las **4.521 filas de `banca_test.csv` están contenidas en `banca_train.csv`**. El motivo es que
`bank.csv` es una muestra de `bank-full.csv` en el dataset *Bank Marketing* de UCI.

- Se eliminaron del train, que quedó en 40.690 filas, para que el hold-out sea independiente.
- El detalle de todo lo que se hizo con los datos está en
  [docs/03_preparacion_datos.md](docs/03_preparacion_datos.md).

## Cómo reproducir

Requisitos: Python 3.11 o superior (probado con 3.14.5 en Windows 11).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Copiar `banca_train.csv` y `banca_test.csv` en `data/raw/` (no se versionan). Después, un solo
comando reconstruye todo: datos → EDA → variables → modelos → evaluación → explicabilidad →
notebooks → diagramas → informe.

```bash
python -m bank_captacion.pipeline            # ~5 minutos
python -m bank_captacion.pipeline --stage train   # o una etapa: data, eda, features, train,
                                                  # evaluate, explain, notebooks, report
python -m pytest -q                          # 29 tests
```

En macOS y Linux también funciona `make all`, `make test`, `make api` y `make ui`.

El PDF del informe se genera con `docx2pdf`, que requiere Microsoft Word. Si no está disponible,
se generan igual el `.md` y el `.docx`.

## API y UI

### Sin Docker

Usar dos terminales, las dos con el entorno virtual activado:

```bash
python -m uvicorn app.api.main:app --port 8000        # API → http://localhost:8000/docs
python -m streamlit run app/ui/streamlit_app.py       # UI  → http://localhost:8501
```

La UI toma la URL de la API desde la variable `API_URL` (por defecto `http://localhost:8000`).

### Con Docker

```bash
docker compose -f app/docker-compose.yml up --build
```

Nota: el `Dockerfile` y el `docker-compose.yml` están escritos pero **no se probaron**, porque
Docker no estaba instalado en la máquina de desarrollo. La ejecución sin Docker sí está verificada.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio |
| GET | `/model/info` | Metadata: hiperparámetros, métricas de CV, umbral, hash de los datos |
| POST | `/predict` | Un cliente (JSON) → probabilidad, recomendación, decil, top-3 SHAP y explicación |
| POST | `/predict/batch` | CSV → CSV rankeado con probabilidad, decil y recomendación |

Si `/predict` recibe `duration` en el payload, responde **422**: el modelo es pre-contacto.

## Estructura

```
├── src/bank_captacion/     # código del pipeline
│   ├── config.py           # rutas, semilla, parámetros de negocio (C, V)
│   ├── data.py             # carga, validación, perfil, eliminación del solapamiento
│   ├── eda.py              # análisis exploratorio (figuras y tablas)
│   ├── features.py         # ingeniería de variables + preprocesadores
│   ├── models.py           # catálogo de 10 modelos y espacios de búsqueda
│   ├── train.py            # CV, desbalance, Optuna, matriz de decisión, umbral
│   ├── evaluate.py         # evaluación única en hold-out, negocio, segmentos
│   ├── explain.py          # SHAP, coeficientes de la logística, árbol de reglas
│   ├── business.py         # lift, gain, deciles, beneficio, sensibilidad
│   ├── inference.py        # inferencia liviana (usada por la API)
│   ├── diagram.py          # diagramas del pipeline (PNG)
│   ├── export_report.py    # plantilla → informe .md / .docx / .pdf
│   └── pipeline.py         # orquestador (comando único)
├── notebooks/              # 01_eda, 02_modelado, 03_evaluacion_y_negocio (ejecutados)
├── models/                 # model_final.joblib, metadata.json, variantes de comparación
├── reports/                # informe, figuras y tablas
├── docs/                   # negocio, datos, preparación, decision log, diagramas
├── app/                    # API FastAPI, UI Streamlit, Docker
└── tests/                  # features, leakage, modelo, API
```

## Documentación

- [docs/01_negocio.md](docs/01_negocio.md): problema, métricas y marco económico.
- [docs/02_datos.md](docs/02_datos.md): comprensión de los datos.
- [docs/03_preparacion_datos.md](docs/03_preparacion_datos.md): **cada operación sobre el dataset**,
  con preguntas y respuestas.
- [docs/decision_log.md](docs/decision_log.md): cada decisión con sus alternativas, justificación y
  evidencia.
- [docs/diagrama_pipeline.md](docs/diagrama_pipeline.md): diagramas Mermaid y PNG.
