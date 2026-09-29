# Decision log

Registro de cada decisión metodológica del trabajo. Formato: **contexto**, **alternativas**,
**elección**, **justificación** y **evidencia** (tabla o figura generada por el pipeline).

---

## Fase 0 — Datos

### D01 · Eliminación del solapamiento entre train y test

- **Contexto.** Al comparar los archivos se detectó que las **4.521 filas de
  `banca_test.csv` (100 %) aparecen idénticas en `banca_train.csv`**. Los archivos
  corresponden al dataset *Bank Marketing* de UCI: `banca_train` = `bank-full.csv` (45.211
  filas) y `banca_test` = `bank.csv` (4.521 filas), que es una muestra del 10 % de bank-full.
- **Alternativas.** (a) Usar los archivos tal cual. (b) Descartar `banca_test` y armar un
  hold-out propio desde train. (c) Quitar de train las filas que están en test.
- **Elección.** (c): train pasa de 45.211 a **40.690 filas**; test queda con sus 4.521.
- **Justificación.** Con (a) el modelo vería durante el entrenamiento exactamente los casos
  con los que después se lo evalúa: las métricas del hold-out serían optimistas y no medirían
  generalización (un árbol profundo o un KNN con k=1 casi "memorizarían" el test). La opción
  (b) desaprovecha el archivo de test que provee la consigna. La (c) respeta la consigna,
  conserva el 90 % de los datos para entrenar y garantiza un hold-out independiente.
- **Evidencia.** `reports/tables/data_preparation_steps.csv` (paso 6),
  `reports/tables/data_summary.csv` (`test_rows_found_in_train = 4521`); el código verifica
  con un `assert` que tras la limpieza el solapamiento es 0 (`data.py`).

### D02 · Naturaleza de la partición: aleatoria, no temporal

- **Contexto.** bank-full está ordenado cronológicamente (mayo 2008 a noviembre 2010). Si el
  test fuera un corte temporal habría que evitar features que usen información del futuro y
  validar con un esquema temporal.
- **Alternativas.** Tratar la partición como temporal (validación *walk-forward*) o como
  aleatoria (validación cruzada estratificada).
- **Elección.** Aleatoria → `StratifiedKFold` de 5 folds.
- **Justificación.** Las posiciones de las filas de test dentro del archivo de train son
  compatibles con una distribución uniforme (Kolmogorov-Smirnov: D = 0,014, p = 0,37;
  posición media relativa 0,495, se esperaría ~1 si fuera el final). Las distribuciones de
  `month` (χ² p = 0,47), `contact` (p = 0,54), `poutcome` (p = 0,20) y del target
  (p = 0,72) no difieren entre train limpio y test.
- **Evidencia.** `reports/tables/split_checks.csv`.
- **Limitación a documentar.** Aunque la partición es aleatoria, los datos reales son una
  serie temporal (2008–2010, crisis financiera). Un despliegue real requeriría validar sobre
  un período posterior; con esta partición no es posible medir la degradación temporal.

### D03 · Faltantes codificados, no nulos

- **Contexto.** No hay valores nulos (`NaN`) en ningún archivo. La información faltante está
  codificada como la categoría `unknown` (`job` 0,6 %, `education` 4,1 %, `contact` 28,8 %,
  `poutcome` 81,7 % en train) y como `pdays = -1` (cliente nunca contactado).
- **Elección.** `unknown` se mantiene como **categoría propia**; no se imputa a la moda.
- **Justificación.** El faltante no es aleatorio: por ejemplo, `poutcome = unknown` equivale a
  "sin campaña previa", y `contact = unknown` se concentra en los primeros meses del registro.
  Imputar destruiría esa información. La evidencia bivariada (tasa de conversión por categoría)
  se presenta en la Fase 2.
- **Evidencia.** `reports/tables/data_profile.csv` (columna `pct_unknown`).

### D04 · Seguimiento de las transformaciones sobre los datos

- **Elección.** Todo paso que modifica o verifica filas queda en
  `reports/tables/data_preparation_steps.csv`, y todo paso que crea o transforma columnas
  queda en `reports/tables/feature_catalog.csv` (Fase 3). La explicación narrativa completa
  está en `docs/03_preparacion_datos.md`.

---

## Fase 1 — Negocio

### D05 · Enfoque de ranking y métricas primarias

- **Contexto.** El problema de negocio es priorizar a quién llamar con presupuesto limitado.
- **Alternativas.** Evaluar como clasificador (accuracy, F1 a umbral 0,5) o como ranking
  (PR-AUC, lift/gain).
- **Elección.** Ranking: PR-AUC y lift/gain al 10/20/30 % como métricas primarias.
- **Justificación.** Ver `docs/01_negocio.md`: con 88,3 % de negativos, accuracy premia al
  modelo trivial; PR-AUC y lift miden directamente lo que el banco necesita.

### D06 · Parámetros económicos C = 1, V = 20

- **Elección.** C = 1, V = 20 (umbral económico teórico p > 0,05) con sensibilidad V/C ∈
  {5, 10, 20, 50}.
- **Justificación.** Son supuestos explícitos y parametrizados; el análisis de sensibilidad
  muestra cómo cambia la decisión si el banco usa valores reales.

---

## Fase 3 — Preparación de los datos

### D07 · Variables derivadas

Cada variable, su origen y su justificación están en `reports/tables/feature_catalog.csv`.
Resumen:

| Variable | Decisión | Evidencia del EDA |
|---|---|---|
| `previously_contacted`, `pdays_clean` | Separar la bandera "nunca contactado" del número de días (NaN si -1) | Las 33.249 filas con pdays = -1 son exactamente las de previous = 0; conversión 9,2 % vs 23,1 % |
| `campaign_w`, `previous_w` | Winsorizar al p99 **aprendido en cada fold** (en train completo: campaign ≤ 17, previous ≤ 9) | Máximo de 63 contactos; cola larga |
| `balance_log`, `balance_negative`, `balance_zero` | Log con signo + banderas; **no se eliminan outliers** | Asimetría 8,5; 10,4 % de outliers IQR que son saldos reales |
| `age_group` | Bins además de la edad numérica | Conversión alta en < 25 y 65+ |
| `contact_known`, `prev_campaign_success` | Banderas binarias | contact = unknown: 4,0 %; poutcome = success: 64,8 % |
| `season`, `day_sin`, `day_cos` | Estación y codificación cíclica del día | Meses con poco volumen; el día 31 está al lado del 1 |
| `n_credit_products` | housing + loan + default | Los préstamos reducen la conversión |

- **Alternativa descartada: eliminar outliers de `balance`.** Se perdería un 10 % de
  clientes reales y el modelo no sabría puntuarlos en producción.
- **Alternativa descartada: imputar `unknown` a la moda.** Ver D03.
- **Nota técnica.** La bandera se llamó inicialmente `poutcome_success`, pero ese nombre chocaba
  con la columna one-hot de la categoría `success` de `poutcome`. Se renombró a
  `prev_campaign_success`. La redundancia con la columna one-hot no afecta a los árboles; en la
  logística queda absorbida por la regularización L2.

### D08 · Preprocesamiento por familia de modelo

- **Lineales, KNN, MLP y SVM**: imputación por mediana (solo afecta a `pdays_clean`),
  `StandardScaler` y one-hot con `handle_unknown='ignore'` (67 columnas). Estos modelos son
  sensibles a la escala.
- **Árboles y boosting con one-hot**: imputación con -1 y sin escalado, porque los árboles son
  invariantes a transformaciones monótonas.
- **LightGBM y XGBoost con categorías nativas**: las categóricas pasan como dtype `category` y
  los NaN quedan sin imputar.
- **Naive Bayes**: numéricas continuas discretizadas en quintiles y categóricas ordinales, porque
  CategoricalNB requiere variables discretas.
- **Todo dentro de un `Pipeline`**, ajustado solo con los folds de entrenamiento. El modelo
  final recibe datos crudos; la API no reimplementa el preprocesamiento.

### D09 · Dos conjuntos de variables

- `pre_contact` (entregable): 24 variables, **sin `duration`**. `FeatureEngineer` descarta
  `duration` aunque venga en los datos (lo verifica `tests/test_leakage.py`).
- `with_duration`: la misma configuración más `duration`, solo para cuantificar el leakage (D18).

---

## Fase 4 — Modelado

### D10 · Protocolo de validación

- `StratifiedKFold(5, shuffle=True, random_state=42)` sobre las 40.690 filas.
- Todos los modelos usan exactamente la misma partición (`reports/tables/cv_folds.csv`).
- Se reporta media ± desvío por fold (`cv_results.csv`, `cv_results_folds.csv`).

### D11 · Algoritmos comparados (10) y por qué

| Modelo | Familia | Por qué se prueba | PR-AUC CV |
|---|---|---|---|
| Dummy (prior) | Referencia | Piso: PR-AUC = prevalencia | 0,117 |
| Regresión logística (balanced) | Lineal | Baseline interpretable | 0,412 ± 0,012 |
| Árbol de decisión (prof. 6) | Árboles | Reglas explicables | 0,351 ± 0,020 |
| KNN (k = 51) | Instancias | No paramétrico, por similitud | 0,401 ± 0,020 |
| Naive Bayes categórico | Probabilístico generativo | Simple y rápido; referencia | 0,395 ± 0,019 |
| Random Forest | Bagging | Robusto, baja varianza | 0,446 ± 0,023 |
| LightGBM | Boosting | Estado del arte en datos tabulares | **0,457 ± 0,020** |
| XGBoost (scale_pos_weight) | Boosting | Boosting regularizado | 0,454 ± 0,021 |
| MLP (64-32) | Redes neuronales | Representante de redes neuronales | 0,430 ± 0,020 |
| SVM RBF (submuestra 10k) | Kernels | Frontera no lineal de máximo margen | 0,362 ± 0,020 |

La SVM tiene costo O(n²), por eso se entrenó sobre una submuestra estratificada de 10.000
filas. Se evalúa sobre el fold de validación completo usando `decision_function`, así que no
tiene Brier.

### D12 · Codificación de categóricas en boosting

- **LightGBM**: la codificación nativa (0,4580) superó al one-hot (0,4567) → se usa **nativa**.
- **XGBoost**: el one-hot (0,4541) superó a la nativa (0,4509) por más de la tolerancia de 0,003
  → se usa **one-hot**.
- Evidencia: `encoding_comparison.csv`.

### D13 · Estrategia de desbalance

- **Contexto.** Se compararon tres estrategias sobre los 3 mejores modelos (LightGBM, XGBoost y
  Random Forest):
  - (a) sin tratamiento, con ajuste de umbral;
  - (b) pesos de clase;
  - (c) SMOTE dentro del pipeline.
- **Resultados.**

  | Modelo | Métrica | (a) Sin tratamiento | (b) Pesos | (c) SMOTE |
  |---|---|---|---|---|
  | LightGBM | PR-AUC | 0,459 | 0,458 | 0,455 |
  | LightGBM | Brier | 0,081 | 0,144 | 0,082 |
  | Random Forest | PR-AUC | 0,451 | 0,446 | 0,436 |

- **Elección.** **(a) sin tratamiento + ajuste de umbral**, para los tres modelos.
- **Justificación.**
  - Las diferencias de PR-AUC son menores que un desvío entre folds.
  - Los pesos de clase **descalibran** las probabilidades: el Brier casi se duplica. El umbral y
    el beneficio esperado dependen de probabilidades bien calibradas.
  - SMOTE interpola clientes sintéticos sobre variables one-hot, algo que no tiene
    interpretación, y además empeora al Random Forest.
  - Regla aplicada: si la diferencia de PR-AUC es menor a 0,003, se prefiere la estrategia que
    no distorsiona las probabilidades.
- **Evidencia.** `imbalance_results.csv`, `imbalance_comparison.png`.

### D14 · Tuning

- **Método.** Optuna con sampler TPE (semilla 42), 60 trials por modelo y tope de 15 minutos.
  El objetivo es la PR-AUC media en los mismos 5 folds.
- **Resultados.**
  - LightGBM: de 0,4589 a **0,4660** (≈ 67 s).
  - XGBoost: de 0,4522 a 0,4629.
- **Mejores hiperparámetros de LightGBM.** n_estimators 500, learning_rate 0,0117, num_leaves 44,
  min_child_samples 20, subsample 0,68, colsample_bytree 0,63 y regularización L1/L2 baja.
- **Espacios de búsqueda.** `models.suggest_params`.
- **Evidencia.** `tuning_summary.csv`, `tuning_trials_*.csv`, `tuning_convergence_*.png`.
- **Limitación.** La PR-AUC del mejor trial es optimista porque se eligió mirando los mismos
  folds. La estimación sin sesgo es la del hold-out.

### D15 · Selección final (matriz de decisión)

- **Criterios y pesos.**
  - PR-AUC: 35 %
  - lift@20 %: 25 %
  - estabilidad (desvío de PR-AUC): 15 %
  - interpretabilidad (escala 1–5): 15 %
  - velocidad de inferencia: 10 %
- **Normalización.** Min-max entre candidatos.
- **Puntaje ponderado.**

  | Candidato | Puntaje |
  |---|---|
  | **LightGBM tuneado** | **0,690** |
  | XGBoost tuneado | 0,647 |
  | XGBoost por defecto | 0,640 |
  | LightGBM por defecto | 0,632 |
  | Regresión logística | 0,400 |

- **Evidencia.** `decision_matrix.csv`.
- **Por qué no la logística.** Es la más estable e interpretable, pero pierde 0,05 de PR-AUC
  (≈ 12 % relativo) y 0,3 de lift@20 %. La interpretabilidad de LightGBM se recupera con SHAP.

### D16 · Umbral operativo

- **Método.** Se eligió con las predicciones out-of-fold del modelo final, maximizando
  V·TP − C·(TP+FP) con V = 20 y C = 1.
- **Umbral elegido: 0,050.** Coincide con el umbral teórico C/V, lo que indica probabilidades
  bien calibradas.
  - En OOF contacta al 61,9 %, con recall 0,891 y beneficio 59.772.
  - Contactar a todos daría 54.670.
- **Umbral que maximiza F1: 0,215.**
  - Contacta al 13,4 % y el beneficio baja a 45.363.
  - F1 pondera igual precisión y recall, e ignora que una conversión vale 20 veces una llamada.
- **Evidencia.** `threshold_choice.json`, `threshold_analysis.csv`, `sensitivity_oof.csv`,
  `threshold_selection.png`.

---

## Fase 5 — Evaluación

### D17 · Uso único del hold-out

- `evaluate.py` y `explain.py` son los únicos módulos que leen el test.
- Cargan el modelo, los hiperparámetros y el umbral ya fijados en la Fase 4.
- El umbral de la logística se eligió con sus propias predicciones OOF de train.
- Ninguna decisión se tomó después de ver los resultados de test.
- Evidencia: `reports/tables/holdout_usage.json`, que registra fecha y hash del modelo evaluado.

### D18 · Cuantificación del leakage

| Conjunto | ROC-AUC CV | PR-AUC CV | ROC-AUC test | PR-AUC test |
|---|---|---|---|---|
| pre_contact (entregable) | 0,806 | 0,466 | 0,784 | 0,435 |
| with_duration | 0,937 | 0,638 | 0,933 | 0,607 |

El modelo con `duration` **no se entrega**. Evidencia: `leakage_cv.csv`, `test_leakage.csv`.

### D19 · Advertencia sobre `month`

- `month` es la variable más importante (15 % de la importancia SHAP).
- El EDA mostró que refleja el volumen de la operación y el período, no una característica del
  cliente.
- Se mantiene porque se conoce antes de llamar y la partición es aleatoria.
- Se recomienda monitorear su efecto y reentrenar con datos recientes antes de usar el modelo en
  campañas nuevas.
