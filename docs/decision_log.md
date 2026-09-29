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
