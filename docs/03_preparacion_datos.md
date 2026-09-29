# Preparación de los datos: qué se hizo con el dataset, paso a paso

Este documento reúne **todas** las operaciones que se aplicaron a los datos, en orden, con sus
números y el archivo que los respalda. Está pensado para responder cualquier pregunta sobre el
tratamiento del dataset.

Hay dos tipos de operaciones:

- **Sobre filas** (qué registros entran): se hacen una sola vez, en `src/bank_captacion/data.py`.
  Registro: `reports/tables/data_preparation_steps.csv`.
- **Sobre columnas** (qué variables ve el modelo): viven **dentro del pipeline** del modelo, en
  `src/bank_captacion/features.py`. Se reaprenden en cada fold de validación y se guardan dentro
  de `models/model_final.joblib`. Registro: `reports/tables/feature_catalog.csv`.

---

## Parte A — Operaciones sobre filas

| # | Paso | Train (filas) | Test (filas) | Resultado |
|---|------|---------------|--------------|-----------|
| 1 | Lectura de `banca_train.csv` | 45.211 | — | Separador `;`, encoding ASCII, 17 columnas |
| 2 | Lectura de `banca_test.csv` | — | 4.521 | Mismo formato |
| 3 | Validación de esquema | 45.211 | 4.521 | Columnas, tipos, niveles de categorías y target (`yes`/`no`) según el diccionario. Sin diferencias |
| 4 | Búsqueda de duplicados exactos dentro de cada archivo | 45.211 | 4.521 | 0 duplicados en ambos. No se elimina nada |
| 5 | Búsqueda de nulos | 45.211 | 4.521 | 0 nulos. Los faltantes están codificados como `unknown` y como `pdays = -1` |
| 6 | **Eliminación del solapamiento train/test** | 45.211 → **40.690** | 4.521 | Las 4.521 filas de test (100 %) estaban idénticas en train. Se quitaron de train |
| 7 | Verificación | 40.690 | 4.521 | `assert`: solapamiento = 0 |
| 8 | Guardado | 40.690 | 4.521 | `data/processed/train_clean.csv` y `test_clean.csv` (sin transformar columnas) |

**No se eliminó ninguna otra fila.** No se descartaron outliers, filas con `unknown` ni filas
con `duration = 0` (hay 3).

### A.1 El hallazgo crítico: test contenido en train

- **Qué se encontró:** al comparar fila por fila las 17 columnas, **las 4.521 filas de test
  aparecen idénticas en train.** Los archivos son los del dataset *Bank Marketing* de UCI:
  `bank-full.csv` (45.211 filas) y `bank.csv` (4.521 filas). El segundo es una muestra
  aleatoria del 10 % del primero.
- **Por qué importa:** si se entrena con el train original, el modelo ya "vio" todos los casos
  de test. La evaluación final mediría memoria, no generalización.
  - Un árbol profundo o un KNN con k = 1 obtendrían métricas casi perfectas en test.
- **Qué se hizo:** se quitaron esas filas del train. Así el test queda como un hold-out
  independiente y se respeta que la consigna entrega dos archivos separados.
- **Qué NO se hizo, y por qué:**
  - No se descartó el test para armar otro hold-out desde train: se habría ignorado el archivo
    que da la consigna.
  - No se usó el test tal cual: habría sido fuga de información.
- **Costo:** se pierde el 10 % de las filas para entrenar (quedan 40.690). Es un costo menor
  frente a tener una evaluación válida.
- **Evidencia:**
  - `data_summary.csv` (columna `test_rows_found_in_train = 4521`);
  - `data_preparation_steps.csv` (paso 6).

### A.2 ¿La partición es aleatoria o temporal?

- **Por qué se pregunta:** bank-full está **ordenado cronológicamente** (mayo 2008 a noviembre
  2010). Si el test fuera "los últimos meses", la validación debería ser temporal.
- **Qué se verificó** (`reports/tables/split_checks.csv`):
  - Posición de cada fila de test dentro del archivo de train: es uniforme.
    - Kolmogorov-Smirnov D = 0,014, p = 0,37.
    - Posición media relativa 0,495; si el test fuera el final del período sería ≈ 1.
  - Distribuciones de mes (χ² p = 0,47), tipo de contacto (p = 0,54), resultado previo
    (p = 0,20) y target (p = 0,72): no difieren entre train y test.
- **Conclusión:** la partición es **aleatoria**, por eso se usa validación cruzada estratificada.
- **Limitación:** con esta partición no se puede medir cómo se degrada el modelo en el tiempo.

### A.3 Balance del target después de la limpieza

| Conjunto | Filas | Conversiones | Tasa |
|---|---|---|---|
| Train original | 45.211 | 5.289 | 11,70 % |
| Train limpio | 40.690 | 4.768 | 11,72 % |
| Test | 4.521 | 521 | 11,52 % |

La limpieza no alteró la proporción de clases.

---

## Parte B — Operaciones sobre columnas (dentro del pipeline)

Ninguna de estas transformaciones se aplica "a mano" sobre el CSV: están en `FeatureEngineer` y
en el preprocesador de cada familia de modelo.

- **En validación cruzada:** se ajustan solo con los 4 folds de entrenamiento de cada
  iteración. Por ejemplo, el percentil 99 de `campaign` se calcula sin mirar el fold de
  validación.
- **En producción:** la API recibe los datos crudos y el pipeline guardado aplica las mismas
  transformaciones.

### B.1 Columnas originales

| Columna | Tratamiento en el modelo final |
|---|---|
| `age` | Se mantiene + se crea `age_group` |
| `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome` | Se mantienen como categóricas; **`unknown` es una categoría más** |
| `balance` | Se reemplaza por `balance_log` + banderas `balance_negative`, `balance_zero` |
| `day` | Se reemplaza por `day_sin`, `day_cos` |
| `campaign` | Se reemplaza por `campaign_w` (winsorizada) |
| `pdays` | Se reemplaza por `pdays_clean` + `previously_contacted` |
| `previous` | Se reemplaza por `previous_w` (winsorizada) |
| `duration` | **Excluida** (fuga de información) |
| `y` | Target: `yes` → 1, `no` → 0 |

### B.2 Variables derivadas (13 numéricas + 2 categóricas nuevas)

| Variable | Cálculo | Por qué (evidencia del EDA) |
|---|---|---|
| `pdays_clean` | `pdays`, pero NaN cuando vale -1 | -1 significa "nunca contactado", no "-1 días". Tratado como número, un modelo lineal lo pondría al lado de 0 días |
| `previously_contacted` | 1 si `pdays != -1` | Conversión 23,1 % vs 9,2 % en los nunca contactados |
| `campaign_w` | `campaign` recortado al percentil 99 del train (= 17) | Máximo de 63 contactos: la cola larga distorsiona a los modelos lineales |
| `previous_w` | `previous` recortado al percentil 99 (= 9) | Máximo de 275: mismo criterio |
| `balance_log` | signo(x) · log(1 + \|x\|) | Asimetría de 8,5; rango de −8.019 a 102.127 |
| `balance_negative` | 1 si saldo < 0 | 8,4 % de clientes en descubierto |
| `balance_zero` | 1 si saldo = 0 | 7,8 % sin saldo |
| `day_sin`, `day_cos` | sen/cos(2π · día / 31) | El día es cíclico: el 31 está cerca del 1 |
| `contact_known` | 1 si `contact != unknown` | `unknown` convierte 4,0 % vs 14,8 % |
| `prev_campaign_success` | 1 si `poutcome = success` | 64,8 % de conversión (5,5 veces la base) |
| `n_credit_products` | cantidad de `yes` en housing + loan + default | Los préstamos reducen la conversión casi a la mitad |
| `age_group` | Bins <25, 25-34, 35-44, 45-54, 55-64, 65+ | Efecto no lineal: estudiantes y jubilados convierten más |
| `season` | Estación según el mes (hemisferio norte) | Agrupa meses de bajo volumen |

Total: **24 variables** en el modelo final: 13 numéricas y 11 categóricas.

### B.3 Codificación según el modelo

| Familia | Numéricas | Categóricas | Columnas finales |
|---|---|---|---|
| Logística, KNN, MLP, SVM | Imputación por mediana (solo `pdays_clean`) + estandarización | One-hot (`handle_unknown='ignore'`) | 67 |
| Árboles, RF, XGBoost | Imputación con -1, sin escalar | One-hot | 67 |
| **LightGBM (modelo final)** | Sin imputar: LightGBM maneja NaN | **Categorías nativas** (dtype `category`) | 24 |
| Naive Bayes | Discretización en quintiles | Codificación ordinal | 24 |

### B.4 Lo que deliberadamente NO se hizo

- **No se imputaron los `unknown` a la moda.** No son faltantes al azar:
  - `contact = unknown` es el 100 % del primer 20 % del archivo (un período sin registro del
    canal);
  - `education = unknown` convierte 14,0 %, más que la moda (`secondary`, 10,5 %).
- **No se eliminaron outliers.** Son saldos y cantidades de contactos reales. Se atenuaron con
  log y winsorización.
- **No se balancearon las clases con SMOTE ni con pesos en el modelo final.** Se compararon las
  tres estrategias: la diferencia de PR-AUC fue mínima y los pesos descalibraban las
  probabilidades (decisión D13).
- **No se usó `duration`.** Solo se conoce después de la llamada (decisión D09/D18).
- **No se escaló para LightGBM.** Los árboles son invariantes a transformaciones monótonas.

---

## Parte C — Preguntas probables y respuestas cortas

**¿Por qué tienen menos filas de train que el archivo original?**
Porque las 4.521 filas de test estaban repetidas dentro de train. Se sacaron para que la
evaluación final sea sobre datos que el modelo nunca vio.

**¿Cómo detectaron el solapamiento?**
Con un *merge* por las 17 columnas entre los dos archivos. Coincidieron las 4.521 filas de test.
El código lo verifica en cada ejecución (`data.find_overlap`).

**¿Cómo saben que no es un split temporal?**
Por el test de Kolmogorov-Smirnov sobre las posiciones y los χ² de las distribuciones
(sección A.2). Ninguno detecta diferencias.

**¿Qué hicieron con los valores faltantes?**
No había nulos. Los faltantes venían como `unknown` y se dejaron como categoría propia porque
tienen tasas de conversión distintas (A y B.4). `pdays = -1` se convirtió en una bandera más un
NaN.

**¿Qué hicieron con los outliers?**
Se conservaron:

- `balance`: transformación logarítmica con signo;
- `campaign` y `previous`: recorte al percentil 99, calculado dentro de cada fold.

**¿Por qué no usaron `duration`, si es la variable más predictiva?**
Porque se conoce recién cuando termina la llamada, y el modelo decide *a quién llamar*.

- Con `duration` el ROC-AUC en test sube de 0,784 a 0,933, pero ese modelo no se puede usar en
  la práctica.
- Además, una llamada larga es en gran parte consecuencia del interés del cliente, no su causa.

**¿Hubo fuga de información en la preparación?**
Se controló en tres puntos:

1. Se sacó el solapamiento train/test.
2. Toda transformación con parámetros aprendidos (percentiles, medianas, escalas, categorías)
   está dentro del pipeline y se ajusta solo con el fold de entrenamiento.
3. `duration` está excluida, con tests automáticos que lo verifican (`tests/test_leakage.py`).

**¿Por qué `month` pesa tanto si no es una característica del cliente?**
El EDA mostró que los meses con pocas llamadas tienen tasas altísimas (marzo 53 %) y mayo, con
el 30 % de las llamadas, tiene 6,7 %: correlación volumen–tasa de −0,84. Refleja cómo operaba el
banco en cada período. Se conoce antes de llamar, así que es válida, pero es el principal riesgo
si cambian las campañas (decisión D19).

**¿El test se usó para alguna decisión?**
No. El modelo, los hiperparámetros y el umbral se eligieron solo con validación cruzada sobre
train. El test se evaluó una vez al final (`reports/tables/holdout_usage.json`).
