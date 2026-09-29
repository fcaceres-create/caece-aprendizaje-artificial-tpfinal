# Fase 2 — Comprensión de los datos

Fuente: dataset *Bank Marketing* (UCI Machine Learning Repository; Moro, Cortez y Rita, 2014).
Campañas telefónicas de un banco portugués entre mayo de 2008 y noviembre de 2010.
Notebook con todo el detalle: `notebooks/01_eda.ipynb`. Todas las cifras provienen de
`reports/tables/eda_*.csv` y `reports/tables/eda_key_numbers.json`.

## 2.1 Archivos recibidos y verificaciones

| Archivo | Equivale a | Filas | Columnas | Separador | Encoding | Conversión |
|---|---|---|---|---|---|---|
| `banca_train.csv` | `bank-full.csv` | 45.211 | 17 | `;` | ASCII | 11,70 % |
| `banca_test.csv` | `bank.csv` | 4.521 | 17 | `;` | ASCII | 11,52 % |

- Nombres de columnas idénticos al diccionario de la consigna. Target codificado `yes`/`no`.
- 0 valores nulos y 0 duplicados exactos dentro de cada archivo.
- **Las 4.521 filas de test están todas contenidas en train** → se quitaron de train, que
  quedó en **40.690 filas** (4.768 conversiones, 11,72 %). Ver decisión D01.
- La partición es **aleatoria** (D02): test de Kolmogorov-Smirnov sobre las posiciones de las
  filas de test en el archivo de train, p = 0,37; χ² de `month`, `contact`, `poutcome` y `y`
  entre train y test, todos con p > 0,20.

## 2.2 Tipos de variables

| Tipo | Variables |
|---|---|
| Numéricas | `age`, `balance`, `day`, `duration`*, `campaign`, `pdays`, `previous` |
| Categóricas nominales | `job` (12), `marital` (3), `education` (4), `contact` (3), `month` (12), `poutcome` (4) |
| Binarias | `default`, `housing`, `loan` |
| Target | `y` (yes/no) |

\* `duration` es leakage: solo se usa en el análisis complementario.

## 2.3 Hallazgos principales

1. **Desbalance**: 11,7 % de conversiones → métricas de ranking, no accuracy.
2. **`unknown` no es aleatorio** (`eda_unknown_analysis.csv`, `eda_unknown_by_file_position.png`):
   - `contact = unknown` (28,7 %) ocupa el 100 % del primer 20 % del archivo y casi
     desaparece después del 30 %: es un período (inicio de 2008) sin registro del canal.
     Conversión 4,0 % vs 14,8 % del resto.
   - `poutcome = unknown` (81,7 %) = cliente nunca contactado antes. Conversión 9,2 % vs 23,1 %.
   - `education = unknown` (4,1 %) convierte 14,0 %, por encima de la base: no se parece a la
     moda (`secondary`, 10,5 %), así que imputarlo sería incorrecto.
3. **`pdays = -1`, `previous = 0` y `poutcome = unknown` son la misma información**: las
   33.249 filas con `pdays = -1` tienen `previous = 0` y `poutcome = unknown`. Solo 5 filas
   tienen contacto previo con `poutcome = unknown`. Conversión de nunca contactados: 9,2 %;
   de contactados antes: 23,1 %; con `poutcome = success`: **64,8 %**.
4. **Drift temporal**: la conversión sube de 2,9 % en el primer décimo del archivo a
   47,5 % en el último. El archivo está ordenado cronológicamente.
5. **`month`**: la correlación de Spearman entre volumen y tasa de conversión por mes es
   **−0,84** (p < 0,001). Mayo concentra el 30,4 % de los contactos con 6,7 % de conversión,
   mientras que marzo, septiembre, octubre y diciembre tienen tasas del 43–53 % con 0,5–1,6 %
   del volumen cada uno. Es estacionalidad de la operación (y proxy del período), no del cliente.
6. **Outliers**:
   - `balance` va de −8.019 a 102.127, con asimetría 8,5, 10,4 % de outliers (regla IQR),
     8,4 % de negativos y 7,8 % de ceros. No son errores → transformación, no eliminación.
   - `campaign` va de 1 a 63 contactos, p99 = 17. La conversión cae de 14,7 % con 1
     contacto a 6,4 % con 5 o más.
7. **`duration`**: ROC-AUC de 0,81 por sí sola. Mediana 164 s en no-conversiones y 424 s en
   conversiones. Llamadas < 60 s: 0,2 % de conversión. Hay 3 filas con `duration = 0`,
   ninguna convierte. Se conoce después de la llamada → **leakage**.
8. **Ranking preliminar** (información mutua, sin `duration`): `poutcome` > `pdays` >
   `month` > `balance` > `previous` > `contact` > `age` > `housing`. `default` casi no aporta
   (V de Cramér 0,02).
9. Sin multicolinealidad severa entre numéricas; la única asociación marcada es
   `pdays`–`previous`, que es estructural (hallazgo 3).

## 2.4 Implicancias para la preparación (Fase 3)

| Hallazgo | Acción |
|---|---|
| `unknown` informativo | Mantener como categoría |
| `pdays = -1` no numérico | `previously_contacted` + `pdays_clean` (NaN si -1) |
| `balance` asimétrico con negativos/ceros | Log con signo + banderas `balance_negative`, `balance_zero` |
| `campaign` con cola larga | Winsorizar al p99 (ajustado dentro de cada fold) |
| Edad con efecto no lineal | `age_group` además de `age` |
| `poutcome = success` muy predictivo | Bandera `poutcome_success` |
| `contact = unknown` = período sin registro | Bandera `contact_known` |
| `month` = volumen/período | `month` como categoría + `season` |
| `day` cíclico | Seno/coseno |
| Préstamos reducen conversión | `n_credit_products` |
| `duration` es leakage | Excluir del modelo entregable |
