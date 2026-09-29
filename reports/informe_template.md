# Predicción de captación de clientes en campañas de marketing bancario

<!-- center -->**Trabajo Práctico Final — Aprendizaje Artificial**

<!-- center -->Maestría en Gestión y Desarrollo de Inteligencia Artificial · Universidad CAECE

<!-- center -->Docentes: Juan Azcurra · Paul Pablo Hernán

<!-- center -->Autor: Fernando Caceres

<!-- center -->Fecha: {{fecha}}

<!-- pagebreak -->

## Resumen ejecutivo

El trabajo construye un modelo que **ordena a los clientes potenciales según su probabilidad de
convertirse en clientes del banco**, para decidir a quién contactar en una campaña telefónica con
presupuesto limitado. Se siguió la metodología CRISP-DM de punta a punta: comprensión del negocio y
de los datos, preparación, modelado, evaluación y despliegue.

**Tratamiento de los datos.** Durante la inspección se detectó que **las {{overlap}} filas del archivo
de test estaban contenidas en el archivo de entrenamiento** (el dataset *Bank Marketing* de UCI
distribuye `bank.csv` como una muestra de `bank-full.csv`). Esas filas se eliminaron del
entrenamiento, que quedó en {{train_clean}} registros, para que la evaluación final se hiciera sobre
datos no vistos.

**Modelo pre-contacto.** Se excluyó la variable `duration` (duración de la llamada): solo se conoce
después de llamar, por lo que usarla constituiría una fuga de información.

**Modelos comparados.** Se compararon diez algoritmos de distintas familias con el mismo protocolo
de validación cruzada estratificada de 5 folds. También se evaluaron:

- tres estrategias de desbalance;
- dos codificaciones de variables categóricas;
- un ajuste de hiperparámetros con Optuna.

**Modelo elegido.** Mediante una matriz de decisión ponderada se eligió **{{model_label}}** sin
tratamiento de desbalance y con categorías nativas. Obtuvo en validación cruzada:

- PR-AUC de {{cv_final_pr}} ± {{cv_final_pr_sd}};
- ROC-AUC de {{cv_final_roc}}.

**Resultados en el hold-out** ({{test_rows}} clientes, evaluado una única vez):

| Métrica | Modelo final | Regresión logística | Azar |
|---|---|---|---|
| ROC-AUC | {{test_final_roc_auc}} | {{test_logreg_roc_auc}} | — |
| PR-AUC | {{test_final_pr_auc}} | {{test_logreg_pr_auc}} | {{test_dummy_pr_auc}} |

Estos valores están dentro del rango de referencia para este dataset sin `duration` (0,75–0,80).

**Lectura de negocio:**

- Contactando al 10 % de clientes mejor rankeado se captura el {{gain10}} de las conversiones,
  con un lift de {{dec1_lift}}.
- Contactando al 20 % se captura el {{gain20}}.
- Con los supuestos económicos adoptados (valor de una conversión = 20 veces el costo de una
  llamada), el umbral de máximo beneficio es {{thr}}. Con él se contacta al {{test_contacted}} de
  los clientes y se captura el {{test_rec}} de las conversiones.
- El beneficio esperado supera en {{biz_uplift_pct}} al de contactar a todos.
- El valor del modelo crece cuando la llamada es más cara en relación con la conversión: con un
  ratio 10:1 el modelo triplica el beneficio de la estrategia de contactar a todos.

**Explicabilidad y despliegue:**

- La explicabilidad (SHAP, regresión logística y un árbol de reglas) coincide en las variables
  clave: el mes del contacto, el canal, el saldo, la tenencia de préstamos y el éxito en campañas
  anteriores.
- El modelo se despliega mediante una API REST (FastAPI) y una interfaz web (Streamlit) que
  permiten puntuar clientes individuales o listados completos.

<!-- pagebreak -->

## 1. Comprensión del negocio

### 1.1 Problema y decisión

Un banco realiza campañas de marketing telefónico para captar clientes para un depósito a plazo.
Cada contacto tiene un costo y solo una minoría de los contactados se convierte
({{rate_train}} en los datos de entrenamiento).

La decisión que apoya el modelo es la siguiente:

> Dado un listado de clientes potenciales y un presupuesto de contactos, ordenarlos según su
> probabilidad de conversión y contactar a los primeros *k*.

Por lo tanto, el modelo se evalúa principalmente como una **herramienta de ranking**. La decisión
binaria (llamar o no llamar) surge de aplicar un umbral económico sobre la probabilidad estimada.

### 1.2 Restricción: información disponible antes del contacto

El modelo se utiliza **antes** de llamar. En consecuencia, se excluye toda variable que solo se
conoce después del contacto. El caso central es `duration`:

- la duración de la llamada se conoce al finalizarla;
- además, es en buena medida *consecuencia* del interés del cliente.

El modelo entregable se denomina **modelo pre-contacto**. Se entrenó una variante con `duration`
exclusivamente para cuantificar el efecto de esa fuga (sección 5.5).

### 1.3 Métricas

| Tipo | Métrica | Justificación |
|---|---|---|
| Primaria | PR-AUC (average precision) | Resume la calidad del ranking sobre la clase minoritaria; su valor al azar es la prevalencia (~0,117) |
| Primaria | Lift y gain en el top 10 %, 20 % y 30 % | Traducen el ranking a la decisión: qué proporción de las conversiones se captura contactando a una fracción de los clientes |
| Secundaria | ROC-AUC | Permite contrastar con la literatura |
| Secundaria | Precision, recall y F1 al umbral operativo | Describen la decisión binaria |
| Secundaria | Brier y curva de calibración | Verifican que las probabilidades puedan interpretarse como tales, lo cual es necesario para el cálculo económico |

**Por qué no accuracy.** Con 88,3 % de clientes que no se convierten, un modelo que responde
siempre "no" alcanza 88,3 % de accuracy sin identificar a ningún cliente. Accuracy premia acertar
la clase mayoritaria, mientras que el valor de negocio está en encontrar a la minoría que se
convierte.

### 1.4 Marco económico

Se definió un costo por contacto **C = 1** y un valor esperado por conversión **V = 20**. El
beneficio esperado de contactar a un conjunto de clientes es:

$$\text{Beneficio} = V \cdot TP - C \cdot (TP + FP)$$

Contactar a un cliente con probabilidad *p* conviene si *p·V > C*, es decir, si *p > C/V = 0,05*.

Los valores C y V son **supuestos ilustrativos**; lo relevante es su cociente. Por ello se analiza
la sensibilidad para V/C = 5, 10, 20 y 50. El banco puede reemplazarlos por valores reales:

- **C**: costo del operador, de la telefonía y del desgaste por contacto;
- **V**: margen neto del producto más el valor de vida del cliente.

Ambos valores se modifican en `config.py` y el pipeline recalcula el umbral y el análisis económico.

<!-- pagebreak -->

## 2. Comprensión de los datos

### 2.1 Archivos y verificaciones

**Archivos recibidos:**

| Archivo | Equivale a | Filas | Tasa de conversión |
|---|---|---|---|
| `banca_train.csv` | `bank-full.csv` | {{train_raw}} | {{rate_train_raw}} |
| `banca_test.csv` | `bank.csv` | {{test_rows}} | {{rate_test}} |

Ambos archivos tienen 17 columnas, separador `;` y encoding ASCII. Los datos corresponden al
dataset *Bank Marketing* (Moro, Cortez y Rita, 2014; UCI Machine Learning Repository), con
campañas realizadas entre mayo de 2008 y noviembre de 2010.

**Verificaciones realizadas:**

- El esquema coincide con el diccionario de la consigna.
- No hay valores nulos ni duplicados exactos dentro de cada archivo.

**Hallazgo crítico: solapamiento.**

- Una comparación fila por fila mostró que las {{overlap}} filas de test (100 %) aparecen idénticas
  en train.
- Utilizar los archivos tal como fueron entregados habría implicado evaluar el modelo sobre datos
  vistos durante el entrenamiento.
- Esas filas se eliminaron del conjunto de entrenamiento, que quedó en {{train_clean}} registros
  ({{pos_train}} conversiones, {{rate_train}}).
- El proceso completo está registrado en la tabla 1 y en `docs/03_preparacion_datos.md`.

**Naturaleza de la partición.** El archivo de entrenamiento está ordenado cronológicamente, por
lo que se verificó si el test correspondía a un período posterior:

- Las posiciones de las filas de test dentro del archivo son compatibles con una distribución
  uniforme (Kolmogorov-Smirnov D = {{ks_d}}, p = {{ks_p}}).
- Las distribuciones de mes, canal, resultado previo y target no difieren entre ambos conjuntos
  (χ² con p = {{chi_month_p}}, {{chi_contact_p}}, {{chi_pout_p}} y {{chi_y_p}}).
- Conclusión: la partición es **aleatoria**, y se valida con validación cruzada estratificada.

*Tabla 1. Registro de operaciones sobre filas.*

{{table:prep_steps}}

### 2.2 Análisis exploratorio

El análisis exploratorio se realizó solo sobre el conjunto de entrenamiento limpio
(`notebooks/01_eda.ipynb`).

![Tasa de conversión por categoría](figures/eda_conversion_by_category.png)

**Valores `unknown`.** No constituyen faltantes aleatorios:

- **`contact = unknown`** ({{pct_unk_contact}} de los registros):
  - convierte {{unk_contact_conv}} frente a {{unk_contact_rest}} del resto;
  - ocupa el 100 % del primer 20 % del archivo, es decir, corresponde a un período sin registro
    del canal.
- **`poutcome = unknown`** ({{pct_unk_poutcome}}) equivale a "nunca contactado en una campaña
  previa".
- **`education = unknown`** convierte {{unk_education_conv}}, más que la moda.

En consecuencia, se mantienen como una categoría propia.

![Unknown y conversión a lo largo del archivo](figures/eda_unknown_by_file_position.png)

**`pdays = -1`.** Las {{n_pdays_neg}} filas con `pdays = -1` ({{pct_pdays_neg}}) son exactamente
las que tienen `previous = 0` y `poutcome = unknown`. El valor -1 es un código, no una cantidad de
días. Tasas de conversión:

- nunca contactados: {{conv_never}};
- contactados en una campaña previa: {{conv_before}};
- con resultado previo exitoso: **{{conv_success}}**.

**Mes y volumen.**

- Mayo concentra el {{may_share}} de los contactos con una tasa de {{may_rate}}.
- Los meses con pocas llamadas (marzo, septiembre, octubre, diciembre) superan el 40 %.
- La correlación de Spearman entre volumen y tasa es {{spearman_month}}.

El efecto refleja la operación del banco y el período, más que una estacionalidad del cliente.

![Volumen y conversión por mes](figures/eda_month_volume_rate.png)

**Outliers:**

- **`balance`**:
  - varía entre {{bal_min}} y {{bal_max}}, con asimetría {{bal_skew}} y {{bal_iqr}} de outliers
    según la regla IQR;
  - tiene {{bal_neg}} de saldos negativos y {{bal_zero}} de saldos nulos;
  - son valores reales, por lo que se transforman en lugar de eliminarse.
- **`campaign`**:
  - llega a {{camp_max}} contactos (p99 = {{camp_p99}});
  - la conversión cae de {{conv_camp1}} con un contacto a {{conv_camp5}} con cinco o más.

**`duration` como leakage:**

- Por sí sola alcanza un ROC-AUC de {{dur_auc}}.
- La mediana es de {{dur_med_no}} s en los que no se convierten y de {{dur_med_yes}} s en los que
  se convierten.
- Las llamadas de menos de 60 s convierten {{dur_short_conv}}.
- Hay {{dur_zero}} llamadas con duración 0.

![Duration como leakage](figures/eda_duration_leakage.png)

![Ranking preliminar de variables](figures/eda_feature_ranking.png)

<!-- pagebreak -->

## 3. Preparación de los datos

Todas las transformaciones de columnas se implementaron dentro de un `Pipeline` de scikit-learn o
imbalanced-learn. Esto tiene dos consecuencias:

- en cada fold de validación se ajustan únicamente con los datos de entrenamiento, lo que evita
  fugas de información;
- el modelo final guardado recibe los datos crudos y aplica exactamente las mismas
  transformaciones.

*Tabla 2. Variables derivadas y su justificación.*

{{table:features}}

**Preprocesamiento según la familia de modelo:**

- **Modelos lineales, KNN, MLP y SVM**: imputación por mediana, estandarización y codificación
  one-hot (67 columnas).
- **Árboles y boosting**: sin escalado (son invariantes a transformaciones monótonas), con one-hot
  o con categorías nativas.
- **Naive Bayes categórico**: discretización en quintiles.

**Decisiones deliberadas:**

- no se imputaron los `unknown`;
- no se eliminaron outliers;
- no se utilizó `duration`.

El documento `docs/03_preparacion_datos.md` detalla cada operación.

<!-- pagebreak -->

## 4. Modelado

### 4.1 Protocolo

Se utilizó `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre los {{train_clean}}
registros. Todos los modelos usan exactamente la misma partición y se reporta la media ± el desvío
estándar entre folds. El conjunto de test no interviene en ninguna decisión de esta fase.

### 4.2 Algoritmos y justificación

Se compararon diez algoritmos que representan las familias vistas en la materia:

| Algoritmo | Familia | Por qué se incluye |
|---|---|---|
| Dummy (prior) | Referencia | Piso de comparación |
| Regresión logística | Lineal | Baseline interpretable |
| Árbol de decisión | Árboles | Genera reglas explicables |
| KNN (k = 51) | Basado en instancias | Clasificación por similitud |
| Naive Bayes categórico | Probabilístico | Modelo generativo |
| Random Forest | Bagging | Reduce varianza |
| LightGBM y XGBoost | Boosting | Estado del arte en datos tabulares |
| MLP | Redes neuronales | Representante de la familia |
| SVM RBF | Kernels | Frontera no lineal de máximo margen |

La SVM se entrenó sobre una submuestra estratificada de 10.000 filas por su costo cuadrático.

*Tabla 3. Resultados de validación cruzada (modelo pre-contacto).*

{{table:cv}}

![Comparación de algoritmos](figures/model_cv_comparison.png)

**Interpretación de los resultados:**

- **Ensambles de árboles**: forman el grupo superior (PR-AUC {{cv_random_forest_pr}}–{{cv_lightgbm_pr}}),
  casi cuatro veces el valor del Dummy ({{cv_dummy_pr}}).
- **Regresión logística** ({{cv_logreg_pr}}): la brecha con los ensambles mide el valor de capturar
  no linealidades e interacciones.
- **KNN y Naive Bayes**: rinden menos. KNN se ve afectado por la alta dimensionalidad del one-hot y
  Naive Bayes por la suposición de independencia, dado que existen variables redundantes.
- **SVM**: presenta un ROC-AUC competitivo pero la peor PR-AUC entre los modelos no triviales;
  ordena peor la parte alta del ranking.

### 4.3 Codificación de categóricas y desbalance

*Tabla 4. Codificación en boosting.*

{{table:encoding}}

**Codificación.** Se eligió la codificación nativa para LightGBM y one-hot para XGBoost. La regla
fue preferir la opción nativa salvo que la otra superara la PR-AUC en más de 0,003.

*Tabla 5. Estrategias de desbalance sobre los tres mejores modelos.*

{{table:imbalance}}

![Estrategias de desbalance](figures/imbalance_comparison.png)

**Desbalance.** Las tres estrategias difieren en PR-AUC menos que un desvío estándar entre folds.
Sin embargo:

- Los **pesos de clase descalibran** las probabilidades: el Brier de LightGBM pasa de
  {{imb_lightgbm_none_brier}} a {{imb_lightgbm_weights_brier}}.
- **SMOTE** genera clientes sintéticos por interpolación entre variables one-hot, lo cual carece
  de interpretación, y empeora al Random Forest.

Como el umbral y el análisis económico requieren probabilidades calibradas, se eligió **no aplicar
tratamiento y ajustar el umbral**.

### 4.4 Ajuste de hiperparámetros

Se utilizó Optuna con sampler TPE (semilla 42), 60 trials por modelo y un tope de 15 minutos por
búsqueda. El objetivo fue la PR-AUC media en los mismos 5 folds.

*Tabla 6. Resultados del tuning.*

{{table:tuning}}

![Convergencia Optuna LightGBM](figures/tuning_convergence_lightgbm.png)

**Lectura:**

- La mejora obtenida es pequeña: de {{tune_lightgbm_default}} a {{tune_lightgbm_best}} en LightGBM.
  El techo lo impone la información disponible antes del contacto, más que el algoritmo.
- La PR-AUC del mejor trial es levemente optimista, porque se seleccionó sobre los mismos folds.
  La estimación insesgada es la del conjunto de test.

### 4.5 Selección del modelo final

**Criterios de la matriz de decisión** (cada uno normalizado entre candidatos):

| Criterio | Peso |
|---|---|
| PR-AUC | 35 % |
| Lift@20 % | 25 % |
| Estabilidad entre folds | 15 % |
| Interpretabilidad | 15 % |
| Velocidad de inferencia | 10 % |

*Tabla 7. Matriz de decisión.*

{{table:decision}}

**Resultado:**

- Se seleccionó **{{model_label}} tuneado** (puntaje {{dm_winner_score}}), con los siguientes
  hiperparámetros:

  | Hiperparámetro | Valor |
  |---|---|
  | Árboles | {{hp_n_estimators}} |
  | learning rate | {{hp_lr}} |
  | Hojas | {{hp_leaves}} |
  | min_child_samples | {{hp_min_child}} |
  | subsample | {{hp_subsample}} |
  | colsample_bytree | {{hp_colsample}} |

- La regresión logística ({{dm_logreg_score}}) es la más estable e interpretable, pero pierde
  alrededor de 0,05 de PR-AUC. Se conserva como baseline y como contraste de explicabilidad.
- La menor interpretabilidad de LightGBM se compensa con SHAP.

### 4.6 Umbral operativo

El umbral se eligió con las predicciones *out-of-fold* del modelo final, maximizando el beneficio
esperado con V = 20 y C = 1.

**Umbral de máximo beneficio: {{thr}}.** Coincide con el umbral teórico C/V, lo que confirma
probabilidades bien calibradas. Con este umbral:

- se contacta al {{thr_contacted}} de los clientes;
- el recall es {{thr_recall}};
- el beneficio es {{thr_profit}}, frente a {{thr_profit_all}} si se contactara a todos.

**Umbral de máximo F1: {{thr_f1}}.**

- Contacta solo al {{thr_f1_contacted}} y reduce el beneficio a {{thr_f1_profit}}.
- F1 pondera por igual precision y recall, e ignora que una conversión vale veinte veces lo que
  cuesta una llamada.

![Selección del umbral](figures/threshold_selection.png)

*Tabla 8. Sensibilidad del umbral al ratio V/C (predicciones OOF de train).*

{{table:sens_oof}}

<!-- pagebreak -->

## 5. Evaluación y análisis

### 5.1 Resultados en el hold-out

**Condiciones de la evaluación:**

- El modelo, los hiperparámetros y el umbral quedaron fijados antes de leer el conjunto de test.
- La evaluación se realizó **una única vez** (`reports/tables/holdout_usage.json`).
- El umbral de la regresión logística también se eligió con sus propias predicciones *out-of-fold*.

*Tabla 9. Métricas en test.*

{{table:test}}

*Tabla 10. Validación cruzada vs test (modelo final).*

{{table:test_vs_cv}}

La PR-AUC en test es {{gap_pr}} menor que en validación cruzada. La diferencia está dentro de dos
desvíos y es esperable por dos razones:

- el leve optimismo del tuning;
- el menor tamaño del test ({{pos_test}} conversiones).

![Curvas ROC y PR](figures/test_roc_pr_curves.png)

### 5.2 Decisión al umbral operativo

![Matriz de confusión](figures/test_confusion_matrix.png)

**Resultados con umbral {{thr}}:**

- el modelo recomienda contactar al {{test_contacted}} de los clientes;
- captura {{cm_tp}} de las {{pos_test}} conversiones (recall {{test_rec}}, precision {{test_prec}});
- genera {{cm_fp}} falsos positivos y {{cm_fn}} falsos negativos.

Los falsos positivos son aceptables por diseño: una llamada sin conversión cuesta 1 y una
conversión perdida cuesta 20.

### 5.3 Ranking, deciles y beneficio

*Tabla 11. Deciles en test (decil 1 = mayor probabilidad).*

{{table:deciles}}

![Gain y lift](figures/test_gain_lift.png)

*Tabla 12. Traducción a negocio (test, V = 20, C = 1).*

{{table:business}}

**Lectura de negocio:**

- **Top 10 %**: el decil superior concentra {{dec1_conv}} conversiones, con una tasa de {{dec1_rate}}
  y un lift de {{dec1_lift}}.
- **Top 20 %**: contactando al 20 % mejor rankeado se captura el {{gain20}} de las conversiones.
- **Umbral operativo**: el beneficio esperado es {{biz_profit}} frente a {{biz_all}} de contactar a
  todos (+{{biz_uplift}}, {{biz_uplift_pct}}).

*Tabla 13. Sensibilidad V/C en test (umbrales elegidos con train).*

{{table:sens_test}}

![Curvas de beneficio](figures/test_profit_curves.png)

**El valor del modelo depende del ratio V/C:**

| V/C | Contactar a todos | Modelo | Lectura |
|---|---|---|---|
| 5 | {{sens5_all}} (pérdida) | {{sens5_model}} | El modelo convierte una pérdida en ganancia |
| 10 | {{sens10_all}} | {{sens10_model}} | El modelo triplica el beneficio |
| 50 | — | — | Conviene contactar a casi todos (el modelo contacta al {{sens50_contacted}}) |

Cuando existe un presupuesto fijo de contactos, lo relevante es la calidad del ranking.

### 5.4 Calibración

![Calibración](figures/test_calibration.png)

**LightGBM** (Brier {{test_final_brier}}): las probabilidades siguen de cerca la diagonal, lo que
permite usarlas para estimar conversiones y beneficio esperados.

**Regresión logística con pesos de clase** (Brier {{test_logreg_brier}}): sobreestima
sistemáticamente la probabilidad. Por ese motivo su umbral óptimo es {{test_logreg_thr}}.

### 5.5 Cuantificación del leakage

*Tabla 14. Pre-contacto vs con `duration` (test).*

{{table:leakage}}

**Magnitud del efecto:**

| Métrica | Pre-contacto | Con `duration` |
|---|---|---|
| ROC-AUC en test | {{test_final_roc_auc}} | {{test_with_duration_roc_auc}} |
| ROC-AUC en CV | {{leak_cv_pre_roc}} | {{leak_cv_dur_roc}} |

**Por qué la mejora es engañosa:**

- Para conocer la duración es necesario haber realizado la llamada, que es justamente lo que el
  modelo debe decidir.
- La relación causal está invertida: la llamada es larga porque el cliente está interesado.

Un modelo con `duration` exhibiría métricas excelentes en validación y no podría utilizarse en
producción.

### 5.6 Explicabilidad

![SHAP global](figures/shap_summary.png)

*Tabla 15. Importancia global (media del valor absoluto de SHAP, test).*

{{table:shap}}

**Variables más influyentes:**

- **{{shap1}}** ({{shap1_pct}}):
  - marzo, septiembre, octubre y diciembre aumentan fuertemente la probabilidad;
  - enero, mayo y agosto la reducen;
  - como se observó en el EDA, refleja la operación y el período. Es el principal riesgo al
    aplicar el modelo en campañas futuras.
- **Canal de contacto** ({{shap_contact_share}}, entre `contact` y `contact_known`): `unknown`
  reduce la probabilidad.
- **Saldo**: tiene un efecto creciente a partir de saldos positivos.
- **Préstamos**: tener uno o más reduce la probabilidad.
- **Éxito en la campaña anterior**: afecta a pocos clientes, pero con un efecto muy grande.
- **Cantidad de contactos en la campaña**: un número alto la reduce.

![Dependencias SHAP](figures/shap_dependence.png)

**Explicaciones locales:**

- **Verdadero positivo típico**: su probabilidad se explica principalmente por el mes (octubre)
  y la ausencia de préstamos.
- **Falso positivo de mayor probabilidad**: un jubilado con éxito en la campaña anterior,
  contactado en marzo. Todas las señales indicaban conversión, por lo que el error es razonable.
- **Falso negativo de menor probabilidad**: un cliente con contacto `unknown` en mayo, saldo cero y
  dos préstamos, es decir, el perfil de menor conversión del dataset.

![Falso positivo](figures/shap_local_fp.png)

![Falso negativo](figures/shap_local_fn.png)

**Contraste con la regresión logística.** Los coeficientes de mayor magnitud también corresponden
al mes, al grupo de 65 años o más y al saldo. La coincidencia entre dos familias de modelos da
confianza en que las relaciones son reales.

*Tabla 16. Coeficientes principales de la regresión logística.*

{{table:logreg}}

**Reglas de negocio.** Un árbol de profundidad 3 entrenado con train (ROC-AUC en test
{{rules_roc}}) resume las reglas en un formato comunicable:

- éxito en la campaña anterior y sin hipoteca → más del 65 % de conversión;
- sin éxito previo, con préstamos y contacto no celular → alrededor del 4 %.

*Tabla 17. Reglas del árbol de profundidad 3.*

{{table:rules}}

![Árbol de reglas](figures/rules_tree_depth3.png)

### 5.7 Análisis de errores y por segmento

*Tabla 18. Perfil de aciertos y errores (test).*

{{table:errors}}

**Falsos negativos.** Se parecen a los verdaderos negativos:

| Rasgo | Falsos negativos | Verdaderos positivos |
|---|---|---|
| Contacto `unknown` | {{fn_unknown}} | {{tp_unknown}} |
| Contactados en mayo | {{fn_may}} | {{tp_may}} |
| Saldo medio | {{fn_balance}} | {{tp_balance}} |
| Contactos en la campaña | {{fn_camp}} | {{tp_camp}} |
| Contactados antes | {{fn_prev}} | {{tp_prev}} |

Son conversiones que ocurrieron dentro de campañas masivas sin señales previas que las distingan.

*Tabla 19. Desempeño por segmento (segmentos con al menos 20 conversiones).*

{{table:segments}}

![Métricas por segmento](figures/segment_metrics.png)

**Segmentos con peor desempeño:**

- **`contact = unknown`**: ROC-AUC {{seg_unknown_roc}}, {{seg_unknown_n}} clientes y
  {{seg_unknown_rate}} de conversión.
- **Mayores de 65 años**: ROC-AUC {{seg_65_roc}}. Convierten mucho en promedio ({{seg_65_rate}}),
  pero el modelo discrimina poco dentro del grupo.

Para esos segmentos convendría incorporar información adicional.

### 5.8 Razonabilidad de los resultados

*Tabla 20. Chequeos de razonabilidad.*

{{table:reasonableness}}

Se cumplen {{reason_ok}} chequeos:

- El ROC-AUC en test ({{test_final_roc_auc}}) está dentro del rango de referencia de 0,75–0,80 para
  este dataset sin `duration`.
- La inclusión de `duration` lleva el valor por encima de 0,93.
- Un modelo pre-contacto con ROC-AUC mayor a 0,90 habría sido una señal de fuga de información.

<!-- pagebreak -->

## 6. Despliegue

El modelo final se serializó como un único pipeline (`models/model_final.joblib`) junto con su
metadata (`models/metadata.json`). La metadata incluye:

- hiperparámetros;
- métricas de validación cruzada;
- umbral;
- cortes de deciles;
- hash de los datos;
- versiones de las librerías.

![Pipeline del modelo final](figures/diagrama_pipeline_modelo.png)

**API REST con FastAPI** (`app/api/main.py`, documentación automática en `/docs`):

| Endpoint | Función |
|---|---|
| `GET /health` | Estado del servicio |
| `GET /model/info` | Metadata del modelo |
| `POST /predict` | Recibe un cliente en JSON y devuelve la probabilidad, la recomendación, el decil, las tres variables SHAP de mayor peso y una explicación en texto |
| `POST /predict/batch` | Recibe un CSV y devuelve el mismo CSV rankeado con probabilidad, decil y recomendación |

La validación de las entradas se realiza con pydantic:

- enumeraciones para las variables categóricas;
- rangos para las numéricas;
- consistencia entre `pdays` y `previous`;
- **rechazo explícito de `duration`**.

**Interfaz con Streamlit** (`app/ui/streamlit_app.py`). Consume la API a través de la variable de
entorno `API_URL` y tiene dos pestañas:

- **Cliente individual**: formulario, resultado y explicación.
- **Campaña**:
  - carga de un CSV y ranking;
  - selección del porcentaje de clientes o del presupuesto;
  - conversiones y beneficio esperados;
  - descarga del listado priorizado.

**Contenedores.** El `Dockerfile` y el `docker-compose.yml` levantan la API y la UI. También se
documenta la ejecución local sin Docker.

**Pruebas.** La suite de tests automáticos (`pytest`) verifica:

- las transformaciones de variables;
- la ausencia de fugas: `duration` fuera del modelo e independencia entre train y test;
- la consistencia del modelo y de la metadata;
- todos los endpoints de la API, incluido el rechazo de `duration`.

<!-- pagebreak -->

## 7. Conclusiones y trabajo futuro

**Conclusiones:**

1. Con la información disponible **antes** del contacto es posible priorizar clientes con un
   poder de discriminación moderado y consistente: ROC-AUC {{test_final_roc_auc}}, PR-AUC
   {{test_final_pr_auc}} y lift de {{dec1_lift}} en el decil superior.
2. El valor de negocio depende del cociente entre el valor de una conversión y el costo de un
   contacto:
   - es alto cuando las llamadas son caras o el presupuesto es limitado;
   - es marginal cuando conviene contactar a casi todos.
3. Las decisiones metodológicas más relevantes no fueron algorítmicas sino de tratamiento de los
   datos:
   - detectar y eliminar el solapamiento entre train y test;
   - excluir `duration`;
   - preferir probabilidades calibradas en lugar de técnicas de rebalanceo.
4. Los ensambles de árboles superan a los modelos lineales, pero la mejora por tuning es pequeña:
   el límite lo impone la información disponible.

**Limitaciones y trabajo futuro:**

- **Temporalidad.** Los datos son una serie temporal (2008–2010) con un fuerte drift en la tasa
  de conversión, y la partición provista es aleatoria. Antes de usar el modelo en campañas nuevas
  se recomienda:
  - validar con un esquema temporal;
  - monitorear el efecto de `month`.
- **Parámetros económicos.** Reemplazar los supuestos de C y V por valores reales del banco.
- **Nuevas fuentes de información.** Incorporar variables transaccionales y de relación con el
  cliente para mejorar los segmentos con peor desempeño.
- **Uplift modeling.** Estimar el efecto causal de la llamada, en lugar de la probabilidad de
  conversión, requeriría datos de un grupo de control no contactado.

<!-- pagebreak -->

## Anexos

### Anexo A. Registro de decisiones (decision log)

{{include:docs/decision_log.md}}

### Anexo B. Diagrama del flujo completo

![Flujo del proyecto](figures/diagrama_flujo_proyecto.png)

### Anexo C. Instrucciones de reproducción

```
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
# copiar banca_train.csv y banca_test.csv en data/raw/
python -m bank_captacion.pipeline # reconstruye datos, EDA, modelos, evaluación, notebooks e informe
python -m pytest -q               # tests
python -m uvicorn app.api.main:app --port 8000
python -m streamlit run app/ui/streamlit_app.py
```

Todas las tablas completas están en `reports/tables/` y todas las figuras en `reports/figures/`.
