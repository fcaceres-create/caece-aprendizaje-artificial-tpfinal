"""Contenido de los notebooks 02 (modelado) y 03 (evaluación y negocio)."""

from __future__ import annotations

from .notebooks_builder import SETUP, _nb

COLS = ("['model','label','family','roc_auc_fmt','pr_auc_fmt','lift@20_fmt','brier_fmt',"
        "'gain@20_mean','fit_time_s_mean','predict_ms_per_1k_mean']")


def modelado_notebook():
    return _nb([
        ("md", "# 02 · Modelado\n\nFase 4 de CRISP-DM. La lógica está en `src/bank_captacion/train.py` y "
               "`models.py`; este notebook presenta los resultados que generó el pipeline "
               "(`python -m bank_captacion.pipeline`). Todo se evalúa sobre el **train limpio (40.690 filas)** "
               "con la misma partición; el hold-out no se usa en esta fase."),
        ("code", SETUP),
        ("md", "## 1. Protocolo de validación\n\n`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`: "
               "cada fold conserva la proporción de conversiones (~11,7 %). Todo el preprocesamiento "
               "(winsorización, imputación, escalado, codificación, SMOTE) está dentro del pipeline y se ajusta "
               "solo con los 4 folds de entrenamiento de cada iteración."),
        ("code", "table('cv_folds.csv')"),
        ("md", "## 2. Comparación de 10 algoritmos (modelo pre-contacto, sin `duration`)\n\n"
               "Cada modelo representa una familia distinta vista en la materia. En esta primera comparación se "
               "usan pesos de clase en los modelos que los admiten."),
        ("code", f"cv = pd.read_csv(cfg.TABLES_DIR / 'cv_results.csv')\ndisplay(cv[{COLS}])\n"
                 "show('model_cv_comparison.png')"),
        ("md", "**Lectura:**\n\n"
               "- Los tres ensambles de árboles (LightGBM, XGBoost, Random Forest) forman el grupo superior "
               "(PR-AUC ≈ 0,45–0,46, ROC-AUC ≈ 0,80), casi 4 veces el piso del Dummy (0,117).\n"
               "- La MLP queda cerca (0,43), pero sin ventaja y con menor interpretabilidad.\n"
               "- La regresión logística (0,41) es el mejor modelo interpretable y el baseline de referencia: "
               "la brecha con boosting (~0,05 de PR-AUC) mide el valor de capturar no linealidades e interacciones.\n"
               "- KNN y Naive Bayes rinden menos: KNN sufre con 67 dimensiones one-hot y NB con la suposición de "
               "independencia (hay variables redundantes como pdays/previous/poutcome).\n"
               "- SVM RBF entrenado con 10.000 filas tiene buen ROC-AUC pero peor PR-AUC: ordena bien en general "
               "pero peor en la parte alta del ranking, que es la que importa.\n"
               "- El árbol de profundidad 6 es el más débil pero es el más explicable (se usa como \"reglas de "
               "negocio\" en la Fase 5)."),
        ("md", "## 3. Codificación de categóricas en boosting: one-hot vs nativa"),
        ("code", "table('encoding_comparison.csv')[['model','kind','pr_auc_fmt','roc_auc_fmt','lift@20_fmt']]"),
        ("md", "En LightGBM la codificación nativa es levemente mejor (y más simple: 24 columnas en vez de 67), así "
               "que se adopta. En XGBoost el one-hot es mejor por más que la tolerancia de 0,003, así que se "
               "mantiene one-hot."),
        ("md", "## 4. Estrategias de desbalance (sobre los 3 mejores)\n\n"
               "(a) sin tratamiento (el umbral se ajusta después), (b) pesos de clase, (c) SMOTE dentro del pipeline "
               "(solo sobre los folds de entrenamiento)."),
        ("code", "imb = table('imbalance_results.csv')[['model','imbalance','kind','pr_auc_fmt','roc_auc_fmt',"
                 "'lift@20_fmt','brier_fmt']]\nshow('imbalance_comparison.png')"),
        ("md", "**Resultado:** las tres estrategias quedan dentro de un desvío estándar entre sí: el desbalance "
               "(11,7 %) es moderado y los modelos de ranking no lo necesitan. Pero **pesos y SMOTE empeoran la "
               "calibración** (Brier de LightGBM: 0,081 sin tratamiento vs 0,144 con pesos), porque inflan "
               "artificialmente la probabilidad de la clase positiva. Como el umbral y el beneficio esperado se "
               "calculan con probabilidades, se elige **sin tratamiento + ajuste de umbral**. Regla aplicada: si la "
               "diferencia de PR-AUC es menor a 0,003 se prefiere la estrategia que no distorsiona las "
               "probabilidades. SMOTE además agrega costo de cómputo y genera clientes sintéticos sobre variables "
               "one-hot, que no tienen interpretación (interpolar entre \"married\" y \"single\")."),
        ("md", "## 5. Tuning con Optuna (2 mejores modelos)\n\nSampler TPE con semilla 42, 60 trials por modelo, "
               "tope de 15 minutos, objetivo = PR-AUC media en los mismos 5 folds."),
        ("code", "tun = table('tuning_summary.csv')\nfor _, r in tun.iterrows():\n"
                 "    print(r['model'], json.loads(r['best_params']))\n"
                 "show('tuning_convergence_lightgbm.png')\nshow('tuning_convergence_xgboost.png')"),
        ("md", "La mejora por tuning es chica (≈ +0,007 en LightGBM y +0,011 en XGBoost): los valores por defecto "
               "ya eran razonables y el techo lo pone la información disponible antes de llamar, no el algoritmo. "
               "Advertencia: el PR-AUC del mejor trial es optimista porque se eligió mirando esos mismos folds; la "
               "estimación sin sesgo es la del hold-out (Fase 5)."),
        ("md", "## 6. Selección final: matriz de decisión\n\nCriterios y pesos: PR-AUC 35 %, lift@20 % 25 %, "
               "estabilidad (desvío de PR-AUC entre folds) 15 %, interpretabilidad 15 %, velocidad de inferencia "
               "10 %. Cada criterio se normaliza 0–1 entre candidatos."),
        ("code", "dm = table('decision_matrix.csv')[['candidate','pr_auc','pr_auc_std','roc_auc','lift@20','brier',"
                 "'predict_ms_per_1k','interpretability','weighted_score']]"),
        ("md", "Gana **LightGBM tuneado** (sin tratamiento de desbalance, categorías nativas): mejor PR-AUC y lift, "
               "inferencia rápida. La regresión logística es la más estable e interpretable, pero pierde ~0,05 de "
               "PR-AUC: se conserva como baseline y como contraste de explicabilidad. La falta de interpretabilidad "
               "de LightGBM se compensa con SHAP (Fase 5)."),
        ("md", "## 7. Umbral operativo"),
        ("code", "thr = json.loads((cfg.TABLES_DIR / 'threshold_choice.json').read_text())\nprint(json.dumps(thr, indent=2))\n"
                 "show('threshold_selection.png')\ntable('sensitivity_oof.csv')"),
        ("md", "- El umbral que **maximiza el beneficio** con V = 20 y C = 1 es **0,050**, igual al umbral teórico C/V: "
               "consistente con probabilidades bien calibradas. Contacta al ~62 % de los clientes y captura el 89 % "
               "de las conversiones.\n"
               "- El umbral que **maximiza F1** es 0,215: contacta solo al ~13 % y deja muchas conversiones "
               "rentables sin llamar. F1 pondera igual precisión y recall, sin considerar que una conversión vale "
               "20 veces lo que cuesta una llamada; por eso su beneficio es menor.\n"
               "- La sensibilidad muestra que el umbral depende del ratio V/C: con V/C = 5 conviene contactar al "
               "16 %; con V/C = 50, a casi todos."),
        ("md", "## 8. Efecto de `duration` (leakage) en CV"),
        ("code", "table('leakage_cv.csv')[['feature_set','roc_auc_fmt','pr_auc_fmt','lift@20_fmt']]"),
        ("md", "Agregar `duration` sube el ROC-AUC de 0,81 a 0,94 y el PR-AUC de 0,47 a 0,64. Es una mejora "
               "**ilusoria**: esa información no existe cuando hay que decidir a quién llamar."),
    ])


def evaluacion_notebook():
    return _nb([
        ("md", "# 03 · Evaluación en hold-out y análisis de negocio\n\nFase 5 de CRISP-DM. **Única** evaluación sobre "
               "`banca_test.csv` (4.521 filas, independiente del train tras la limpieza de la Fase 0), con el "
               "modelo ya elegido y el umbral ya fijado. Lógica en `evaluate.py`, `explain.py` y `business.py`."),
        ("code", SETUP),
        ("code", "print((cfg.TABLES_DIR / 'holdout_usage.json').read_text())"),
        ("md", "## 1. Métricas en test: modelo final vs baselines"),
        ("code", "table('test_metrics.csv')\nshow('test_roc_pr_curves.png')"),
        ("md", '- **Modelo final (LightGBM pre-contacto)**: ROC-AUC 0,784, PR-AUC 0,435, lift@10 % = 4,2. Supera claramente a la regresión logística (PR-AUC 0,349) y al Dummy (0,115 = prevalencia).\n- Frente a la CV (PR-AUC 0,466 ± 0,020) el test da algo menos (−0,03, dentro de 2 desvíos). Es la caída esperable: la CV del modelo tuneado es levemente optimista porque los hiperparámetros se eligieron mirando esos folds, y el test tiene solo 521 conversiones (más varianza).\n- La variante con `duration` llega a ROC-AUC 0,933, pero no es utilizable (ver sección 6).'),
        ("md", "## 2. Matriz de confusión al umbral operativo"),
        ("code", "table('test_confusion_matrix.csv')\nshow('test_confusion_matrix.png')"),
        ("md", 'Con el umbral de 0,05 el modelo recomienda llamar al 61 % de los clientes y captura 451 de las 521 conversiones (recall 86,6 %), con 2.311 falsos positivos. Esos falsos positivos son **aceptables por diseño**: con V/C = 20 una llamada sin conversión cuesta 1 y una conversión perdida cuesta 20.'),
        ("md", "## 3. Deciles, gain y lift"),
        ("code", "table('test_deciles.csv')\nshow('test_gain_lift.png')"),
        ("md", 'El primer decil concentra 220 de las 521 conversiones (42 %, tasa 48,6 %, lift 4,2). Con el 20 % mejor rankeado se captura el 60 % de las conversiones y con el 30 %, el 68 %. Los deciles 8 a 10 casi no tienen conversiones (2–4 %).'),
        ("md", "## 4. Beneficio esperado y sensibilidad V/C"),
        ("code", "table('test_business_summary.csv')\ntable('test_sensitivity.csv')\nshow('test_profit_curves.png')"),
        ("md", '**Traducción a negocio.** Contactando al 20 % mejor rankeado se captura el 60 % de las conversiones, con una eficiencia 3 veces mayor que al azar. Con los supuestos V = 20 y C = 1, el máximo beneficio se logra con el umbral operativo (61 % contactado): 6.258 vs 5.899 de contactar a todos (+6 %).\n\nEl valor del modelo **depende del ratio V/C**: con V/C = 5 contactar a todos da pérdida (−1.916) y el modelo da ganancia (+718); con V/C = 10 el modelo triplica el beneficio (2.195 vs 689); con V/C = 50 conviene llamar a casi todos y el modelo apenas agrega valor. El modelo es más valioso cuanto más cara es la llamada respecto del valor de la conversión, o cuando hay un **presupuesto fijo** de llamadas (ahí importa el ranking: lift 4,2 en el top 10 %).'),
        ("md", "## 5. Calibración"),
        ("code", "show('test_calibration.png')"),
        ("md", "LightGBM sin tratamiento de desbalance está bien calibrado (Brier 0,082; puntos cerca de la diagonal), lo que valida usar sus probabilidades para estimar conversiones y beneficio. La regresión logística con `class_weight='balanced'` sobreestima sistemáticamente (Brier 0,183): por eso su umbral óptimo es 0,275 y no 0,05."),
        ("md", "## 6. Leakage: pre-contacto vs con `duration`"),
        ("code", "table('test_leakage.csv')"),
        ("md", 'Con `duration`: ROC-AUC 0,933 y PR-AUC 0,607 (vs 0,784 y 0,435). La mejora es **ilusoria**: para conocer la duración hay que haber hecho la llamada, justo lo que el modelo debe decidir. Además la causalidad está invertida: la llamada es larga porque el cliente está interesado. Ese modelo mostraría métricas excelentes en validación y no podría usarse en producción.'),
        ("md", "## 7. Explicabilidad global (SHAP)"),
        ("code", "show('shap_summary.png')\nshow('shap_importance.png')\nshow('shap_dependence.png')\n"
                 "table('shap_importance.csv')"),
        ("md", '- **Mes** es la variable más influyente (15 % de la importancia): marzo, septiembre, octubre y diciembre suben fuerte la probabilidad; mayo, enero y agosto la bajan. Como se vio en el EDA, refleja la operación y el período más que al cliente: es el principal riesgo para campañas futuras.\n- **Tipo de contacto / canal conocido** (juntas ≈ 23 %): `unknown` baja la probabilidad. Son dos codificaciones de la misma información y se reparten la importancia.\n- **Saldo**: efecto creciente a partir de saldos positivos; negativos y ceros restan.\n- **Nº de préstamos**: sin préstamos suma; con 1 o más resta.\n- **Éxito previo**: afecta a pocos clientes pero con efecto enorme (+1,0 a +1,7 log-odds).\n- **Contactos en la campaña**: muchos contactos restan (insistir no funciona). La **edad** alta suma.'),
        ("md", "## 8. Explicaciones locales (verdadero positivo, falso positivo, falso negativo)"),
        ("code", "for c in ['tp', 'fp', 'fn']:\n    show(f'shap_local_{c}.png')\n"
                 "print((cfg.TABLES_DIR / 'shap_local_cases.md').read_text(encoding='utf-8'))"),
        ("md", '- **Verdadero positivo**: cliente contactado por celular en octubre y sin préstamos; el mes explica casi toda la suba (probabilidad 0,28, muy por encima del umbral).\n- **Falso positivo** (el de mayor probabilidad): jubilado con **éxito en la campaña anterior**, contactado en marzo. Todo indicaba conversión (0,81) y no convirtió: un error razonable, cualquier analista lo habría llamado.\n- **Falso negativo** (el de menor probabilidad): contacto `unknown` en mayo, saldo cero, hipoteca y préstamo personal. Es el perfil de menor conversión del dataset; su conversión no era predecible con la información previa a la llamada.'),
        ("md", "## 9. Contraste: coeficientes de la regresión logística"),
        ("code", "table('logreg_coefficients.csv').head(20)\nshow('logreg_coefficients.png')"),
        ("md", 'La regresión logística coincide en lo esencial con SHAP: los coeficientes más grandes son de **mes** (marzo OR 2,8, diciembre 2,6, octubre 1,9; noviembre 0,37, enero 0,43, mayo 0,54), el grupo de 65+ (OR 2,6) y el saldo. Que dos familias de modelos distintas lleguen a las mismas variables da confianza en que las relaciones son reales.'),
        ("md", "## 10. Reglas de negocio: árbol de profundidad 3"),
        ("code", "show('rules_tree_depth3.png')\nprint((cfg.TABLES_DIR / 'rules_tree_depth3.txt').read_text(encoding='utf-8'))\n"
                 "table('rules_tree_leaves.csv')"),
        ("md", 'Reglas simples para el equipo comercial (tasas medidas en train):\n\n1. **Éxito en la campaña anterior y sin hipoteca** → 66–75 % de conversión (lift 5,6–6,4). Llamar siempre.\n2. Éxito anterior con hipoteca → 46–61 %.\n3. Sin éxito previo, **sin préstamos** y fuera del verano → 22 % (lift 1,9).\n4. Sin éxito previo, **con préstamos** y contacto no celular → 4 % (lift 0,36). Última prioridad.\n\nEl árbol de profundidad 3 obtiene ROC-AUC 0,675 en test: sirve para comunicar, no para reemplazar al modelo.'),
        ("md", "## 11. Análisis de errores"),
        ("code", "table('error_profiles.csv')\ntable('error_categorical_profiles.csv').head(40)"),
        ("md", '- **Falsos negativos (70)**: se parecen a los verdaderos negativos y no a los verdaderos positivos: 61 % con contacto `unknown` (vs 4 % en TP), 49 % en mayo (vs 13 %), saldo medio 563 (vs 1.729), más contactos en la campaña (3,6 vs 2,1) y solo 10 % contactados antes (vs 39 %). Son conversiones inesperadas dentro de campañas masivas; no hay señal previa que las distinga.\n- **Falsos positivos (2.311)**: se parecen a los TP (canal celular 84 %, la mitad sin hipoteca). Con un umbral de 0,05 son la contracara buscada de un recall alto.'),
        ("md", "## 12. Métricas por segmento"),
        ("code", "table('segment_metrics.csv')\nshow('segment_metrics.png')"),
        ("md", '- El segmento más débil es **contact = unknown** (ROC-AUC 0,65; 1.324 clientes, 4,6 % de conversión).\n- **Mayores de 65** (0,62) y **menores de 25** (0,68, poco confiable): convierten mucho en promedio (36 % y 19 %) y el modelo los pone arriba a todos, pero discrimina poco dentro del grupo.\n- Por ocupación, blue-collar y self-employed (0,73) son los más difíciles con muestra suficiente; management y services, los mejores (> 0,80).\n- Para esos segmentos convendría sumar información (por ejemplo, historial transaccional).'),
        ("md", "## 13. Razonabilidad frente a la literatura"),
        ("code", "table('reasonableness.csv')"),
        ("md", 'Los cinco chequeos de razonabilidad se cumplen. El ROC-AUC de 0,78 en test está dentro del rango de referencia 0,75–0,80 para este dataset sin `duration`; con `duration` supera 0,93. Si el modelo pre-contacto hubiera dado ROC-AUC > 0,90, habría sido una señal de fuga de información.'),
    ])


BUILDERS = {"02_modelado.ipynb": modelado_notebook, "03_evaluacion_y_negocio.ipynb": evaluacion_notebook}
