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
        ("md", "## 2. Matriz de confusión al umbral operativo"),
        ("code", "table('test_confusion_matrix.csv')\nshow('test_confusion_matrix.png')"),
        ("md", "## 3. Deciles, gain y lift"),
        ("code", "table('test_deciles.csv')\nshow('test_gain_lift.png')"),
        ("md", "## 4. Beneficio esperado y sensibilidad V/C"),
        ("code", "table('test_business_summary.csv')\ntable('test_sensitivity.csv')\nshow('test_profit_curves.png')"),
        ("md", "## 5. Calibración"),
        ("code", "show('test_calibration.png')"),
        ("md", "## 6. Leakage: pre-contacto vs con `duration`"),
        ("code", "table('test_leakage.csv')"),
        ("md", "## 7. Explicabilidad global (SHAP)"),
        ("code", "show('shap_summary.png')\nshow('shap_importance.png')\nshow('shap_dependence.png')\n"
                 "table('shap_importance.csv')"),
        ("md", "## 8. Explicaciones locales (verdadero positivo, falso positivo, falso negativo)"),
        ("code", "for c in ['tp', 'fp', 'fn']:\n    show(f'shap_local_{c}.png')\n"
                 "print((cfg.TABLES_DIR / 'shap_local_cases.md').read_text(encoding='utf-8'))"),
        ("md", "## 9. Contraste: coeficientes de la regresión logística"),
        ("code", "table('logreg_coefficients.csv').head(20)\nshow('logreg_coefficients.png')"),
        ("md", "## 10. Reglas de negocio: árbol de profundidad 3"),
        ("code", "show('rules_tree_depth3.png')\nprint((cfg.TABLES_DIR / 'rules_tree_depth3.txt').read_text(encoding='utf-8'))\n"
                 "table('rules_tree_leaves.csv')"),
        ("md", "## 11. Análisis de errores"),
        ("code", "table('error_profiles.csv')\ntable('error_categorical_profiles.csv').head(40)"),
        ("md", "## 12. Métricas por segmento"),
        ("code", "table('segment_metrics.csv')\nshow('segment_metrics.png')"),
        ("md", "## 13. Razonabilidad frente a la literatura"),
        ("code", "table('reasonableness.csv')"),
    ])


BUILDERS = {"02_modelado.ipynb": modelado_notebook, "03_evaluacion_y_negocio.ipynb": evaluacion_notebook}
