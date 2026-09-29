# Diagramas del pipeline

Los PNG se generan con `python -m bank_captacion.diagram`, que los dibuja con matplotlib porque
no hay Mermaid CLI instalado. Los diagramas Mermaid de abajo son equivalentes y se ven directo en
GitHub o en VS Code.

## 1. Flujo completo del proyecto

![Flujo del proyecto](../reports/figures/diagrama_flujo_proyecto.png)

```mermaid
flowchart TD
    subgraph D["1-3 · Datos y preparación"]
        A1["banca_train.csv<br/>45.211 filas"] --> A2["Quitar solapamiento<br/>−4.521 filas repetidas en test<br/>train limpio = 40.690"]
        T0["banca_test.csv<br/>4.521 filas"] -. verificación .-> A2
        A2 --> A3["EDA (solo train)"]
        A3 --> A4["FeatureEngineer<br/>24 variables pre-contacto"]
    end
    subgraph M["4 · Modelado (solo train)"]
        B1["CV estratificada 5 folds<br/>10 algoritmos"] --> B2["Codificación + desbalance"]
        B2 --> B3["Optuna (2 mejores)"]
        B3 --> B4["Matriz de decisión<br/>→ LightGBM"]
        B4 --> B5["Umbral OOF<br/>máx. beneficio = 0,050"]
    end
    subgraph E["5 · Evaluación (hold-out, uso único)"]
        C1["Métricas, deciles, lift,<br/>beneficio, calibración"] --> C2["Leakage: con vs sin duration"]
        C2 --> C3["SHAP, logística, árbol de reglas"]
        C3 --> C4["Errores y segmentos"]
    end
    subgraph P["6 · Despliegue"]
        D1["model_final.joblib<br/>+ metadata.json"] --> D2["API FastAPI"] --> D3["UI Streamlit"]
    end
    A4 --> B1
    B5 --> C1
    T0 --> C1
    B5 --> D1
```

## 2. Pipeline del modelo final

![Pipeline del modelo](../reports/figures/diagrama_pipeline_modelo.png)

```mermaid
flowchart LR
    X["Cliente: 15 columnas crudas<br/>(sin duration)"] --> F["FeatureEngineer<br/>derivadas · winsorización p99 ·<br/>log saldo · banderas · seno/coseno"]
    F --> C["CategoryCaster<br/>11 categóricas → category"]
    C --> L["LGBMClassifier<br/>500 árboles · 44 hojas · lr 0,0117"]
    L --> O["Probabilidad<br/>≥ 0,050 → llamar<br/>+ decil + top-3 SHAP"]
```
