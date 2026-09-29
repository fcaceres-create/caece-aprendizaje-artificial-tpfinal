"""Genera los notebooks del proyecto (se ejecutan después con papermill desde el pipeline).

Los notebooks no contienen lógica propia: importan funciones de ``bank_captacion`` y
muestran tablas y figuras ya generadas, con el comentario analítico en español.
"""

from __future__ import annotations

import nbformat as nbf

from . import config as cfg

SETUP = """\
import json
import pandas as pd
from IPython.display import Image, display, Markdown
from bank_captacion import config as cfg
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

def show(fig):
    display(Image(filename=str(cfg.FIGURES_DIR / fig)))

def table(name, **kw):
    df = pd.read_csv(cfg.TABLES_DIR / name, **kw)
    display(df)
    return df
"""


def _nb(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return nb


def eda_notebook() -> nbf.NotebookNode:
    return _nb([
        ("md", "# 01 · Análisis exploratorio de datos (EDA)\n\n"
               "Fase 2 de CRISP-DM: comprensión de los datos. Todo el análisis se hace **solo sobre el train "
               "limpio** (40.690 filas, sin las filas que estaban repetidas en test). El hold-out no se mira "
               "para no sesgar decisiones.\n\n"
               "La lógica está en `src/bank_captacion/eda.py`; este notebook la ejecuta y comenta los resultados."),
        ("code", SETUP),
        ("code", "from bank_captacion.eda import run_eda\nkey = run_eda()\nprint(json.dumps(key, indent=2, default=float))"),
        ("md", "## 1. Preparación previa del dataset (Fase 0)\n\n"
               "Antes del EDA se verificó el esquema y se eliminó el solapamiento train/test. "
               "La tabla siguiente es el registro completo de pasos aplicados a las filas."),
        ("code", "table('data_preparation_steps.csv')\ntable('data_summary.csv')"),
        ("md", "La partición train/test es **aleatoria**: las filas de test están distribuidas uniformemente a lo "
               "largo del archivo (ordenado cronológicamente) y las distribuciones de mes, tipo de contacto, "
               "resultado previo y target no difieren (p > 0,05 en todos los tests)."),
        ("code", "table('split_checks.csv')"),
        ("md", "## 2. Variable objetivo\n\nDesbalance moderado: 11,7 % de conversiones. Un clasificador trivial "
               "que responde siempre \"no\" tendría 88,3 % de accuracy, por eso accuracy no se usa."),
        ("code", "show('eda_target_balance.png')"),
        ("md", "## 3. Variables numéricas"),
        ("code", "show('eda_numeric_distributions.png')\ntable('eda_numeric_by_class.csv')"),
        ("md", "- **Edad**: los que convierten están sobrerrepresentados en los extremos (< 30 y > 60 años: "
               "estudiantes y jubilados). Relación no lineal → justifica `age_group` además de `age`.\n"
               "- **Saldo**: muy asimétrico (asimetría ≈ 8,5), con negativos y ceros. Los que convierten tienen "
               "saldos algo mayores.\n"
               "- **Contactos en la campaña**: más contactos se asocian a menor conversión (insistir no ayuda).\n"
               "- **pdays**: entre los contactados antes, picos en ~90 y ~180 días con alta conversión "
               "(probablemente seguimientos trimestrales/semestrales)."),
        ("md", "## 4. Variables categóricas: tasa de conversión por categoría"),
        ("code", "show('eda_conversion_by_category.png')\nconv = table('eda_conversion_by_category.csv')"),
        ("md", "- **poutcome = success**: 64,8 % de conversión (5,5 veces la base). Es la señal más fuerte "
               "disponible antes de llamar.\n"
               "- **Ocupación**: estudiantes (29 %) y jubilados (23 %) convierten mucho más que blue-collar (7 %).\n"
               "- **Préstamos**: tener hipoteca o préstamo personal reduce la conversión casi a la mitad.\n"
               "- **contact = unknown**: solo 4 % de conversión (ver sección 5)."),
        ("md", "## 5. `unknown` como categoría con significado propio"),
        ("code", "table('eda_unknown_analysis.csv')\nshow('eda_unknown_by_file_position.png')\n"
                 "table('eda_unknown_by_file_position.csv')"),
        ("md", "`unknown` **no es un faltante aleatorio**:\n\n"
               "- `contact = unknown` ocupa el 100 % del primer 20 % del archivo y desaparece después del 30 %: "
               "corresponde a un período (mayo–junio 2008) en que no se registraba el canal. Su baja conversión "
               "(4 %) refleja ese período, no el canal.\n"
               "- `poutcome = unknown` = cliente nunca contactado en una campaña previa (ver sección 6).\n"
               "- La tasa de conversión crece a lo largo del archivo (de 2,9 % a 47,5 %): **hay drift temporal** "
               "fuerte en los datos. Con una partición aleatoria esto no afecta la evaluación, pero es una "
               "limitación para producción.\n\n"
               "Decisión: mantener `unknown` como categoría propia (no imputar a la moda)."),
        ("md", "## 6. `pdays = -1`, `previous` y `poutcome`"),
        ("code", "table('eda_pdays_previous_poutcome.csv')\nshow('eda_pdays_poutcome.png')"),
        ("md", "Las tres variables codifican lo mismo cuando el cliente nunca fue contactado: las 33.249 filas "
               "con `pdays = -1` tienen `previous = 0` y `poutcome = unknown`. Solo 5 filas tienen contacto previo "
               "pero `poutcome = unknown` (inconsistencia menor, se conservan). El -1 **no es un número**: "
               "tratarlo como tal le diría a un modelo lineal que \"-1 días\" está cerca de \"0 días\". Por eso "
               "en la Fase 3 se crea la bandera `previously_contacted` y `pdays_clean` (NaN cuando -1)."),
        ("md", "## 7. `month`: estacionalidad aparente vs volumen"),
        ("code", "show('eda_month_volume_rate.png')\ntable('eda_month_volume_rate.csv')"),
        ("md", "Mayo concentra el 30 % de los contactos con la menor tasa (6,7 %), mientras que marzo, "
               "septiembre, octubre y diciembre tienen tasas del 43–53 % con muy poco volumen (1–2 % cada uno). "
               "La correlación de Spearman entre volumen y tasa es −0,84 (p < 0,001).\n\n"
               "Interpretación: no es estacionalidad del cliente sino de **la operación**: los meses con campañas "
               "masivas tienen tasas bajas y los meses con pocas llamadas (dirigidas, o del final del período "
               "2010) tienen tasas altas. Como el dataset no trae el año, `month` actúa en parte como proxy del "
               "período. Se usa como variable (se conoce antes de llamar), pero se advierte que su efecto puede "
               "no trasladarse a campañas futuras."),
        ("md", "## 8. Outliers: `balance` y `campaign`"),
        ("code", "show('eda_outliers_balance_campaign.png')\nprint(json.dumps(key['outliers'], indent=2))"),
        ("md", "- `balance`: 10,4 % de outliers por la regla IQR, valores entre −8.019 y 102.127. No son errores "
               "(saldos altos existen), por eso **no se eliminan**: se aplica transformación logarítmica con signo "
               "y se agregan banderas de saldo negativo (8,4 %) y cero (7,8 %).\n"
               "- `campaign`: máximo 63 contactos, p99 = 17. La conversión cae de 14,7 % (1 contacto) a 6,4 % "
               "(≥ 5). Se **winsoriza al p99** (calculado dentro de cada fold) para que valores extremos no "
               "dominen los modelos lineales."),
        ("md", "## 9. `duration`: por qué es leakage"),
        ("code", "show('eda_duration_leakage.png')\nprint(json.dumps(key['duration'], indent=2))"),
        ("md", "`duration` por sí sola tiene ROC-AUC ≈ 0,81: más que muchos modelos completos sin ella. La mediana "
               "de duración es 164 s en los que no convierten y 424 s en los que convierten; las llamadas de menos "
               "de 60 s casi nunca convierten (0,2 %). Solo hay 3 llamadas con duración 0.\n\n"
               "El problema: **la duración se conoce recién al terminar la llamada**, y una llamada larga es en "
               "buena medida *consecuencia* del interés del cliente. Un modelo que la use no puede decidir a quién "
               "llamar. Se excluye del modelo entregable y se usa solo para cuantificar el efecto (Fase 5)."),
        ("md", "## 10. Ranking preliminar de variables"),
        ("code", "show('eda_feature_ranking.png')\ntable('eda_feature_ranking.csv')"),
        ("md", "Sin contar `duration`, las variables más informativas son `poutcome`, `pdays`, `month`, `balance`, "
               "`previous`, `contact`, `age` y `housing`. `default` es prácticamente no informativa (V ≈ 0,02). "
               "Nota: la V de Cramér de `pdays` es baja porque el 82 % de sus valores es −1 y al discretizar en "
               "deciles queda casi en un único grupo; la información mutua sí la detecta."),
        ("code", "show('eda_numeric_correlation.png')"),
        ("md", "Correlaciones bajas entre numéricas, salvo `pdays`–`previous` (misma información de contacto previo). "
               "No hay multicolinealidad severa que obligue a descartar variables."),
    ])


def write_notebooks(which: list[str] | None = None) -> None:
    builders = {"01_eda.ipynb": eda_notebook}
    try:
        from . import notebooks_builder_model as nbm  # noqa: F401  (se agrega en fases posteriores)
        builders.update(nbm.BUILDERS)
    except ImportError:
        pass
    for name, builder in builders.items():
        if which and name not in which:
            continue
        nbf.write(builder(), cfg.NOTEBOOKS_DIR / name)


if __name__ == "__main__":
    write_notebooks()
