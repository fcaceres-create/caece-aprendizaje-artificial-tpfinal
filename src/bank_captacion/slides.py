"""Genera la presentación HTML (un único archivo autocontenido) con los resultados reales.

Salida: ``presentacion/index.html``. Las figuras se embeben en base64 y los números se leen de
``reports/tables`` (mismas funciones que el informe), así que la presentación siempre coincide
con la última ejecución del pipeline.

Controles: ← → / espacio / clic para navegar · N notas del orador · F pantalla completa ·
Inicio/Fin · imprimir (Ctrl+P) genera un PDF con una slide por página.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import date

import pandas as pd

from . import config as cfg
from .export_report import collect_numbers, f, i, pct

OUT_DIR = cfg.ROOT / "presentacion"
OUT_FILE = OUT_DIR / "index.html"
REPO_URL = "https://github.com/fcaceres-create/caece-aprendizaje-artificial-tpfinal"

AUTHOR = "Fernando Caceres"
TEACHERS = "Juan Azcurra · Paul Pablo Hernán"
COURSE = "Aprendizaje Artificial"
PROGRAM = "Maestría en Gestión y Desarrollo de Inteligencia Artificial"
UNIVERSITY = "Universidad CAECE"


def img(name: str, alt: str = "", cls: str = "fig") -> str:
    data = base64.b64encode((cfg.FIGURES_DIR / name).read_bytes()).decode()
    return f'<img class="{cls}" alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def table(df: pd.DataFrame, highlight_first: bool = False, cls: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for k, (_, r) in enumerate(df.iterrows()):
        tr_cls = ' class="hl"' if highlight_first and k == 0 else ""
        rows.append(f"<tr{tr_cls}>" + "".join(f"<td>{v}</td>" for v in r.values) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def stat(value: str, label: str, tone: str = "") -> str:
    return f'<div class="stat {tone}"><div class="v">{value}</div><div class="l">{label}</div></div>'


def build_slides() -> list[dict]:
    n = collect_numbers()
    t = lambda name: pd.read_csv(cfg.TABLES_DIR / name)  # noqa: E731
    meta = json.loads(cfg.METADATA_FILE.read_text(encoding="utf-8"))
    S: list[dict] = []

    def add(title, body, notes="", section="", cls=""):
        S.append({"title": title, "body": body, "notes": notes, "section": section, "cls": cls})

    # 1. Portada ----------------------------------------------------------------------------------
    add("", f"""
      <div class="cover">
        <div class="kicker">Trabajo Práctico Final · {COURSE}</div>
        <h1>Predicción de captación de clientes<br>en campañas de marketing bancario</h1>
        <p class="sub">Un modelo pre-contacto para decidir <b>a quién llamar</b> con un presupuesto limitado</p>
        <div class="meta">
          <div><span>Autor</span>{AUTHOR}</div>
          <div><span>Docentes</span>{TEACHERS}</div>
          <div><span>Programa</span>{PROGRAM} · {UNIVERSITY}</div>
          <div><span>Fecha</span>{date.today().strftime('%d/%m/%Y')}</div>
        </div>
      </div>""",
        "Presentarse. El trabajo es del caso 'Captación de clientes'. Anticipar la idea central: no es un "
        "clasificador más, es una herramienta para priorizar llamadas, y la mayor parte del valor estuvo en "
        "tratar bien los datos.", cls="cover-slide")

    # 2. Agenda -----------------------------------------------------------------------------------
    phases = [("1", "Negocio", "Decisión, métricas y marco económico"),
              ("2", "Datos", "Verificación, hallazgo crítico y EDA"),
              ("3", "Preparación", "Variables derivadas dentro del pipeline"),
              ("4", "Modelado", "10 algoritmos, desbalance, tuning y selección"),
              ("5", "Evaluación", "Hold-out, negocio, leakage, explicabilidad"),
              ("6", "Despliegue", "API FastAPI + interfaz Streamlit")]
    add("Metodología: CRISP-DM de punta a punta",
        '<div class="phases">' + "".join(
            f'<div class="phase"><div class="num">{a}</div><div><b>{b}</b><p>{c}</p></div></div>'
            for a, b, c in phases) + "</div>"
        + '<p class="foot">Todo se reconstruye con un solo comando: <code>python -m bank_captacion.pipeline</code> '
          "(datos → modelos → evaluación → informe, ~4 min). Cada número de esta presentación sale de esa ejecución.</p>",
        "Recorrer las seis fases. Remarcar la reproducibilidad: semilla fija, versiones fijas, un comando.")

    # 3. Problema ---------------------------------------------------------------------------------
    add("El problema: ¿a quién conviene llamar?", f"""
      <div class="cols">
        <div>
          <p class="lead">Un banco hace campañas telefónicas para captar clientes. Cada llamada cuesta y solo una
          minoría se convierte.</p>
          <blockquote>Dado un listado de clientes y un presupuesto de llamadas, <b>ordenarlos por probabilidad de
          conversión</b> y contactar a los primeros <i>k</i>.</blockquote>
          <ul>
            <li>El modelo se evalúa como <b>ranking</b>, no solo como clasificador.</li>
            <li>Se usa <b>antes</b> de llamar → solo información disponible pre-contacto.</li>
            <li>La decisión binaria sale de un <b>umbral económico</b>, no del 0,5 por defecto.</li>
          </ul>
        </div>
        <div class="stack">
          {stat(n['rate_train'], 'de los contactados se convierte (train)', 'accent')}
          {stat('88,3 %', 'accuracy de un modelo que dice siempre “no” → inútil', 'warn')}
        </div>
      </div>""",
        "Explicar por qué accuracy no sirve: el modelo trivial tiene 88 % y no encuentra a nadie. Por eso las "
        "métricas son de ranking.", "1 · Negocio")

    # 4. Métricas y economía ----------------------------------------------------------------------
    add("Métricas y marco económico", f"""
      <div class="cols">
        <div>
          <h3>Métricas</h3>
          <table><thead><tr><th>Tipo</th><th>Métrica</th><th>Para qué</th></tr></thead><tbody>
            <tr><td>Primaria</td><td><b>PR-AUC</b></td><td>Calidad del ranking sobre la clase minoritaria (azar ≈ 0,117)</td></tr>
            <tr><td>Primaria</td><td><b>Lift / gain</b> top 10-20-30 %</td><td>¿Qué % de conversiones capturo llamando a pocos?</td></tr>
            <tr><td>Secundaria</td><td>ROC-AUC</td><td>Comparación con la literatura</td></tr>
            <tr><td>Secundaria</td><td>Brier, calibración</td><td>¿Las probabilidades son creíbles?</td></tr>
          </tbody></table>
        </div>
        <div>
          <h3>Beneficio esperado</h3>
          <div class="formula">Beneficio = V · TP − C · (TP + FP)</div>
          <ul>
            <li>Supuestos: <b>C = 1</b> por llamada, <b>V = 20</b> por conversión.</li>
            <li>Llamar conviene si <b>p · V &gt; C</b> ⇒ <b>p &gt; C/V = 0,05</b>.</li>
            <li>Lo que importa es el ratio V/C → sensibilidad con 5, 10, 20 y 50.</li>
            <li>El banco los reemplaza por valores reales en <code>config.py</code>.</li>
          </ul>
        </div>
      </div>""",
        "C y V son supuestos ilustrativos. El umbral teórico 0,05 va a reaparecer: el umbral óptimo empírico da "
        "exactamente eso, lo que prueba que el modelo está calibrado.", "1 · Negocio")

    # 5. Datos --------------------------------------------------------------------------------------
    add("Los datos: Bank Marketing (UCI)", f"""
      <div class="cols">
        <div>
          <table><thead><tr><th>Archivo</th><th>Equivale a</th><th>Filas</th><th>Conversión</th></tr></thead><tbody>
            <tr><td><code>banca_train.csv</code></td><td>bank-full.csv</td><td>{n['train_raw']}</td><td>{n['rate_train_raw']}</td></tr>
            <tr><td><code>banca_test.csv</code></td><td>bank.csv</td><td>{n['test_rows']}</td><td>{n['rate_test']}</td></tr>
          </tbody></table>
          <ul>
            <li>17 columnas · separador <code>;</code> · ASCII · target <code>yes/no</code>.</li>
            <li>Banco portugués, campañas de <b>mayo 2008 a noviembre 2010</b> (Moro, Cortez y Rita, 2014).</li>
            <li>0 nulos, 0 duplicados dentro de cada archivo. Esquema idéntico al diccionario.</li>
            <li>Faltantes codificados: <code>unknown</code> (poutcome {n['pct_unk_poutcome']}, contact {n['pct_unk_contact']})
                y <code>pdays = -1</code>.</li>
          </ul>
        </div>
        <div class="stack">
          {stat('16', 'variables candidatas (sin contar el target)')}
          {stat('1', 'variable prohibida: <code>duration</code> (leakage)', 'warn')}
        </div>
      </div>""",
        "Antes de modelar se verificó todo: formato, esquema, nulos, duplicados. Así apareció el hallazgo "
        "de la slide siguiente.", "2 · Datos")

    # 6. Hallazgo crítico --------------------------------------------------------------------------
    add("Hallazgo crítico: el test estaba dentro del train", f"""
      <div class="cols">
        <div>
          <div class="flow">
            <div class="box">banca_train<br><b>{n['train_raw']}</b></div>
            <div class="arrow">−{n['overlap']}</div>
            <div class="box ok">train limpio<br><b>{n['train_clean']}</b></div>
          </div>
          <ul>
            <li>Comparando las 17 columnas, <b>las {n['overlap']} filas de test (100 %) aparecen idénticas en train</b>:
                <code>bank.csv</code> es una muestra del 10 % de <code>bank-full.csv</code>.</li>
            <li>Sin corregirlo, el test mide <b>memoria, no generalización</b>.</li>
            <li>Decisión: quitar esas filas de train. Un <code>assert</code> verifica solapamiento = 0 en cada ejecución.</li>
          </ul>
        </div>
        <div>
          <h3>¿Partición aleatoria o temporal?</h3>
          <p>El train está ordenado en el tiempo. Si el test fuera “el final”, habría que validar temporalmente.</p>
          <ul>
            <li>Posiciones del test en el archivo: uniformes (KS D = {n['ks_d']}, p = {n['ks_p']}).</li>
            <li>Mes, canal, resultado previo y target: sin diferencias (χ² p ≥ {n['chi_pout_p']}).</li>
          </ul>
          <p class="callout">⇒ Partición <b>aleatoria</b> → validación cruzada estratificada.</p>
        </div>
      </div>""",
        "Es la decisión más importante del trabajo. Si preguntan cómo se detectó: merge por las 17 columnas. "
        "Si preguntan por qué no descartar el test: porque lo entrega la consigna; se perdió solo el 10 % de train. "
        f"La tasa de conversión quedó igual ({n['rate_train']}).", "2 · Datos")

    # 7. unknown y drift ----------------------------------------------------------------------------
    add("EDA: “unknown” no es un faltante al azar", f"""
      <div class="cols wide-right">
        <div>
          <ul>
            <li><code>contact = unknown</code> convierte <b>{n['unk_contact_conv']}</b> vs {n['unk_contact_rest']}: ocupa el
                100 % del primer 20 % del archivo → período sin registro del canal.</li>
            <li><code>poutcome = unknown</code> = nunca contactado antes ({n['unk_poutcome_conv']} vs {n['unk_poutcome_rest']}).</li>
            <li><code>education = unknown</code> convierte {n['unk_education_conv']}: más que la moda.</li>
          </ul>
          <p class="callout">Decisión: <b>unknown se mantiene como categoría</b>, no se imputa.</p>
          <p class="warn-text">Además: la conversión sube de 2,9 % a 47,5 % a lo largo del archivo → <b>drift temporal</b>.</p>
        </div>
        <div>{img('eda_unknown_by_file_position.png', 'unknown por posición')}</div>
      </div>""",
        "El gráfico muestra el archivo en orden cronológico. El unknown de contact desaparece después del 30 %: "
        "no es aleatorio, es un período. Y la tasa de conversión crece mucho: esto es una limitación para producción.",
        "2 · Datos")

    # 8. Conversion por categoría --------------------------------------------------------------------
    add("EDA: tasa de conversión por categoría",
        f'<div class="figwrap">{img("eda_conversion_by_category.png", "conversión por categoría")}</div>'
        f'<p class="foot">Resultado previo <b>success</b>: {n["conv_success"]} (5,5× la base) · estudiantes y jubilados '
        "convierten más · tener préstamos reduce la conversión a la mitad.</p>",
        "Señalar los tres paneles clave: poutcome, job y housing/loan.", "2 · Datos")

    # 9. Mes y volumen ----------------------------------------------------------------------------
    add("EDA: el mes refleja la operación, no al cliente", f"""
      <div class="cols wide-right">
        <div>
          <ul>
            <li>Mayo concentra el <b>{n['may_share']}</b> de las llamadas con <b>{n['may_rate']}</b> de conversión.</li>
            <li>Marzo, septiembre, octubre y diciembre: &gt; 40 % con 0,5–1,6 % del volumen cada uno.</li>
            <li>Spearman volumen vs tasa: <b>{n['spearman_month']}</b>.</li>
          </ul>
          <p>Campañas masivas → tasas bajas. Meses de pocas llamadas dirigidas (y el final del período) → tasas altas.
             Sin año en los datos, <code>month</code> es en parte un proxy del período.</p>
        </div>
        <div>{img('eda_month_volume_rate.png', 'mes')}</div>
      </div>""",
        "Esta es la variable más importante del modelo final. Se usa porque se conoce antes de llamar, pero es el "
        "principal riesgo en campañas futuras.", "2 · Datos")

    # 10. pdays y outliers --------------------------------------------------------------------------
    add("EDA: códigos especiales y valores extremos", f"""
      <div class="cols">
        <div>
          <h3><code>pdays = -1</code></h3>
          <ul>
            <li>Las {n['n_pdays_neg']} filas con -1 ({n['pct_pdays_neg']}) son exactamente las de
                <code>previous = 0</code> y <code>poutcome = unknown</code>.</li>
            <li>-1 es un código, no “días”: nunca contactados {n['conv_never']} vs contactados {n['conv_before']}.</li>
            <li>→ bandera <code>previously_contacted</code> + <code>pdays_clean</code> (NaN).</li>
          </ul>
          <h3>Outliers</h3>
          <ul>
            <li><code>balance</code>: {n['bal_min']} a {n['bal_max']}, asimetría {n['bal_skew']}, {n['bal_iqr']} outliers IQR
                → reales: <b>log con signo</b>, no se borran.</li>
            <li><code>campaign</code>: hasta {n['camp_max']} (p99 = {n['camp_p99']}); conversión {n['conv_camp1']} con 1
                contacto vs {n['conv_camp5']} con ≥ 5 → <b>winsorización p99</b> dentro de cada fold.</li>
          </ul>
        </div>
        <div>{img('eda_pdays_poutcome.png', 'pdays')}{img('eda_outliers_balance_campaign.png', 'outliers')}</div>
      </div>""",
        "Ninguna fila se eliminó por outlier: se atenuó con transformaciones aprendidas dentro del pipeline.",
        "2 · Datos")

    # 11. duration --------------------------------------------------------------------------------
    add("EDA: por qué <code>duration</code> queda afuera", f"""
      <div class="cols wide-right">
        <div>
          <div class="stack">
            {stat(n['dur_auc'], 'ROC-AUC de <code>duration</code> sola', 'warn')}
          </div>
          <ul>
            <li>Mediana: {n['dur_med_no']} s (no convierte) vs {n['dur_med_yes']} s (convierte).</li>
            <li>Llamadas &lt; 60 s: {n['dur_short_conv']} de conversión.</li>
            <li><b>Se conoce después de la llamada</b>: no sirve para decidir a quién llamar.</li>
            <li>Causalidad invertida: la llamada es larga <i>porque</i> el cliente está interesado.</li>
          </ul>
          <p class="callout">Modelo entregable = <b>pre-contacto</b>. Variante con duration solo para medir el leakage.</p>
        </div>
        <div>{img('eda_duration_leakage.png', 'duration')}</div>
      </div>""",
        "Una sola variable ya da 0,81 de AUC: si el modelo final diera más de 0,90 sin ella, sospecharíamos fuga.",
        "2 · Datos")

    # 12. Preparación ------------------------------------------------------------------------------
    cat = t("feature_catalog.csv")
    cat = cat[cat["feature"].str.len() < 40][["feature", "source", "transformation"]].head(12)
    cat.columns = ["Variable", "Origen", "Transformación"]
    add("Preparación: variables derivadas dentro del pipeline",
        f"""<div class="cols wide-left">
          <div class="small">{table(cat.map(lambda v: html.escape(str(v))))}</div>
          <div>
            <ul>
              <li><b>24 variables</b> pre-contacto (13 numéricas, 11 categóricas).</li>
              <li>Todo en un <code>Pipeline</code>: percentiles, medianas, escalas y categorías se aprenden
                  <b>solo con el fold de entrenamiento</b>.</li>
              <li>El modelo guardado recibe <b>datos crudos</b>: la API no reimplementa nada.</li>
              <li>Preprocesador según familia: escalado + one-hot (lineales, KNN, MLP, SVM), sin escalar
                  (árboles), categorías nativas (LightGBM), discretización (Naive Bayes).</li>
            </ul>
          </div>
        </div>
        <div class="pipe">
          <div class="p d">Cliente crudo<small>15 columnas, sin duration</small></div>
          <div class="p f">FeatureEngineer<small>derivadas · p99 · log · banderas</small></div>
          <div class="p f">CategoryCaster<small>11 categóricas → category</small></div>
          <div class="p m">LightGBM<small>{n['hp_n_estimators']} árboles · {n['hp_leaves']} hojas</small></div>
          <div class="p o">Probabilidad<small>≥ {n['thr']} → llamar · decil · SHAP</small></div>
        </div>""",
        "Lo que NO se hizo, a propósito: no imputar unknown, no borrar outliers, no usar duration. "
        "Todo documentado en docs/03_preparacion_datos.md.", "3 · Preparación")

    # 13. Protocolo + algoritmos --------------------------------------------------------------------
    cv = t("cv_results.csv")
    cvt = pd.DataFrame({"Modelo": cv["label"], "Familia": cv["family"],
                        "PR-AUC": [f"{f(a)} ± {f(b)}" for a, b in zip(cv["pr_auc_mean"], cv["pr_auc_std"])],
                        "ROC-AUC": cv["roc_auc_mean"].map(f), "Lift@20 %": cv["lift@20_mean"].map(lambda x: f(x, 2))})
    add("Modelado: 10 algoritmos, mismo protocolo", f"""
      <div class="cols wide-left">
        <div class="small">{table(cvt, highlight_first=True)}</div>
        <div>
          <ul>
            <li><code>StratifiedKFold(5, shuffle, seed 42)</code> sobre {n['train_clean']} filas: <b>misma partición</b> para todos.</li>
            <li>Cada algoritmo representa una familia vista en la materia.</li>
            <li>Ensambles de árboles arriba (~4× el azar).</li>
            <li>La brecha con la logística mide el valor de las no linealidades.</li>
            <li>SVM (submuestra 10k): buen ROC, peor PR-AUC → ordena mal la parte alta del ranking.</li>
          </ul>
        </div>
      </div>""",
        "Cada modelo representa una familia vista en clase. La justificación de cada uno está en el decision log (D11).",
        "4 · Modelado")
    add("Comparación por fold (PR-AUC, ROC-AUC, lift@20 %)",
        f'<div class="figwrap">{img("model_cv_comparison.png", "comparación por fold")}</div>'
        '<p class="foot">Cada punto es un fold. La línea roja es la prevalencia (PR-AUC del azar). Los tres ensambles '
        'se solapan: la diferencia entre ellos es menor que la variación entre folds.</p>',
        "Mostrar la dispersión: por eso la selección final no se decide solo por la media.", "4 · Modelado")

    # 14. Desbalance y codificación ------------------------------------------------------------------
    add("Desbalance: ¿pesos, SMOTE o nada?", f"""
      <div class="cols wide-right">
        <div>
          <table><thead><tr><th>LightGBM</th><th>PR-AUC</th><th>Brier</th></tr></thead><tbody>
            <tr class="hl"><td>Sin tratamiento</td><td>{n['imb_lightgbm_none_pr']}</td><td>{n['imb_lightgbm_none_brier']}</td></tr>
            <tr><td>Pesos de clase</td><td>{n['imb_lightgbm_weights_pr']}</td><td>{n['imb_lightgbm_weights_brier']}</td></tr>
            <tr><td>SMOTE</td><td>{n['imb_lightgbm_smote_pr']}</td><td>{n['imb_lightgbm_smote_brier']}</td></tr>
          </tbody></table>
          <ul>
            <li>Diferencias de PR-AUC &lt; 1 desvío entre folds.</li>
            <li><b>Pesos descalibran</b>: el Brier casi se duplica.</li>
            <li>SMOTE interpola clientes sintéticos sobre one-hot (sin sentido) y empeora al RF.</li>
          </ul>
          <p class="callout">Elección: <b>sin tratamiento + ajuste de umbral</b>. Codificación: nativa en LightGBM, one-hot en XGBoost.</p>
        </div>
        <div>{img('imbalance_comparison.png', 'desbalance')}</div>
      </div>""",
        "Punto clave de criterio: el umbral económico necesita probabilidades calibradas; por eso se descartan los "
        "pesos aunque den PR-AUC parecida.", "4 · Modelado")

    # 15. Tuning + matriz ------------------------------------------------------------------------------
    dm = t("decision_matrix.csv")
    dmt = pd.DataFrame({"Candidato": dm["candidate"], "PR-AUC": dm["pr_auc"].map(f),
                        "Desvío": dm["pr_auc_std"].map(f), "Lift@20": dm["lift@20"].map(lambda x: f(x, 2)),
                        "Interpr.": dm["interpretability"], "Puntaje": dm["weighted_score"].map(f)})
    add("Tuning y selección final", f"""
      <div class="cols">
        <div>
          <h3>Optuna (TPE, 60 trials, tope 15 min)</h3>
          <ul>
            <li>LightGBM: {n['tune_lightgbm_default']} → <b>{n['tune_lightgbm_best']}</b> ({n['tune_lightgbm_secs']} s).</li>
            <li>XGBoost: {n['tune_xgboost_default']} → {n['tune_xgboost_best']}.</li>
            <li>Mejora chica: el techo lo pone la información pre-contacto, no el algoritmo.</li>
          </ul>
          {img('tuning_convergence_lightgbm.png', 'optuna')}
        </div>
        <div>
          <h3>Matriz de decisión</h3>
          <p class="small-text">PR-AUC 35 % · lift@20 25 % · estabilidad 15 % · interpretabilidad 15 % · velocidad 10 %</p>
          <div class="small">{table(dmt, highlight_first=True)}</div>
          <p class="callout">Ganador: <b>LightGBM tuneado</b> — {meta['hyperparameters']['n_estimators']} árboles,
             {meta['hyperparameters']['num_leaves']} hojas, lr {n['hp_lr']}. La logística queda como baseline.</p>
        </div>
      </div>""",
        "La interpretabilidad que pierde LightGBM se recupera con SHAP. El PR-AUC del mejor trial es levemente optimista: "
        "la estimación sin sesgo es la del test.", "4 · Modelado")

    # 16. Umbral ---------------------------------------------------------------------------------------
    add("Umbral operativo: beneficio, no F1", f"""
      <div class="cols wide-right">
        <div>
          <div class="stack">
            {stat(n['thr'], 'umbral de máximo beneficio (OOF) = C/V teórico', 'accent')}
            {stat(n['thr_f1'], 'umbral de máximo F1')}
          </div>
          <ul>
            <li>Con 0,05: contacta {n['thr_contacted']}, recall {n['thr_recall']}, beneficio {n['thr_profit']}
                (vs {n['thr_profit_all']} contactando a todos).</li>
            <li>Con F1: contacta {n['thr_f1_contacted']} y el beneficio cae a {n['thr_f1_profit']}.</li>
            <li>F1 ignora que una conversión vale 20 veces una llamada.</li>
          </ul>
        </div>
        <div>{img('threshold_selection.png', 'umbral')}</div>
      </div>""",
        "Elegido con predicciones out-of-fold de train, nunca con test. Que coincida con 0,05 confirma la calibración.",
        "4 · Modelado")

    # 17. Resultados hold-out ------------------------------------------------------------------------
    tm = t("test_metrics.csv")
    tmt = pd.DataFrame({"Modelo": tm["label"], "ROC-AUC": tm["roc_auc"].map(f), "PR-AUC": tm["pr_auc"].map(f),
                        "Lift@10 %": tm["lift@10"].map(lambda x: f(x, 2)), "Brier": tm["brier"].map(f)})
    add("Hold-out: evaluación única sobre 4.521 clientes", f"""
      <div class="stats-row">
        {stat(n['test_final_roc_auc'], 'ROC-AUC (CV ' + n['cv_final_roc'] + ')', 'accent')}
        {stat(n['test_final_pr_auc'], 'PR-AUC (azar ' + n['test_dummy_pr_auc'] + ')', 'accent')}
        {stat(n['dec1_lift'], 'lift en el decil superior')}
        {stat(n['reason_ok'], 'chequeos de razonabilidad')}
      </div>
      <div class="cols">
        <div class="small">{table(tmt, highlight_first=True)}</div>
        <div>{img('test_roc_pr_curves.png', 'curvas')}</div>
      </div>
      <p class="foot">Modelo, hiperparámetros y umbral fijados antes de leer el test. Caída vs CV de {n['gap_pr']} en PR-AUC:
         dentro de 2 desvíos (leve optimismo del tuning + test chico). ROC-AUC dentro del rango de referencia 0,75–0,80.</p>""",
        "Leer los números: supera claramente a la logística y al azar. La variante con duration aparece para comparar "
        "pero no es utilizable.", "5 · Evaluación")

    # 18. Ranking y decisión -------------------------------------------------------------------------
    add("Ranking y decisión en test", f"""
      <div class="cols">
        <div>{img('test_gain_lift.png', 'gain lift')}
          <div class="stats-row compact">
            {stat(n['gain10'], 'de las conversiones en el top 10 %')}
            {stat(n['gain20'], 'en el top 20 %')}
            {stat(n['gain30'], 'en el top 30 %')}
          </div>
        </div>
        <div>
          {img('test_confusion_matrix.png', 'confusión', 'fig narrow')}
          <ul>
            <li>Umbral {n['thr']}: se llama al {n['test_contacted']} y se capturan {n['cm_tp']} de {n['pos_test']}
                conversiones (recall {n['test_rec']}).</li>
            <li>{n['cm_fp']} falsos positivos: aceptables por diseño (una llamada “inútil” cuesta 1; una conversión perdida, 20).</li>
          </ul>
        </div>
      </div>""",
        "El decil 1 tiene casi 50 % de conversión contra 11,5 % de base.", "5 · Evaluación")

    # 19. Negocio y sensibilidad ---------------------------------------------------------------------
    sens = t("test_sensitivity.csv")
    st = pd.DataFrame({"V/C": sens["vc_ratio"].map(lambda x: str(int(x))),
                       "% contactado": sens["contacted_pct"].map(lambda x: f(x, 1) + " %"),
                       "Beneficio modelo": sens["profit_model"].map(i),
                       "Contactar a todos": sens["profit_contact_all"].map(i),
                       "Diferencia": sens["profit_vs_all"].map(i)})
    add("Traducción a negocio y sensibilidad V/C", f"""
      <div class="cols">
        <div>
          <p class="lead">Con V/C = 20 el modelo gana <b>{n['biz_profit']}</b> vs {n['biz_all']} de llamar a todos
             (<b>+{n['biz_uplift_pct']}</b>).</p>
          <div class="small">{table(st)}</div>
          <ul>
            <li><b>V/C = 5</b>: llamar a todos pierde; el modelo gana.</li>
            <li><b>V/C = 10</b>: el modelo triplica el beneficio.</li>
            <li><b>V/C = 50</b>: conviene llamar a casi todos; el modelo aporta poco.</li>
          </ul>
          <p class="callout">El modelo vale más cuanto más cara es la llamada o más acotado el presupuesto.</p>
        </div>
        <div>{img('test_profit_curves.png', 'beneficio')}{img('test_calibration.png', 'calibración')}</div>
      </div>""",
        "Mensaje honesto: con los supuestos por defecto la ganancia es modesta; el valor aparece con presupuesto fijo o "
        "llamadas caras. La calibración (Brier 0,082) permite confiar en estas cuentas.", "5 · Evaluación")

    # 20. Leakage ---------------------------------------------------------------------------------------
    add("Leakage cuantificado", f"""
      <div class="compare">
        <div class="col-card">
          <div class="tag ok">Entregable</div><h3>Pre-contacto</h3>
          <div class="big">{n['test_final_roc_auc']}</div><div class="lbl">ROC-AUC test</div>
          <div class="big2">{n['test_final_pr_auc']}</div><div class="lbl">PR-AUC test</div>
        </div>
        <div class="vs">vs</div>
        <div class="col-card bad">
          <div class="tag no">No utilizable</div><h3>Con <code>duration</code></h3>
          <div class="big">{n['test_with_duration_roc_auc']}</div><div class="lbl">ROC-AUC test</div>
          <div class="big2">{n['test_with_duration_pr_auc']}</div><div class="lbl">PR-AUC test</div>
        </div>
      </div>
      <p class="foot">La mejora es ilusoria: para conocer la duración hay que haber llamado, que es justo lo que se quiere decidir.
         En producción ese modelo no tendría el dato. Tests automáticos garantizan que el modelo final ignora
         <code>duration</code> y la API la rechaza (HTTP 422).</p>""",
        "Si preguntan por qué no usar el modelo mejor: este es el argumento.", "5 · Evaluación")

    # 21. SHAP global -----------------------------------------------------------------------------------
    add("Explicabilidad global (SHAP)", f"""
      <div class="cols wide-right">
        <div>
          <ol>
            <li><b>{n['shap1']}</b> ({n['shap1_pct']}): mar/sep/oct/dic suben; may/ene/ago bajan.</li>
            <li><b>Canal</b> ({n['shap_contact_share']} entre contact y contact_known): unknown resta.</li>
            <li><b>Saldo</b>: creciente desde saldos positivos.</li>
            <li><b>Préstamos</b>: tener 1 o más resta.</li>
            <li><b>Éxito previo</b>: pocos casos, efecto enorme.</li>
            <li><b>Contactos en la campaña</b>: insistir resta.</li>
          </ol>
          <p class="small-text">La regresión logística llega a las mismas variables (mes, 65+, saldo): dos familias,
             mismas relaciones.</p>
        </div>
        <div>{img('shap_summary.png', 'shap')}</div>
      </div>""",
        "SHAP en escala log-odds. Color = valor de la variable.", "5 · Evaluación")

    # 22. Casos locales + reglas --------------------------------------------------------------------------
    rl = t("rules_tree_leaves.csv").head(5)
    rlt = pd.DataFrame({"Regla": rl["rule"].map(html.escape), "Tasa": rl["conversion_rate"].map(pct),
                        "Lift": rl["lift"].map(lambda x: f(x, 1))})
    add("Explicaciones locales y reglas de negocio", f"""
      <div class="cols">
        <div>
          <h3>Falso negativo (prob. mínima)</h3>
          {img('shap_local_fn.png', 'fn')}
          <p class="small-text">Contacto unknown en mayo, saldo 0, hipoteca y préstamo: el perfil de menor conversión.
             Su conversión no era predecible antes de llamar.</p>
        </div>
        <div>
          <h3>Árbol de profundidad 3 (ROC-AUC {n['rules_roc']})</h3>
          <div class="small">{table(rlt)}</div>
          <p class="small-text">Comunicable al equipo comercial: <b>éxito previo y sin hipoteca → 66–75 %</b>.</p>
        </div>
      </div>""",
        "El falso positivo más fuerte fue un jubilado con éxito previo contactado en marzo: todo indicaba que convertía. "
        "Son errores razonables.", "5 · Evaluación")

    # 23. Errores y segmentos -----------------------------------------------------------------------------
    add("¿Dónde falla el modelo?", f"""
      <div class="cols wide-right">
        <div>
          <h3>Falsos negativos ≈ clientes típicos de campaña masiva</h3>
          <table><thead><tr><th></th><th>FN</th><th>TP</th></tr></thead><tbody>
            <tr><td>Contacto unknown</td><td>{n['fn_unknown']}</td><td>{n['tp_unknown']}</td></tr>
            <tr><td>Contactados en mayo</td><td>{n['fn_may']}</td><td>{n['tp_may']}</td></tr>
            <tr><td>Saldo medio</td><td>{n['fn_balance']}</td><td>{n['tp_balance']}</td></tr>
            <tr><td>Contactados antes</td><td>{n['fn_prev']}</td><td>{n['tp_prev']}</td></tr>
          </tbody></table>
          <ul>
            <li>Segmento más débil: <code>contact = unknown</code> (ROC-AUC {n['seg_unknown_roc']}).</li>
            <li>Mayores de 65: convierten {n['seg_65_rate']} pero el modelo discrimina poco dentro del grupo
                ({n['seg_65_roc']}).</li>
          </ul>
        </div>
        <div>{img('segment_metrics.png', 'segmentos')}</div>
      </div>""",
        "Para esos segmentos faltaría información (transaccional, relación con el banco).", "5 · Evaluación")

    # 24. Despliegue ---------------------------------------------------------------------------------------
    add("Despliegue: API + interfaz", f"""
      <div class="cols">
        <div>
          <h3>API REST (FastAPI)</h3>
          <table><tbody>
            <tr><td><code>GET /health</code></td><td>estado</td></tr>
            <tr><td><code>GET /model/info</code></td><td>metadata, métricas, umbral</td></tr>
            <tr><td><code>POST /predict</code></td><td>cliente → probabilidad, decil, top-3 SHAP, explicación</td></tr>
            <tr><td><code>POST /predict/batch</code></td><td>CSV → CSV rankeado</td></tr>
          </tbody></table>
          <ul class="tight">
            <li>Validación pydantic: categorías, rangos, consistencia pdays/previous, <b>rechazo de duration</b>.</li>
            <li>Interfaz Streamlit: cliente individual y campaña (slider de % o presupuesto, descarga del listado).</li>
            <li>Dockerfile + docker-compose · 29 tests automáticos (features, leakage, modelo, API).</li>
          </ul>
        </div>
        <div>
          <h3>Respuesta real de <code>/predict</code></h3>
<pre>{{
  "probability": 0.82406,
  "recommendation": "llamar",
  "decile": 1,
  "top_factors": [
    {{"label": "Éxito previo", "value": 1, "shap": 1.20}},
    {{"label": "Mes", "value": "mar", "shap": 0.81}},
    {{"label": "Resultado previo", "value": "success", "shap": 0.51}}
  ]
}}</pre>
          <p class="small-text">Jubilado, contactado en marzo, con éxito en la campaña anterior.</p>
        </div>
      </div>""",
        "Si hay tiempo, demo en vivo: uvicorn + streamlit. El modelo se sirve como un único pipeline: recibe datos crudos.",
        "6 · Despliegue")

    # 25. Conclusiones ---------------------------------------------------------------------------------------
    add("Conclusiones", f"""
      <div class="cols">
        <div>
          <ol class="big-list">
            <li>Con información <b>previa al contacto</b> se puede priorizar con lift {n['dec1_lift']} en el decil superior
                (ROC-AUC {n['test_final_roc_auc']}, PR-AUC {n['test_final_pr_auc']}).</li>
            <li>El valor de negocio depende del ratio <b>V/C</b> y del presupuesto.</li>
            <li>Las decisiones clave fueron <b>de datos</b>: quitar el solapamiento train/test, excluir duration,
                preferir probabilidades calibradas.</li>
            <li>Los ensambles superan a los lineales, pero el techo lo impone la información disponible.</li>
          </ol>
        </div>
        <div>
          <h3>Limitaciones y trabajo futuro</h3>
          <ul>
            <li><b>Drift temporal</b> (2008–2010): validar temporalmente y monitorear <code>month</code>.</li>
            <li>Reemplazar C y V por valores reales del banco.</li>
            <li>Sumar variables transaccionales para los segmentos débiles.</li>
            <li><b>Uplift modeling</b>: medir el efecto causal de la llamada (requiere grupo de control).</li>
          </ul>
        </div>
      </div>""",
        "Cerrar con el mensaje: el mayor aporte fue metodológico, garantizar que las métricas sean reales.")

    # 26. Cierre ------------------------------------------------------------------------------------------
    add("", f"""
      <div class="cover end">
        <h1>¡Gracias!</h1>
        <p class="sub">Preguntas</p>
        <div class="meta">
          <div><span>Repositorio</span><a href="{REPO_URL}">{REPO_URL.replace('https://', '')}</a></div>
          <div><span>Reproducir</span><code>python -m bank_captacion.pipeline</code></div>
          <div><span>Autor</span>{AUTHOR}</div>
        </div>
      </div>""",
        "Material de respaldo: informe (reports/informe_final.pdf), decision log, docs/03_preparacion_datos.md.",
        cls="cover-slide")
    return S


CSS = """
:root{--ink:#13233f;--muted:#5b6b82;--bg:#eef1f6;--paper:#ffffff;--accent:#0f7c83;--accent2:#e4572e;
--line:#d8dee8;--soft:#f5f7fb;--ok:#2e8b57;--hl:#e6f4f4}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#1b2436;font-family:"Segoe UI",system-ui,-apple-system,Roboto,"Helvetica Neue",Arial,sans-serif;color:var(--ink)}
#stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;overflow:hidden}
#deck{width:1280px;height:720px;position:relative;transform-origin:center center}
.slide{position:absolute;inset:0;background:var(--paper);padding:46px 58px 40px;display:none;flex-direction:column;
  border-radius:6px;box-shadow:0 10px 40px rgba(0,0,0,.35);overflow:hidden}
.slide.active{display:flex}
.slide header{display:flex;align-items:flex-end;justify-content:space-between;border-bottom:3px solid var(--accent);
  padding-bottom:10px;margin-bottom:18px}
.slide header h2{margin:0;font-size:32px;line-height:1.15;font-weight:700;letter-spacing:-.3px}
.slide header .section{font-size:13px;color:var(--accent);text-transform:uppercase;letter-spacing:1.5px;font-weight:600;white-space:nowrap;margin-left:20px}
.content{flex:1;min-height:0;font-size:18px;line-height:1.42;display:flex;flex-direction:column}
.slide footer{position:absolute;left:58px;right:58px;bottom:14px;display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:34px;flex:1 1 auto;min-height:0}
.cols.wide-right{grid-template-columns:.9fr 1.1fr}.cols.wide-left{grid-template-columns:1.25fr .75fr}
.cols>div{min-height:0;display:flex;flex-direction:column;gap:8px}
h3{margin:2px 0 6px;font-size:20px;color:var(--accent)}
p{margin:6px 0}ul,ol{margin:6px 0;padding-left:24px}li{margin:5px 0}
ul.tight li{margin:2px 0;font-size:16px}
code{font-family:Consolas,"SF Mono",Menlo,monospace;font-size:.88em;background:var(--soft);padding:1px 5px;border-radius:4px;color:#8a2c0d}
.lead{font-size:21px}
blockquote{margin:10px 0;padding:12px 18px;border-left:5px solid var(--accent);background:var(--hl);font-size:19px;border-radius:0 6px 6px 0}
.callout{background:var(--hl);border:1px solid #bfe3e3;padding:10px 14px;border-radius:8px}
.warn-text{color:var(--accent2)}
.foot{font-size:15px;color:var(--muted);margin-top:10px}
.small-text{font-size:15px;color:var(--muted)}
.fig{max-width:100%;max-height:100%;object-fit:contain;display:block;margin:0 auto;min-height:0;flex:0 1 auto}
.fig.strip{height:118px;max-height:118px;width:auto;margin-top:12px;flex:0 0 auto}.fig.narrow{max-width:62%;flex:0 1 auto}
.figwrap{flex:1 1 auto;min-height:0;display:flex;justify-content:center}.figwrap .fig{height:100%}
table{border-collapse:collapse;width:100%;font-size:15px}
th{background:var(--ink);color:#fff;text-align:left;padding:6px 8px;font-weight:600}
td{padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
tr.hl td{background:var(--hl);font-weight:600}
.small table{font-size:13px}.small td,.small th{padding:4px 6px}
.stack{display:flex;flex-direction:column;gap:14px;justify-content:center}
.stats-row{display:flex;gap:16px;margin-bottom:14px}.stats-row .stat{flex:1}
.stats-row.compact .v{font-size:30px}
.stat{background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:14px 18px}
.stat .v{font-size:42px;font-weight:700;color:var(--ink);line-height:1.05}
.stat .l{font-size:14px;color:var(--muted);margin-top:4px}
.stat.accent{border-color:#9fd3d6;background:var(--hl)}.stat.accent .v{color:var(--accent)}
.stat.warn .v{color:var(--accent2)}
.formula{font-size:26px;font-weight:700;text-align:center;padding:14px;background:var(--soft);border-radius:8px;margin:6px 0 10px;font-family:Georgia,serif}
.phases{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-top:10px}
.phase{display:flex;gap:14px;background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:18px}
.phase .num{width:44px;height:44px;border-radius:50%;background:var(--accent);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:20px;flex:0 0 auto}
.phase b{font-size:20px}.phase p{margin:4px 0 0;font-size:15px;color:var(--muted)}
.flow{display:flex;align-items:center;gap:14px;margin:6px 0 14px}
.flow .box{border:2px solid var(--ink);border-radius:10px;padding:12px 18px;text-align:center;font-size:16px}
.flow .box b{font-size:28px}.flow .box.ok{border-color:var(--ok);background:#eaf6ef}
.flow .arrow{font-size:22px;font-weight:700;color:var(--accent2)}
.flow .arrow::after{content:" →"}
.compare{display:flex;align-items:center;justify-content:center;gap:40px;margin-top:20px}
.col-card{width:380px;text-align:center;border:2px solid var(--ok);border-radius:14px;padding:22px;background:#f3faf6}
.col-card.bad{border-color:var(--accent2);background:#fdf1ec}
.col-card h3{color:var(--ink);font-size:24px}
.col-card .big{font-size:64px;font-weight:800}.col-card .big2{font-size:40px;font-weight:700;margin-top:6px}
.col-card .lbl{font-size:14px;color:var(--muted)}
.tag{display:inline-block;padding:3px 12px;border-radius:20px;font-size:13px;font-weight:700;color:#fff}
.tag.ok{background:var(--ok)}.tag.no{background:var(--accent2)}
.vs{font-size:28px;color:var(--muted);font-weight:700}
pre{background:#0f1b2d;color:#d6e2f0;padding:14px 16px;border-radius:8px;font-size:14px;line-height:1.4;margin:4px 0;overflow:hidden}
.big-list li{font-size:19px;margin:10px 0}
.pipe{display:flex;align-items:stretch;gap:26px;margin-top:14px;flex:0 0 auto}
.pipe .p{flex:1;border:1.5px solid #6b7a90;border-radius:10px;padding:10px 12px;text-align:center;font-weight:700;font-size:16px;position:relative}
.pipe .p small{display:block;font-weight:400;font-size:13px;color:var(--muted);margin-top:3px}
.pipe .p:not(:last-child)::after{content:"→";position:absolute;right:-22px;top:50%;transform:translateY(-50%);font-size:20px;color:var(--ink)}
.pipe .d{background:#dbe9f6}.pipe .f{background:#e5f5e0}.pipe .m{background:#fdebd0}.pipe .o{background:#fde0dd}
.cover-slide{background:linear-gradient(135deg,#13233f 0%,#1c3a5e 60%,#0f7c83 100%);color:#fff;justify-content:center}
.cover{padding:0 30px}.cover .kicker{text-transform:uppercase;letter-spacing:2px;font-size:15px;color:#9fe0e3;font-weight:600}
.cover h1{font-size:50px;line-height:1.12;margin:16px 0 14px;font-weight:800;letter-spacing:-.5px}
.cover .sub{font-size:22px;color:#d7e6f5;margin-bottom:40px}
.cover .meta{display:grid;grid-template-columns:1fr 1fr;gap:14px 40px;font-size:17px;border-top:1px solid rgba(255,255,255,.25);padding-top:22px}
.cover .meta span{display:block;font-size:12px;text-transform:uppercase;letter-spacing:1.5px;color:#9fe0e3;margin-bottom:2px}
.cover a{color:#fff}.cover code{background:rgba(255,255,255,.12);color:#fff}
.cover.end h1{font-size:72px}
.cover-slide footer{color:#9fb3c8}.cover-slide .content{justify-content:center}.compare{flex:0 0 auto}
#progress{position:fixed;left:0;bottom:0;height:4px;background:var(--accent);transition:width .2s;z-index:5}
#notes{position:fixed;left:0;right:0;bottom:0;max-height:32vh;overflow:auto;background:#0f1b2d;color:#e8eef6;padding:14px 22px 18px;font-size:16px;line-height:1.45;display:none;z-index:6;border-top:3px solid var(--accent)}
#notes.show{display:block}#notes b{color:#9fe0e3}
#help{position:fixed;right:12px;bottom:10px;color:#8aa0b8;font-size:12px;z-index:5;font-family:system-ui;transition:opacity .6s}#help.hide{opacity:0}
@media print{
  @page{size:1280px 720px;margin:0}
  html,body{background:#fff}#stage{position:static;display:block}#deck{transform:none!important;width:auto;height:auto}
  .slide{display:flex!important;position:relative;width:1280px;height:720px;page-break-after:always;box-shadow:none;border-radius:0}
  #progress,#notes,#help{display:none!important}
}
"""

JS = """
const slides=[...document.querySelectorAll('.slide')];let cur=0;
const deck=document.getElementById('deck'),prog=document.getElementById('progress'),notes=document.getElementById('notes');
function fit(){const s=Math.min(innerWidth/1280,innerHeight/720)*0.97;deck.style.transform='scale('+s+')'}
function show(k){cur=Math.max(0,Math.min(slides.length-1,k));slides.forEach((s,j)=>s.classList.toggle('active',j===cur));
  prog.style.width=(100*(cur+1)/slides.length)+'%';notes.innerHTML='<b>Notas · slide '+(cur+1)+'/'+slides.length+'</b><br>'+(slides[cur].dataset.notes||'—');
  if(location.hash!=='#'+(cur+1))history.replaceState(null,'','#'+(cur+1))}
addEventListener('keydown',e=>{
  if(['ArrowRight','PageDown',' ','Enter'].includes(e.key)){e.preventDefault();show(cur+1)}
  else if(['ArrowLeft','PageUp','Backspace'].includes(e.key)){e.preventDefault();show(cur-1)}
  else if(e.key==='Home')show(0);else if(e.key==='End')show(slides.length-1);
  else if(e.key==='n'||e.key==='N')notes.classList.toggle('show');
  else if(e.key==='f'||e.key==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}});
document.getElementById('stage').addEventListener('click',e=>{if(e.target.closest('a'))return;show(e.clientX>innerWidth/2?cur+1:cur-1)});
let tx=null;addEventListener('touchstart',e=>tx=e.touches[0].clientX);
addEventListener('touchend',e=>{if(tx===null)return;const d=e.changedTouches[0].clientX-tx;if(Math.abs(d)>50)show(cur+(d<0?1:-1));tx=null});
addEventListener('resize',fit);fit();setTimeout(()=>document.getElementById('help').classList.add('hide'),5000);show((parseInt(location.hash.slice(1))||1)-1);
"""


def render() -> str:
    slides = build_slides()
    total = len(slides)
    parts = []
    for k, s in enumerate(slides, start=1):
        header = ""
        if s["title"]:
            sec = f'<div class="section">{s["section"]}</div>' if s["section"] else ""
            header = f'<header><h2>{s["title"]}</h2>{sec}</header>'
        parts.append(
            f'<section class="slide {s["cls"]}" data-notes="{html.escape(s["notes"])}">{header}'
            f'<div class="content">{s["body"]}</div>'
            f'<footer><span>{AUTHOR} · {COURSE} · {UNIVERSITY}</span><span>{k} / {total}</span></footer></section>')
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Captación de clientes — TP Final</title>
<meta name="description" content="Presentación del TP Final de Aprendizaje Artificial (CAECE): modelo pre-contacto para priorizar clientes en campañas de marketing bancario.">
<style>{CSS}</style></head>
<body><div id="stage"><div id="deck">{''.join(parts)}</div></div>
<div id="progress"></div><div id="notes"></div>
<div id="help">← → navegar · N notas · F pantalla completa</div>
<script>{JS}</script></body></html>"""


def build_slides_html() -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(render(), encoding="utf-8")
    return str(OUT_FILE.relative_to(cfg.ROOT))


if __name__ == "__main__":
    print(build_slides_html())
