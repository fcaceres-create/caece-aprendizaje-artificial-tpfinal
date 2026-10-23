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

import numpy as np
import pandas as pd

from . import config as cfg
from .export_report import collect_numbers, f, i, pct
from .glossary import TIP_CSS, TIP_JS, annotate
from .slides_prep import PREP_CSS, block, prep_slides

OUT_DIR = cfg.ROOT / "presentacion"
OUT_FILE = OUT_DIR / "index.html"
REPO_URL = "https://github.com/fcaceres-create/caece-aprendizaje-artificial-tpfinal"
APP_URL = "https://captacion-clientes.onrender.com"

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


def crisp_svg() -> str:
    """Diagrama del ciclo CRISP-DM (SVG en línea): flujo principal en el anillo y vueltas atrás punteadas."""
    import math
    cx, cy, r = 240, 222, 160
    names = ["Negocio", "Datos", "Preparación", "Modelado", "Evaluación", "Despliegue"]
    ang = [-90 + 60 * k for k in range(6)]
    pos = [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in ang]
    pt = lambda a, rr=r: (cx + rr * math.cos(math.radians(a)), cy + rr * math.sin(math.radians(a)))
    parts = [f'<svg class="crisp-svg" viewBox="0 0 480 444" role="img" aria-label="Ciclo CRISP-DM">',
             '<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
             '<path d="M0,0 L10,5 L0,10 z" fill="#0f7c83"/></marker>'
             '<marker id="ahd" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
             '<path d="M0,0 L10,5 L0,10 z" fill="#e4572e"/></marker></defs>']
    # Anillo: flujo principal (sentido horario), un arco por cada transición.
    for k in range(6):
        a0, a1 = ang[k] + 21, ang[k] + 60 - 21
        x0, y0 = pt(a0)
        x1, y1 = pt(a1)
        parts.append(f'<path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 0 1 {x1:.1f},{y1:.1f}" fill="none" stroke="#0f7c83" '
                     f'stroke-width="2.5" marker-end="url(#ah)"/>')
    # Vueltas atrás (iteración): curvas punteadas hacia adentro.
    def back(i, j, bend):
        (x0, y0), (x1, y1) = pos[i], pos[j]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        qx, qy = mx + (cx - mx) * bend, my + (cy - my) * bend
        # recortar extremos para no tapar las cajas
        def trim(xa, ya, xb, yb, d=34):
            L = math.hypot(xb - xa, yb - ya) or 1
            return xa + (xb - xa) * d / L, ya + (yb - ya) * d / L
        sx, sy = trim(x0, y0, qx, qy)
        ex, ey = trim(x1, y1, qx, qy)
        return (f'<path d="M{sx:.1f},{sy:.1f} Q{qx:.1f},{qy:.1f} {ex:.1f},{ey:.1f}" fill="none" stroke="#e4572e" '
                f'stroke-width="2" stroke-dasharray="5 4" marker-end="url(#ahd)"/>')
    parts += [back(1, 0, 0.55), back(3, 2, 0.55), back(4, 0, 0.35)]
    # Centro.
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="46" fill="#e3f3f3" stroke="#0f7c83" stroke-width="1.5"/>'
                 f'<text x="{cx}" y="{cy + 6}" text-anchor="middle" font-size="17" font-weight="700" fill="#13233f">Datos</text>')
    # Nodos.
    for k, ((x, y), name) in enumerate(zip(pos, names)):
        w = 138
        parts.append(f'<rect x="{x - w / 2:.1f}" y="{y - 21:.1f}" width="{w}" height="42" rx="21" fill="#13233f"/>'
                     f'<circle cx="{x - w / 2 + 21:.1f}" cy="{y:.1f}" r="13" fill="#0f7c83"/>'
                     f'<text x="{x - w / 2 + 21:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-size="14" font-weight="700" fill="#fff">{k + 1}</text>'
                     f'<text x="{x - w / 2 + 40:.1f}" y="{y + 5:.1f}" font-size="14.5" font-weight="600" fill="#fff">{name}</text>')
    parts.append('<g font-size="12.5" fill="#5b6b82"><line x1="60" y1="430" x2="88" y2="430" stroke="#0f7c83" stroke-width="2.5"/>'
                 '<text x="94" y="434">flujo principal</text>'
                 '<line x1="220" y1="430" x2="248" y2="430" stroke="#e4572e" stroke-width="2" stroke-dasharray="5 4"/>'
                 '<text x="254" y="434">vuelta atrás (iteración)</text></g></svg>')
    return "".join(parts)


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

    # 2. Metodología: CRISP-DM con diagrama del ciclo ------------------------------------------------
    phases = [("1", "Negocio", "Decidir a quién llamar; métricas de ranking y beneficio (V, C)"),
              ("2", "Datos", "Verificación, hallazgo del solapamiento train/test y EDA"),
              ("3", "Preparación", "24 variables pre-contacto dentro de un pipeline sin fugas"),
              ("4", "Modelado", "10 algoritmos, desbalance, tuning, matriz de decisión y umbral"),
              ("5", "Evaluación", "Única evaluación en test, negocio, leakage, SHAP y errores"),
              ("6", "Despliegue", "API + app web publicada, Docker y tests")]
    add("Metodología: CRISP-DM",
        '<p class="lead crisp-sub"><b>C</b>ross-<b>I</b>ndustry <b>S</b>tandard <b>P</b>rocess for <b>D</b>ata '
        '<b>M</b>ining: el estándar para organizar proyectos de ciencia de datos. El modelo es solo una de las '
        'seis fases, y el proceso es <b>iterativo</b>.</p>'
        f'<div class="cols crisp">{crisp_svg()}<div class="crisp-list">' + "".join(
            f'<div class="crisp-item"><span class="num">{a}</span><div><b>{b}</b><p>{c}</p></div></div>'
            for a, b, c in phases) + "</div></div>"
        + '<p class="foot">Todo se reconstruye con un solo comando (<code>python -m bank_captacion.pipeline</code>, ~4 min): '
          "cada número de esta presentación sale de esa ejecución.</p>",
        "CRISP-DM = Cross-Industry Standard Process for Data Mining. Mensaje: el modelo es solo una de seis fases; "
        "primero entender la decisión de negocio y los datos. Las flechas punteadas son las vueltas atrás: en este "
        "trabajo, el hallazgo de la fase 2 (test dentro del train) cambió la preparación, y la evaluación (efecto del mes) "
        "lleva a recomendar volver a los datos con información reciente. Por eso el foco de la presentación está en los "
        "datos y en justificar cada decisión.")

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

    # 4a. Conceptos base: matriz de confusión, precision, recall y accuracy -----------------------------
    _tp, _fp = int(n["cm_tp"].replace(".", "")), int(n["cm_fp"].replace(".", ""))
    _fn, _tn = int(n["cm_fn"].replace(".", "")), int(n["cm_tn"].replace(".", ""))
    _tot = _tp + _fp + _fn + _tn
    add("Conceptos base: ¿cómo se mide un clasificador?", f"""
      <div class="cols">
        <div>
          <h3>Matriz de confusión <span class="small-text">(resultados reales del test, umbral {n['thr']})</span></h3>
          <div class="cm">
            <div></div><div class="cm-h">Decisión: <b>llamar</b></div><div class="cm-h">Decisión: <b>no llamar</b></div>
            <div class="cm-r">Real: <b>convirtió</b></div>
            <div class="cm-c good"><span>TP</span><b>{i(_tp)}</b><small>verdaderos positivos<br>llamé y convirtió</small></div>
            <div class="cm-c bad"><span>FN</span><b>{i(_fn)}</b><small>falsos negativos<br>no llamé y hubiera convertido</small></div>
            <div class="cm-r">Real: <b>no convirtió</b></div>
            <div class="cm-c bad"><span>FP</span><b>{i(_fp)}</b><small>falsos positivos<br>llamé y no convirtió</small></div>
            <div class="cm-c good"><span>TN</span><b>{i(_tn)}</b><small>verdaderos negativos<br>no llamé y no convertía</small></div>
          </div>
          <p class="small-text">Positivo = “convierte”. Los nombres cruzan acierto/error (verdadero/falso) con la decisión
             del modelo (positivo/negativo).</p>
        </div>
        <div>
          <div class="def"><b>Precision</b> = TP / (TP + FP) <span>“puntería”</span>
            <p>De los que llamé, ¿cuántos convirtieron? {i(_tp)} / {i(_tp + _fp)} = <b>{pct(_tp / (_tp + _fp))}</b></p></div>
          <div class="def"><b>Recall</b> = TP / (TP + FN) <span>“cobertura” o sensibilidad</span>
            <p>De todos los que iban a convertir, ¿a cuántos llamé? {i(_tp)} / {i(_tp + _fn)} = <b>{pct(_tp / (_tp + _fn))}</b></p></div>
          <p class="small-text"><b>Trade-off:</b> bajar el umbral hace llamar a más gente → sube el recall y baja la precision.
             El umbral se elige con criterio económico (siguiente slides).</p>
          <div class="def warnbox"><b>Accuracy</b> = (TP + TN) / total <span>la métrica más intuitiva… y engañosa aquí</span>
            <p>Este modelo: ({i(_tp)} + {i(_tn)}) / {i(_tot)} = <b>{pct((_tp + _tn) / _tot)}</b>.
               Un modelo que <b>nunca llama</b>: {pct(1 - (_tp + _fn) / _tot)}. ¡Gana el que no sirve para nada!</p></div>
        </div>
      </div>""",
        "Explicar la matriz con los números reales. Precision = puntería, recall = cobertura. Remate: con 11,5 % de "
        "positivos, accuracy premia al modelo que nunca llama (88,5 %) por encima del nuestro (47 %), que encuentra el 87 % "
        "de los clientes. Por eso usamos métricas de ranking y un umbral económico.", "1 · Negocio")

    # 4a bis. Las métricas del trabajo (siglas y nombres completos) ----------------------------------
    add("Las métricas del trabajo: siglas y significado", f"""
      <table class="explain"><thead><tr><th>Sigla</th><th>Nombre completo</th><th>Qué responde, en palabras simples</th>
        <th>Cómo se lee</th><th>En test</th></tr></thead><tbody>
        <tr class="hl"><td><b>PR-AUC</b><br><small>principal</small></td>
            <td><i>Area Under the Precision-Recall Curve</i> (también <i>Average Precision</i>)</td>
            <td>De los que el modelo pone arriba del ranking, ¿cuántos convierten? Resume la precision a lo largo de todo el ranking.</td>
            <td>0 a 1. Al azar = tasa base</td><td><b>{n['test_final_pr_auc']}</b><br><small>azar {n['test_dummy_pr_auc']}</small></td></tr>
        <tr class="hl"><td><b>Gain @ k %</b><br><small>principal</small></td>
            <td><i>Cumulative gain</i> (ganancia acumulada) en el k %</td>
            <td>Llamando solo al k % mejor rankeado, ¿qué % del total de conversiones capturo?</td>
            <td>Al azar = k %</td><td><b>{n['gain20']}</b><br><small>en el top 20 %</small></td></tr>
        <tr class="hl"><td><b>Lift @ k %</b><br><small>principal</small></td>
            <td><i>Lift</i> (elevación) en el k %</td>
            <td>¿Cuántas veces más conversiones hay en ese grupo que eligiendo al azar?</td>
            <td>Al azar = 1</td><td><b>{n['dec1_lift']}×</b><br><small>en el top 10 %</small></td></tr>
        <tr><td><b>ROC-AUC</b><br><small>secundaria</small></td>
            <td><i>Area Under the Receiver Operating Characteristic Curve</i></td>
            <td>Con un cliente que convierte y uno que no, ¿qué probabilidad hay de que el modelo ponga primero al que convierte?</td>
            <td>0,5 = azar · 1 = perfecto</td><td><b>{n['test_final_roc_auc']}</b></td></tr>
        <tr><td><b>Brier</b><br><small>secundaria</small></td>
            <td><i>Brier score</i> (Glenn Brier, 1950, pronósticos del clima)</td>
            <td>¿Las probabilidades son creíbles? Si dice “30 %”, ¿convierten ~30 %?</td>
            <td>0 = perfecto · menor es mejor</td><td><b>{n['test_final_brier']}</b></td></tr>
      </tbody></table>
      <div class="cols two-notes">
        <p class="callout"><b>¿Por qué PR-AUC es la principal?</b> Con 11,7 % de positivos, ROC-AUC puede verse bien gracias a
           los muchos negativos fáciles de ordenar. PR-AUC mira solo lo que importa: si los que pongo arriba convierten.</p>
        <p class="callout"><b>AUC</b> = área bajo la curva: resume en un número cómo rinde el modelo en <b>todos</b> los umbrales
           posibles. <b>Gain y lift</b> vienen del marketing directo: traducen el ranking a “a cuántos llamo”.</p>
      </div>""",
        "Nombres estándar de la literatura y de scikit-learn. ROC viene del análisis de radares en la Segunda Guerra Mundial. "
        "PR-AUC es la principal porque con datos desbalanceados es más exigente que ROC-AUC. Gain y lift son el puente con "
        "la decisión de negocio. Brier verifica que las probabilidades sirvan para calcular el beneficio.", "1 · Negocio")

    # 4b. Fórmula del beneficio explicada ------------------------------------------------------------
    tp = int(n["cm_tp"].replace(".", ""))
    fp = int(n["cm_fp"].replace(".", ""))
    pos = int(n["pos_test"].replace(".", ""))
    tot = int(n["test_rows"].replace(".", ""))
    V, C = int(cfg.VALUE_PER_CONVERSION), int(cfg.COST_PER_CONTACT)
    add("¿Cuánto gana el banco? El beneficio esperado", f"""
      <div class="formula">Beneficio = <span class="v">V</span> · <span class="t">TP</span> − <span class="c">C</span> · (<span class="t">TP</span> + <span class="f">FP</span>)</div>
      <div class="cols">
        <div>
          <div class="terms">
            <div class="term"><span class="v">V</span><div><b>Valor de una conversión</b><br>Lo que gana el banco cada vez que un
                 cliente llamado acepta el producto. Supuesto: <b>V = {V}</b>.</div></div>
            <div class="term"><span class="c">C</span><div><b>Costo de una llamada</b><br>Operador, telefonía, molestia al cliente.
                 Supuesto: <b>C = {C}</b>.</div></div>
            <div class="term"><span class="t">TP</span><div><b>Verdaderos positivos</b><br>Clientes que <b>llamamos</b> y
                 <b>sí</b> se convirtieron.</div></div>
            <div class="term"><span class="f">FP</span><div><b>Falsos positivos</b><br>Clientes que <b>llamamos</b> y
                 <b>no</b> se convirtieron (llamada sin resultado).</div></div>
          </div>
          <p class="small-text">En palabras: <b>ingresos de las conversiones logradas − costo de todas las llamadas
             hechas</b> (TP + FP = total de llamadas). A quien no se llama no cuesta ni genera nada.</p>
        </div>
        <div>
          <div class="example">
            <h3>Ejemplo con los resultados reales del test</h3>
            <p>El modelo recomendó llamar a {i(tp + fp)} clientes: <b>{i(tp)}</b> se convirtieron (TP) y
               <b>{i(fp)}</b> no (FP).</p>
            <div class="calc">{V} × {i(tp)} − {C} × ({i(tp)} + {i(fp)})<br>= {i(V * tp)} − {i(C * (tp + fp))} = <b>{i(V * tp - C * (tp + fp))}</b></div>
            <p>Llamar a los {i(tot)}: {V} × {i(pos)} − {C} × {i(tot)} = <b>{i(V * pos - C * tot)}</b>.</p>
          </div>
          <div class="callout"><b>Regla de decisión:</b> llamar a un cliente con probabilidad <b>p</b> de convertir conviene si
             lo que se espera ganar (<b>p × V</b>) supera lo que cuesta (<b>C</b>):
             p × {V} &gt; {C} ⇒ <b>p &gt; {C}/{V} = 0,05</b> (5 %).</div>
        </div>
      </div>""",
        "V y C son supuestos: el banco no los informa. Lo que importa es la relación entre ambos (V/C = 20), por eso "
        "después se muestra qué pasa con 5, 10, 20 y 50. El banco puede poner sus valores reales en config.py y todo "
        "se recalcula. El umbral teórico de 5 % reaparece más adelante: el óptimo empírico da exactamente eso.",
        "1 · Negocio")

    # 5. Datos (con muestra real del archivo y lupa) ---------------------------------------------------
    _raw = pd.read_csv(cfg.TRAIN_FILE, sep=cfg.CSV_SEP).head(5)
    # Diccionario de columnas (según la consigna), agrupado por tipo de información.
    _groups = [
        ("Perfil del cliente", "g1", [("age", "edad"), ("job", "tipo de trabajo"), ("marital", "estado civil"),
                                     ("education", "nivel educativo")]),
        ("Situación financiera", "g2", [("default", "¿tiene préstamos impagos?"), ("balance", "saldo anual promedio"),
                                       ("housing", "¿tiene préstamo hipotecario?"), ("loan", "¿tiene préstamo personal?")]),
        ("Contacto en esta campaña", "g3", [("contact", "tipo de contacto (celular / fijo)"),
                                            ("day", "día del mes del último contacto"), ("month", "mes del último contacto"),
                                            ("duration", "duración de la llamada (s) — leakage"),
                                            ("campaign", "cantidad de contactos en esta campaña")]),
        ("Campañas anteriores y target", "g4", [("pdays", "días desde el contacto previo (-1 = nunca)"),
                                                ("previous", "contactos antes de esta campaña"),
                                                ("poutcome", "resultado de la campaña anterior"),
                                                ("y", "TARGET: ¿se convirtió en cliente?")])]
    _dict_html = "".join(
        f'<div class="dict-g {g}"><div class="dict-h">{t}</div>'
        + "".join(f'<div class="dict-i"><code>{c}</code><span>{d}</span></div>' for c, d in items) + "</div>"
        for t, g, items in _groups)
    add("Los datos: Bank Marketing (UCI)", f"""
      <div class="cols data-top">
        <div>
          <table><thead><tr><th>Archivo</th><th>Equivale a</th><th>Filas</th><th>Conversión</th></tr></thead><tbody>
            <tr><td><code>banca_train.csv</code></td><td>bank-full.csv</td><td>{n['train_raw']}</td><td>{n['rate_train_raw']}</td></tr>
            <tr><td><code>banca_test.csv</code></td><td>bank.csv</td><td>{n['test_rows']}</td><td>{n['rate_test']}</td></tr>
          </tbody></table>
          <ul class="tight">
            <li>17 columnas · separador <code>;</code> · ASCII · target <code>y</code> = <code>yes/no</code>.</li>
            <li>Banco portugués, campañas de <b>mayo 2008 a noviembre 2010</b> (Moro, Cortez y Rita, 2014).</li>
            <li>0 nulos y 0 duplicados en cada archivo; esquema idéntico al diccionario de la consigna.</li>
            <li>Faltantes codificados: <code>unknown</code> y <code>pdays = -1</code>.</li>
          </ul>
        </div>
        <div class="stack">
          {stat('16', 'variables candidatas + 1 target (<code>y</code>)')}
          {stat('1', 'variable prohibida: <code>duration</code> (leakage)', 'warn')}
        </div>
      </div>
      <div class="zoomable">
        <div class="db-cap">Muestra real: primeras 5 filas de <b>banca_train.csv</b> (17 columnas)</div>
        {block(_raw, list(_raw.columns), {"duration": "drop", "y": "chg"}, idx_label="fila",
               hit={"contact": lambda v: v == "unknown", "poutcome": lambda v: v == "unknown", "pdays": lambda v: v == -1})}
        <div class="dict">{_dict_html}</div>
      </div>""",
        "Mostrar la muestra: cada fila es un cliente contactado; y = si se convirtió. Ya en estas 5 filas se ven los "
        "faltantes codificados (contact y poutcome = unknown, pdays = -1, resaltados) y la columna duration (tachada, "
        "se excluye). Debajo, el significado de cada columna. Antes de modelar se verificó formato, esquema, nulos y duplicados: así "
        "apareció el hallazgo de la slide siguiente.", "2 · Datos")

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

    # 7. unknown del canal: una marca de época (explicado en 3 pasos simples) ---------------------------
    from .data import load_clean as _load_clean
    _d = _load_clean()[0]  # train limpio: conserva el orden cronológico del archivo
    _N = len(_d)
    _pos = np.arange(_N) / _N
    _tramos = [(0, .1, "Primer 10 %"), (.1, .2, "10 % – 20 %"), (.2, .3, "20 % – 30 %"), (.3, .4, "30 % – 40 %"),
               (.4, 1.0001, "Resto del archivo")]
    _mes_es = {"jan": "ene", "feb": "feb", "mar": "mar", "apr": "abr", "may": "may", "jun": "jun", "jul": "jul",
               "aug": "ago", "sep": "sep", "oct": "oct", "nov": "nov", "dec": "dic"}
    _rows = []
    for lo, hi, lab in _tramos:
        g = _d[(_pos >= lo) & (_pos < hi)]
        meses = " – ".join(_mes_es[m] for m in g.month.value_counts().head(2).index)
        pu = (g.contact == "unknown").mean()
        cls = ' class="hl-warn"' if pu > .5 else ""
        _rows.append(f"<tr{cls}><td>{lab}</td><td>{meses}{' 2008' if hi <= .4 else ' … 2010'}</td>"
                     f"<td><b>{pct(pu, 0)}</b></td><td>{pct((g.y == 'yes').mean())}</td></tr>")
    _unk = _d[_d.contact == "unknown"]
    _early = _d[_pos < .3]
    _share_early = (np.flatnonzero((_d.contact == "unknown").to_numpy()) < .3 * _N).mean()
    _c_known_all = (_d[_d.contact != "unknown"].y == "yes").mean()
    _c_unk_all = (_unk.y == "yes").mean()
    _c_known_early = (_early[_early.contact != "unknown"].y == "yes").mean()
    _c_unk_early = (_early[_early.contact == "unknown"].y == "yes").mean()
    add("EDA: el canal “unknown” es una marca de época", f"""
      <div class="cols unk3">
        <div class="step">
          <div class="step-n">1</div><h3>El dato</h3>
          <p><code>contact</code> = por qué canal se llamó (celular o fijo). En el <b>{n['pct_unk_contact']}</b> de las filas dice
             <code>unknown</code>: no se registró.</p>
          <div class="vs2"><div><b>{pct(_c_unk_all)}</b><span>convierten si el canal es <i>unknown</i></span></div>
            <div><b>{pct(_c_known_all)}</b><span>convierten si el canal está registrado</span></div></div>
          <p class="small-text">Parecería que “no saber el canal” predice que el cliente no compra… pero eso no tiene sentido.
             Hay que mirar <b>cuándo</b> aparecen.</p>
        </div>
        <div class="step">
          <div class="step-n">2</div><h3>¿Cuándo aparecen?</h3>
          <p class="small-text">El archivo está <b>ordenado en el tiempo</b> (mayo 2008 → noviembre 2010). Cortado en tramos:</p>
          <table class="mini tramos"><thead><tr><th>Tramo del archivo</th><th>Meses</th><th>% unknown</th><th>Conversión</th></tr></thead>
            <tbody>{''.join(_rows)}</tbody></table>
          <p class="small-text">El <b>{pct(_share_early, 0)}</b> de los <i>unknown</i> está al principio: <b>mayo–junio 2008</b>.
             En esos meses el banco <b>no anotaba el canal</b>. Y en esa época convertía poca gente.</p>
        </div>
        <div class="step">
          <div class="step-n">3</div><h3>¿Qué significa?</h3>
          <p class="small-text">Comparando <b>dentro de la misma época</b> (primer 30 % del archivo):</p>
          <div class="vs2"><div><b>{pct(_c_unk_early)}</b><span>canal unknown</span></div>
            <div><b>{pct(_c_known_early)}</b><span>canal registrado</span></div></div>
          <p class="small-text">La diferencia casi desaparece: <b>la baja conversión la explica la época, no el canal</b>.</p>
          <div class="callout"><b>Decisión:</b> no imputar (sería inventar un dato y mezclar 2008 con 2010). Se deja como
            categoría, sabiendo que marca un período, igual que el mes.</div>
        </div>
      </div>
      <p class="foot">Otros <i>unknown</i>: <code>poutcome</code> = nunca contactado antes ({n['unk_poutcome_conv']} vs {n['unk_poutcome_rest']});
         <code>education</code> convierte {n['unk_education_conv']}, más que la moda. Y la conversión general sube de ~3 % a ~47 %
         a lo largo del archivo: <b>los datos cambian con el tiempo</b> (limitación para usar el modelo en el futuro).</p>""",
        "Contarlo como historia en 3 pasos. 1) Los unknown del canal convierten 4 % contra 15 %. 2) Pero el archivo está "
        "ordenado en el tiempo y casi todos los unknown son de mayo-junio 2008: en esos meses el banco no anotaba el canal, y "
        "en esa época convertía poca gente. 3) Comparando dentro de la misma época, la diferencia casi desaparece: es la "
        "época, no el canal. Por eso no se imputa: se deja como categoría, sabiendo que marca un período (como el mes). "
        "Frase para la defensa: 'los unknown no son faltantes al azar, son una marca del período en que no se registraba "
        "el canal'.", "2 · Datos")

    # 8. Conversion por categoría --------------------------------------------------------------------
    _panels = ["poutcome", "job", "month", "contact", "housing", "loan", "education", "marital", "default"]
    add("EDA: tasa de conversión por categoría",
        '<div class="panel-grid">' + "".join(
            f'<div class="zoomable panel"><button class="zoom-btn sm" type="button" title="Ampliar">🔍</button>'
            f'{img(f"eda_conv_{v}.png", f"conversión por {v}", "fig panel-img")}</div>' for v in _panels) + "</div>"
        f'<p class="foot">Rojo = <code>unknown</code> · línea punteada = tasa base ({n["rate_train"]}). Resultado previo '
        f'<b>success</b>: {n["conv_success"]} (5,5× la base) · estudiantes y jubilados convierten más · tener préstamos '
        "reduce la conversión casi a la mitad. Usar 🔍 para ampliar cada gráfico.</p>",
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

    # 10. Códigos especiales y valores extremos (explicado en términos simples) --------------------------
    from .data import load_clean as _lc
    _t = _lc()[0]
    _bal_med = _t.balance.median()
    _bal_2000 = (_t.balance <= 2000).mean()
    _camp_le3 = (_t.campaign <= 3).mean()
    _camp_gt = (_t.campaign > 17).mean()
    add("EDA: un código disfrazado de número y valores extremos", f"""
      <div class="cols ex2">
        <div class="step">
          <div class="step-n">A</div><h3>Un número que en realidad es un código</h3>
          <p><code>pdays</code> = <b>días desde que se contactó al cliente en una campaña anterior</b>. Si nunca fue contactado,
             el archivo pone <b>-1</b>.</p>
          <table class="mini ex"><thead><tr><th>Cliente</th><th>pdays</th><th>Qué significa</th></tr></thead><tbody>
            <tr><td>A</td><td>151</td><td>contactado hace 151 días</td></tr>
            <tr><td>B</td><td>3</td><td>contactado hace 3 días</td></tr>
            <tr class="hl-warn"><td>C</td><td><b>-1</b></td><td><b>nunca</b> contactado</td></tr>
          </tbody></table>
          <p class="small-text"><b>El riesgo:</b> el modelo no sabe que -1 es un código. Lo trataría como un número y vería al
             cliente C “al lado” del B (contactado hace 3 días). Falso.</p>
          <p class="small-text">Es el {n['pct_pdays_neg']} de los clientes. Y la diferencia importa: los nunca contactados
             convierten <b>{n['conv_never']}</b>; los contactados antes, <b>{n['conv_before']}</b>.</p>
          <div class="callout"><b>Cómo se resolvió (P3):</b> separar en dos columnas: “¿fue contactado antes? sí/no” y
            “¿hace cuántos días?” (vacía si no aplica).</div>
        </div>
        <div class="step">
          <div class="step-n">B</div><h3>Valores extremos: pocos casos, números enormes</h3>
          <div class="ext"><b>Saldo (balance)</b>
            <p>La mitad de los clientes tiene <b>{i(_bal_med)} o menos</b> y el {pct(_bal_2000, 0)} no pasa de 2.000…
               pero algunos llegan a <b>{n['bal_max']}</b> y otros están en negativo (hasta {n['bal_min']}).</p></div>
          <div class="ext"><b>Contactos en la campaña (campaign)</b>
            <p>El {pct(_camp_le3, 0)} recibió <b>3 llamadas o menos</b>, pero a algunos se los llamó hasta
               <b>{n['camp_max']} veces</b>. Y llamar más no ayuda: {n['conv_camp1']} aceptan con 1 llamada vs
               {n['conv_camp5']} con 5 o más.</p></div>
          <p class="small-text"><b>El riesgo:</b> unos pocos números gigantes “tiran” de los modelos que hacen cuentas con
             distancias o promedios, y tapan el comportamiento de la mayoría.</p>
          <div class="callout"><b>Cómo se resolvió (P4, P5):</b> no se borró ningún cliente (son casos reales). Se
            <b>comprimió</b> la escala del saldo (102.127 → 11,5) y se <b>recortó</b> la cantidad de llamadas a un tope de
            {n['camp_p99']} (solo el {pct(_camp_gt, 1)} lo superaba).</div>
        </div>
        <div class="ex-figs">
          <div class="zoomable panel"><button class="zoom-btn sm" type="button" title="Ampliar">🔍</button>
            {img('eda_pdays_poutcome.png', 'conversión según contacto previo', 'fig panel-img')}</div>
          <div class="zoomable panel"><button class="zoom-btn sm" type="button" title="Ampliar">🔍</button>
            {img('eda_outliers_balance_campaign.png', 'saldo y contactos', 'fig panel-img')}</div>
        </div>
      </div>""",
        "Dos problemas típicos de los datos reales. A) Un código disfrazado de número: -1 en pdays significa 'nunca', "
        "y si no se separa el modelo lo confunde con 'hace muy poco'. B) Valores extremos: casi todos los clientes tienen "
        "saldos chicos y pocas llamadas, pero unos pocos tienen valores enormes que distorsionan. No se borra a nadie: se "
        "comprime (saldo) o se recorta a un tope (llamadas). El detalle de cada solución está en P3, P4 y P5.",
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

    # 12. Sección completa de preparación del dataset (ver slides_prep.py) ----------------------------
    prep_slides(add, n)

    # 13. Comparación de 10 algoritmos, con explicación simple de cada modelo ----------------------------
    MODEL_TIPS = {
        "lightgbm": "Muchos árboles de decisión chicos, uno después del otro: cada uno corrige los errores del anterior "
                    "(boosting). Rápido y muy preciso en datos de tablas.",
        "xgboost": "Otro algoritmo de boosting (árboles en cadena que corrigen errores), con frenos extra para no "
                   "memorizar los datos.",
        "random_forest": "Cientos de árboles entrenados por separado, cada uno con una muestra distinta de clientes. "
                         "La predicción es el promedio de todos (como una votación).",
        "mlp": "Red neuronal: capas de “neuronas” que combinan las variables con pesos aprendidos. Muy flexible, pero "
               "difícil de explicar.",
        "logreg": "Suma ponderada de las variables que una curva en forma de S convierte en probabilidad. Simple y "
                  "explicable: cada peso dice cuánto suma o resta una variable.",
        "knn": "Busca los 51 clientes más parecidos y predice según cuántos de ellos convirtieron.",
        "naive_bayes": "Combina la probabilidad que aporta cada variable por separado, suponiendo que no se influyen "
                       "entre sí. Muy simple y rápido.",
        "svm_rbf": "Busca la frontera que mejor separa a los que convierten de los que no, con el mayor margen posible "
                   "(la frontera puede ser curva).",
        "tree": "Una serie de preguntas sí/no (¿tuvo éxito antes? ¿tiene hipoteca?) que termina en una predicción. "
                "El más fácil de explicar.",
        "dummy": "Referencia que no mira los datos: le da a todos la misma probabilidad (la tasa base). Es el piso "
                 "que cualquier modelo debe superar.",
    }
    WHY_SHORT = {'lightgbm': 'estado del arte en datos de tablas', 'xgboost': '2.ª implementación de boosting, para confirmar', 'random_forest': 'ensamble por votación, robusto', 'mlp': 'representante de las redes neuronales', 'logreg': 'baseline simple e interpretable', 'knn': 'métodos por similitud entre casos', 'naive_bayes': 'modelo probabilístico simple', 'svm_rbf': 'márgenes y fronteras no lineales', 'tree': 'reglas explicables al negocio', 'dummy': 'piso de referencia'}
    cv = t("cv_results.csv")
    MODEL_WHY = {'lightgbm': 'Representa el boosting, el estado del arte en datos de tablas. Admite categorías nativas.', 'xgboost': 'Segunda implementación de boosting, para confirmar que el resultado no depende de una librería.', 'random_forest': 'Representa los ensambles por votación (bagging): robusto y difícil de sobreajustar.', 'mlp': 'Representa a las redes neuronales vistas en la materia.', 'logreg': 'Es el baseline clásico, simple e interpretable: todo modelo complejo debe superarlo para justificarse.', 'knn': 'Representa los métodos basados en similitud entre casos.', 'naive_bayes': 'Representa los modelos probabilísticos: simple, rápido y buena referencia.', 'svm_rbf': 'Representa los métodos de márgenes con kernels (fronteras no lineales).', 'tree': 'Genera reglas que se le pueden explicar al negocio.', 'dummy': 'Sirve como piso: muestra cuánto aporta realmente cada modelo.'}
    _name = lambda r: (f'<span class="gl" data-tip="{html.escape(MODEL_TIPS[r.model] + " — Por qué lo incluí: " + MODEL_WHY[r.model], quote=True)}">'  # noqa: E731
                       f'{html.escape(r.label)}<sup class="q">?</sup></span>')
    cvt = pd.DataFrame({"#": range(1, len(cv) + 1), "Modelo": [_name(r) for r in cv.itertuples()],
                        "Familia · por qué lo incluí": [f'{fam}<br><span class="why-in">{WHY_SHORT[m]}</span>'
                                                        for fam, m in zip(cv["family"], cv["model"])],
                        "PR-AUC (± variación)": [f"<b>{f(a_)}</b> ± {f(b_)}" for a_, b_ in zip(cv["pr_auc_mean"], cv["pr_auc_std"])],
                        "ROC-AUC": cv["roc_auc_mean"].map(f), "Lift top 20 %": cv["lift@20_mean"].map(lambda x: f(x, 2))})
    _top = cv.iloc[0]
    _second = cv.iloc[1]
    _lr = cv[cv.model == "logreg"].iloc[0]
    add("Modelado: 10 algoritmos compitiendo en igualdad de condiciones", f"""
      <div class="cols wide-left">
        <div class="small">{table(cvt, highlight_first=True)}
          <p class="small-text">Pasar el mouse por el <b>?</b> de cada modelo para ver cómo funciona y el detalle de por qué se incluyó.</p></div>
        <div>
          <div class="def tools"><b>¿Con qué se entrenó?</b>
            <p>Un programa propio en <b>Python</b> con la librería <b>scikit-learn</b> (más LightGBM, XGBoost,
               imbalanced-learn y Optuna). Un solo comando entrena y evalúa todo; el mismo código corre en
               <b>Google Colab</b>.</p>
            <div class="chips"><span>Python</span><span>scikit-learn</span><span>LightGBM</span><span>XGBoost</span>
              <span>pandas</span><span>Colab</span></div></div>
          <div class="def"><b>¿Cómo se probó?</b>
            <p>Validación cruzada: aprende con el 80 % de los {n['train_clean']} clientes, se evalúa con el 20 % que no
               vio y se rota 5 veces. <b>Mismas particiones para los 10.</b> El “±” es cuánto varía entre las 5 pruebas.</p></div>
          <ul class="tight">
            <li><b>Ganan los árboles combinados</b>: casi <b>4 veces</b> mejor que el azar
                ({f(cv[cv.model == 'dummy'].pr_auc_mean.iloc[0])}).</li>
            <li><b>LightGBM primero, pero por poco</b> ({f(_top.pr_auc_mean)} vs {f(_second.pr_auc_mean)}): se confirma
                después con la matriz de decisión.</li>
          </ul>
        </div>
      </div>""",
        "Es una competencia justa: mismos datos, mismas 5 particiones, misma métrica. Pasar el mouse por los ? si "
        "preguntan por algún modelo. Mensaje: los modelos de árboles combinados son los mejores para este problema; "
        "LightGBM queda primero por poco (diferencia menor que la variación entre pruebas), así que la decisión final se "
        "toma más adelante con la matriz de decisión. El Dummy es el piso; la logística es el baseline interpretable.",
        "4 · Modelado")
    # 13 bis. Por qué cada modelo rindió así -------------------------------------------------------------
    _pr = dict(zip(cv.model, cv.pr_auc_mean))
    _why = [
        ("win", "LightGBM · XGBoost · Random Forest",
         f"{f(min(_pr['random_forest'], _pr['xgboost'], _pr['lightgbm']))} – {f(_pr['lightgbm'])}",
         "Combinan muchos árboles: captan <b>combinaciones</b> de variables (p. ej. “éxito previo <i>y</i> sin hipoteca”), "
         "manejan bien categorías y valores extremos. Es el tipo de modelo que mejor funciona en datos de tablas."),
        ("mid", "Red neuronal (MLP)", f(_pr["mlp"]),
         "Muy flexible, pero necesita muchos más datos para aprovecharlo. Con 40 mil filas no supera a los árboles y es "
         "difícil de explicar."),
        ("mid", "Regresión logística", f(_pr["logreg"]),
         "Solo <b>suma efectos de a uno</b> (“tener hipoteca resta X”), no capta combinaciones. La brecha con los árboles "
         "mide cuánto valen esas interacciones."),
        ("low", "K vecinos (KNN)", f(_pr["knn"]),
         "Compara clientes por <b>distancia</b>. Con 67 columnas (one-hot) la distancia pierde sentido: todos parecen igual "
         "de lejos."),
        ("low", "Naive Bayes", f(_pr["naive_bayes"]),
         "Supone que las variables <b>no se relacionan</b>, pero <code>pdays</code>, <code>previous</code> y "
         "<code>poutcome</code> dicen casi lo mismo: cuenta tres veces la misma evidencia."),
        ("low", "SVM", f(_pr["svm_rbf"]),
         "Por su costo de cálculo se entrenó con solo 10 mil clientes. Ordena bien en general, pero peor justo "
         "<b>arriba del ranking</b>, que es lo que importa."),
        ("low", "Árbol de decisión", f(_pr["tree"]),
         "Un solo árbol de 6 niveles es demasiado simple e inestable. Su valor es que se explica fácil (se usa para las "
         "reglas de negocio)."),
        ("base", "Dummy", f(_pr["dummy"]),
         "No mira los datos: le da a todos la tasa base. Es el <b>piso</b> que cualquier modelo útil debe superar."),
    ]
    _rows_why = "".join(
        f'<tr class="why-{k}"><td><b>{m}</b></td><td class="num">{v}</td><td>{d}</td></tr>' for k, m, v, d in _why)
    add("¿Por qué cada modelo rindió así?", f"""
      <table class="why"><thead><tr><th>Modelo</th><th>PR-AUC</th><th>Por qué rinde así (en simple)</th></tr></thead>
        <tbody>{_rows_why}</tbody></table>
      <p class="callout" style="margin-top:10px"><b>En una frase:</b> ganan los modelos que captan <b>combinaciones</b> entre
         variables y toleran categorías y valores extremos; pierden los que suman efectos de a uno, los que dependen de
         <b>distancias</b> y los que suponen <b>independencia</b> entre variables.</p>""",
        "Cada resultado se explica por cómo 'piensa' cada modelo y cómo son estos datos. Árboles combinados: captan "
        "interacciones. Logística: solo efectos de a uno (la brecha mide el valor de las interacciones). KNN: la distancia "
        "pierde sentido con 67 columnas. Naive Bayes: supone independencia y hay variables redundantes. SVM: entrenado con "
        "menos datos. Árbol simple: demasiado simple. Dummy: el piso.", "4 · Modelado")
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
            <li>Frontend web propio servido por la API · Docker · 31 tests automáticos.</li>
            <li>Publicada en <a href="{APP_URL}">{APP_URL.replace('https://', '')}</a>.</li>
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
          <div><span>App web</span><a href="{APP_URL}">{APP_URL.replace('https://', '')}</a></div>
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
.def.tools{background:var(--hl);border-color:#bfe3e3}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.chips span{font-family:Consolas,Menlo,monospace;font-size:12px;background:#fff;border:1px solid #bfe3e3;border-radius:6px;padding:1px 7px;color:var(--ink)}
.why-in{font-size:11.5px;color:var(--accent);font-style:italic;line-height:1.1}
table.why{font-size:14.5px}table.why td{padding:6px 10px;vertical-align:middle}table.why td.num{font-weight:700;white-space:nowrap}
table.why td:first-child{white-space:nowrap;border-left:5px solid transparent}
tr.why-win td{background:#eaf6ef}tr.why-win td:first-child{border-left-color:#1f7a4d}
tr.why-mid td:first-child{border-left-color:#eda100}tr.why-low td:first-child{border-left-color:#e4572e}
tr.why-base td{color:var(--muted)}tr.why-base td:first-child{border-left-color:#8c9bab}
.cols.ex2{grid-template-columns:1fr 1fr .85fr;gap:14px}
.ex2 .step p{font-size:14.5px;margin:4px 0}.ex2 .callout{font-size:13.5px;padding:7px 10px;margin-top:6px}
table.mini.ex{font-size:13px;margin:6px 0}table.mini.ex td,table.mini.ex th{padding:3px 7px}
.ext{background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 10px;margin:6px 0}
.ext b{font-size:14.5px;color:var(--accent)}.ext p{font-size:13.5px !important;margin:2px 0 0 !important}
.ex-figs{display:grid;grid-template-rows:1fr 1fr;gap:10px;min-height:0}
.panel-grid{display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(3,1fr);gap:8px;flex:1 1 auto;min-height:0}
.zoomable.panel{padding:4px;border-style:solid;display:flex;align-items:center;justify-content:center;min-height:0}
.panel-img{max-height:100%;max-width:100%;object-fit:contain}
.zoom-btn.sm{padding:3px 7px;font-size:14px;top:4px;right:4px;opacity:.85}
#zoom .zoom-inner img{display:block;max-width:90vw;max-height:84vh;width:auto;height:auto}
.cols.unk3{grid-template-columns:1fr 1.25fr 1fr;gap:16px}
.step{background:var(--soft);border:1px solid var(--line);border-radius:12px;padding:12px 14px;position:relative}
.step h3{margin:0 0 6px 34px}.step p{font-size:15px}
.step-n{position:absolute;top:10px;left:12px;width:26px;height:26px;border-radius:50%;background:var(--accent);color:#fff;
  font-weight:800;display:flex;align-items:center;justify-content:center;font-size:14px}
.vs2{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0}
.vs2>div{background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px;text-align:center}
.vs2 b{display:block;font-size:24px}.vs2 span{font-size:12px;color:var(--muted);line-height:1.2;display:block}
.vs2>div:first-child b{color:var(--accent2)}.vs2>div:last-child b{color:var(--accent)}
table.tramos{font-size:12.5px}table.tramos td,table.tramos th{padding:4px 6px}
table.tramos tr.hl-warn td{background:#fdf1ec}
.step .callout{font-size:13.5px;padding:8px 10px;margin-top:6px}
.dict{display:grid;grid-template-columns:1fr 1fr 1.15fr 1.15fr;gap:8px;margin-top:8px}
.dict-g{background:var(--soft);border-radius:8px;padding:6px 9px;border-left:4px solid #8c9bab}
.dict-g.g1{border-left-color:#2a78d6}.dict-g.g2{border-left-color:#1baf7a}.dict-g.g3{border-left-color:#eda100}.dict-g.g4{border-left-color:#6250d6}
.dict-h{font-size:11px;text-transform:uppercase;letter-spacing:.8px;font-weight:700;color:var(--muted);margin-bottom:3px}
.dict-i{display:flex;gap:6px;font-size:12px;line-height:1.35;align-items:baseline}
.dict-i code{font-size:11px;flex:0 0 auto}.dict-i span{color:var(--ink)}
#zoom .zoom-inner .dict-i{font-size:clamp(11px,1vw,18px)}#zoom .zoom-inner .dict-i code{font-size:clamp(10px,.92vw,16px)}
#zoom .zoom-inner .dict-h{font-size:clamp(10px,.85vw,14px)}
.cols.data-top{flex:0 0 auto;grid-template-columns:1.5fr 1fr;margin-bottom:10px}.data-top .stat .v{font-size:32px}
.zoomable{position:relative;border:1px dashed var(--line);border-radius:10px;padding:8px 10px;flex:0 0 auto;overflow:hidden}
.zoomable table.db{font-size:11px}.zoomable table.db th,.zoomable table.db td{padding:4px 6px}
.zoom-btn{position:absolute;top:6px;right:8px;font:inherit;font-size:13px;background:var(--ink);color:#fff;border:0;border-radius:8px;
  padding:5px 10px;cursor:zoom-in;z-index:2}
.zoom-btn:hover{background:var(--accent)}
pre.rawcsv{background:#0f1b2d;color:#d6e2f0;font-size:10.5px;padding:8px 10px;margin:0;white-space:pre;overflow:hidden;border-radius:6px}
#zoom{position:fixed;inset:0;z-index:50;background:rgba(10,16,28,.82);display:none;align-items:center;justify-content:center;cursor:zoom-out}
#zoom.show{display:flex}
#zoom .zoom-inner{background:#fff;border-radius:12px;padding:22px 26px;max-width:96vw;max-height:92vh;overflow:auto;cursor:default}
#zoom .zoom-inner table.db{font-size:clamp(11px,1.02vw,19px)}#zoom .zoom-inner table.db th,#zoom .zoom-inner table.db td{padding:.5em .7em}
#zoom .zoom-inner pre.rawcsv{font-size:clamp(10px,.95vw,17px);white-space:pre-wrap}
#zoom .zoom-inner .db-cap{font-size:17px}
#zoom .zoom-close{position:fixed;top:14px;right:20px;color:#fff;font-size:15px;opacity:.85}
.cm{display:grid;grid-template-columns:120px 1fr 1fr;gap:6px;margin:6px 0 8px}
.cm-h{font-size:14px;text-align:center;align-self:end;color:var(--muted)}
.cm-r{font-size:14px;color:var(--muted);align-self:center;text-align:right;padding-right:6px}
.cm-c{border-radius:10px;padding:10px 12px;text-align:center;display:flex;flex-direction:column;gap:1px}
.cm-c span{font-weight:800;font-size:14px;letter-spacing:1px}
.cm-c b{font-size:28px;line-height:1.1}
.cm-c small{font-size:12.5px;color:var(--muted);line-height:1.25}
.cm-c.good{background:#e6f4ec;border:1px solid #b9e0c8}.cm-c.good span{color:#2e8b57}
.cm-c.bad{background:#fdf0eb;border:1px solid #f3c9b8}.cm-c.bad span{color:#c4441f}
.def{background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:9px 14px;margin-bottom:9px;font-size:16px}
.def>b{font-size:18px;color:var(--accent)}.def>span{color:var(--muted);font-size:13.5px;margin-left:6px}
.def p{margin:3px 0 0;font-size:15.5px}
.def.warnbox{background:#fdf1ec;border-color:#f3c9b8}.def.warnbox>b{color:var(--accent2)}
.cols.two-notes{margin-top:10px;gap:16px;flex:0 0 auto}.two-notes .callout{margin:0;font-size:15px}
table.explain td small{color:var(--muted)}
.crisp-sub{font-size:18px;margin:0 0 6px}
.cols.crisp{grid-template-columns:1fr 1.05fr;gap:24px;align-items:center}
.crisp-svg{width:100%;max-height:440px}
.crisp-list{display:flex;flex-direction:column;gap:9px}
.crisp-item{display:flex;gap:12px;align-items:flex-start;background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:9px 12px}
.crisp-item .num{width:30px;height:30px;border-radius:50%;background:var(--accent);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;flex:0 0 auto}
.crisp-item b{font-size:17px}.crisp-item p{margin:1px 0 0;font-size:14.5px;color:var(--muted)}
table.explain td{font-size:15px;padding:7px 9px}table.explain td:first-child{white-space:nowrap}table.explain td small{color:var(--muted);font-weight:400}
.formula .v{color:#0f7c83}.formula .c{color:#e4572e}.formula .t{color:#2e8b57}.formula .f{color:#8a5a00}
.terms{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.term{display:flex;gap:10px;align-items:flex-start;background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:15px;line-height:1.35}
.term>span{font-family:Georgia,serif;font-weight:700;font-size:26px;min-width:44px;text-align:center}
.term .v{color:#0f7c83}.term .c{color:#e4572e}.term .t{color:#2e8b57}.term .f{color:#8a5a00}
.example{background:#fffaf0;border:1px solid #f0dcb4;border-radius:10px;padding:10px 16px;font-size:16px;margin-bottom:10px}
.example h3{margin-top:0;font-size:18px}
.calc{font-family:Consolas,Menlo,monospace;font-size:17px;background:#fff;border-radius:6px;padding:8px 12px;margin:6px 0;border:1px solid #f0dcb4}
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
const zoomEl=document.getElementById('zoom');
document.querySelectorAll('.zoom-btn').forEach(b=>b.addEventListener('click',e=>{e.stopPropagation();const c=b.parentElement.cloneNode(true);c.querySelector('.zoom-btn').remove();c.classList.remove('zoomable');const inner=zoomEl.querySelector('.zoom-inner');inner.replaceChildren(c);zoomEl.classList.add('show')}));
zoomEl.addEventListener('click',e=>{if(!e.target.closest('.zoom-inner')||e.target===zoomEl)zoomEl.classList.remove('show')});
function fit(){const s=Math.min(innerWidth/1280,innerHeight/720)*0.97;deck.style.transform='scale('+s+')'}
function show(k){cur=Math.max(0,Math.min(slides.length-1,k));slides.forEach((s,j)=>s.classList.toggle('active',j===cur));
  prog.style.width=(100*(cur+1)/slides.length)+'%';notes.innerHTML='<b>Notas · slide '+(cur+1)+'/'+slides.length+'</b><br>'+(slides[cur].dataset.notes||'—');
  if(location.hash!=='#'+(cur+1))history.replaceState(null,'','#'+(cur+1))}
addEventListener('keydown',e=>{
  if(zoomEl.classList.contains('show')){if(e.key==='Escape'){zoomEl.classList.remove('show');return}zoomEl.classList.remove('show')}
  if(['ArrowRight','PageDown'].includes(e.key)){e.preventDefault();show(cur+1)}
  else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(cur-1)}
  else if(e.key==='Home')show(0);else if(e.key==='End')show(slides.length-1);
  else if(e.key==='n'||e.key==='N')notes.classList.toggle('show');
  else if(e.key==='f'||e.key==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}});
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
            f'<div class="content">{annotate(s["body"])}</div>'
            f'<footer><span>{AUTHOR} · {COURSE} · {UNIVERSITY}</span><span>{k} / {total}</span></footer></section>')
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Captación de clientes — TP Final</title>
<meta name="description" content="Presentación del TP Final de Aprendizaje Artificial (CAECE): modelo pre-contacto para priorizar clientes en campañas de marketing bancario.">
<style>{CSS}{PREP_CSS}{TIP_CSS}</style></head>
<body><div id="stage"><div id="deck">{''.join(parts)}</div></div>
<div id="progress"></div><div id="notes"></div><div id="tip"></div><div id="zoom"><span class="zoom-close">Esc o clic para cerrar</span><div class="zoom-inner"></div></div>
<div id="help">← → navegar · N notas · F pantalla completa</div>
<script>{JS}{TIP_JS}</script></body></html>"""


def build_slides_html() -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(render(), encoding="utf-8")
    return str(OUT_FILE.relative_to(cfg.ROOT))


if __name__ == "__main__":
    print(build_slides_html())
