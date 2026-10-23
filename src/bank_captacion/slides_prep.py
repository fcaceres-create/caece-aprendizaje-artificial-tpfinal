"""Sección "Preparación del dataset" de la presentación.

Cada slide muestra un bloque de filas REALES del dataset antes y después del ajuste. Las filas se eligen
con criterios fijos (no al azar), así que la presentación es reproducible.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd

from . import config as cfg
from .data import find_overlap, load_clean, load_raw
from .export_report import f, i, pct
from .features import FeatureEngineer, make_preprocessor

SECTION = "3 · Preparación"

# Clases de columna en los bloques: "new" = creada, "chg" = transformada, "drop" = eliminada, "key" = referencia.


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<span class="nan">NaN</span>'
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer():
            return i(v) if abs(v) >= 1000 else str(int(v))
        return f(v, 3) if abs(v) < 1000 else f(v, 1)
    if isinstance(v, (int, np.integer)):
        return i(v) if abs(v) >= 1000 else str(int(v))
    return html.escape(str(v))


def block(df: pd.DataFrame, cols: list[str], kinds: dict | None = None, idx_label: str = "fila",
          hit: dict | None = None, caption: str = "") -> str:
    """Tabla tipo "bloque de datos" con columnas marcadas (nueva, transformada, eliminada)."""
    kinds = kinds or {}
    hit = hit or {}
    head = f'<th class="idx">{idx_label}</th>' + "".join(
        f'<th class="{kinds.get(c, "")}">{html.escape(c)}</th>' for c in cols)
    rows = []
    for ix, r in df.iterrows():
        cells = "".join(
            f'<td class="{kinds.get(c, "")}{" hit" if hit.get(c) and hit[c](r[c]) else ""}">{_fmt(r[c])}</td>'
            for c in cols)
        rows.append(f'<tr><td class="idx">{_fmt(ix)}</td>{cells}</tr>')
    cap = f'<div class="db-cap">{caption}</div>' if caption else ""
    return f'{cap}<table class="db"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def legend() -> str:
    return ('<div class="db-legend"><span><i class="k-chg"></i>transformada</span><span><i class="k-new"></i>nueva</span>'
            '<span><i class="k-drop"></i>eliminada</span><span><i class="k-hit"></i>valor a observar</span></div>')


def prep_slides(add, n: dict) -> None:
    train_raw = load_raw(cfg.TRAIN_FILE)
    train, test = load_clean()
    fe = FeatureEngineer().fit(train)
    Z = fe.transform(train)
    both = train.join(Z.drop(columns=[c for c in Z.columns if c in train.columns]))
    conv = lambda mask: (train.loc[mask, cfg.TARGET] == "yes").mean()  # noqa: E731

    # ---- 0. Portada de la sección -------------------------------------------------------------------
    items_rows = [("P1", "Solapamiento train / test", "se quitaron 4.521 filas de train")]
    items_cols = [("P2", "Faltantes codificados (unknown)", "se conservan + bandera"),
                  ("P3", "pdays = -1 y campaña previa", "código → bandera + NaN"),
                  ("P4", "Saldo (balance)", "log con signo + banderas"),
                  ("P5", "Valores extremos de contactos", "winsorización al p99"),
                  ("P6", "Calendario, edad y crédito", "cíclicas, grupos y conteo"),
                  ("P7", "duration (leakage)", "excluida del modelo"),
                  ("P8", "Codificación según el modelo", "one-hot, escala o nativa")]
    li = lambda its: "".join(f'<div class="pi"><span class="pk">{a}</span><div><b>{b}</b><p>{c}</p></div></div>'  # noqa: E731
                             for a, b, c in its)
    add("Preparación del dataset", f"""
      <p class="lead" style="font-size:18px;margin:0 0 4px">De los CSV originales a las variables que ve el modelo: <b>8 ajustes</b>, cada uno motivado por un
         hallazgo del análisis exploratorio y justificado con evidencia.</p>
      <div class="cols prep-map">
        <div><h3>Sobre las filas (una sola vez)</h3>{li(items_rows)}
          <div class="principle"><b>Principio rector:</b> todo lo que se <i>aprende</i> de los datos (percentiles, medianas,
            escalas, categorías) vive dentro de un <code>Pipeline</code> y se ajusta <b>solo con train</b> (en cada fold de
            validación). Así no se filtra información del test ni de la validación.</div>
          <h3 style="margin-top:12px">Lo que deliberadamente NO se hizo</h3>
          <ul class="tight">
            <li>No se imputaron los <code>unknown</code> → tienen significado propio.</li>
            <li>No se eliminaron outliers → son clientes reales.</li>
            <li>No se rebalancearon las clases → descalibra las probabilidades.</li>
            <li>No se usó <code>duration</code> → se conoce después de llamar.</li>
          </ul>
        </div>
        <div><h3>Sobre las columnas (dentro del pipeline)</h3>{li(items_cols)}</div>
      </div>""",
        "Portada de la sección. Mensaje: cada ajuste responde a algo que se vio en los datos. Dos niveles: filas (una vez, "
        "antes de todo) y columnas (dentro del pipeline, aprendido solo con train). Recorrer rápido el mapa y anticipar lo "
        "que NO se hizo, que también es una decisión.", SECTION)

    # ---- P1. Solapamiento ---------------------------------------------------------------------------
    cols_show = ["age", "job", "marital", "balance", "contact", "day", "month", "duration", "y"]
    raw_i = train_raw.reset_index().rename(columns={"index": "fila_train"})
    te_i = test.reset_index().rename(columns={"index": "fila_test"})
    m = te_i.merge(raw_i, on=list(test.columns), how="inner").head(4)
    tshow = m.set_index("fila_test")[cols_show + ["fila_train"]]
    add("P1 · Solapamiento: el test estaba dentro del train", f"""
      <div class="cols wide-left">
        <div>
          {block(tshow, cols_show + ["fila_train"], {"fila_train": "chg"}, idx_label="fila test",
                 hit={"fila_train": lambda v: True},
                 caption="Primeras filas de <b>banca_test.csv</b> y la fila <b>idéntica</b> (las 17 columnas) que aparece en banca_train.csv")}
          <p class="small-text">Se compararon las 17 columnas de cada fila de test contra train (<i>merge</i> exacto):
             <b>las {n['overlap']} filas de test aparecen en train</b>. bank.csv es una muestra del 10 % de bank-full.csv
             (UCI); los hashes SHA-256 de los archivos coinciden con los originales de UCI.</p>
        </div>
        <div>
          <div class="flow"><div class="box">train original<br><b>{n['train_raw']}</b></div><div class="arrow">−{n['overlap']}</div>
            <div class="box ok">train limpio<br><b>{n['train_clean']}</b></div></div>
          <ul class="tight">
            <li><b>Problema:</b> evaluar con datos ya vistos mide memoria, no generalización.</li>
            <li><b>Decisión:</b> quitar esas filas de train (no del test, que es el hold-out de la consigna).</li>
            <li><b>Costo:</b> −10 % de datos de entrenamiento; la tasa de conversión no cambia ({n['rate_train_raw']} → {n['rate_train']}).</li>
            <li><b>Control:</b> un <code>assert</code> verifica solapamiento = 0 en cada ejecución.</li>
            <li><b>Evidencia de partición aleatoria:</b> KS p = {n['ks_p']}; χ² sin diferencias.</li>
          </ul>
        </div>
      </div>""",
        "Mostrar la tabla: la fila 0 del test es la fila X del train, idéntica en todo. Por eso se quitaron. Si preguntan por "
        "qué no se descartó el test: es el hold-out que da la consigna. Si preguntan cómo se supo que es aleatorio: posiciones "
        "uniformes en el archivo (KS) y distribuciones iguales (χ²).", SECTION)

    # ---- P2. unknown -------------------------------------------------------------------------------
    pick = []
    for cond in [(train.contact == "unknown") & (train.poutcome == "unknown"), train.education == "unknown",
                 train.job == "unknown", (train.contact == "cellular") & (train.poutcome == "success")]:
        pick.append(train.index[cond][0])
    s2 = both.loc[pick]
    unk = pd.read_csv(cfg.TABLES_DIR / "eda_unknown_analysis.csv")
    unk_rows = "".join(
        f"<tr><td><code>{v}</code></td><td>{f(100 * unk[(unk.variable == v) & unk.is_unknown].conversion_rate.iloc[0], 1)} %</td>"
        f"<td>{f(100 * unk[(unk.variable == v) & ~unk.is_unknown].conversion_rate.iloc[0], 1)} %</td>"
        f"<td>{n['pct_unk_' + v]}</td></tr>" for v in ["contact", "poutcome", "education", "job"])
    add("P2 · Faltantes codificados: “unknown” se conserva", f"""
      {block(s2, ["job", "education", "contact", "poutcome", "contact_known"], {"contact_known": "new"},
             hit={"job": lambda v: v == "unknown", "education": lambda v: v == "unknown",
                  "contact": lambda v: v == "unknown", "poutcome": lambda v: v == "unknown"},
             caption="Filas reales de train: los <b>unknown</b> quedan como categoría y se agrega la bandera <b>contact_known</b>")}
      <div class="cols" style="margin-top:10px">
        <div>
          <table class="mini"><thead><tr><th>Variable</th><th>Conversión si unknown</th><th>Si conocido</th><th>% unknown</th></tr></thead>
            <tbody>{unk_rows}</tbody></table>
        </div>
        <div>
          <ul class="tight">
            <li>No hay nulos (<code>NaN</code>): los faltantes vienen escritos como <code>unknown</code>.</li>
            <li><b>No son al azar:</b> convierten distinto que el resto. El caso del canal: casi todos los
                <code>unknown</code> son de mayo–junio 2008, cuando el banco <b>no anotaba el canal</b> (ver slide del EDA).</li>
            <li><b>Decisión:</b> mantenerlos como categoría propia. Imputar a la moda (p. ej. <code>secondary</code>)
                borraría esa señal y mezclaría grupos que se comportan distinto.</li>
          </ul>
        </div>
      </div>""",
        "Señalar las celdas resaltadas: unknown en distintas columnas. En la tabla de abajo, contact unknown convierte 4 % "
        "contra 15 %: es información, no ruido. Por eso no se imputa.", SECTION)

    # ---- P3. pdays ----------------------------------------------------------------------------------
    never = train.index[train.pdays == -1][:2].tolist()
    fail = train.index[(train.pdays != -1) & (train.poutcome == "failure")][:1].tolist()
    succ = train.index[(train.pdays != -1) & (train.poutcome == "success")][:2].tolist()
    s3 = both.loc[never + fail + succ]
    add("P3 · pdays = -1: un código, no una cantidad de días", f"""
      <div class="cols">
        <div>
          {block(s3, ["pdays", "previous", "poutcome"], {"pdays": "chg"}, hit={"pdays": lambda v: v == -1},
                 caption="Antes (crudo)")}
          <div class="arrow-down">↓ FeatureEngineer</div>
          {block(s3, ["pdays_clean", "previously_contacted", "previous_w", "prev_campaign_success"],
                 {"pdays_clean": "chg", "previously_contacted": "new", "previous_w": "chg", "prev_campaign_success": "new"},
                 hit={"pdays_clean": lambda v: pd.isna(v)}, caption="Después")}
        </div>
        <div>
          <ul>
            <li><code>pdays = -1</code> significa <b>“nunca contactado”</b>. Tratado como número, un modelo lineal lo ubicaría
                al lado de “0 días” (contactado ayer).</li>
            <li>Las {n['n_pdays_neg']} filas con -1 ({n['pct_pdays_neg']}) son exactamente las de <code>previous = 0</code> y
                <code>poutcome = unknown</code>: la misma información en tres columnas.</li>
            <li><b>Ajuste:</b> <code>pdays_clean</code> (NaN cuando -1) + bandera <code>previously_contacted</code>;
                <code>prev_campaign_success</code> marca el éxito previo.</li>
          </ul>
          <div class="stats-row compact">
            <div class="stat"><div class="v">{n['conv_never']}</div><div class="l">conversión nunca contactados</div></div>
            <div class="stat"><div class="v">{n['conv_before']}</div><div class="l">contactados antes</div></div>
            <div class="stat accent"><div class="v">{n['conv_success']}</div><div class="l">con éxito previo</div></div>
          </div>
        </div>
      </div>""",
        "Las dos primeras filas: -1 pasa a NaN y la bandera queda en 0. Las otras: días reales y bandera 1. El NaN después "
        "se imputa según el modelo (LightGBM lo maneja nativo). El éxito previo es la señal más fuerte del dataset.", SECTION)

    # ---- P4. balance ---------------------------------------------------------------------------------
    order = train.balance
    picks = [order.idxmin(), train.index[(order < 0) & (order > -300)][0], train.index[order == 0][0],
             train.index[(order > 1400) & (order < 1600)][0], order.idxmax()]
    s4 = both.loc[picks]
    add("P4 · Saldo: escala logarítmica con signo, sin eliminar outliers", f"""
      <div class="cols">
        <div>
          {block(s4, ["balance", "balance_log", "balance_negative", "balance_zero"],
                 {"balance": "chg", "balance_log": "new", "balance_negative": "new", "balance_zero": "new"},
                 hit={"balance_negative": lambda v: v == 1, "balance_zero": lambda v: v == 1},
                 caption="Filas reales: el mínimo, un negativo chico, un cero, un valor típico y el máximo")}
          <div class="formula small-f">balance_log = signo(x) · log(1 + |x|)</div>
        </div>
        <div>
          <ul>
            <li>Rango de <b>{n['bal_min']}</b> a <b>{n['bal_max']}</b>, asimetría {n['bal_skew']}: unos pocos saldos enormes
                dominarían a un modelo lineal o a KNN.</li>
            <li>{n['bal_iqr']} son outliers por la regla IQR, pero <b>son clientes reales</b>: borrarlos dejaría al modelo sin
                saber puntuarlos en producción.</li>
            <li>El log comprime la escala (102.127 → 11,5) y conserva el orden. El signo mantiene los descubiertos
                ({n['bal_neg']} de los clientes) y la bandera marca los saldos cero ({n['bal_zero']}).</li>
          </ul>
          <p class="callout">Transformar en lugar de eliminar: se conserva el 100 % de los clientes.</p>
        </div>
      </div>""",
        "Leer la tabla de arriba abajo: el saldo máximo pasa de 102 mil a 11,5 en escala log. Los negativos quedan negativos "
        "y con bandera. Ningún cliente se elimina.", SECTION)

    # ---- P5. winsorización ---------------------------------------------------------------------------
    c = train.campaign
    picks = [c.idxmax(), train.index[(c >= 25) & (c <= 35)][0], train.index[c == 17][0], train.index[c == 3][0],
             train.previous.idxmax()]
    s5 = both.loc[picks]
    add("P5 · Valores extremos: winsorización al percentil 99", f"""
      <div class="cols">
        <div>
          {block(s5, ["campaign", "campaign_w", "previous", "previous_w"],
                 {"campaign": "chg", "campaign_w": "new", "previous": "chg", "previous_w": "new"},
                 hit={"campaign_w": lambda v: v == fe.campaign_cap_, "previous_w": lambda v: v == fe.previous_cap_},
                 caption=f"Filas reales: los valores por encima del p99 se recortan al tope (campaign {f(fe.campaign_cap_, 0)}, previous {f(fe.previous_cap_, 0)})")}
        </div>
        <div>
          <ul>
            <li><code>campaign</code> (contactos en esta campaña) llega a <b>{n['camp_max']}</b>, pero el 99 % está en
                <b>{n['camp_p99']} o menos</b>. Igual con <code>previous</code> (máx. 275, p99 = 9).</li>
            <li><b>Winsorizar</b> = recortar al percentil 99: el valor sigue siendo “muy alto”, pero ya no distorsiona.</li>
            <li>El tope se <b>aprende en <code>fit</code></b> con los datos de entrenamiento de cada fold: el fold de
                validación nunca influye en el p99 (sin fuga).</li>
            <li>La señal se conserva: {n['conv_camp1']} de conversión con 1 contacto vs {n['conv_camp5']} con 5 o más.</li>
          </ul>
        </div>
      </div>""",
        "Mostrar que 63 contactos pasa a 17 y 275 contactos previos pasa a 9. Remarcar que el tope se calcula dentro del "
        "pipeline, solo con train: detalle técnico que muestra cuidado contra el leakage.", SECTION)

    # ---- P6. calendario, edad, crédito -------------------------------------------------------------
    picks = [train.index[(train.day == 1)][0], train.index[(train.day == 15) & (train.age < 25)][0],
             train.index[(train.day == 31)][0], train.index[(train.age >= 65)][0]]
    s6 = both.loc[picks]
    add("P6 · Calendario, edad y crédito: variables derivadas", f"""
      {block(s6, ["day", "day_sin", "day_cos", "month", "season", "age", "age_group", "housing", "loan", "default", "n_credit_products"],
             {"day_sin": "new", "day_cos": "new", "season": "new", "age_group": "new", "n_credit_products": "new"},
             caption="Filas reales de train con las variables derivadas (en verde)")}
      <div class="grid3">
        <div class="def"><b>Día cíclico</b> <span>sen / cos (2π · día / 31)</span>
          <p>El día 31 queda “al lado” del día 1. Como número, 1 y 31 estarían en extremos opuestos.</p></div>
        <div class="def"><b>Edad en grupos</b> <span>&lt;25 … 65+</span>
          <p>El efecto no es lineal: convierten más los muy jóvenes (estudiantes) y los mayores (jubilados). Se conserva
             además la edad numérica.</p></div>
        <div class="def"><b>Crédito y estación</b> <span>conteo · agrupación</span>
          <p><code>n_credit_products</code> = hipoteca + préstamo + impago (tener préstamos reduce la conversión a la mitad).
             <code>season</code> agrupa meses de poco volumen.</p></div>
      </div>""",
        "Ejemplos concretos: día 1 y día 31 tienen seno/coseno parecidos. La edad se agrupa porque el efecto es en U. El "
        "conteo de préstamos resume tres columnas que apuntan a lo mismo.", SECTION)

    # ---- P7. duration ---------------------------------------------------------------------------------
    s7 = train.head(4)
    add("P7 · duration: la variable que se excluye", f"""
      <div class="cols">
        <div>
          {block(s7, ["age", "job", "contact", "month", "campaign", "duration", "y"], {"duration": "drop"},
                 caption="La columna <b>duration</b> existe en los datos, pero el modelo entregable no la usa")}
          <div class="stats-row compact" style="margin-top:12px">
            <div class="stat warn"><div class="v">{n['dur_auc']}</div><div class="l">ROC-AUC de duration sola</div></div>
            <div class="stat"><div class="v">{n['test_with_duration_roc_auc']}</div><div class="l">modelo con duration (test)</div></div>
            <div class="stat accent"><div class="v">{n['test_final_roc_auc']}</div><div class="l">modelo entregable</div></div>
          </div>
        </div>
        <div>
          <ul>
            <li><code>duration</code> = duración de la llamada en segundos. <b>Se conoce recién al cortar.</b></li>
            <li>El modelo decide <b>a quién llamar</b>: en ese momento la duración no existe → usarla es <b>leakage</b>.</li>
            <li>Además, la causalidad está invertida: la llamada es larga <i>porque</i> el cliente está interesado.</li>
            <li>Medianas: {n['dur_med_no']} s (no convierte) vs {n['dur_med_yes']} s (convierte).</li>
            <li><b>Controles:</b> <code>FeatureEngineer</code> la descarta aunque venga en los datos, hay tests automáticos y la
                API la rechaza (error 422).</li>
          </ul>
        </div>
      </div>""",
        "La columna tachada. Con ella el modelo parece mucho mejor (0,93 vs 0,78), pero es una mejora ilusoria: en producción "
        "no tendría el dato. Variante con duration solo para cuantificar el efecto.", SECTION)

    # ---- P8. Codificación según el modelo -----------------------------------------------------------
    prep = make_preprocessor("linear").fit(Z)
    rows8 = Z.loc[[train.index[(train.job == "management") & (train.age == 35)][0],
                    train.index[(train.job == "retired") & (train.age >= 65)][0],
                    train.index[(train.job == "student") & (train.age <= 22)][0]]]
    lin = pd.DataFrame(prep.transform(rows8), index=rows8.index, columns=prep.get_feature_names_out())
    oh_cols = ["job_admin.", "job_management", "job_retired", "job_student"]
    lin_show = rows8[["job", "age"]].join(lin[["age"] + oh_cols].rename(columns={"age": "age (escalada)"}))
    add("P8 · Codificación según la familia de modelo", f"""
      <div class="cols">
        <div>
          {block(lin_show, ["job", "age", "age (escalada)"] + oh_cols,
                 {"age (escalada)": "chg", **{c: "new" for c in oh_cols}},
                 hit={c: (lambda v: v == 1) for c in oh_cols},
                 caption="Para modelos lineales, KNN, MLP y SVM: <b>escalado</b> + <b>one-hot</b> (una columna 0/1 por categoría; se muestran 4 de 12)")}
          <p class="small-text">Edad escalada = (edad − media) / desvío, con media y desvío aprendidos en train.</p>
        </div>
        <div>
          <table class="mini"><thead><tr><th>Familia</th><th>Numéricas</th><th>Categóricas</th><th>Columnas</th></tr></thead><tbody>
            <tr><td>Lineal, KNN, MLP, SVM</td><td>mediana + escala</td><td>one-hot</td><td>67</td></tr>
            <tr><td>Árboles, RF, XGBoost</td><td>sin escalar</td><td>one-hot</td><td>67</td></tr>
            <tr class="hl"><td><b>LightGBM (final)</b></td><td>sin escalar, NaN nativo</td><td><b>categoría nativa</b></td><td>24</td></tr>
            <tr><td>Naive Bayes</td><td>quintiles</td><td>ordinal</td><td>24</td></tr>
          </tbody></table>
          <ul class="tight">
            <li>Los modelos de distancia y gradiente necesitan escalas comparables; los árboles no (cortan por umbrales).</li>
            <li>LightGBM trabaja con categorías nativas: 24 columnas en vez de 67 y algo mejor en validación.</li>
            <li><code>handle_unknown='ignore'</code>: una categoría nueva en producción no rompe el modelo.</li>
          </ul>
        </div>
      </div>""",
        "Mismo dato, distinto formato según el algoritmo. Ejemplo: 'management' se vuelve un 1 en job_management y 0 en el "
        "resto. Para LightGBM no hace falta: entiende categorías directamente.", SECTION)

    # ---- Resultado: de 16 columnas crudas a 24 variables -------------------------------------------
    r0 = train.index[(train.pdays != -1) & (train.balance > 0)][0]
    raw_row = train.drop(columns=[cfg.TARGET]).loc[r0]
    eng_row = Z.loc[r0]
    def vlist(series, kinds):
        return "".join(f'<tr class="{kinds.get(k, "")}"><td>{html.escape(k)}</td><td>{_fmt(v)}</td></tr>' for k, v in series.items())
    raw_k = {"duration": "drop", "balance": "chg", "pdays": "chg", "campaign": "chg", "previous": "chg", "day": "chg"}
    new_k = {c: "new" for c in eng_row.index if c not in raw_row.index}
    add("Resultado: un cliente, antes y después de la preparación", f"""
      <div class="cols three">
        <div><div class="db-cap">Crudo (16 columnas, fila {r0} de train)</div><table class="vt">{vlist(raw_row, raw_k)}</table></div>
        <div class="mid-arrow">→<small>Pipeline<br>ajustado<br>solo con<br>train</small></div>
        <div><div class="db-cap">Lo que ve el modelo (24 variables: 13 numéricas + 11 categóricas)</div>
          <div class="vt-split"><table class="vt">{vlist(eng_row.iloc[:13], new_k)}</table><table class="vt">{vlist(eng_row.iloc[13:], new_k)}</table></div></div>
      </div>
      {legend()}""",
        "Cierre de la sección con un cliente real completo: a la izquierda las 16 columnas crudas (duration tachada), a la "
        "derecha las 24 variables que recibe el modelo. Todo esto está guardado dentro del modelo: la API recibe datos crudos "
        "y aplica exactamente las mismas transformaciones.", SECTION)


PREP_CSS = """
.prep-map{grid-template-columns:1fr 1fr;gap:28px}
.pi{display:flex;gap:10px;align-items:center;background:var(--soft);border:1px solid var(--line);border-radius:9px;padding:3px 11px;margin-bottom:4px}
.pi .pk{background:var(--accent);color:#fff;font-weight:800;font-size:13px;border-radius:6px;padding:3px 7px;flex:0 0 auto}
.pi b{font-size:15px}.pi p{margin:0;font-size:12.5px;color:var(--muted)}
.prep-map h3{margin:2px 0 6px}
.principle{background:var(--hl);border:1px solid #bfe3e3;border-radius:10px;padding:9px 12px;font-size:14.5px;margin-top:8px}
table.db{align-self:flex-start;border-collapse:separate;border-spacing:0;width:auto;font-family:Consolas,"SF Mono",Menlo,monospace;font-size:13.5px;
  border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff}
table.db th{background:#2b3a52;color:#fff;font-weight:600;padding:6px 10px;text-align:right;white-space:nowrap}
table.db td{padding:5px 10px;border-bottom:1px solid #eef1f5;text-align:right;white-space:nowrap}
table.db tr:last-child td{border-bottom:0}
table.db .idx{background:#f1f3f7;color:var(--muted);text-align:center}
table.db th.idx{background:#1a2638;color:#b8c4d6}
table.db th.new{background:#1f7a4d}table.db td.new{background:#eaf6ef;color:#14532d;font-weight:600}
table.db th.chg{background:#0f7c83}table.db td.chg{background:#e6f4f4}
table.db th.drop{background:#b23a1e;text-decoration:line-through}table.db td.drop{background:#fbe9e3;color:#b23a1e;text-decoration:line-through}
table.db td.hit{outline:2px solid #e4572e;outline-offset:-2px;font-weight:800}
table.db .nan{color:#b23a1e;font-style:italic}
.db-cap{font-size:13.5px;color:var(--ink);margin:2px 0 6px}
.arrow-down{font-size:14px;color:var(--accent);font-weight:700;margin:6px 0}
.db-legend{display:flex;gap:18px;font-size:13px;color:var(--muted);margin-top:10px}
.db-legend i{display:inline-block;width:14px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.k-chg{background:#0f7c83}.k-new{background:#1f7a4d}.k-drop{background:#b23a1e}.k-hit{border:2px solid #e4572e}
table.mini{font-size:14px}table.mini td,table.mini th{padding:5px 8px}
.formula.small-f{font-size:19px;padding:8px;margin-top:10px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px}.grid3 .def{margin:0}
.cols.three{grid-template-columns:.8fr 60px 1.6fr;gap:12px;align-items:start}
.vt-split{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.mid-arrow{font-size:40px;color:var(--accent);text-align:center;align-self:center;font-weight:700}
.mid-arrow small{display:block;font-size:11.5px;color:var(--muted);font-weight:400;line-height:1.3}
table.vt{border-collapse:collapse;font-family:Consolas,Menlo,monospace;font-size:12.5px;width:100%}
table.vt td{padding:2px 8px;border-bottom:1px solid #eef1f5}
table.vt td:first-child{color:var(--ink-2,#44546b);width:52%}
table.vt td:last-child{text-align:right}
table.vt tr.new td{background:#eaf6ef;color:#14532d;font-weight:600}
table.vt tr.chg td{background:#e6f4f4}
table.vt tr.drop td{background:#fbe9e3;color:#b23a1e;text-decoration:line-through}
table.vt.two{column-count:1}
"""
