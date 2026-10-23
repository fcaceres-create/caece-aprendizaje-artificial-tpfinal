"""Glosario de la presentación: términos técnicos con su explicación en lenguaje simple.

``annotate`` marca la primera aparición de cada término en una slide con un "?" que muestra la
definición al pasar el mouse. Solo modifica texto visible (nunca dentro de etiquetas, código o SVG).
"""

from __future__ import annotations

import html
import re

# (patrón regex, definición). El orden importa: primero los más largos/específicos.
GLOSSARY: list[tuple[str, str]] = [
    (r"tasa de conversión", "Porcentaje de clientes contactados que dijo que sí (aceptó el producto). "
                            "Ejemplo: 4.768 de 40.690 = 11,7 %."),
    (r"tasa base", "Porcentaje de clientes que convierte en promedio (11,7 %). Es la referencia: un grupo por encima "
                   "acepta más que el promedio; por debajo, menos."),
    (r"Spearman", "Correlación de Spearman: mide si dos variables suben y bajan juntas (según su orden). Va de −1 "
                  "(cuando una sube, la otra baja) a +1 (suben juntas). 0 = sin relación."),
    (r"PR-AUC", "Área bajo la curva Precision-Recall: resume qué tan arriba del ranking quedan los clientes que "
                "convierten. Al azar vale la tasa base (~0,12); 1 = perfecto."),
    (r"ROC-AUC", "Probabilidad de que el modelo ponga a un cliente que convierte por encima de uno que no. "
                 "0,5 = azar; 1 = perfecto."),
    (r"[Ll]ift", "Cuántas veces más conversiones hay en un grupo que eligiendo clientes al azar. "
                 "Lift 4 = cuatro veces mejor que el azar; lift 1 = igual que el azar."),
    (r"[Gg]ain", "Ganancia acumulada: qué % del total de conversiones se captura llamando solo al k % de clientes mejor "
                 "rankeados."),
    (r"Brier", "Error de las probabilidades: promedio de (probabilidad dicha − resultado real)². 0 = perfecto; "
               "más bajo es mejor."),
    (r"SHAP", "Método que reparte cada predicción entre las variables: cuánto sumó o restó cada una a la "
              "probabilidad de ese cliente, respecto de un cliente promedio."),
    (r"log-odds", "Escala en la que trabaja el modelo internamente (logaritmo de p / (1 − p)). Valores positivos "
                  "suben la probabilidad; negativos la bajan."),
    (r"[Ww]insoriz\w*", "Winsorizar: recortar los valores extremos a un tope (acá el percentil 99) en lugar de "
                        "borrarlos. El cliente se conserva, pero su valor ya no distorsiona."),
    (r"p99|percentil 99", "Percentil 99: el valor por debajo del cual está el 99 % de los datos. Solo el 1 % lo supera."),
    (r"IQR", "Rango intercuartil: el ancho del 50 % central de los datos. Se considera outlier a lo que queda muy "
             "lejos de ese rango."),
    (r"[Oo]utliers?", "Valores extremos, muy alejados del resto (por ejemplo, un saldo de 102.127)."),
    (r"one-hot", "Convertir una categoría en varias columnas 0/1, una por cada valor posible "
                 "(job_management = 1, el resto = 0)."),
    (r"[Ll]eakage", "Fuga de información: usar un dato que no estaría disponible en el momento de decidir. "
                    "Produce métricas engañosamente buenas."),
    (r"[Pp]ipeline", "Cadena de pasos (preparación de datos + modelo) que se entrena como una sola pieza, solo con "
                     "datos de entrenamiento, y se aplica igual a los datos nuevos."),
    (r"[Hh]old-out", "Datos separados que el modelo nunca vio, usados una única vez para la evaluación final "
                     "(acá, banca_test.csv)."),
    (r"validación cruzada", "Dividir el train en 5 partes: entrenar con 4 y evaluar con la restante, rotando 5 veces. "
                            "Da una estimación más estable que una sola prueba."),
    (r"folds?", "Cada una de las 5 partes en que se divide el train para la validación cruzada."),
    (r"\bOOF\b|out-of-fold", "Predicciones out-of-fold: cada cliente es evaluado por un modelo que no lo vio al "
                             "entrenar. Permite elegir el umbral sin usar el test."),
    (r"\bKS\b", "Test de Kolmogorov-Smirnov: compara dos distribuciones. Con p > 0,05 no hay diferencia "
                "significativa."),
    (r"χ²", "Test chi-cuadrado: compara si las proporciones de las categorías son iguales en dos grupos. "
            "Con p > 0,05 no difieren."),
    (r"Optuna", "Librería que prueba combinaciones de hiperparámetros de forma inteligente (aprende de los intentos "
                "anteriores) para encontrar la mejor."),
    (r"SMOTE", "Técnica que crea clientes sintéticos de la clase minoritaria (los que convierten) para balancear las "
               "clases."),
    (r"calibra\w*", "Probabilidades calibradas: significan lo que dicen. De los clientes con 30 % de probabilidad, "
                    "convierte efectivamente ~30 %."),
    (r"\bEDA\b", "Análisis exploratorio de datos: explorar, contar y graficar los datos antes de modelar."),
    (r"LightGBM", "Algoritmo de gradient boosting: combina muchos árboles de decisión chicos, donde cada uno corrige "
                  "los errores de los anteriores."),
    (r"[Uu]mbral", "Probabilidad mínima a partir de la cual se recomienda llamar (acá 0,05 = 5 %)."),
    (r"[Bb]anderas?", "Bandera (flag): columna que solo vale 1 (sí) o 0 (no), como una casilla de verificación. "
                      "Ej.: previously_contacted = 1 si el cliente fue contactado antes."),
    (r"NaN", "Not a Number: celda vacía, \"no aplica\". Ej.: pdays_clean queda vacío cuando el cliente nunca fue "
             "contactado (en lugar del código -1)."),
    (r"[Hh]iperparámetros", "Configuraciones del algoritmo que se fijan antes de entrenar (cantidad de árboles, "
                            "profundidad, tasa de aprendizaje…)."),
]

_SKIP_TAGS = {"code", "pre", "svg", "button", "script", "style", "th", "title", "span-gl"}
_COMBINED = re.compile("|".join(rf"(?P<t{k}>(?<![\w-])(?:{pat})(?![\w-]))" for k, (pat, _) in enumerate(GLOSSARY)))


def annotate(body: str) -> str:
    """Marca la primera aparición de cada término del glosario en el texto visible de una slide."""
    parts = re.split(r"(<[^>]+>)", body)
    stack: list[str] = []
    used: set[str] = set()

    def repl(m: re.Match) -> str:
        key = m.lastgroup
        if key in used:
            return m.group(0)
        used.add(key)
        tip = html.escape(GLOSSARY[int(key[1:])][1], quote=True)
        return f'<span class="gl" data-tip="{tip}">{m.group(0)}<sup class="q">?</sup></span>'

    out = []
    for part in parts:
        if part.startswith("<"):
            m = re.match(r"<\s*(/)?\s*([a-zA-Z0-9]+)", part)
            if m:
                tag = m.group(2).lower()
                if m.group(1):
                    if tag == "span" and stack and stack[-1] in ("span", "span-gl"):
                        stack.pop()
                    elif tag in stack:
                        while stack and stack.pop() != tag:
                            pass
                elif not part.endswith("/>") and tag not in {"br", "img", "input", "meta", "link", "hr", "path",
                                                             "circle", "rect", "line"}:
                    stack.append("span-gl" if tag == "span" and 'class="gl"' in part else tag)
            out.append(part)
        elif any(t in _SKIP_TAGS for t in stack) or not part.strip():
            out.append(part)
        else:
            out.append(_COMBINED.sub(repl, part))
    return "".join(out)


TIP_CSS = """
.gl{border-bottom:1px dotted var(--accent);cursor:help}
.gl .q{display:inline-block;width:14px;height:14px;line-height:14px;text-align:center;border-radius:50%;background:var(--accent);
  color:#fff;font-size:10px;font-weight:700;margin-left:2px;vertical-align:super;font-style:normal}
#tip{position:fixed;z-index:60;max-width:340px;background:#13233f;color:#fff;font-size:14px;line-height:1.4;padding:10px 13px;
  border-radius:9px;box-shadow:0 6px 24px rgba(0,0,0,.35);pointer-events:none;display:none}
#tip b{color:#9fe0e3;display:block;margin-bottom:3px}
"""

TIP_JS = """
const tipEl=document.getElementById('tip');
document.addEventListener('mouseover',e=>{const t=e.target.closest('.gl');if(!t)return;
  tipEl.innerHTML='';const b=document.createElement('b');b.textContent=t.firstChild.textContent;
  tipEl.append(b,document.createTextNode(t.dataset.tip));tipEl.style.display='block';
  const r=t.getBoundingClientRect(),w=tipEl.offsetWidth,h=tipEl.offsetHeight;
  let x=Math.min(innerWidth-w-8,Math.max(8,r.left+r.width/2-w/2)),y=r.top-h-8;if(y<8)y=r.bottom+8;
  tipEl.style.left=x+'px';tipEl.style.top=y+'px'});
document.addEventListener('mouseout',e=>{if(e.target.closest('.gl'))tipEl.style.display='none'});
"""
