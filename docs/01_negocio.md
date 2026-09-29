# Fase 1 — Comprensión del negocio

## 1.1 Contexto

Un banco portugués realiza campañas de marketing telefónico para captar clientes para un
producto financiero (depósito a plazo, según la fuente original del dataset: Moro, Cortez
y Rita, 2014). Cada llamada tiene un costo: tiempo del operador, costo de la
comunicación y desgaste de la relación con el cliente. Solo una minoría de los contactados
termina convirtiéndose: en el conjunto de entrenamiento la tasa de conversión es del
**11,7 %** (ver `reports/tables/data_summary.csv`).

## 1.2 Decisión que apoya el modelo

> Dado un listado de clientes potenciales y un presupuesto limitado de contactos,
> **ordenar a los clientes según su probabilidad de conversión y contactar a los primeros k**.

La salida principal del modelo es un **puntaje (score)** que permite priorizar. La
clasificación binaria (llamar/no llamar) se obtiene aplicando un umbral operativo sobre ese
puntaje, y el umbral se elige con criterio económico (sección 1.5).

Consecuencia: el modelo se evalúa como una **herramienta de ranking**. Lo importante es que
los clientes que efectivamente se convierten queden en los primeros lugares del listado.

## 1.3 Restricción fundamental: solo información disponible antes de llamar

El modelo se usa **antes** de contactar al cliente. Toda variable que solo se conoce
después del contacto queda excluida del modelo entregable. El caso central es `duration`
(duración de la llamada): se conoce recién cuando la llamada terminó y, además, una llamada
larga es en gran medida consecuencia de que el cliente esté interesado. Usarla sería una
**fuga de información (data leakage)**. Se entrena una variante con `duration` solo para
cuantificar ese efecto (Fase 5).

Por lo tanto, el modelo entregable se denomina **modelo pre-contacto**.

## 1.4 Métricas de evaluación

| Tipo | Métrica | Por qué |
|------|---------|---------|
| Primaria | **PR-AUC (average precision)** | Resume la calidad del ranking sobre la clase positiva minoritaria. Su piso (modelo aleatorio) es la prevalencia (~0,117), lo que la hace informativa con desbalance. |
| Primaria | **Lift y gain en el top 10 %, 20 % y 30 %** | Traducen el ranking a la decisión real: "si llamo al 20 % mejor rankeado, ¿qué porcentaje de las conversiones capturo y cuántas veces mejor que al azar?". |
| Secundaria | ROC-AUC | Estándar de la literatura; permite contrastar con valores publicados (razonabilidad). |
| Secundaria | Precision, recall y F1 en el umbral operativo | Describen la decisión binaria finalmente tomada. |
| Secundaria | Brier score y curva de calibración | Indican si las probabilidades pueden leerse como probabilidades reales (necesario para estimar conversiones y beneficio esperados). |

### ¿Por qué no accuracy?

Con un 88,3 % de negativos, un modelo que responde "no se convierte" para todos los
clientes obtiene **88,3 % de accuracy** y no sirve para nada: no identifica a ningún
cliente. Accuracy premia acertar la clase mayoritaria, mientras que el valor de negocio está
en encontrar a la minoría que se convierte. Además, depende de un umbral fijo (0,5) que
no tiene relación con la economía de la campaña.

## 1.5 Marco económico

Se parametriza en `src/bank_captacion/config.py`:

- **C** = costo por contacto = **1** unidad monetaria.
- **V** = valor esperado de una conversión = **20** unidades monetarias.

Beneficio esperado de contactar a un conjunto de clientes:

$$\text{Beneficio} = V \cdot \text{TP} - C \cdot (\text{TP} + \text{FP})$$

donde TP son las conversiones logradas entre los contactados y TP + FP es la cantidad de
contactos realizados. No contactar no tiene costo (tampoco genera ingresos).

Implicancia: con una probabilidad de conversión *p*, contactar conviene si
*p · V > C*, es decir *p > C / V = 0,05*. El umbral económico teórico es 5 %. Si el
modelo está bien calibrado, el umbral óptimo empírico (elegido con predicciones
out-of-fold) debería estar cerca de ese valor.

### Supuestos y sensibilidad

Los valores C = 1 y V = 20 son **supuestos ilustrativos**: el banco no los informa. Lo
relevante para la decisión es el **ratio V/C**, no los valores absolutos. Por eso se
analiza la sensibilidad para ratios **5, 10, 20 y 50**. Con un ratio bajo (conversiones
poco valiosas o llamadas caras) conviene contactar a pocos clientes; con un ratio alto
conviene contactar a casi todos.

**Cómo los reemplazaría el banco:**

- *C*: costo del minuto de operador × duración promedio de una llamada + costo de
  telefonía + un costo de oportunidad o desgaste por contacto no deseado.
- *V*: margen neto esperado del producto (por ejemplo, el spread del depósito a plazo
  durante su vida promedio) más el valor de vida del cliente captado.

Basta con cambiar `COST_PER_CONTACT` y `VALUE_PER_CONVERSION` en `config.py` y volver a
ejecutar el pipeline: el umbral operativo y el análisis económico se recalculan solos.

## 1.6 Criterio de éxito

1. El modelo pre-contacto supera claramente al azar y a un baseline interpretable
   (regresión logística) en PR-AUC y lift@20 %, con validación cruzada y en el hold-out.
2. Las métricas son razonables frente a la literatura sobre este dataset sin `duration`
   (ROC-AUC en torno a 0,75–0,80).
3. El análisis económico muestra un beneficio esperado superior al de las dos estrategias
   triviales: contactar a todos y no contactar a nadie.
