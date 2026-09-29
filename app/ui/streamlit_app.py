"""Interfaz Streamlit que consume la API del modelo.

Ejecutar (con la API levantada):  python -m streamlit run app/ui/streamlit_app.py
La URL de la API se configura con la variable de entorno API_URL (por defecto http://localhost:8000).
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

LEVELS = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["married", "single", "divorced"],
    "education": ["secondary", "tertiary", "primary", "unknown"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["unknown", "failure", "other", "success"],
}
MONTHS_ES = dict(zip(LEVELS["month"], ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
                                       "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]))

st.set_page_config(page_title="Captación de clientes", page_icon="📞", layout="wide")


@st.cache_data(ttl=60)
def get_model_info():
    r = requests.get(f"{API_URL}/model/info", timeout=10)
    r.raise_for_status()
    return r.json()


try:
    info = get_model_info()
except requests.RequestException as exc:
    st.error(f"No se pudo conectar con la API en {API_URL}. ¿Está levantada? Detalle: {exc}")
    st.stop()

st.title("📞 Priorización de clientes para campañas de marketing")
st.caption(f"Modelo: **{info['model_label']}** (pre-contacto, sin `duration`) · "
           f"PR-AUC CV {info['cv_metrics_mean']['pr_auc']:.3f} · ROC-AUC CV {info['cv_metrics_mean']['roc_auc']:.3f} · "
           f"umbral operativo {info['threshold']:.3f} · API: {API_URL}")

with st.sidebar:
    st.header("Parámetros económicos")
    cost = st.number_input("Costo por contacto (C)", min_value=0.01, value=float(info["business_params"]["cost_per_contact"]))
    value = st.number_input("Valor de una conversión (V)", min_value=0.01,
                            value=float(info["business_params"]["value_per_conversion"]))
    st.markdown(f"Umbral económico teórico C/V = **{cost / value:.3f}**: conviene llamar si la probabilidad "
                "estimada lo supera (con probabilidades calibradas).")

tab1, tab2 = st.tabs(["👤 Cliente individual", "📋 Campaña (lote CSV)"])

with tab1:
    with st.form("cliente"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Edad", 18, 100, 35)
            job = st.selectbox("Ocupación", LEVELS["job"], index=4)
            marital = st.selectbox("Estado civil", LEVELS["marital"], index=1)
            education = st.selectbox("Educación", LEVELS["education"], index=1)
            balance = st.number_input("Saldo promedio anual", -20000, 200000, 1500)
        with c2:
            default = st.radio("¿Préstamos impagos?", ["no", "yes"], horizontal=True)
            housing = st.radio("¿Préstamo hipotecario?", ["no", "yes"], horizontal=True)
            loan = st.radio("¿Préstamo personal?", ["no", "yes"], horizontal=True)
            contact = st.selectbox("Tipo de contacto", LEVELS["contact"])
            month = st.selectbox("Mes del contacto", LEVELS["month"], index=9, format_func=MONTHS_ES.get)
        with c3:
            day = st.number_input("Día del mes", 1, 31, 15)
            campaign = st.number_input("Contactos en esta campaña (incluido este)", 1, 100, 1)
            previous = st.number_input("Contactos en campañas previas", 0, 300, 0)
            pdays = st.number_input("Días desde el último contacto previo (-1 = nunca)", -1, 999,
                                    -1 if previous == 0 else 90)
            poutcome = st.selectbox("Resultado de la campaña previa", LEVELS["poutcome"])
        submitted = st.form_submit_button("Estimar probabilidad", type="primary")

    if submitted:
        payload = dict(age=age, job=job, marital=marital, education=education, default=default, balance=balance,
                       housing=housing, loan=loan, contact=contact, day=day, month=month, campaign=campaign,
                       pdays=pdays, previous=previous, poutcome=poutcome)
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        if r.status_code != 200:
            st.error(f"La API rechazó los datos ({r.status_code}): {r.json().get('detail')}")
        else:
            res = r.json()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Probabilidad de conversión", f"{100 * res['probability']:.1f} %")
            m2.metric("Decil (1 = mejor)", res["decile"])
            m3.metric("Recomendación", res["recommendation"].upper())
            ev = value * res["probability"] - cost
            m4.metric("Valor esperado de llamar", f"{ev:,.2f}")
            (st.success if res["contact_recommended"] else st.warning)(res["explanation"])
            st.subheader("Variables con mayor peso para este cliente (SHAP)")
            fac = pd.DataFrame(res["top_factors"])[["label", "value", "shap", "effect"]].astype({"value": str})
            fac.columns = ["Variable", "Valor", "Contribución (log-odds)", "Efecto"]
            st.dataframe(fac, hide_index=True, width="stretch")
            st.caption("Contribución positiva = aumenta la probabilidad respecto del cliente promedio.")

with tab2:
    st.markdown("Subí un CSV con los clientes a evaluar (mismas columnas que `banca_train.csv`; "
                "`duration` e `y` se ignoran si vienen). Separador `,` o `;`.")
    up = st.file_uploader("Archivo CSV", type=["csv"])
    if up is not None:
        r = requests.post(f"{API_URL}/predict/batch", files={"file": (up.name, up.getvalue(), "text/csv")}, timeout=300)
        if r.status_code != 200:
            st.error(f"La API rechazó el archivo ({r.status_code}): {r.json().get('detail')}")
        else:
            ranked = pd.read_csv(io.StringIO(r.text))
            ignored = r.headers.get("X-Ignored-Columns", "")
            if ignored:
                st.info(f"Columnas ignoradas por el modelo: {ignored}")
            n = len(ranked)
            mode = st.radio("Definir el tamaño de la campaña por", ["% de clientes", "presupuesto"], horizontal=True)
            if mode == "% de clientes":
                default_pct = int(round(100 * (ranked["recommendation"] == "llamar").mean()))
                pct = st.slider("% de clientes a contactar (mejor rankeados)", 1, 100, max(default_pct, 1))
                k = int(np.ceil(pct / 100 * n))
            else:
                budget = st.number_input("Presupuesto disponible", min_value=0.0, value=float(0.2 * n * cost))
                k = int(min(n, budget // cost))
            sel = ranked.head(k)
            exp_conv = sel["probability"].sum()
            exp_profit = value * exp_conv - cost * k
            all_profit = value * ranked["probability"].sum() - cost * n
            a, b, c, d = st.columns(4)
            a.metric("Clientes a contactar", f"{k:,} de {n:,}")
            b.metric("Conversiones esperadas", f"{exp_conv:,.0f}",
                     f"{100 * exp_conv / ranked['probability'].sum():.0f} % del total esperado")
            c.metric("Beneficio esperado", f"{exp_profit:,.0f}")
            d.metric("vs contactar a todos", f"{exp_profit - all_profit:,.0f}")
            if "y" in ranked.columns:
                real = (ranked["y"].astype(str).isin(["yes", "1"])).astype(int)
                st.caption(f"El archivo trae el resultado real: entre los {k} seleccionados hay "
                           f"{int(real.head(k).sum())} conversiones reales de {int(real.sum())} "
                           f"({100 * real.head(k).sum() / max(real.sum(), 1):.0f} %).")
            curve = pd.DataFrame({"% contactado": 100 * np.arange(1, n + 1) / n,
                                  "Beneficio esperado": value * ranked["probability"].cumsum().values
                                  - cost * np.arange(1, n + 1)})
            st.line_chart(curve, x="% contactado", y="Beneficio esperado")
            st.dataframe(sel, hide_index=True, width="stretch")
            st.download_button("⬇️ Descargar listado priorizado", sel.to_csv(index=False).encode("utf-8"),
                               "clientes_a_contactar.csv", "text/csv")
