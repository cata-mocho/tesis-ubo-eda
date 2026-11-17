"""Plataforma Integral de Permanencia Estudiantil — UBO.

Ejecutar con:  streamlit run app.py
"""

import streamlit as st

from ubo_app.config import TITULO_APP
from ubo_app.paginas import pagina_alerta, pagina_exploracion, pagina_inicio, pagina_modelo

st.set_page_config(
    page_title="Permanencia Estudiantil | UBO",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
)

paginas = {
    "inicio": st.Page(pagina_inicio, title="Inicio", icon=":material/home:", url_path="inicio", default=True),
    "exploracion": st.Page(pagina_exploracion, title="Análisis exploratorio", icon=":material/bar_chart:", url_path="exploratorio"),
    "alerta": st.Page(pagina_alerta, title="Alerta temprana 2025", icon=":material/notifications_active:", url_path="alerta"),
    "modelo": st.Page(pagina_modelo, title="El modelo de la tesis", icon=":material/model_training:", url_path="modelo"),
}
st.session_state["_paginas"] = paginas

navegacion = st.navigation(
    {
        "Plataforma": [paginas["inicio"]],
        "Módulos": [paginas["exploracion"], paginas["alerta"]],
        "Tesis": [paginas["modelo"]],
    }
)

st.sidebar.caption(TITULO_APP)
navegacion.run()

st.divider()
st.caption("Plataforma de tesis · Universidad Bernardo O'Higgins · Facultad de Ingeniería, Ciencia y Tecnología · "
           "Ingeniería en Informática · Catalina Baeza González, 2026.")
