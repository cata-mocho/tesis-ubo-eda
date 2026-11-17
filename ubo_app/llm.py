"""Asistente LLM especializado en la tesis.

- Usa Claude (Anthropic) u OpenAI si hay una clave en `st.secrets`, en variables de
  entorno o escrita en la barra lateral; si no, usa un motor local basado en reglas.
- Nunca envía registros individuales ni identificadores (MRUN, RBD): solo tablas
  agregadas del gráfico o el perfil anónimo de un estudiante.
"""

from __future__ import annotations

import hashlib
import os

import pandas as pd
import streamlit as st

from .config import CONTEXTO_TESIS

MODELO_ANTHROPIC = "claude-haiku-5-5"
MODELO_OPENAI = "gpt-4o-mini"

SISTEMA_GRAFICOS = f"""
Eres el asistente analítico de una tesis sobre permanencia estudiantil en la
Universidad Bernardo O'Higgins (UBO). Interpretas gráficos del análisis exploratorio
para una comisión evaluadora y para equipos de acompañamiento estudiantil.

Contexto de la tesis (úsalo para conectar el gráfico con los resultados):
{CONTEXTO_TESIS}

Reglas:
1. Usa solo las cifras de la tabla y los hallazgos entregados, más el contexto de la tesis.
   No inventes cifras. Si algo no se puede concluir con los datos, dilo.
2. Distingue asociación de causalidad. Los gráficos describen matrícula, no deserción,
   salvo que el contexto de la tesis lo diga explícitamente.
3. Los índices IVE e IVM son del establecimiento o del territorio: no los atribuyas al
   estudiante individual (falacia ecológica).
4. Señala si un grupo es pequeño y sus porcentajes pueden ser inestables.
5. Responde en español de Chile, con coma decimal, en un máximo de 220 palabras y
   con exactamente estas tres secciones en negrita:
   **Lo que muestra el gráfico**, **Relación con la tesis**, **Implicancias para la permanencia**.
""".strip()

SISTEMA_PLAN = f"""
Eres un especialista en acompañamiento estudiantil de primer año en la UBO. Recibes el
perfil anónimo de un estudiante con su probabilidad estimada de no continuidad y los
factores que más elevaron su puntaje según SHAP. Redactas un plan de acompañamiento
preventivo para el equipo de apoyo.

Contexto de la tesis:
{CONTEXTO_TESIS}

Reglas:
1. El puntaje es una estimación para priorizar, no un diagnóstico: no etiquetes al
   estudiante ni supongas su situación personal.
2. Los índices de vulnerabilidad describen su colegio o territorio; propone verificar la
   situación real en una entrevista antes de actuar.
3. No propongas acciones distintas por género; el género no define una intervención.
4. Las acciones orientan apoyos voluntarios; nunca restringen matrícula ni beneficios.
5. Responde en español de Chile, máximo 300 palabras, con estas secciones en negrita:
   **Lectura del riesgo**, **Primeras dos semanas**, **Seguimiento del semestre**,
   **Señales de alerta a vigilar**, **Cómo medir si funciona**.
""".strip()


# ---------------------------------------------------------------------------
# Configuración del proveedor
# ---------------------------------------------------------------------------
def _secreto(nombre: str) -> str | None:
    try:
        valor = st.secrets.get(nombre)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001  (sin secrets.toml)
        valor = None
    return valor or os.environ.get(nombre)


def proveedor_activo() -> tuple[str | None, str | None, str | None]:
    """Devuelve (proveedor, clave, modelo) o (None, None, None) si no hay clave."""
    manual = st.session_state.get("llm_clave_manual", "").strip()
    elegido = st.session_state.get("llm_proveedor_manual", "Claude (Anthropic)")
    if manual:
        if elegido.startswith("Claude"):
            return "anthropic", manual, _secreto("ANTHROPIC_MODEL") or MODELO_ANTHROPIC
        return "openai", manual, _secreto("OPENAI_MODEL") or MODELO_OPENAI
    if clave := _secreto("ANTHROPIC_API_KEY"):
        return "anthropic", clave, _secreto("ANTHROPIC_MODEL") or MODELO_ANTHROPIC
    if clave := _secreto("OPENAI_API_KEY"):
        return "openai", clave, _secreto("OPENAI_MODEL") or MODELO_OPENAI
    return None, None, None


def panel_configuracion() -> None:
    """Controles de la barra lateral para el asistente."""
    with st.sidebar.expander("Asistente de análisis (IA)", expanded=False):
        proveedor, _, modelo = proveedor_activo()
        if proveedor and not st.session_state.get("llm_clave_manual"):
            st.success(f"Conectado con la clave del servidor ({modelo}).")
        st.selectbox("Proveedor", ["Claude (Anthropic)", "OpenAI"], key="llm_proveedor_manual")
        st.text_input("Clave de API (opcional)", type="password", key="llm_clave_manual",
                      help="Se usa solo en esta sesión. Sin clave, el análisis lo genera el motor local.")
        st.caption("Al asistente solo se envían tablas agregadas o perfiles anónimos, nunca el MRUN.")


def llamar(sistema: str, mensaje: str, max_tokens: int = 700) -> str:
    proveedor, clave, modelo = proveedor_activo()
    if proveedor == "anthropic":
        import anthropic

        cliente = anthropic.Anthropic(api_key=clave)
        r = cliente.messages.create(model=modelo, max_tokens=max_tokens, temperature=0.2,
                                    system=sistema, messages=[{"role": "user", "content": mensaje}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
    if proveedor == "openai":
        import openai

        cliente = openai.OpenAI(api_key=clave)
        r = cliente.chat.completions.create(model=modelo, max_tokens=max_tokens, temperature=0.2,
                                            messages=[{"role": "system", "content": sistema},
                                                      {"role": "user", "content": mensaje}])
        return (r.choices[0].message.content or "").strip()
    raise RuntimeError("No hay un proveedor de IA configurado.")


# ---------------------------------------------------------------------------
# Análisis de gráficos
# ---------------------------------------------------------------------------
def _tabla_texto(tabla: pd.DataFrame, max_filas: int = 25) -> str:
    if tabla is None or len(tabla) == 0:
        return "(sin datos)"
    return tabla.head(max_filas).to_csv(index=False, sep=";", decimal=",", float_format="%.2f")


def mensaje_grafico(vista, tabla: pd.DataFrame, hallazgos: list[str], filtros: str) -> str:
    return (
        f"Gráfico: {vista.titulo}\nPregunta que responde: {vista.pregunta}\n"
        f"Qué muestra: {vista.muestra}\nCómo se lee: {vista.como_leer}\n"
        f"Relación conocida con la tesis: {vista.tesis}\n"
        f"Filtros activos: {filtros}\n\n"
        f"Hallazgos calculados:\n- " + "\n- ".join(hallazgos) +
        f"\n\nDatos agregados del gráfico (CSV, separador ;):\n{_tabla_texto(tabla)}"
    )


def analisis_local_grafico(vista, hallazgos: list[str]) -> str:
    """Respaldo sin IA externa: ordena los hallazgos y los conecta con la tesis."""
    lineas = "\n".join(f"- {h}" for h in hallazgos)
    return (
        f"**Lo que muestra el gráfico**\n\n{vista.muestra}\n\n{lineas}\n\n"
        f"**Relación con la tesis**\n\n{vista.tesis}\n\n"
        "**Implicancias para la permanencia**\n\n"
        "Este gráfico describe la composición de la matrícula; no mide por sí solo el riesgo. "
        "Úsalo para dimensionar a qué grupos dirigir el acompañamiento y contrástalo con el "
        "Módulo 2, que estima el riesgo de cada estudiante de la cohorte 2025."
    )


def bloque_analisis(vista, tabla: pd.DataFrame, hallazgos: list[str], filtros: str) -> None:
    """Botón + resultado del análisis, cacheado por vista y datos."""
    clave = hashlib.md5((vista.id + filtros + _tabla_texto(tabla)).encode()).hexdigest()
    cache = st.session_state.setdefault("llm_cache", {})
    proveedor, _, modelo = proveedor_activo()

    col_b, col_t = st.columns([1, 3])
    with col_b:
        pulsado = st.button("Analizar este gráfico", key=f"llm_{vista.id}", type="primary", use_container_width=True)
    with col_t:
        st.caption(f"Motor: {modelo}" if proveedor else "Motor: análisis local (sin clave de IA configurada)")

    if pulsado:
        if proveedor:
            with st.spinner("Interpretando el gráfico a la luz de la tesis…"):
                try:
                    cache[clave] = llamar(SISTEMA_GRAFICOS, mensaje_grafico(vista, tabla, hallazgos, filtros))
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"No se pudo conectar con el proveedor de IA ({exc}). Se muestra el análisis local.")
                    cache[clave] = analisis_local_grafico(vista, hallazgos)
        else:
            cache[clave] = analisis_local_grafico(vista, hallazgos)

    if clave in cache:
        with st.container(border=True):
            st.markdown(cache[clave])
            st.caption("Interpretación generada automáticamente: revísala antes de citarla.")
