"""Asistente LLM para la redacción de planes de acompañamiento.

- Usa Claude (Anthropic) u OpenAI si hay una clave en `st.secrets`, en variables de
  entorno o escrita en la barra lateral.
- Nunca envía registros individuales ni identificadores (MRUN, RBD): solo el perfil
  anónimo de un estudiante.
"""

from __future__ import annotations

import os

import streamlit as st

from .config import CONTEXTO_TESIS

MODELO_ANTHROPIC = "claude-haiku-5-5"
MODELO_OPENAI = "gpt-4o-mini"

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
    with st.sidebar.expander("Asistente de IA (plan de acompañamiento)", expanded=False):
        proveedor, _, modelo = proveedor_activo()
        if proveedor and not st.session_state.get("llm_clave_manual"):
            st.success(f"Conectado con la clave del servidor ({modelo}).")
        st.selectbox("Proveedor", ["Claude (Anthropic)", "OpenAI"], key="llm_proveedor_manual")
        st.text_input("Clave de API (opcional)", type="password", key="llm_clave_manual",
                      help="Se usa solo en esta sesión. Sin clave, el plan por reglas sigue disponible.")
        st.caption("Al asistente solo se envía el perfil anónimo de un estudiante, nunca el MRUN.")


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
