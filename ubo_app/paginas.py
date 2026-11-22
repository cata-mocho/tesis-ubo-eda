"""Páginas de la plataforma."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from . import acompanamiento as ac
from . import datos, llm
from .config import (
    COLORES_TRAMO, IMPORTANCIA_SHAP, ORDEN_TRAMO, RESULTADOS_TESIS, SUBTITULO_APP, TASA_ESPERADA_2025,
    TASA_REFERENCIA_2022_2024, TITULO_APP,
)
from .vistas_eda import POR_ID, SECCIONES, VISTAS, estilo, fmt

PRIVACIDAD = (
    "Los puntajes orientan apoyos con revisión humana y **nunca** se usan para restringir matrícula, "
    "beneficios ni acceso. En pantalla se muestra un código seudónimo (EST-…) en lugar del MRUN."
)


def mostrar_mrun() -> bool:
    try:
        return bool(st.secrets.get("MOSTRAR_MRUN", False))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return False


def pct(x: float, dec: int = 1) -> str:
    return fmt(x * 100, dec) + " %"


# =============================================================================
# Inicio
# =============================================================================
def pagina_inicio() -> None:
    st.title(TITULO_APP)

    st.markdown(
        "Esta plataforma acompaña la tesis y tiene dos módulos: un **análisis exploratorio** de la "
        "matrícula 2015–2025 y un **sistema de alerta temprana** que aplica el modelo a la cohorte 2025 "
        "y propone un plan de acompañamiento para cada estudiante priorizado."
    )
    r = RESULTADOS_TESIS
    c = st.columns(4)
    c[0].metric("Estudiantes en el modelado", fmt(r["poblacion"]), help="Primer año, cohortes 2015–2024")
    c[1].metric("Tasa de no continuidad", fmt(r["tasa_no_continuidad"], 2) + " %")
    c[2].metric("AP en prueba", fmt(r["ap_prueba"], 3), f"{fmt(r['ap_prueba'] / r['prevalencia'], 1)} × el azar", delta_color="off")
    c[3].metric("ROC-AUC temporal", fmt(r["roc_auc_temporal"], 3), help="Prediciendo cohortes futuras, 2018–2024")

    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        st.subheader("Módulo 1 · Análisis exploratorio")
        st.write("Catorce vistas organizadas en cuatro secciones: composición de la matrícula, carreras y "
                 "jornadas, trayectoria escolar, y territorio y vulnerabilidad. Cada vista indica el tipo de "
                 "gráfico y explica cómo interpretarlo.")
        st.page_link(st.session_state["_paginas"]["exploracion"], label="Ir al análisis exploratorio", icon=":material/bar_chart:")
    with col2, st.container(border=True):
        st.subheader("Módulo 2 · Alerta temprana 2025")
        st.write("Resumen de riesgo de la cohorte 2025, listado priorizado y una ficha por estudiante con su "
                 "pronóstico, los factores que explican su puntaje y un plan de acompañamiento preventivo.")
        st.page_link(st.session_state["_paginas"]["alerta"], label="Ir a la alerta temprana", icon=":material/notifications_active:")

    st.info(PRIVACIDAD)


# =============================================================================
# Módulo 1 · Análisis exploratorio
# =============================================================================
def _filtros(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    sb = st.sidebar
    sb.header("Filtros")
    anios = sorted(df["CAT_PERIODO"].dropna().astype(int).unique().tolist())
    rango = sb.slider("Año de matrícula", anios[0], anios[-1], (anios[0], anios[-1])) if len(anios) > 1 else (anios[0], anios[0])
    areas = sorted(df["AREA_CONOCIMIENTO"].unique().tolist())
    areas_sel = sb.multiselect("Área del conocimiento", areas, placeholder="Todas")
    jornadas = sorted(df["JORNADA"].unique().tolist())
    jornada_sel = sb.multiselect("Jornada", jornadas, placeholder="Todas")
    generos = sorted(df["GEN_ALU"].unique().tolist())
    genero_sel = sb.multiselect("Género", generos, placeholder="Todos")
    carreras = sorted(df["NOMB_CARRERA"].unique().tolist())
    carreras_sel = sb.multiselect("Carrera", carreras, placeholder="Todas")
    unidad = sb.radio(
        "Contar",
        ["estudiantes", "registros"],
        index=0,
        horizontal=True,
        format_func=lambda u: "Estudiantes únicos" if u == "estudiantes" else "Registros de matrícula",
    )
    sb.caption("Un estudiante puede tener varios registros, uno por cada año matriculado.")

    f = df[df["CAT_PERIODO"].between(rango[0], rango[1])]
    partes = [f"años {rango[0]}–{rango[1]}"]
    for col, sel, nombre in [("AREA_CONOCIMIENTO", areas_sel, "área"), ("JORNADA", jornada_sel, "jornada"),
                             ("GEN_ALU", genero_sel, "género"), ("NOMB_CARRERA", carreras_sel, "carrera")]:
        if sel:
            f = f[f[col].isin(sel)]
            partes.append(f"{nombre}: {', '.join(sel)}")
    partes.append("conteo: " + ("estudiantes únicos" if unidad == "estudiantes" else "registros de matrícula"))
    return f, unidad, "; ".join(partes)


def pagina_exploracion() -> None:
    df = datos.obtener_eda()
    if df is None:
        st.stop()
    filtrado, unidad, texto_filtros = _filtros(df)

    st.title("Análisis exploratorio de la matrícula")
    c = st.columns(4)
    c[0].metric("Registros de matrícula", fmt(len(filtrado)))
    c[1].metric("Estudiantes únicos", fmt(filtrado["MRUN"].nunique()))
    c[2].metric("Carreras", fmt(filtrado["NOMB_CARRERA"].nunique()))
    c[3].metric("Establecimientos de egreso", fmt(filtrado["RBD"].nunique()))

    if filtrado.empty:
        st.info("No hay registros con los filtros seleccionados.")
        st.stop()

    col_s, col_v = st.columns([1, 2])
    with col_s:
        seccion = st.selectbox("Sección", SECCIONES, key="eda_seccion")
    opciones = [v.id for v in VISTAS if v.seccion == seccion]
    with col_v:
        vista_id = st.selectbox("Gráfico", opciones, format_func=lambda i: POR_ID[i].titulo, key=f"eda_vista_{seccion}")
    vista = POR_ID[vista_id]

    parametros = {"unidad": unidad, **vista.parametros}
    if vista.id == "top_carreras":
        parametros["top_n"] = st.radio("Carreras a mostrar", [10, 15, 20], index=1, horizontal=True)
    if vista.id == "mapa":
        parametros["cobertura"] = st.radio("Cobertura", ["Todo Chile", "Región Metropolitana"], horizontal=True)

    st.divider()
    st.subheader(vista.titulo)
    st.caption(vista.pregunta)

    fig, tabla, hallazgos = vista.construir(filtrado, parametros)

    col_g, col_l = st.columns([1.7, 1])
    with col_g:
        if fig is None:
            st.info(hallazgos[0])
        else:
            st.plotly_chart(fig, use_container_width=True, key=f"fig_{vista.id}")
    with col_l, st.container(border=True):
        st.markdown(f"**Tipo de gráfico:** {vista.tipo}")
        st.divider()
        st.markdown("**Cómo interpretarlo**")
        st.markdown(vista.explicacion)
        if fig is not None:
            st.divider()
            st.markdown("**Con los filtros actuales**")
            st.markdown("\n".join(f"- {h}" for h in hallazgos))

    if fig is not None:
        with st.expander("Datos del gráfico"):
            st.dataframe(tabla, use_container_width=True, hide_index=True)
            st.download_button("Descargar CSV", tabla.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"eda_{vista.id}.csv", mime="text/csv", key=f"dl_{vista.id}")


# =============================================================================
# Módulo 2 · Alerta temprana
# =============================================================================
def _filtros_sat(df: pd.DataFrame) -> pd.DataFrame:
    sb = st.sidebar
    sb.header("Filtros")
    tramos = sb.multiselect("Tramo de riesgo", ORDEN_TRAMO, default=ORDEN_TRAMO)
    areas = sb.multiselect("Área", sorted(df["AREA_CONOCIMIENTO"].dropna().unique()) if "AREA_CONOCIMIENTO" in df else [], placeholder="Todas")
    carreras = sb.multiselect("Carrera", sorted(df["NOMB_CARRERA"].dropna().unique()), placeholder="Todas")
    jornadas = sb.multiselect("Jornada", sorted(df["JORNADA"].dropna().unique()), placeholder="Todas")
    f = df[df["TRAMO_RIESGO"].isin(tramos)] if tramos else df.iloc[0:0]
    if areas:
        f = f[f["AREA_CONOCIMIENTO"].isin(areas)]
    if carreras:
        f = f[f["NOMB_CARRERA"].isin(carreras)]
    if jornadas:
        f = f[f["JORNADA"].isin(jornadas)]
    return f


def _resumen(df: pd.DataFrame, f: pd.DataFrame) -> None:
    n = len(f)
    c = st.columns(4)
    c[0].metric("Estudiantes", fmt(n))
    c[1].metric("Casos esperados", fmt(f["PROB_DESERCION"].sum(), 0), help="Suma de las probabilidades individuales: es una estimación, no casos observados.")
    c[2].metric("Tasa esperada", pct(f["PROB_DESERCION"].mean() if n else 0, 2), help=f"Recalibrada a la tasa 2022–2024 ({pct(TASA_REFERENCIA_2022_2024, 2)}).")
    alertas = int(f["ALERTA_UMBRAL_OPTIMO"].sum()) if "ALERTA_UMBRAL_OPTIMO" in f else int((f["TRAMO_RIESGO"] == "Alto").sum())
    c[3].metric("Alertas (umbral 0,140)", fmt(alertas), pct(alertas / n if n else 0), delta_color="off")

    col1, col2 = st.columns(2)
    with col1:
        t = f["TRAMO_RIESGO"].value_counts().reindex(ORDEN_TRAMO).fillna(0).reset_index()
        t.columns = ["Tramo", "Estudiantes"]
        fig = px.bar(t, x="Tramo", y="Estudiantes", color="Tramo", color_discrete_map=COLORES_TRAMO, text="Estudiantes")
        fig.update_layout(showlegend=False)
        st.markdown("**Estudiantes por tramo de riesgo**")
        st.plotly_chart(estilo(fig, 380), use_container_width=True)
    with col2:
        fig = px.histogram(f, x="PROB_DESERCION", nbins=40, color="TRAMO_RIESGO", color_discrete_map=COLORES_TRAMO,
                           category_orders={"TRAMO_RIESGO": ORDEN_TRAMO}, labels={"PROB_DESERCION": "Probabilidad estimada", "TRAMO_RIESGO": "Tramo"})
        fig.update_xaxes(tickformat=".0%")
        st.markdown("**Distribución de la probabilidad estimada**")
        st.plotly_chart(estilo(fig, 380), use_container_width=True)

    por_carrera = (f.groupby("NOMB_CARRERA").agg(Estudiantes=("ID", "size"), Alto=("TRAMO_RIESGO", lambda s: (s == "Alto").sum()),
                                                 Tasa_esperada=("PROB_DESERCION", "mean"))
                   .sort_values("Alto", ascending=False).head(12).reset_index())
    por_carrera["Tasa_esperada"] = (por_carrera["Tasa_esperada"] * 100).round(1)
    st.markdown("**Carreras con más estudiantes en tramo alto**")
    fig = px.bar(por_carrera.sort_values("Alto"), x="Alto", y="NOMB_CARRERA", orientation="h", color_discrete_sequence=[COLORES_TRAMO["Alto"]],
                 hover_data={"Estudiantes": True, "Tasa_esperada": True}, labels={"NOMB_CARRERA": "", "Alto": "Estudiantes en tramo alto"})
    st.plotly_chart(estilo(fig, 460), use_container_width=True)



def _abrir_ficha(id_estudiante: str) -> None:
    """Se ejecuta al pulsar el botón: fija al estudiante y cambia a la vista de la ficha."""
    st.session_state["ficha_id"] = id_estudiante
    st.session_state["ficha_select"] = id_estudiante
    st.session_state["alerta_vista"] = "Ficha y plan de acompañamiento"


def _priorizacion(f: pd.DataFrame) -> None:
    st.markdown("Listado ordenado por probabilidad estimada. Selecciona **una fila** y pulsa el botón para abrir su ficha.")
    cols = ["PRIORIDAD", "ID"] + (["MRUN"] if mostrar_mrun() else []) + [
        "NOMB_CARRERA", "JORNADA", "TRAMO_RIESGO", "PROB_DESERCION", "PERCENTIL_RIESGO_COHORTE",
        "PROM_NOTAS_ALU", "FACTOR_RIESGO_1", "FACTOR_RIESGO_2", "FACTOR_RIESGO_3"]
    cols = [c for c in cols if c in f.columns]
    vista = f[cols].reset_index(drop=True)
    vista["PROB_DESERCION"] = (vista["PROB_DESERCION"] * 100).round(2)
        vista["TRAMO_RIESGO"] = vista["TRAMO_RIESGO"].map(
        {"Alto": "🔴 Alto", "Medio": "🟡 Medio", "Bajo": "🟢 Bajo"}
    ).fillna(vista["TRAMO_RIESGO"])
    evento = st.dataframe(
        vista, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="tabla_sat",
        column_config={
            "PROB_DESERCION": st.column_config.ProgressColumn("Probabilidad (%)", format="%.1f", min_value=0, max_value=50),
            "PERCENTIL_RIESGO_COHORTE": st.column_config.NumberColumn("Percentil", format="%.1f"),
            "PROM_NOTAS_ALU": st.column_config.NumberColumn("NEM", format="%.2f"),
            "NOMB_CARRERA": "Carrera", "JORNADA": "Jornada", "TRAMO_RIESGO": "Tramo",
            "FACTOR_RIESGO_1": "Factor 1", "FACTOR_RIESGO_2": "Factor 2", "FACTOR_RIESGO_3": "Factor 3",
        },
    )
    filas = evento.selection.rows if evento and hasattr(evento, "selection") else []
    if filas:
        id_sel = vista.iloc[filas[0]]["ID"]
        st.button(f"Abrir ficha y plan de acompañamiento de {id_sel}", type="primary",
                  on_click=_abrir_ficha, args=(id_sel,), key="btn_abrir_ficha")
    else:
        st.caption("Selecciona una fila de la tabla para habilitar el botón de la ficha.")

    c1, c2 = st.columns(2)
    c1.download_button("Descargar listado priorizado (CSV)", f[[c for c in cols if c != "ID"] + ["ID"]].to_csv(index=False).encode("utf-8-sig"),
                       file_name="priorizacion_alerta_temprana_2025.csv", mime="text/csv")
    planes = ac.tabla_planes(f)
    if not mostrar_mrun() and "MRUN" in planes:
        planes = planes.drop(columns="MRUN")
    c2.download_button("Descargar planes de acompañamiento, tramos alto y medio (CSV)", planes.to_csv(index=False).encode("utf-8-sig"),
                       file_name="planes_acompanamiento_2025.csv", mime="text/csv", disabled=planes.empty)


def _ficha(df: pd.DataFrame, f: pd.DataFrame) -> None:
    base = f if not f.empty else df
    ids = base["ID"].tolist()
    # Aseguramos que el valor guardado del selector exista en la lista actual.
    if st.session_state.get("ficha_select") not in ids:
        actual = st.session_state.get("ficha_id")
        st.session_state["ficha_select"] = actual if actual in ids else ids[0]
    etiqueta = {r.ID: f"#{r.PRIORIDAD} · {r.ID} · {r.NOMB_CARRERA} · {pct(r.PROB_DESERCION)} ({r.TRAMO_RIESGO})"
                for r in base.itertuples()}
    sel = st.selectbox("Estudiante (ordenados por prioridad)", ids, format_func=etiqueta.get, key="ficha_select")
    st.session_state["ficha_id"] = sel
    fila = base[base["ID"] == sel].iloc[0]
    pr = ac.pronostico(fila)

    col1, col2 = st.columns([1, 1.4])
    with col1, st.container(border=True):
        color = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}.get(pr["tramo"], "⚪")
        st.markdown(f"### {color} Riesgo {pr['tramo']} · {pr['prioridad']}")
        st.metric("Probabilidad estimada de no continuar en 2026", pct(pr["probabilidad"]),
                  f"{fmt(pr['veces_cohorte'], 1)} × la tasa esperada de la cohorte ({pct(TASA_ESPERADA_2025, 2)})", delta_color="inverse")
        st.progress(min(pr["probabilidad"] / 0.5, 1.0), text=f"Percentil {fmt(pr['percentil'], 1)} de la cohorte")
        st.write(f"**Carrera:** {fila.get('NOMB_CARRERA', 'S/I')}")
        st.write(f"**Área:** {fila.get('AREA_CONOCIMIENTO', 'S/I')} · **Jornada:** {fila.get('JORNADA', 'S/I')}")
        st.write(f"**NEM:** {fmt(float(fila['PROM_NOTAS_ALU']), 2) if pd.notna(fila.get('PROM_NOTAS_ALU')) else 'S/I'}")
        if mostrar_mrun():
            st.write(f"**MRUN:** {fila['MRUN']}")
        st.caption(f"De cada 100 estudiantes con un perfil similar, se esperaría que unos {pr['de_cada_100']} no continúen.")
    with col2, st.container(border=True):
        st.markdown("#### Factores que más elevaron su puntaje")
        for dim, factor in ac.dimensiones_estudiante(fila):
            st.markdown(f"- **{factor}** → dimensión *{dim.nombre}*")
        st.caption("Describen cómo el modelo calculó el puntaje, no causas. Los índices IVE e IVM son del colegio o del territorio: confirmar en entrevista.")
        st.markdown("#### Intensidad del seguimiento")
        st.markdown(f"- **Modalidad:** {pr['modalidad']}\n- **Primer contacto:** {pr['primer_contacto']}\n"
                    f"- **Frecuencia:** {pr['frecuencia']}\n- **Revisión:** {pr['revision']}")

    st.markdown("### Plan de acompañamiento preventivo")
    if pr["tramo"] == "Bajo":
        st.info("Tramo bajo: el estudiante participa del apoyo universal de primer año. El plan se activa si aparece una señal de alerta en el semestre.")
    pestanas = st.tabs(["Acciones por dimensión", "Calendario y señales de alerta", "Plan redactado con IA"])
    with pestanas[0]:
        for dim, factor in ac.dimensiones_estudiante(fila):
            with st.container(border=True):
                st.markdown(f"**{dim.nombre}** · _{factor}_")
                st.caption(dim.lectura)
                st.markdown("\n".join(f"- {a}" for a in dim.acciones))
                st.markdown(f"**Responsable:** {dim.responsable}  \n**Indicador de seguimiento:** {dim.indicador}")
    with pestanas[1]:
        cal = pd.DataFrame(ac.CALENDARIO, columns=["Momento", "Acción", "Señal de alerta"])
        st.dataframe(cal, use_container_width=True, hide_index=True)
    with pestanas[2]:
        proveedor, _, modelo = llm.proveedor_activo()
        st.caption(f"Se envía solo el perfil anónimo (sin MRUN ni RBD). Motor: {modelo or 'no configurado'}.")
        with st.expander("Ver el perfil que se envía"):
            st.code(ac.perfil_anonimo(fila), language=None)
        cache = st.session_state.setdefault("plan_cache", {})
        if st.button("Redactar plan personalizado", type="primary", disabled=proveedor is None, key=f"plan_{sel}"):
            with st.spinner("Redactando el plan…"):
                try:
                    cache[sel] = llm.llamar(llm.SISTEMA_PLAN, ac.perfil_anonimo(fila), max_tokens=900)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"No se pudo conectar con el proveedor de IA ({exc}).")
        if proveedor is None:
            st.info("Configura una clave de IA en la barra lateral para redactar el plan. El plan por reglas de las otras pestañas no la necesita.")
        if sel in cache:
            with st.container(border=True):
                st.markdown(cache[sel])
                st.caption("Borrador generado con IA: revísalo antes de usarlo con el estudiante.")

    plan = ac.plan_markdown(fila)
    if sel in st.session_state.get("plan_cache", {}):
        plan += "\n\n## Plan redactado con IA (borrador)\n\n" + st.session_state["plan_cache"][sel]
    st.download_button("Descargar plan del estudiante (.md)", plan.encode("utf-8"), file_name=f"plan_{sel}.md", mime="text/markdown")


def _catalogo() -> None:
    st.markdown("Cada factor del modelo se asocia a una dimensión de apoyo. La intensidad depende del tramo de riesgo.")
    filas = []
    vistos = set()
    for var, dim in ac.MAPA_FACTORES.items():
        filas.append({"Factor del modelo": var, "Dimensión": dim.nombre})
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
    for dim in ac.MAPA_FACTORES.values():
        if dim.nombre in vistos:
            continue
        vistos.add(dim.nombre)
        with st.expander(dim.nombre):
            st.caption(dim.lectura)
            st.markdown("\n".join(f"- {a}" for a in dim.acciones))
            st.markdown(f"**Responsable:** {dim.responsable}  \n**Indicador:** {dim.indicador}")
    st.markdown("**Intensidad por tramo**")
    st.dataframe(pd.DataFrame(ac.INTENSIDAD).T.rename_axis("Tramo").reset_index(), use_container_width=True, hide_index=True)


def pagina_alerta() -> None:
    df = datos.obtener_sat()
    llm.panel_configuracion()
    if df is None:
        st.stop()
    f = _filtros_sat(df)
    st.title("Sistema de alerta temprana · Cohorte 2025")

    secciones = ["Resumen de riesgo", "Priorización", "Ficha y plan de acompañamiento", "Catálogo de intervenciones"]
    # Valor inicial y protección: el control no debe quedar sin selección.
    if st.session_state.get("alerta_vista") not in secciones:
        st.session_state["alerta_vista"] = secciones[0]
    seccion = st.segmented_control("Sección", secciones, key="alerta_vista", label_visibility="collapsed")
    if seccion is None:
        seccion = st.session_state["alerta_vista"] = secciones[0]
    st.divider()

    if seccion == "Resumen de riesgo":
        if f.empty:
            st.info("No hay estudiantes con los filtros seleccionados.")
        else:
            _resumen(df, f)
    elif seccion == "Priorización":
        _priorizacion(f)
    elif seccion == "Ficha y plan de acompañamiento":
        if f.empty:
            st.info("No hay estudiantes con los filtros seleccionados.")
        else:
            _ficha(df, f)
    else:
        _catalogo()


# =============================================================================
# Modelo y metodología
# =============================================================================
def pagina_modelo() -> None:
    st.title("El modelo de la tesis")
    r = RESULTADOS_TESIS
    st.markdown(
        "Se integraron registros de matrícula de la UBO, egreso de enseñanza media (MINEDUC) e índices de "
        "vulnerabilidad de JUNAEB (IVE e IVM), siguiendo CRISP-DM. Se compararon regresión logística, Random "
        "Forest, XGBoost y LightGBM, con validación cruzada, prueba independiente y evaluación temporal."
    )
    c = st.columns(4)
    c[0].metric("AP en prueba", fmt(r["ap_prueba"], 3), "azar = 0,126", delta_color="off")
    c[1].metric("ROC-AUC en prueba", fmt(r["roc_auc_prueba"], 3))
    c[2].metric("Recall con umbral 0,160", fmt(r["recall"], 1) + " %")
    c[3].metric("Precisión de las alertas", fmt(r["precision"], 1) + " %")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Variables con mayor contribución (SHAP)**")
        shap = pd.DataFrame(IMPORTANCIA_SHAP, columns=["Variable", "% de la importancia", "Dirección"])
        fig = px.bar(shap.sort_values("% de la importancia"), x="% de la importancia", y="Variable", orientation="h",
                     hover_data={"Dirección": True}, color_discrete_sequence=["#053380"])
        st.plotly_chart(estilo(fig, 420), use_container_width=True)
    with col2:
        st.markdown("**Evaluación temporal (entrenando con cohortes anteriores)**")
        temp = pd.DataFrame({"Cohorte": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
                             "ROC-AUC": [0.629, 0.667, 0.577, 0.640, 0.632, 0.534, 0.625]})
        fig = px.line(temp, x="Cohorte", y="ROC-AUC", markers=True, color_discrete_sequence=["#053380"])
        fig.add_hline(y=0.650, line_dash="dash", annotation_text="Partición aleatoria 0,650")
        fig.add_hline(y=0.5, line_dash="dot", annotation_text="Azar 0,500")
        st.plotly_chart(estilo(fig, 420), use_container_width=True)

    st.markdown("**Cómo leer los resultados**")
    st.markdown(
        f"- Capacidad discriminativa **moderada**: el tramo alto concentra {fmt(r['tasa_tramo_alto'], 1)} % de no "
        f"continuidad, pero el {fmt(r['casos_tramo_bajo'], 1)} % de los casos queda en el tramo bajo. Sirve para priorizar, no para diagnosticar.\n"
        "- Los ensambles no superaron de forma significativa a la regresión logística (p = 0,470).\n"
        "- La vulnerabilidad se asocia al riesgo, pero no agregó poder predictivo detectable (p = 0,092).\n"
        "- El resultado es la no continuidad **institucional**: no distingue traslado de abandono.\n"
        "- Los factores SHAP describen el modelo, no causas."
    )
    st.info(PRIVACIDAD)
