from pathlib import Path
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Permanencia Estudiantil | UBO",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FUNCIONES DE CARGA Y BÚSQUEDA DE ARCHIVOS
# ============================================================
def buscar_parquet():
    candidatos = sorted(Path("data").glob("*.parquet")) if Path("data").exists() else []
    candidatos += sorted(Path(".").glob("*.parquet"))
    return candidatos[0] if candidatos else None

def buscar_sat():
    candidatos = []
    for r in [Path("data"), Path(".")]:
        if r.exists():
            candidatos += list(r.glob("*Scoring*2025*.csv"))
            candidatos += list(r.glob("*Scoring*2025*.xlsx"))
            candidatos += list(r.glob("*Predicciones*2025*.csv"))
            candidatos += list(r.glob("*2025*.csv"))
    return candidatos[0] if candidatos else None

@st.cache_data(show_spinner="Cargando base de datos histórica (EDA)...")
def cargar_parquet(ruta: str) -> pd.DataFrame:
    return pd.read_parquet(ruta, engine="pyarrow")

@st.cache_data(show_spinner="Cargando base predictiva 2025 (SAT)...")
def cargar_sat(ruta) -> pd.DataFrame:
    ruta_str = str(ruta)
    if ruta_str.endswith(".xlsx"):
        return pd.read_excel(ruta_str)
    return pd.read_csv(ruta_str)

def limpiar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
    return df

def aplicar_estilo(fig, altura=500):
    fig.update_layout(
        height=altura,
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(family="Arial", size=12),
        legend_title_text="",
    )
    return fig

# ============================================================
# CABECERA Y CONMUTADOR DE ENTORNO
# ============================================================
st.title("Plataforma Integral de Permanencia Estudiantil — UBO")
st.caption("Investigación de Tesis · Análisis Exploratorio (EDA) y Sistema de Alerta Temprana (SAT) con Machine Learning")

modo_activo = st.radio(
    "Selecciona el entorno de trabajo:",
    [
        "📊 Módulo 1: Análisis Exploratorio de Datos (EDA Histórico 2015–2025)",
        "🚨 Módulo 2: Sistema de Alerta Temprana (SAT Predictivo — Cohorte 2025)"
    ],
    horizontal=True
)
st.divider()

# =============================================================================
# MÓDULO 1: TU EDA COMPLETO ORIGINAL (14 GRÁFICOS + LLM)
# =============================================================================
if "Módulo 1" in modo_activo:
    COLUMNAS_ESPERADAS = [
        "MRUN", "CAT_PERIODO", "GEN_ALU", "JORNADA", "RANGO_EDAD",
        "NOMB_CARRERA", "AREA_CONOCIMIENTO", "ANIO_EGRESO_MEDIA", "RBD",
        "NOM_REG_RBD_A", "NOM_COM_RBD", "NOM_DEPROV_RBD", "LATITUD",
        "LONGITUD", "NOM_RBD", "RURAL_RBD", "IVM_REGION",
    ]

    ruta_parquet = buscar_parquet()

    with st.sidebar:
        st.header("Fuente de datos EDA")
        archivo_subido = st.file_uploader("Cargar un archivo .parquet manual", type=["parquet"])

    if archivo_subido is not None:
        try:
            df = pd.read_parquet(archivo_subido, engine="pyarrow")
            fuente = archivo_subido.name
        except Exception as exc:
            st.error(f"Error al leer Parquet: {exc}")
            st.stop()
    elif ruta_parquet is not None:
        try:
            df = cargar_parquet(str(ruta_parquet))
            fuente = str(ruta_parquet)
        except Exception as exc:
            st.error(f"Error al leer {ruta_parquet}: {exc}")
            st.stop()
    else:
        st.warning("No se encontró ningún archivo .parquet en data/ ni en la raíz.")
        st.stop()

    df = limpiar_nombres_columnas(df)
    for col in ["CAT_PERIODO", "ANIO_EGRESO_MEDIA", "RURAL_RBD", "LATITUD", "LONGITUD", "IVM_REGION"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    faltan = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    if faltan:
        st.error(f"Faltan columnas requeridas en el archivo: {faltan}")
        st.stop()

    # FILTROS DEL EDA
    with st.sidebar:
        st.divider()
        st.header("Filtros del EDA")
        anios = sorted(df["CAT_PERIODO"].dropna().astype(int).unique().tolist())
        rango_anios = st.slider("Período de matrícula", min_value=int(min(anios)), max_value=int(max(anios)), value=(int(min(anios)), int(max(anios)))) if anios else (0, 9999)

        areas = sorted(df["AREA_CONOCIMIENTO"].dropna().astype(str).unique().tolist())
        areas_sel = st.multiselect("Área del conocimiento", areas, default=areas)

        generos = sorted(df["GEN_ALU"].dropna().astype(str).unique().tolist())
        generos_sel = st.multiselect("Género", generos, default=generos)

        carreras = sorted(df["NOMB_CARRERA"].dropna().astype(str).unique().tolist())
        carreras_sel = st.multiselect("Carrera (opcional)", carreras, default=[])

        st.divider()
        st.header("Configuración LLM")
        api_key_llm = st.text_input("API Key (Opcional)", type="password", help="Si no ingresas clave, se usará el motor analítico integrado.")

    filtrado = df[df["CAT_PERIODO"].between(rango_anios[0], rango_anios[1], inclusive="both")].copy()
    if areas_sel:
        filtrado = filtrado[filtrado["AREA_CONOCIMIENTO"].astype(str).isin(areas_sel)]
    else:
        filtrado = filtrado.iloc[0:0]
    if generos_sel:
        filtrado = filtrado[filtrado["GEN_ALU"].astype(str).isin(generos_sel)]
    else:
        filtrado = filtrado.iloc[0:0]
    if carreras_sel:
        filtrado = filtrado[filtrado["NOMB_CARRERA"].astype(str).isin(carreras_sel)]

    def generar_interpretacion_llm(titulo_grafico, resumen_datos, descripcion_base):
        with st.expander("🤖 Interpretación con Asistente LLM", expanded=True):
            if st.button("Generar análisis profundo del gráfico con IA", key=f"btn_{titulo_grafico}"):
                with st.spinner("Analizando patrones con IA..."):
                    prompt = f"""
                    Eres un asistente de investigación de tesis en educación superior.
                    Analiza el siguiente gráfico: '{titulo_grafico}'.
                    Contexto analítico: {descripcion_base}.
                    Datos clave obtenidos bajo los filtros actuales:
                    {resumen_datos}
                    
                    Redacta un análisis académico de 2 párrafos concisos:
                    1. Interpretación directa de los datos visibles y patrones dominantes.
                    2. Relevancia de estos hallazgos para la gestión de permanencia y equidad universitaria.
                    """
                    key_usar = api_key_llm or st.secrets.get("OPENAI_API_KEY", None)
                    if key_usar:
                        try:
                            import openai
                            client = openai.OpenAI(api_key=key_usar)
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3,
                                max_tokens=350
                            )
                            st.markdown(response.choices[0].message.content)
                            return
                        except Exception as e:
                            st.caption(f"(Conexión externa no disponible: {e}. Se utilizó el motor descriptivo analítico)")

                    st.markdown(f"""
                    **Análisis Estadístico e Interpretación:**
                    * **Diagnóstico de los datos:** En el gráfico **{titulo_grafico}**, los datos reflejan un comportamiento concentrado. {resumen_datos}.
                    * **Implicancia:** La variabilidad observada en esta distribución entrega evidencia empírica directa para el modelado de retención. Diferencias sistemáticas en este atributo sustentan la necesidad de incluirlo como variable predictora o de control en algoritmos de clasificación supervisada.
                    """)

    paginas = [
        "1. Proporción por Género",
        "2. Evolución Anual por Género",
        "3. Matrícula por Área y Género",
        "4. Distribución por Rango Etario",
        "5. Ranking Top Carreras",
        "6. Carreras en Jornada Vespertina",
        "7. Carreras con Ingreso ≥ 25 Años",
        "8. Distribución de Brecha Temporal",
        "9. Dispersión y Outliers de Rezago",
        "10. Procedencia según Entorno Escolar",
        "11. Brecha según Entorno Escolar",
        "12. Vulnerabilidad Regional (IVM_REGION)",
        "13. Territorio y Comunas de Procedencia",
        "14. Densidad Geográfica de Establecimientos"
    ]

    col_nav1, col_nav2 = st.columns([2, 1])
    with col_nav1:
        pagina_actual = st.selectbox(
            "📑 Selecciona el Gráfico / Página a visualizar:",
            paginas,
            index=0,
            help="Elige cualquier gráfico para ver su visualización individual, descripción metodológica y asistente LLM."
        )
    with col_nav2:
        st.write("")
        st.caption("👈 Cambia de visualización del EDA.")

    st.divider()

    if filtrado.empty and pagina_actual != "1. Proporción por Género":
        st.info("No hay registros para los filtros seleccionados.")
        st.stop()

    if pagina_actual == "1. Proporción por Género":
        st.header("Distribución Global de Género")
        genero = filtrado["GEN_ALU"].fillna("Sin información").value_counts().rename_axis("Género").reset_index(name="Matrículas")
        col_g, col_d = st.columns([1.6, 1])
        with col_g:
            fig = px.pie(
                genero, names="Género", values="Matrículas", hole=0.45,
                title="Proporción de estudiantes matriculados según género",
                color="Género",
                color_discrete_map={"Hombre": "#2b5c8f", "Mujer": "#e74c3c", "Sin información": "#95a5a6"}
            )
            st.plotly_chart(aplicar_estilo(fig, 520), use_container_width=True)
        with col_d:
            st.subheader("Descripción Metodológica")
            desc = "Muestra la proporción acumulada de hombres y mujeres matriculados en la institución. Permite evaluar la paridad de género global y detectar sesgos basales de composición estudiantil."
            st.write(desc)
            resumen = f"Total evaluado: {genero['Matrículas'].sum():,} alumnos. Composición: " + ", ".join([f"{r['Género']}: {r['Matrículas']:,} ({r['Matrículas']/genero['Matrículas'].sum()*100:.1f}%)" for _, r in genero.iterrows()])
            st.info(resumen)
            generar_interpretacion_llm("Proporción Global de Género", resumen, desc)

    elif pagina_actual == "2. Evolución Anual por Género":
        st.header("Evolución Temporal de Matrícula por Género")
        anual_genero = filtrado.groupby(["CAT_PERIODO", "GEN_ALU"]).size().reset_index(name="Matrículas")
        col_g, col_d = st.columns([1.6, 1])
        with col_g:
            fig = px.line(
                anual_genero, x="CAT_PERIODO", y="Matrículas", color="GEN_ALU", markers=True,
                title="Evolución anual de matrículas según género (2015-2025)",
                color_discrete_map={"Hombre": "#2b5c8f", "Mujer": "#e74c3c", "Sin información": "#95a5a6"}
            )
            st.plotly_chart(aplicar_estilo(fig, 520), use_container_width=True)
        with col_d:
            st.subheader("Descripción Metodológica")
            desc = "Analiza las tendencias de crecimiento o contracción de la matrícula desagregada por género a lo largo de las cohortes 2015-2025, evidenciando cambios en la demanda formativa institucional."
            st.write(desc)
            resumen = f"Cohortes analizadas: {anual_genero['CAT_PERIODO'].min()} a {anual_genero['CAT_PERIODO'].max()}. Máximo histórico registrado: {anual_genero['Matrículas'].max():,} matrículas en un período."
            st.info(resumen)
            generar_interpretacion_llm("Evolución Anual por Género", resumen, desc)

    elif pagina_actual == "3. Matrícula por Área y Género":
        st.header("Composición por Área del Conocimiento y Género")
        area_genero = filtrado.groupby(["AREA_CONOCIMIENTO", "GEN_ALU"]).size().reset_index(name="Matrículas")
        col_g, col_d = st.columns([1.6, 1])
        with col_g:
            fig = px.bar(
                area_genero, x="AREA_CONOCIMIENTO", y="Matrículas", color="GEN_ALU", barmode="stack",
                title="Distribución de matrícula por área disciplinar y género",
                color_discrete_map={"Hombre": "#2980b9", "Mujer": "#e84393", "Sin información": "#bdc3c7"}
            )
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(aplicar_estilo(fig, 550), use_container_width=True)
        with col_d:
            st.subheader("Descripción Metodológica")
            desc = "Identifica la segregación disciplinar horizontal. Permite constatar qué áreas del conocimiento presentan masculinización o feminización de la matrícula y su volumen relativo dentro de la UBO."
            st.write(desc)
            top_area = area_genero.groupby("AREA_CONOCIMIENTO")["Matrículas"].sum().idxmax()
            resumen = f"Área con mayor volumen: '{top_area}'. Muestra total distribuida en {area_genero['AREA_CONOCIMIENTO'].nunique()} áreas del conocimiento."
            st.info(resumen)
            generar_interpretacion_llm("Matrícula por Área y Género", resumen, desc)

    elif pagina_actual == "4. Distribución por Rango Etario":
        st.header("Distribución de Rangos Etarios al Ingreso")
        edades = filtrado["RANGO_EDAD"].fillna("Sin información").value_counts().rename_axis("Rango de edad").reset_index(name="Matrículas")
        col_g, col_d = st.columns([1.6, 1])
        with col_g:
            fig = px.bar(
                edades, x="Rango de edad", y="Matrículas", title="Matrículas según rango etario al ingreso",
                color="Matrículas", color_continuous_scale="Plasma", text_auto=True
            )
            fig.update_xaxes(categoryorder="total descending", tickangle=-25)
            st.plotly_chart(aplicar_estilo(fig, 520), use_container_width=True)
        with col_d:
            st.subheader("Descripción Metodológica")
            desc = "Cuantifica la estructura etaria estudiantil. Permite diferenciar entre estudiantes de perfil tradicional (egreso escolar inmediato, 18-20 años) y estudiantes de inserción tardía o adultos que compatibilizan trabajo y estudio."
            st.write(desc)
            resumen = f"Rango etario dominante: '{edades.iloc[0]['Rango de edad']}' con {edades.iloc[0]['Matrículas']:,} estudiantes."
            st.info(resumen)
            generar_interpretacion_llm("Distribución de Rangos Etarios", resumen, desc)

    elif pagina_actual == "5. Ranking Top Carreras":
        st.header("Carreras con Mayor Matrícula Acumulada")
        top_n = st.radio("Cantidad de carreras a desplegar:", [10, 15, 20], horizontal=True, index=1)
        top_carreras = filtrado["NOMB_CARRERA"].value_counts().head(top_n).sort_values().reset_index()
        top_carreras.columns = ["Carrera", "Matrículas"]
        col_g, col_d = st.columns([1.6, 1])
        with col_g:
            fig = px.bar(
                top_carreras, x="Matrículas", y="Carrera", orientation="h",
                title=f"Top {top_n} carreras por volumen de matrícula",
                color="Matrículas", color_continuous_scale="Turbo", text_auto=True
            )
            st.plotly_chart(aplicar_estilo(fig, 560), use_container_width=True)
        with col_d:
            st.subheader("Descripción Metodológica")
            desc = "Diagrama de concentración institucional. Revela qué carreras concentran la mayor masa crítica de la universidad y representan los núcleos operativos de mayor impacto para planes de retención institucional."
            st.write(desc)
            resumen = f"La carrera con mayor concentración es '{top_carreras.iloc[-1]['Carrera']}' con {top_carreras.iloc[-1]['Matrículas']:,} matrículas acumuladas."
            st.info(resumen)
            generar_interpretacion_llm(f"Top {top_n} Carreras", resumen, desc)

    elif pagina_actual == "6. Carreras en Jornada Vespertina":
        st.header("Concentración de Matrícula en Jornada Vespertina")
        vespertina = filtrado[filtrado["JORNADA"].astype(str).str.contains("vespert", case=False, na=False)]
        if vespertina.empty:
            st.warning("No hay registros en jornada vespertina para los filtros seleccionados.")
        else:
            datos = vespertina["NOMB_CARRERA"].value_counts().head(10).sort_values().reset_index()
            datos.columns = ["Carrera", "Matrículas vespertinas"]
            col_g, col_d = st.columns([1.6, 1])
            with col_g:
                fig = px.bar(
                    datos, x="Matrículas vespertinas", y="Carrera", orientation="h",
                    title="Top 10 carreras con mayor matrícula vespertina",
                    color="Matrículas vespertinas", color_continuous_scale="Sunset", text_auto=True
                )
                st.plotly_chart(aplicar_estilo(fig, 520), use_container_width=True)
            with col_d:
                st.subheader("Descripción Metodológica")
                desc = "Caracteriza el perfil nocturno/vespertino. Los estudiantes de esta modalidad suelen presentar mayores cargas laborales y familiares, siendo una variable crítica asociada al riesgo de deserción."
                st.write(desc)
                resumen = f"Total matrículas vespertinas analizadas: {len(vespertina):,}. Programa con mayor volumen nocturno: '{datos.iloc[-1]['Carrera']}'."
                st.info(resumen)
                generar_interpretacion_llm("Carreras en Jornada Vespertina", resumen, desc)

    elif pagina_actual == "7. Carreras con Ingreso ≥ 25 Años":
        st.header("Carreras con Mayor Matrícula de Adultos (≥ 25 Años)")
        edad_minima = pd.to_numeric(filtrado["RANGO_EDAD"].astype(str).str.extract(r"(\d{2})", expand=False), errors="coerce")
        mayores = filtrado[edad_minima >= 25].copy()
        if mayores.empty:
            st.warning("No hay registros de edad ≥ 25 años bajo los filtros seleccionados.")
        else:
            datos = mayores["NOMB_CARRERA"].value_counts().head(10).sort_values().reset_index()
            datos.columns = ["Carrera", "Matrículas ≥ 25 años"]
            col_g, col_d = st.columns([1.6, 1])
            with col_g:
                fig = px.bar(
                    datos, x="Matrículas ≥ 25 años", y="Carrera", orientation="h",
                    title="Top 10 carreras: edad de ingreso ≥ 25 años",
                    color="Matrículas ≥ 25 años", color_continuous_scale="Tealrose", text_auto=True
                )
                st.plotly_chart(aplicar_estilo(fig, 520), use_container_width=True)
            with col_d:
                st.subheader("Descripción Metodológica")
                desc = "Monitorea la presencia de estudiantes no tradicionales según edad. Permite comparar qué programas académicos atraen a estudiantes que retornan a la educación superior tras años fuera del sistema."
                st.write(desc)
                resumen = f"Total alumnos con ingreso a los 25 años o más: {len(mayores):,} ({len(mayores)/len(filtrado)*100:.1f}% de la muestra filtrada)."
                st.info(resumen)
                generar_interpretacion_llm("Carreras con Ingreso ≥ 25 Años", resumen, desc)

    elif pagina_actual == "8. Distribución de Brecha Temporal":
        st.header("Distribución de la Brecha Temporal de Egreso (Rezago)")
        datos = filtrado.dropna(subset=["CAT_PERIODO", "ANIO_EGRESO_MEDIA"]).copy()
        datos["BRECHA_TEMPORAL"] = datos["CAT_PERIODO"] - datos["ANIO_EGRESO_MEDIA"]
        brecha = datos["BRECHA_TEMPORAL"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brecha promedio", f"{brecha.mean():.2f} años")
        c2.metric("Mediana", f"{brecha.median():.0f} años")
        c3.metric("Ingreso directo (0–1 año)", f"{brecha.between(0, 1).mean() * 100:.1f}%")
        c4.metric("Rezago severo (> 3 años)", f"{(brecha > 3).mean() * 100:.1f}%")
        col_g, col_d = st.columns([1.6, 1])
        with col_g:
            fig = px.histogram(
                datos, x="BRECHA_TEMPORAL", nbins=30,
                title="Histograma de frecuencias: Años transcurridos entre 4° Medio y Universidad",
                color_discrete_sequence=["#0984e3"]
            )
            st.plotly_chart(aplicar_estilo(fig, 500), use_container_width=True)
        with col_d:
            st.subheader("Descripción Metodológica")
            desc = "Calcula la variable 'BRECHA_TEMPORAL = CAT_PERIODO - ANIO_EGRESO_MEDIA'. Permite dimensionar la deshabituación académica previa al ingreso universitario, factor fuertemente correlacionado con el rendimiento en primer año."
            st.write(desc)
            resumen = f"Promedio de rezago: {brecha.mean():.2f} años. El {brecha.between(0, 1).mean() * 100:.1f}% ingresa de forma inmediata."
            st.info(resumen)
            generar_interpretacion_llm("Distribución de Brecha Temporal", resumen, desc)

    elif pagina_actual == "9. Dispersión y Outliers de Rezago":
        st.header("Dispersión y Valores Atípicos del Rezago Escolar")
        datos = filtrado.dropna(subset=["CAT_PERIODO", "ANIO_EGRESO_MEDIA"]).copy()
        datos["BRECHA_TEMPORAL"] = datos["CAT_PERIODO"] - datos["ANIO_EGRESO_MEDIA"]
        col_g, col_d = st.columns([1.6, 1])
        with col_g:
            fig = px.box(
                datos, y="BRECHA_TEMPORAL", points="outliers",
                title="Diagrama de Caja y Bigotes: Dispersión y casos extremos de brecha",
                color_discrete_sequence=["#d63031"]
            )
            st.plotly_chart(aplicar_estilo(fig, 500), use_container_width=True)
        with col_d:
            st.subheader("Descripción Metodológica")
            desc = "Presenta la dispersión intercuartílica del rezago temporal y detecta valores atípicos (estudiantes que ingresan tras 15, 20 o más años de haber finalizado la enseñanza media)."
            st.write(desc)
            resumen = f"Mediana de rezago: {datos['BRECHA_TEMPORAL'].median():.1f} años. Rango intercuartílico: Q1={datos['BRECHA_TEMPORAL'].quantile(0.25):.1f}, Q3={datos['BRECHA_TEMPORAL'].quantile(0.75):.1f} años."
            st.info(resumen)
            generar_interpretacion_llm("Dispersión y Outliers de Rezago", resumen, desc)

    elif pagina_actual == "10. Procedencia según Entorno Escolar":
        st.header("Entorno Escolar de Procedencia (Urbano vs Rural)")
        rural = filtrado["RURAL_RBD"].map({0: "Urbano (RURAL_RBD = 0)", 1: "Rural (RURAL_RBD = 1)"}).fillna("Sin información")
        distribucion = rural.value_counts().rename_axis("Tipo de entorno").reset_index(name="Matrículas")
        col_g, col_d = st.columns([1.6, 1])
        with col_g:
            fig = px.pie(
                distribucion, names="Tipo de entorno", values="Matrículas", hole=0.45,
                title="Distribución de procedencia según entorno del establecimiento escolar",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(aplicar_estilo(fig, 500), use_container_width=True)
        with col_d:
            st.subheader("Descripción Metodológica")
            desc = "Mide la proporción de estudiantes que egresaron de establecimientos en entornos rurales frente a zonas urbanas según la clasificación oficial del MINEDUC."
            st.write(desc)
            resumen = ", ".join([f"{r['Tipo de entorno']}: {r['Matrículas']:,} ({r['Matrículas']/distribucion['Matrículas'].sum()*100:.1f}%)" for _, r in distribucion.iterrows()])
            st.info(resumen)
            generar_interpretacion_llm("Procedencia según Entorno Escolar", resumen, desc)

    elif pagina_actual == "11. Brecha según Entorno Escolar":
        st.header("Brecha Temporal Promedio según Entorno Escolar")
        datos = filtrado.dropna(subset=["CAT_PERIODO", "ANIO_EGRESO_MEDIA"]).copy()
        datos["BRECHA_TEMPORAL"] = datos["CAT_PERIODO"] - datos["ANIO_EGRESO_MEDIA"]
        datos["ENTORNO_ESCOLAR"] = datos["RURAL_RBD"].map({0: "Urbano", 1: "Rural"}).fillna("Sin información")
        resumen_df = datos.groupby("ENTORNO_ESCOLAR")["BRECHA_TEMPORAL"].agg(Mediana="median", Promedio="mean", Registros="count").reset_index()
        col_g, col_d = st.columns([1.6, 1])
        with col_g:
            fig = px.bar(
                resumen_df, x="ENTORNO_ESCOLAR", y="Promedio", text_auto=".2f",
                title="Brecha de rezago promedio según entorno escolar (Urbano vs Rural)",
                color="ENTORNO_ESCOLAR", color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(aplicar_estilo(fig, 500), use_container_width=True)
        with col_d:
            st.subheader("Descripción Metodológica")
            desc = "Contrasta si los estudiantes provenientes de colegios rurales tardan más años en incorporarse a la universidad que aquellos de origen urbano, reflejando inequidades de acceso territorial."
            st.write(desc)
            resumen = "Resumen por entorno: " + ", ".join([f"{r['ENTORNO_ESCOLAR']}: Promedio {r['Promedio']:.2f} años" for _, r in resumen_df.iterrows()])
            st.info(resumen)
            generar_interpretacion_llm("Brecha según Entorno Escolar", resumen, desc)

    elif pagina_actual == "12. Vulnerabilidad Regional (IVM_REGION)":
        st.header("Índice de Vulnerabilidad Multidimensional Regional (IVM_REGION)")
        ivm = filtrado.dropna(subset=["IVM_REGION"]).copy()
        if ivm.empty:
            st.warning("No se registran datos válidos de IVM_REGION para el filtro seleccionado.")
        else:
            col_g, col_d = st.columns([1.6, 1])
            with col_g:
                fig = px.box(
                    ivm, x="NOM_REG_RBD_A", y="IVM_REGION", points=False,
                    title="Distribución de IVM_REGION según región de procedencia escolar",
                    color="NOM_REG_RBD_A", color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig.update_xaxes(tickangle=-30)
                st.plotly_chart(aplicar_estilo(fig, 540), use_container_width=True)
            with col_d:
                st.subheader("Descripción Metodológica")
                desc = "El IVM_REGION es el indicador oficial de JUNAEB que mide carencias socioeducativas complejas a nivel regional. Permite evaluar el nivel de vulnerabilidad territorial basal de los estudiantes."
                st.write(desc)
                resumen = f"Media global de IVM: {ivm['IVM_REGION'].mean():.2f}%. Regiones evaluadas: {ivm['NOM_REG_RBD_A'].nunique()}."
                st.info(resumen)
                generar_interpretacion_llm("Vulnerabilidad Regional (IVM_REGION)", resumen, desc)

    elif pagina_actual == "13. Territorio y Comunas de Procedencia":
        st.header("Comunas de Procedencia Escolar")
        comunas = filtrado["NOM_COM_RBD"].fillna("Sin información").value_counts().head(15).sort_values().reset_index()
        comunas.columns = ["Comuna", "Matrículas"]
        col_g, col_d = st.columns([1.6, 1])
        with col_g:
            fig = px.bar(
                comunas, x="Matrículas", y="Comuna", orientation="h", text_auto=True,
                title="Top 15 comunas de egreso escolar con mayor cantidad de matriculados",
                color="Matrículas", color_continuous_scale="Spectral"
            )
            st.plotly_chart(aplicar_estilo(fig, 560), use_container_width=True)
        with col_d:
            st.subheader("Descripción Metodológica")
            desc = "Mapeo comunal de los colegios de egreso de los alumnos. Permite comprobar la zona de influencia territorial directa de la UBO en la Región Metropolitana y otras regiones."
            st.write(desc)
            resumen = f"Comuna con mayor aporte de estudiantes: '{comunas.iloc[-1]['Comuna']}' con {comunas.iloc[-1]['Matrículas']:,} alumnos."
            st.info(resumen)
            generar_interpretacion_llm("Territorio y Comunas de Procedencia", resumen, desc)

    elif pagina_actual == "14. Densidad Geográfica de Establecimientos":
        st.header("Densidad Geoespacial de Establecimientos Escolares")
        opcion_mapa = st.radio("Cobertura del mapa:", ["Todo Chile", "Región Metropolitana"], horizontal=True)
        mapa = filtrado.dropna(subset=["LATITUD", "LONGITUD"]).copy()
        mapa = mapa[mapa["LATITUD"].between(-57, -17) & mapa["LONGITUD"].between(-76, -66)]
        if opcion_mapa == "Región Metropolitana":
            mapa = mapa[mapa["NOM_REG_RBD_A"].astype(str).str.contains(r"\bRM\b|METROPOLITANA", case=False, na=False, regex=True)]
        if len(mapa) > 15000:
            mapa = mapa.sample(15000, random_state=42)
        fig = px.density_mapbox(
            mapa, lat="LATITUD", lon="LONGITUD", radius=10,
            center={"lat": -33.45, "lon": -70.66} if opcion_mapa == "Región Metropolitana" else {"lat": -35.5, "lon": -71.0},
            zoom=8 if opcion_mapa == "Región Metropolitana" else 3.5,
            mapbox_style="open-street-map",
            color_continuous_scale="Rainbow",
            hover_name="NOM_RBD",
            hover_data={"NOM_COM_RBD": True, "NOM_REG_RBD_A": True},
            title=f"Mapa de Calor de Establecimientos Escolares — {opcion_mapa}",
        )
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Nota técnica: El mapa representa la localización geográfica de los colegios de egreso de los estudiantes.")
        desc = f"Mapa de densidad territorial de colegios para {opcion_mapa}. Muestra las zonas geográficas con mayor concentración de egresados matriculados en la universidad."
        resumen = f"Puntos geoespaciales mapeados: {len(mapa):,} establecimientos/matrículas."
        generar_interpretacion_llm("Densidad Geográfica de Establecimientos", resumen, desc)

# =============================================================================
# MÓDULO 2: SISTEMA DE ALERTA TEMPRANA (SAT — COHORTE 2025)
# =============================================================================
else:
    ruta_sat = buscar_sat()
    
    with st.sidebar:
        st.header("Fuente de Datos SAT 2025")
        archivo_subido_sat = st.file_uploader("Cargar Scoring 2025 manual", type=["csv", "xlsx"])

    if archivo_subido_sat is not None:
        df_sat = pd.read_excel(archivo_subido_sat) if archivo_subido_sat.name.endswith(".xlsx") else pd.read_csv(archivo_subido_sat)
    elif ruta_sat is not None:
        df_sat = cargar_sat(ruta_sat)
    else:
        st.warning("⚠️ No se encontró el archivo de Scoring 2025 en data/. Sube 'Scoring_Cohorte2025_Riesgo_Desercion.csv' en la carpeta data/ de GitHub.")
        st.stop()

    df_sat.columns = [str(c).strip().upper() for c in df_sat.columns]

    if "PROB_DESERCION" not in df_sat.columns and "PROBABILIDAD_DESERCION" in df_sat.columns:
        df_sat["PROB_DESERCION"] = df_sat["PROBABILIDAD_DESERCION"]
    if "TRAMO_RIESGO" not in df_sat.columns and "NIVEL_RIESGO" in df_sat.columns:
        df_sat["TRAMO_RIESGO"] = df_sat["NIVEL_RIESGO"]
    if "TRAMO_RIESGO" not in df_sat.columns and "PROB_DESERCION" in df_sat.columns:
        df_sat["TRAMO_RIESGO"] = np.select([df_sat["PROB_DESERCION"] >= 0.25, df_sat["PROB_DESERCION"] >= 0.12], ["Alto", "Medio"], default="Bajo")

    with st.sidebar:
        st.divider()
        st.header("Filtros del SAT")
        tramos_disp = sorted(df_sat["TRAMO_RIESGO"].dropna().unique().tolist()) if "TRAMO_RIESGO" in df_sat.columns else []
        tramos_sel = st.multiselect("Nivel de Riesgo:", tramos_disp, default=tramos_disp)

        carreras_sat = sorted(df_sat["NOMB_CARRERA"].dropna().unique().tolist()) if "NOMB_CARRERA" in df_sat.columns else []
        carreras_sat_sel = st.multiselect("Carrera:", carreras_sat, default=[])

        jornadas_sat = sorted(df_sat["JORNADA"].dropna().unique().tolist()) if "JORNADA" in df_sat.columns else []
        jornadas_sat_sel = st.multiselect("Jornada:", jornadas_sat, default=jornadas_sat)

    filtrado_sat = df_sat.copy()
    if tramos_sel:
        filtrado_sat = filtrado_sat[filtrado_sat["TRAMO_RIESGO"].isin(tramos_sel)]
    if carreras_sat_sel:
        filtrado_sat = filtrado_sat[filtrado_sat["NOMB_CARRERA"].isin(carreras_sat_sel)]
    if jornadas_sat_sel:
        filtrado_sat = filtrado_sat[filtrado_sat["JORNADA"].isin(jornadas_sat_sel)]

    subpestanas = st.tabs([
        "📈 Resumen Ejecutivo de Riesgo",
        "📋 Semáforo de Casos y Descarga",
        "👤 Ficha Individual del Estudiante (SHAP)",
        "🎯 Acciones e Intervenciones Sugeridas"
    ])

    with subpestanas[0]:
        st.subheader("Indicadores de Alerta Temprana — Cohorte 2025")
        c1, c2, c3, c4 = st.columns(4)
        total_sat = len(filtrado_sat)
        n_alto = (filtrado_sat["TRAMO_RIESGO"] == "Alto").sum() if "TRAMO_RIESGO" in filtrado_sat.columns else 0
        n_medio = (filtrado_sat["TRAMO_RIESGO"] == "Medio").sum() if "TRAMO_RIESGO" in filtrado_sat.columns else 0
        prob_prom = filtrado_sat["PROB_DESERCION"].mean() if "PROB_DESERCION" in filtrado_sat.columns else 0.0

        c1.metric("Estudiantes Evaluados", f"{total_sat:,}")
        c2.metric("Tasa Deserción Esperada", f"{prob_prom:.2%}")
        c3.metric("Riesgo Alto (Críticos)", f"{n_alto:,}", f"{n_alto/max(total_sat,1):.1%}", delta_color="inverse")
        c4.metric("Riesgo Medio", f"{n_medio:,}", f"{n_medio/max(total_sat,1):.1%}", delta_color="off")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if "TRAMO_RIESGO" in filtrado_sat.columns:
                conteo_tr = filtrado_sat["TRAMO_RIESGO"].value_counts().reindex(["Alto", "Medio", "Bajo"]).fillna(0).reset_index()
                conteo_tr.columns = ["Nivel de Riesgo", "Estudiantes"]
                fig_pie_sat = px.pie(conteo_tr, names="Nivel de Riesgo", values="Estudiantes", hole=0.45,
                                     title="Composición de la Cohorte 2025 según Nivel de Riesgo",
                                     color="Nivel de Riesgo", color_discrete_map={"Alto": "#d63031", "Medio": "#fdcb6e", "Bajo": "#00b894"})
                st.plotly_chart(fig_pie_sat, use_container_width=True)
        with col2:
            if "NOMB_CARRERA" in filtrado_sat.columns and "TRAMO_RIESGO" in filtrado_sat.columns:
                carr_alto = filtrado_sat[filtrado_sat["TRAMO_RIESGO"] == "Alto"]["NOMB_CARRERA"].value_counts().head(10).reset_index()
                carr_alto.columns = ["Carrera", "Estudiantes en Riesgo Alto"]
                fig_carr = px.bar(carr_alto, x="Estudiantes en Riesgo Alto", y="Carrera", orientation="h",
                                  title="Top 10 Carreras con Mayor Masa Crítica de Riesgo Alto",
                                  color="Estudiantes en Riesgo Alto", color_continuous_scale="Reds")
                fig_carr.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_carr, use_container_width=True)

    with subpestanas[1]:
        st.subheader("Semáforo Nominal y Priorización de Alumnos")
        cols_sat = [c for c in ["MRUN", "NOMB_CARRERA", "JORNADA", "TRAMO_RIESGO", "PROB_DESERCION", "PROM_NOTAS_ALU", "BRECHA_REZAGO_ANIOS", "FACTOR_RIESGO_1", "FACTOR_RIESGO_2"] if c in filtrado_sat.columns]
        df_display = filtrado_sat[cols_sat].sort_values(by="PROB_DESERCION", ascending=False).reset_index(drop=True)
        st.dataframe(df_display.style.format({"PROB_DESERCION": "{:.2%}", "PROM_NOTAS_ALU": "{:.2f}"}), use_container_width=True)

        csv_sat = df_display.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📥 Descargar Listado Priorizado (CSV)",
            data=csv_sat,
            file_name="Priorizacion_Alerta_Temprana_2025.csv",
            mime="text/csv"
        )

    with subpestanas[2]:
        st.subheader("Ficha de Alerta Individual y Factores SHAP")
        if "MRUN" in df_sat.columns:
            sel_mrun = st.selectbox("Buscar por MRUN:", df_sat["MRUN"].unique())
            alumno = df_sat[df_sat["MRUN"] == sel_mrun].iloc[0]

            cf1, cf2 = st.columns([1, 1.5])
            with cf1:
                st.markdown(f"### Estudiante: `{sel_mrun}`")
                st.write(f"**Carrera:** {alumno.get('NOMB_CARRERA', 'S/I')}")
                st.write(f"**Jornada:** {alumno.get('JORNADA', 'S/I')}")
                st.write(f"**NEM de Egreso:** {alumno.get('PROM_NOTAS_ALU', 'S/I')}")
                st.write(f"**Rezago Escolar:** {alumno.get('BRECHA_REZAGO_ANIOS', 0)} años")
                
                pr = alumno.get("PROB_DESERCION", 0.0)
                tr = alumno.get("TRAMO_RIESGO", "Bajo")
                color_icono = "🔴" if tr == "Alto" else ("🟡" if tr == "Medio" else "🟢")
                st.markdown(f"#### Estado: {color_icono} **Riesgo {tr}** ({pr:.2%})")

            with cf2:
                st.markdown("#### Factores de Riesgo Determinantes (SHAP Local):")
                st.info(f"1. **{alumno.get('FACTOR_RIESGO_1', 'No reportado')}**")
                st.info(f"2. **{alumno.get('FACTOR_RIESGO_2', 'No reportado')}**")
                st.info(f"3. **{alumno.get('FACTOR_RIESGO_3', 'No reportado')}**")
                st.caption("Los factores identifican las variables que aumentaron la probabilidad de deserción de este estudiante según el modelo de Machine Learning.")

    with subpestanas[3]:
        st.subheader("Catálogo de Intervención Tutorial (UBO)")
        ci1, ci2, ci3 = st.columns(3)
        with ci1:
            st.error("📘 Alerta Académica (NEM / IVE / IVM)")
            st.write("Derivar de forma temprana a tutorías de ciencias básicas, matemáticas y talleres de lectoescritura.")
        with ci2:
            st.warning("⏱️ Alerta de Trayectoria (Rezago / Vespertino)")
            st.write("Acompañamiento en gestión de carga académica, compatibilidad laboral y adaptación a la vida universitaria.")
        with ci3:
            st.success("🌱 Alerta Territorial")
            st.write("Orientación con la Dirección de Asuntos Estudiantiles (DAE) para becas de mantención, transporte y redes de apoyo.")

# ============================================================
# PIE DE PÁGINA
# ============================================================
st.divider()
st.caption("Plataforma de Tesis · Universidad Bernardo O'Higgins · Facultad de Ingeniería, Ciencia y Tecnología.")
