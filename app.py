from pathlib import Path
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="EDA | Matrícula UBO 2015–2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Análisis exploratorio de datos — Matrícula UBO")
st.caption("Caracterización demográfica, trayectoria escolar y contexto territorial · 2015–2025")

COLUMNAS_ESPERADAS = [
    "MRUN", "CAT_PERIODO", "GEN_ALU", "JORNADA", "RANGO_EDAD",
    "NOMB_CARRERA", "AREA_CONOCIMIENTO", "ANIO_EGRESO_MEDIA", "RBD",
    "NOM_REG_RBD_A", "NOM_COM_RBD", "NOM_DEPROV_RBD", "LATITUD",
    "LONGITUD", "NOM_RBD", "RURAL_RBD", "IVM_REGION",
]

# ============================================================
# CARGA DE DATOS
# ============================================================
def buscar_parquet():
    candidatos = sorted(Path("data").glob("*.parquet")) if Path("data").exists() else []
    candidatos += sorted(Path(".").glob("*.parquet"))
    return candidatos[0] if candidatos else None

@st.cache_data(show_spinner="Cargando base de datos...")
def cargar_parquet(ruta: str) -> pd.DataFrame:
    return pd.read_parquet(ruta, engine="pyarrow")

def limpiar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
    return df

ruta_parquet = buscar_parquet()

with st.sidebar:
    st.header("Fuente de datos")
    archivo_subido = st.file_uploader("Cargar un archivo .parquet", type=["parquet"])

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

# ============================================================
# FILTROS GLOBALES
# ============================================================
with st.sidebar:
    st.divider()
    st.header("Filtros del dashboard")
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
    api_key_llm = st.text_input("API Key (Opcional)", type="password", help="Si no ingresas una clave, se usará el motor descriptivo analítico integrado.")

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
# FUNCIÓN ASISTENTE LLM
# ============================================================
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
                # Si el usuario configuró una clave OpenAI en secrets o input
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

                # Motor de interpretación analítica integrado (fallback dinámico)
                st.markdown(f"""
                **Análisis Estadístico e Interpretación:**
                * **Diagnóstico de los datos:** En el gráfico **{titulo_grafico}**, los datos reflejan un comportamiento concentrado. {resumen_datos}.
                * **Implicancia:** La variabilidad observada en esta distribución entrega evidencia empírica directa para el modelado de retención. Diferencias sistemáticas en este atributo sustentan la necesidad de incluirlo como variable predictora o de control en algoritmos de clasificación supervisada.
                """)

# ============================================================
# NAVEGACIÓN DIRECTA VISIBLE EN LA PÁGINA PRINCIPAL
# ============================================================
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

# Selector destacado en el centro de la página (no en la barra lateral oculta)
col_nav1, col_nav2 = st.columns([2, 1])
with col_nav1:
    pagina_actual = st.selectbox(
        "📑 Selecciona el Gráfico / Página a visualizar:",
        paginas,
        index=0,
        help="Elige cualquier gráfico para ver su visualización individual, descripción metodológica y asistente LLM."
    )
with col_nav2:
    st.write("") # Espacio estético
    st.caption("👈 Cambia aquí de gráfico en cualquier momento.")

st.divider()

if filtrado.empty and pagina_actual != "1. Proporción por Género":
    st.info("No hay registros para los filtros seleccionados.")
    st.stop()

# ============================================================
# PÁGINA 1: GÉNERO (PIE)
# ============================================================
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

# ============================================================
# PÁGINA 2: EVOLUCIÓN GÉNERO (LINE)
# ============================================================
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

# ============================================================
# PÁGINA 3: ÁREA Y GÉNERO (BAR STACK)
# ============================================================
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

# ============================================================
# PÁGINA 4: EDAD (BAR)
# ============================================================
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

# ============================================================
# PÁGINA 5: TOP CARRERAS (BAR H)
# ============================================================
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

# ============================================================
# PÁGINA 6: JORNADA VESPERTINA (BAR H)
# ============================================================
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

# ============================================================
# PÁGINA 7: MAYORES DE 25 AÑOS (BAR H)
# ============================================================
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

# ============================================================
# PÁGINA 8: HISTOGRAMA DE BRECHA TEMPORAL
# ============================================================
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

# ============================================================
# PÁGINA 9: BOXPLOT DE BRECHA TEMPORAL
# ============================================================
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

# ============================================================
# PÁGINA 10: ENTORNO ESCOLAR (PIE)
# ============================================================
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

# ============================================================
# PÁGINA 11: BRECHA SEGÚN ENTORNO ESCOLAR (BAR)
# ============================================================
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

# ============================================================
# PÁGINA 12: IVM REGIONAL (BOXPLOT)
# ============================================================
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

# ============================================================
# PÁGINA 13: COMUNAS Y PROVINCIAS (BAR H)
# ============================================================
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

# ============================================================
# PÁGINA 14: MAPA DE DENSIDAD
# ============================================================
elif pagina_actual == "14. Densidad Geográfica de Establecimientos":
    st.header("Densidad Geoespacial de Establecimientos Escolares")
    opcion_mapa = st.radio("Cobertura del mapa:", ["Todo Chile", "Región Metropolitana"], horizontal=True)
    mapa = filtrado.dropna(subset=["LATITUD", "LONGITUD"]).copy()
    mapa = mapa[mapa["LATITUD"].between(-57, -17) & mapa["LONGITUD"].between(-76, -66)]
    if opcion_mapa == "Región Metropolitana":
        mapa = mapa[mapa["NOM_REG_RBD_A"].astype(str).str.contains(r"\bRM\b|METROPOLITANA", case=False, na=False, regex=True)]

    if mapa.empty:
        st.info("No hay coordenadas disponibles bajo los filtros seleccionados.")
    else:
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

# ============================================================
# PIE DE PÁGINA
# ============================================================
st.divider()
st.caption("Dashboard de Tesis · Universidad Bernardo O’Higgins · Facultad de Ingeniería, Ciencia y Tecnología.")
