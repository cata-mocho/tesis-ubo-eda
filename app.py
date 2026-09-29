from pathlib import Path
import re

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# missingno se utiliza para la matriz de integridad de la muestra.
try:
    import missingno as msno
    import matplotlib.pyplot as plt
except ImportError:
    msno = None
    plt = None


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
COLUMNAS_ESCOLARES = [
    "RBD", "IVM_REGION", "LATITUD", "LONGITUD", "NOM_RBD",
    "NOM_COM_RBD", "NOM_DEPROV_RBD", "RURAL_RBD",
]


# ============================================================
# CARGA DE DATOS
# ============================================================
def buscar_parquet():
    """Busca un parquet en data/ o en la raíz del repositorio."""
    candidatos = sorted(Path("data").glob("*.parquet")) if Path("data").exists() else []
    candidatos += sorted(Path(".").glob("*.parquet"))
    return candidatos[0] if candidatos else None


@st.cache_data(show_spinner="Leyendo archivo Parquet...")
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
    st.caption("Si no cargas un archivo aquí, la aplicación buscará uno en data/ o en la raíz del repositorio.")

if archivo_subido is not None:
    try:
        df = pd.read_parquet(archivo_subido, engine="pyarrow")
        fuente = archivo_subido.name
    except Exception as exc:
        st.error(f"No se pudo leer el archivo Parquet: {exc}")
        st.stop()
elif ruta_parquet is not None:
    try:
        df = cargar_parquet(str(ruta_parquet))
        fuente = str(ruta_parquet)
    except Exception as exc:
        st.error(f"No se pudo leer {ruta_parquet}: {exc}")
        st.stop()
else:
    st.warning("No se encontró ningún archivo .parquet. Guarda tu base en data/ o súbela desde el panel lateral.")
    st.code("tesis-ubo-eda/\n├── app.py\n├── requirements.txt\n└── data/\n    └── Tablon_Descriptivo_UBO_79026.parquet")
    st.stop()

# Normalizar nombres y tipos sin modificar el archivo original.
df = limpiar_nombres_columnas(df)
for col in ["CAT_PERIODO", "ANIO_EGRESO_MEDIA", "RURAL_RBD", "LATITUD", "LONGITUD", "IVM_REGION"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

faltan = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
if faltan:
    st.error("El archivo no contiene todas las columnas esperadas por este dashboard.")
    st.write("Columnas faltantes:", faltan)
    st.write("Columnas encontradas:", df.columns.tolist())
    st.stop()

st.sidebar.success(f"Fuente cargada: {fuente}")
st.sidebar.metric("Registros disponibles", f"{len(df):,}".replace(",", "."))

# ============================================================
# FILTROS GLOBALES
# ============================================================
with st.sidebar:
    st.divider()
    st.header("Filtros del dashboard")
    anios = sorted(df["CAT_PERIODO"].dropna().astype(int).unique().tolist())
    if anios:
        rango_anios = st.slider("Período de matrícula", min_value=int(min(anios)), max_value=int(max(anios)), value=(int(min(anios)), int(max(anios))))
    else:
        rango_anios = (0, 9999)

    areas = sorted(df["AREA_CONOCIMIENTO"].dropna().astype(str).unique().tolist())
    areas_sel = st.multiselect("Área del conocimiento", areas, default=areas)

    generos = sorted(df["GEN_ALU"].dropna().astype(str).unique().tolist())
    generos_sel = st.multiselect("Género", generos, default=generos)

    carreras = sorted(df["NOMB_CARRERA"].dropna().astype(str).unique().tolist())
    carreras_sel = st.multiselect("Carrera (opcional)", carreras, default=[])

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


def aplicar_estilo(fig, altura=420):
    fig.update_layout(
        height=altura,
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(family="Arial", size=12),
        legend_title_text="",
    )
    return fig


def mostrar_sin_datos():
    st.info("No hay registros para los filtros seleccionados. Modifica los filtros del panel lateral.")


# ============================================================
# NAVEGACIÓN MEDIANTE PESTAÑAS (TABS)
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 1. Integridad de los datos",
    "👥 2. Perfil demográfico e institucional",
    "🎒 3. Trayectoria escolar y brecha temporal",
    "📍 4. Vulnerabilidad y territorio"
])

# ============================================================
# PESTAÑA 1: DATA HEALTH CHECK
# ============================================================
with tab1:
    st.header("1. Integridad y diagnóstico de la muestra")
    st.write("Esta sección utiliza la base completa, sin aplicar los filtros demográficos del panel lateral.")

    n_registros = len(df)
    n_estudiantes = df["MRUN"].nunique(dropna=True)
    n_periodos = df["CAT_PERIODO"].nunique(dropna=True)
    n_sin_mrun = int(df["MRUN"].isna().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eventos de matrícula", f"{n_registros:,}".replace(",", "."))
    c2.metric("Estudiantes únicos (MRUN)", f"{n_estudiantes:,}".replace(",", "."))
    c3.metric("Períodos distintos", n_periodos)
    c4.metric("MRUN faltantes", n_sin_mrun)

    if n_registros != 79026:
        st.info(f"La base cargada contiene {n_registros:,} registros. El valor de referencia indicado para la tesis es 79.026; verifica que hayas cargado el archivo completo.".replace(",", "."))

    st.subheader("Cobertura de variables y valores faltantes")
    faltantes = pd.DataFrame({
        "Variable": df.columns,
        "Valores faltantes": [int(df[c].isna().sum()) for c in df.columns],
        "% faltante": [float(df[c].isna().mean() * 100) for c in df.columns],
    }).sort_values("% faltante", ascending=False)
    st.dataframe(faltantes.style.format({"% faltante": "{:.2f}%"}), use_container_width=True, hide_index=True)

    st.subheader("Matriz de valores faltantes — variables escolares")
    columnas_presentes = [c for c in COLUMNAS_ESCOLARES if c in df.columns]
    if msno is not None and plt is not None:
        muestra = df[columnas_presentes].sample(min(3000, len(df)), random_state=42) if len(df) else df[columnas_presentes]
        fig, ax = plt.subplots(figsize=(12, 4))
        msno.matrix(muestra, ax=ax, sparkline=False, labels=True, color=(0.18, 0.45, 0.71))
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    else:
        st.warning("No está instalada la librería missingno. Instala las dependencias de requirements.txt para mostrar esta matriz.")

    st.subheader("Registros por año")
    por_anio = df.groupby("CAT_PERIODO", dropna=False).size().reset_index(name="Registros").sort_values("CAT_PERIODO")
    fig = px.bar(
        por_anio, 
        x="CAT_PERIODO", 
        y="Registros", 
        text_auto=True, 
        title="Eventos de matrícula por período", 
        color="Registros",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(aplicar_estilo(fig), use_container_width=True)

# ============================================================
# PESTAÑA 2: DEMOGRAFÍA E INSTITUCIÓN
# ============================================================
with tab2:
    st.header("2. Caracterización demográfica e institucional")
    if filtrado.empty:
        mostrar_sin_datos()
    else:
        st.subheader("Distribución global de género")
        genero = filtrado["GEN_ALU"].fillna("Sin información").value_counts().rename_axis("Género").reset_index(name="Matrículas")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(
                genero, 
                names="Género", 
                values="Matrículas", 
                hole=0.45, 
                title="Proporción por género",
                color="Género",
                color_discrete_map={"Hombre": "#2b5c8f", "Mujer": "#e74c3c", "Sin información": "#95a5a6"}
            )
            st.plotly_chart(aplicar_estilo(fig), use_container_width=True)
        with c2:
            anual_genero = filtrado.groupby(["CAT_PERIODO", "GEN_ALU"]).size().reset_index(name="Matrículas")
            fig = px.line(
                anual_genero, 
                x="CAT_PERIODO", 
                y="Matrículas", 
                color="GEN_ALU", 
                markers=True, 
                title="Evolución anual por género",
                color_discrete_map={"Hombre": "#2b5c8f", "Mujer": "#e74c3c", "Sin información": "#95a5a6"}
            )
            st.plotly_chart(aplicar_estilo(fig), use_container_width=True)

        st.subheader("Composición por área del conocimiento")
        area_genero = filtrado.groupby(["AREA_CONOCIMIENTO", "GEN_ALU"]).size().reset_index(name="Matrículas")
        fig = px.bar(
            area_genero, 
            x="AREA_CONOCIMIENTO", 
            y="Matrículas", 
            color="GEN_ALU", 
            barmode="stack", 
            title="Matrícula por área y género",
            color_discrete_map={"Hombre": "#2980b9", "Mujer": "#e84393", "Sin información": "#bdc3c7"}
        )
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(aplicar_estilo(fig, 500), use_container_width=True)

        st.subheader("Distribución de rangos etarios")
        edades = filtrado["RANGO_EDAD"].fillna("Sin información").value_counts().rename_axis("Rango de edad").reset_index(name="Matrículas")
        fig = px.bar(
            edades, 
            x="Rango de edad", 
            y="Matrículas", 
            title="Matrículas por rango etario", 
            color="Matrículas",
            color_continuous_scale="Plasma",
            text_auto=True
        )
        fig.update_xaxes(categoryorder="total descending", tickangle=-25)
        st.plotly_chart(aplicar_estilo(fig, 480), use_container_width=True)

        st.subheader("Carreras con mayor matrícula acumulada")
        top_n = st.radio("Cantidad de carreras", [10, 15], horizontal=True, index=0)
        top_carreras = filtrado["NOMB_CARRERA"].value_counts().head(top_n).sort_values().reset_index()
        top_carreras.columns = ["Carrera", "Matrículas"]
        fig = px.bar(
            top_carreras, 
            x="Matrículas", 
            y="Carrera", 
            orientation="h", 
            title=f"Top {top_n} carreras", 
            color="Matrículas",
            color_continuous_scale="Turbo",
            text_auto=True
        )
        st.plotly_chart(aplicar_estilo(fig, 520), use_container_width=True)

        st.subheader("Jornada vespertina y rangos de mayor edad")
        edad_minima = filtrado["RANGO_EDAD"].astype(str).str.extract(r"(\d{2})", expand=False)
        edad_minima = pd.to_numeric(edad_minima, errors="coerce")
        mayores = filtrado[edad_minima >= 25].copy()
        vespertina = filtrado[filtrado["JORNADA"].astype(str).str.contains("vespert", case=False, na=False)]
        c1, c2 = st.columns(2)
        with c1:
            datos = vespertina["NOMB_CARRERA"].value_counts().head(10).sort_values().reset_index()
            datos.columns = ["Carrera", "Matrículas vespertinas"]
            fig = px.bar(
                datos, 
                x="Matrículas vespertinas", 
                y="Carrera", 
                orientation="h", 
                title="Top 10 carreras: jornada vespertina", 
                color="Matrículas vespertinas",
                color_continuous_scale="Sunset",
                text_auto=True
            )
            st.plotly_chart(aplicar_estilo(fig, 460), use_container_width=True)
        with c2:
            datos = mayores["NOMB_CARRERA"].value_counts().head(10).sort_values().reset_index()
            datos.columns = ["Carrera", "Matrículas de 25 años o más"]
            fig = px.bar(
                datos, 
                x="Matrículas de 25 años o más", 
                y="Carrera", 
                orientation="h", 
                title="Top 10 carreras: edad de ingreso ≥ 25 años", 
                color="Matrículas de 25 años o más",
                color_continuous_scale="Tealrose",
                text_auto=True
            )
            st.plotly_chart(aplicar_estilo(fig, 460), use_container_width=True)

# ============================================================
# PESTAÑA 3: TRAYECTORIA ESCOLAR Y BRECHA TEMPORAL
# ============================================================
with tab3:
    st.header("3. Trayectoria escolar y brecha temporal")
    datos = filtrado.dropna(subset=["CAT_PERIODO", "ANIO_EGRESO_MEDIA"]).copy()
    datos["BRECHA_TEMPORAL"] = datos["CAT_PERIODO"] - datos["ANIO_EGRESO_MEDIA"]
    if datos.empty:
        mostrar_sin_datos()
    else:
        brecha = datos["BRECHA_TEMPORAL"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brecha promedio", f"{brecha.mean():.2f} años")
        c2.metric("Mediana", f"{brecha.median():.0f} años")
        c3.metric("Ingreso 0–1 año", f"{brecha.between(0, 1).mean() * 100:.1f}%")
        c4.metric("Rezago > 3 años", f"{(brecha > 3).mean() * 100:.1f}%")

        st.caption("Brecha temporal = CAT_PERIODO − ANIO_EGRESO_MEDIA. Los valores negativos se conservan para detectar posibles inconsistencias o casos que requieren revisión.")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                datos, 
                x="BRECHA_TEMPORAL", 
                nbins=30, 
                title="Distribución de la brecha temporal", 
                color_discrete_sequence=["#0984e3"]
            )
            st.plotly_chart(aplicar_estilo(fig), use_container_width=True)
        with c2:
            fig = px.box(
                datos, 
                y="BRECHA_TEMPORAL", 
                points="outliers", 
                title="Boxplot de la brecha temporal", 
                color_discrete_sequence=["#d63031"]
            )
            st.plotly_chart(aplicar_estilo(fig), use_container_width=True)

        st.subheader("Entorno escolar: urbano y rural")
        rural = datos["RURAL_RBD"].map({0: "Urbano (RURAL_RBD = 0)", 1: "Rural (RURAL_RBD = 1)"}).fillna("Sin información / otro código")
        distribucion = rural.value_counts().rename_axis("Tipo de entorno").reset_index(name="Matrículas")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(
                distribucion, 
                names="Tipo de entorno", 
                values="Matrículas", 
                hole=0.4, 
                title="Procedencia según entorno escolar",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(aplicar_estilo(fig), use_container_width=True)
        with c2:
            datos["ENTORNO_ESCOLAR"] = rural
            resumen = datos.groupby("ENTORNO_ESCOLAR")["BRECHA_TEMPORAL"].agg(Mediana="median", Promedio="mean", Registros="count").reset_index()
            fig = px.bar(
                resumen, 
                x="ENTORNO_ESCOLAR", 
                y="Promedio", 
                title="Brecha promedio según entorno escolar", 
                color="ENTORNO_ESCOLAR",
                color_discrete_sequence=px.colors.qualitative.Bold, 
                text_auto=".2f"
            )
            st.plotly_chart(aplicar_estilo(fig), use_container_width=True)
        st.dataframe(resumen, use_container_width=True, hide_index=True)

# ============================================================
# PESTAÑA 4: VULNERABILIDAD Y TERRITORIO
# ============================================================
with tab4:
    st.header("4. Vulnerabilidad y contexto territorial")
    st.subheader("Índice de Vulnerabilidad Multidimensional (IVM_REGION)")
    ivm = filtrado.dropna(subset=["IVM_REGION"]).copy()
    if ivm.empty:
        st.warning("No hay valores de IVM_REGION para los filtros seleccionados.")
    else:
        resumen_ivm = ivm.groupby("NOM_REG_RBD_A")["IVM_REGION"].agg(
            Registros="count", Media="mean", Mediana="median", Desviación_estándar="std",
            Mínimo="min", P25=lambda x: x.quantile(0.25), P75=lambda x: x.quantile(0.75), Máximo="max"
        ).reset_index().sort_values("Media", ascending=False)
        st.dataframe(resumen_ivm.style.format({c: "{:.2f}" for c in ["Media", "Mediana", "Desviación_estándar", "Mínimo", "P25", "P75", "Máximo"]}), use_container_width=True, hide_index=True)
        fig = px.box(
            ivm, 
            x="NOM_REG_RBD_A", 
            y="IVM_REGION", 
            points=False, 
            title="Distribución de IVM_REGION por región", 
            color="NOM_REG_RBD_A",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(aplicar_estilo(fig, 500), use_container_width=True)

    st.subheader("Comunas y provincias de procedencia")
    c1, c2 = st.columns(2)
    with c1:
        comunas = filtrado["NOM_COM_RBD"].fillna("Sin información").value_counts().head(15).sort_values().reset_index()
        comunas.columns = ["Comuna", "Matrículas"]
        fig = px.bar(
            comunas, 
            x="Matrículas", 
            y="Comuna", 
            orientation="h", 
            title="Top 15 comunas", 
            color="Matrículas",
            color_continuous_scale="Spectral",
            text_auto=True
        )
        st.plotly_chart(aplicar_estilo(fig, 520), use_container_width=True)
    with c2:
        provincias = filtrado["NOM_DEPROV_RBD"].fillna("Sin información").value_counts().head(15).sort_values().reset_index()
        provincias.columns = ["Provincia / DEPROV", "Matrículas"]
        fig = px.bar(
            provincias, 
            x="Matrículas", 
            y="Provincia / DEPROV", 
            orientation="h", 
            title="Top 15 provincias / DEPROV", 
            color="Matrículas",
            color_continuous_scale="Viridis",
            text_auto=True
        )
        st.plotly_chart(aplicar_estilo(fig, 520), use_container_width=True)

    st.subheader("Densidad geográfica de establecimientos escolares")
    opcion_mapa = st.radio("Cobertura del mapa", ["Todo Chile", "Región Metropolitana"], horizontal=True)
    mapa = filtrado.dropna(subset=["LATITUD", "LONGITUD"]).copy()
    # Excluir coordenadas fuera de rangos geográficos válidos.
    mapa = mapa[mapa["LATITUD"].between(-57, -17) & mapa["LONGITUD"].between(-76, -66)]
    if opcion_mapa == "Región Metropolitana":
        mapa = mapa[mapa["NOM_REG_RBD_A"].astype(str).str.contains(r"\bRM\b|METROPOLITANA", case=False, na=False, regex=True)]

    if mapa.empty:
        st.info("No existen coordenadas válidas para mostrar con los filtros seleccionados.")
    else:
        if len(mapa) > 15000:
            mapa = mapa.sample(15000, random_state=42)
        fig = px.density_mapbox(
            mapa, lat="LATITUD", lon="LONGITUD", radius=10,
            center={"lat": -33.45, "lon": -70.66} if opcion_mapa == "Región Metropolitana" else {"lat": -35.5, "lon": -71.0},
            zoom=7 if opcion_mapa == "Región Metropolitana" else 3.2,
            mapbox_style="open-street-map",
            color_continuous_scale="Rainbow",
            hover_name="NOM_RBD",
            hover_data={"NOM_COM_RBD": True, "NOM_REG_RBD_A": True, "LATITUD": ":.4f", "LONGITUD": ":.4f"},
            title=f"Densidad de colegios de procedencia — {opcion_mapa}",
        )
        fig.update_layout(height=650, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("El mapa representa la ubicación de los establecimientos escolares asociados a los registros de matrícula, no la residencia de los estudiantes. La densidad puede contar un mismo colegio varias veces si aparece en múltiples matrículas.")

# ============================================================
# PIE DE PÁGINA
# ============================================================
st.divider()
st.caption("EDA de tesis · Universidad Bernardo O’Higgins · Fuente: base de matrícula y variables de contexto escolar proporcionada para el análisis.")
