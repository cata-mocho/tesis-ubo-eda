import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(
    page_title="EDA Tesis UBO - Retención y Trayectoria Estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 1. CARGA DE DATOS ROBUSTA CON CACHE
# -------------------------------------------------------------
@st.cache_data
def cargar_datos():
    rutas = [
        "Tablon_Maestro_Tesis_UBO.parquet",
        "data/Tablon_Maestro_Tesis_UBO.parquet",
        "Tablon_Maestro_Tesis_UBO.xlsx",
        "data/Tablon_Maestro_Tesis_UBO.xlsx"
    ]
    archivo = next((r for r in rutas if os.path.exists(r)), None)
    if not archivo:
        raise FileNotFoundError("No se encontró el archivo de datos ni en raíz ni en 'data/'.")
    
    df = pd.read_parquet(archivo) if archivo.endswith('.parquet') else pd.read_excel(archivo)
    df.columns = df.columns.str.strip().str.upper()

    # Normalización de Género
    col_gen = [c for c in df.columns if 'GEN' in c or 'SEXO' in c]
    if col_gen:
        df['GENERO_TXT'] = df[col_gen[0]].replace({1: 'Hombre', 2: 'Mujer', '1': 'Hombre', '2': 'Mujer'}).fillna('Sin Info')
    else:
        df['GENERO_TXT'] = 'Sin Info'

    # Normalización de Retención (0: Deserta, 1: Permanece)
    col_ret = [c for c in df.columns if 'RETIENE' in c or 'RETENCION' in c]
    if col_ret:
        df['RETENCION_TXT'] = df[col_ret[0]].map({1: 'Permanece', 0: 'Deserta'}).fillna('Sin Registro')
        df['RETENCION_NUM'] = pd.to_numeric(df[col_ret[0]], errors='coerce')
    else:
        df['RETENCION_TXT'] = 'Sin Registro'
        df['RETENCION_NUM'] = np.nan

    # Coordenadas
    if 'LATITUD' in df.columns and 'LONGITUD' in df.columns:
        df['LATITUD'] = pd.to_numeric(df['LATITUD'].astype(str).str.replace(',', '.'), errors='coerce')
        df['LONGITUD'] = pd.to_numeric(df['LONGITUD'].astype(str).str.replace(',', '.'), errors='coerce')

    return df

try:
    df = cargar_datos()
except Exception as e:
    st.error(f"Error cargando los datos: {e}")
    st.stop()

# -------------------------------------------------------------
# 2. BARRA LATERAL: FILTROS
# -------------------------------------------------------------
st.sidebar.markdown("## 🏛️ Filtros Generales")

# Filtro Cohorte
col_anio = [c for c in df.columns if 'ANIO_MATRICULA' in c or 'ANIO_CENSO' in c or 'ANIO_ING' in c]
if col_anio:
    anios = sorted(df[col_anio[0]].dropna().unique().astype(int))
    anios_sel = st.sidebar.multiselect("Año de Matrícula:", anios, default=anios)
    df_filtrado = df[df[col_anio[0]].isin(anios_sel)].copy() if anios_sel else df.copy()
else:
    df_filtrado = df.copy()

# Filtro Carrera Opcional
col_carr = [c for c in df.columns if 'CARRERA' in c or 'NOMB_CARRERA' in c]
if col_carr:
    todas_carr = sorted(df_filtrado[col_carr[0]].dropna().unique().astype(str))
    carr_sel = st.sidebar.multiselect("Filtrar por Carrera (Global):", todas_carr, default=[])
    if carr_sel:
        df_filtrado = df_filtrado[df_filtrado[col_carr[0]].isin(carr_sel)]

if df_filtrado.empty:
    st.warning("No hay registros coincidentes con los filtros seleccionados.")
    st.stop()

# -------------------------------------------------------------
# 3. CABECERA Y METRICAS
# -------------------------------------------------------------
st.title("🎓 Dashboard EDA Tesis: Trayectorias y Permanencia UBO")
st.markdown("Diagnóstico demográfico, territorial, vulnerabilidad escolar y dinámicas de retención estudiantil.")

m1, m2, m3, m4 = st.columns(4)
total_est = len(df_filtrado)
m1.metric("Estudiantes Analizados", f"{total_est:,}")

ret_validos = df_filtrado['RETENCION_NUM'].dropna()
tasa_ret = (ret_validos == 1).mean() * 100 if not ret_validos.empty else 0
m2.metric("Tasa de Retención (1er Año)", f"{tasa_ret:.1f}%")

col_rbd = [c for c in df_filtrado.columns if 'RBD' in c and 'NOM' not in c and 'COD' not in c]
n_colegios = df_filtrado[col_rbd[0]].nunique() if col_rbd else 0
m3.metric("Colegios Escolares (RBD)", f"{n_colegios:,}")

if 'DISTANCIA_CAMPUS_KM' in df_filtrado.columns:
    dist_med = df_filtrado['DISTANCIA_CAMPUS_KM'].median()
    m4.metric("Distancia Mediana Campus", f"{dist_med:.1f} km")

st.markdown("---")

# -------------------------------------------------------------
# 4. PESTAÑAS PRINCIPALES DE VISUALIZACIÓN
# -------------------------------------------------------------
tab_carreras, tab_evolucion, tab_avanzado, tab_geo = st.tabs([
    "📊 1. Áreas y Carreras",
    "📈 2. Tendencia Temporal por Género",
    "🔬 3. Vulnerabilidad y Retención (Tesis)",
    "🗺️ 4. Densidad Geoespacial"
])

# =============================================================
# PESTAÑA 1: ÁREAS Y CARRERAS (DRILL-DOWN INTERACTIVO)
# =============================================================
with tab_carreras:
    st.subheader("Distribución de Áreas del Conocimiento y Desglose por Género")
    st.markdown("Selecciona un área de conocimiento en el selector para visualizar la partición exacta entre hombres y mujeres.")

    col_area = [c for c in df_filtrado.columns if 'AREA' in c and 'RURAL' not in c and 'URB' not in c]
    nombre_col_area = col_area[0] if col_area else None

    c1, c2 = st.columns([1.2, 1])

    with c1:
        if nombre_col_area:
            df_area_counts = df_filtrado[nombre_col_area].astype(str).str.strip().value_counts().reset_index()
            df_area_counts.columns = ['Área', 'Estudiantes']
            fig_area = px.pie(
                df_area_counts,
                names='Área',
                values='Estudiantes',
                title="Distribución por Área de Conocimiento",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("Columna de Área de Conocimiento no detectada en la base.")

    with c2:
        if nombre_col_area:
            areas_disponibles = sorted(df_filtrado[nombre_col_area].dropna().astype(str).unique().tolist())
            area_seleccionada = st.selectbox("Selecciona un Área para ver el Género:", areas_disponibles)
            
            df_sub_area = df_filtrado[df_filtrado[nombre_col_area].astype(str) == area_seleccionada]
            fig_gen_area = px.pie(
                df_sub_area,
                names='GENERO_TXT',
                title=f"Composición de Género en: {area_seleccionada}",
                hole=0.45,
                color='GENERO_TXT',
                color_discrete_map={'Hombre': '#1f77b4', 'Mujer': '#e377c2', 'Sin Info': '#7f7f7f'}
            )
            st.plotly_chart(fig_gen_area, use_container_width=True)

    st.markdown("---")
    st.subheader("Volumen de Estudiantes por Carrera")
    if col_carr:
        conteo_carreras = df_filtrado[col_carr[0]].astype(str).str.strip().value_counts().reset_index()
        conteo_carreras.columns = ['Carrera', 'Estudiantes']
        fig_todas_carr = px.bar(
            conteo_carreras,
            x='Estudiantes',
            y='Carrera',
            orientation='h',
            title=f"Distribución de Matrícula en todas las Carreras ({len(conteo_carreras)} programas)",
            color='Estudiantes',
            color_continuous_scale='Viridis',
            height=max(450, len(conteo_carreras) * 22)
        )
        fig_todas_carr.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_todas_carr, use_container_width=True)

# =============================================================
# PESTAÑA 2: TENDENCIA TEMPORAL CON LÍNEAS DE GÉNERO
# =============================================================
with tab_evolucion:
    st.subheader("Evolución Histórica de la Matrícula por Género")
    st.markdown("Inspecciona la trayectoria anual en el volumen de hombres y mujeres para cualquier carrera de la universidad.")

    if col_carr and col_anio:
        carreras_evo = sorted(df[col_carr[0]].dropna().astype(str).unique().tolist())
        carr_linea = st.selectbox("Selecciona la Carrera a Examinar:", carreras_evo, key="select_evo_carr")

        df_linea = df[df[col_carr[0]].astype(str) == carr_linea].copy()
        
        # Agrupación por Año y Género
        agg_linea = (
            df_linea.groupby([col_anio[0], 'GENERO_TXT'])
            .size()
            .reset_index(name='ESTUDIANTES')
        )
        agg_linea[col_anio[0]] = agg_linea[col_anio[0]].astype(int)

        fig_lineas = px.line(
            agg_linea,
            x=col_anio[0],
            y='ESTUDIANTES',
            color='GENERO_TXT',
            markers=True,
            title=f"Evolución Anual de Matrícula por Género - {carr_linea}",
            color_discrete_map={'Hombre': '#1f77b4', 'Mujer': '#e377c2', 'Sin Info': '#7f7f7f'}
        )
        fig_lineas.update_layout(
            xaxis_title="Año de Matrícula",
            yaxis_title="Cantidad de Estudiantes",
            hovermode="x unified"
        )
        st.plotly_chart(fig_lineas, use_container_width=True)
    else:
        st.info("Variables de año o carrera no disponibles para generar la tendencia.")

# =============================================================
# PESTAÑA 3: 5 GRÁFICOS AVANZADOS DE LA TESIS (RETENCIÓN Y VULNERABILIDAD)
# =============================================================
with tab_avanzado:
    st.subheader("Factores Determinantes de la Permanencia Universitaria")
    st.markdown("Visualizaciones analíticas diseñadas para respaldar las hipótesis del modelo predictivo de permanencia.")

    # Gráfico 1: Rezago vs Retención
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        if 'BRECHA_REZAGO_ANIOS' in df_filtrado.columns:
            fig_box_rezago = px.box(
                df_filtrado,
                x='RETENCION_TXT',
                y='BRECHA_REZAGO_ANIOS',
                color='RETENCION_TXT',
                title="1. Brecha de Rezago Temporal (Años) vs Retención al 1° Año",
                labels={'BRECHA_REZAGO_ANIOS': 'Años fuera del sistema formal', 'RETENCION_TXT': 'Estado'},
                color_discrete_map={'Permanece': '#2ca02c', 'Deserta': '#d62728'}
            )
            st.plotly_chart(fig_box_rezago, use_container_width=True)

    # Gráfico 2: IVM Establecimiento vs Retención
    with g_col2:
        if 'IVM_ESTABLECIMIENTO' in df_filtrado.columns:
            fig_kde_ivm = px.histogram(
                df_filtrado.dropna(subset=['IVM_ESTABLECIMIENTO']),
                x='IVM_ESTABLECIMIENTO',
                color='RETENCION_TXT',
                barmode='overlay',
                marginal='box',
                title="2. Distribución del IVM Escolar según Retención",
                color_discrete_map={'Permanece': '#2ca02c', 'Deserta': '#d62728'}
            )
            st.plotly_chart(fig_kde_ivm, use_container_width=True)

    st.markdown("---")
    g_col3, g_col4 = st.columns(2)

    # Gráfico 3: Heatmap Área vs Tramo ROC IVM
    with g_col3:
        if nombre_col_area and 'IVM_TRAMO_ESTABLECIMIENTO' in df_filtrado.columns:
            tabla_tramo = pd.crosstab(
                df_filtrado[nombre_col_area],
                df_filtrado['IVM_TRAMO_ESTABLECIMIENTO'],
                normalize='index'
            ) * 100
            fig_heat = px.imshow(
                tabla_tramo,
                text_auto='.1f',
                aspect='auto',
                color_continuous_scale='YlOrRd',
                title="3. Concentración de Vulnerabilidad ROC por Área (%)"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # Gráfico 4: Vía de Admisión vs Retención
    with g_col4:
        col_via = [c for c in df_filtrado.columns if 'INGRESO' in c or 'VIA' in c]
        if col_via:
            df_via_agg = df_filtrado.groupby([col_via[0], 'RETENCION_TXT']).size().reset_index(name='TOTAL')
            fig_via = px.bar(
                df_via_agg,
                x=col_via[0],
                y='TOTAL',
                color='RETENCION_TXT',
                barmode='relative',
                title="4. Tasa de Retención según Vía de Admisión",
                color_discrete_map={'Permanece': '#2ca02c', 'Deserta': '#d62728'}
            )
            st.plotly_chart(fig_via, use_container_width=True)

    # Gráfico 5: Distancia al Campus y Abandono Comunal
    st.markdown("---")
    if 'DISTANCIA_CAMPUS_KM' in df_filtrado.columns:
        col_nom_com = [c for c in df_filtrado.columns if 'NOM_COM' in c or 'COMUNA' in c]
        if col_nom_com:
            df_com_scatter = df_filtrado.groupby(col_nom_com[0]).agg(
                DISTANCIA_MEDIA=('DISTANCIA_CAMPUS_KM', 'median'),
                TASA_DESERCION=('RETENCION_NUM', lambda x: (x == 0).mean() * 100 if len(x.dropna()) > 0 else 0),
                TOTAL_ESTUDIANTES=('RETENCION_NUM', 'count')
            ).reset_index()

            df_com_scatter = df_com_scatter[df_com_scatter['TOTAL_ESTUDIANTES'] >= 30]

            fig_dist = px.scatter(
                df_com_scatter,
                x='DISTANCIA_MEDIA',
                y='TASA_DESERCION',
                size='TOTAL_ESTUDIANTES',
                hover_name=col_nom_com[0],
                trendline='ols',
                title="5. Relación entre Distancia al Campus (Km) y Tasa de Deserción Comunal",
                labels={'DISTANCIA_MEDIA': 'Distancia Mediana (Km)', 'TASA_DESERCION': 'Deserción (%)'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

# =============================================================
# PESTAÑA 4: MAPA DE CALOR GEOESPACIAL
# =============================================================
with tab_geo:
    st.subheader("Mapa de Calor y Concentración de Matrícula")
    if 'LATITUD' in df_filtrado.columns and 'LONGITUD' in df_filtrado.columns:
        df_geo = df_filtrado.dropna(subset=['LATITUD', 'LONGITUD']).copy()
        df_geo = df_geo[(df_geo['LATITUD'] >= -56.0) & (df_geo['LATITUD'] <= -17.5)]

        if not df_geo.empty and col_rbd:
            col_nom_c = [c for c in df_geo.columns if 'NOM_RBD' in c or 'NOMBRE_ESTABLE' in c]
            nom_c_col = col_nom_c[0] if col_nom_c else col_rbd[0]

            agg_colegios = (
                df_geo.groupby([col_rbd[0], nom_c_col, 'LATITUD', 'LONGITUD'])
                .size()
                .reset_index(name='ALUMNOS')
            )

            mapa = folium.Map(location=[-33.4569, -70.6631], zoom_start=10, tiles='CartoDB positron')
            heat_data = [[r['LATITUD'], r['LONGITUD'], r['ALUMNOS']] for _, r in agg_colegios.iterrows()]
            HeatMap(heat_data, radius=13, blur=15, max_zoom=13).add_to(mapa)

            cluster = MarkerCluster().add_to(mapa)
            for _, r in agg_colegios.sort_values('ALUMNOS', ascending=False).head(350).iterrows():
                folium.CircleMarker(
                    location=[r['LATITUD'], r['LONGITUD']],
                    radius=4 + min(r['ALUMNOS'], 12),
                    popup=f"<b>RBD:</b> {int(r[col_rbd[0]])}<br><b>Colegio:</b> {r[nom_c_col]}<br><b>Matriculados:</b> {r['ALUMNOS']}",
                    color="#1f77b4",
                    fill=True
                ).add_to(cluster)

            st_folium(mapa, width="100%", height=500)
