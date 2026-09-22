import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

# Configuración general de la página
st.set_page_config(
    page_title="EDA Tesis UBO - Retención y Vulnerabilidad",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 1. CARGA DE DATOS OPTIMIZADA Y TOLERANTE A RUTAS
# -------------------------------------------------------------
@st.cache_data
def cargar_datos():
    candidatos_ruta = [
        "Tablon_Maestro_Tesis_UBO.parquet",
        "data/Tablon_Maestro_Tesis_UBO.parquet",
        "Tablon_Maestro_Tesis_UBO.xlsx",
        "data/Tablon_Maestro_Tesis_UBO.xlsx"
    ]
    archivo_encontrado = next((r for r in candidatos_ruta if os.path.exists(r)), None)
    if not archivo_encontrado:
        raise FileNotFoundError("No se encontró 'Tablon_Maestro_Tesis_UBO.parquet' ni en la raíz ni en 'data/'.")

    df = pd.read_parquet(archivo_encontrado) if archivo_encontrado.endswith('.parquet') else pd.read_excel(archivo_encontrado)
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

    # Normalización segura de coordenadas geoespaciales
    if 'LATITUD' in df.columns and 'LONGITUD' in df.columns:
        df['LATITUD'] = pd.to_numeric(df['LATITUD'].astype(str).str.replace(',', '.'), errors='coerce')
        df['LONGITUD'] = pd.to_numeric(df['LONGITUD'].astype(str).str.replace(',', '.'), errors='coerce')

    return df

try:
    df = cargar_datos()
except Exception as e:
    st.error(f"Error al cargar el conjunto de datos: {e}")
    st.info("Verifica haber subido el archivo 'Tablon_Maestro_Tesis_UBO.parquet' a tu repositorio de GitHub.")
    st.stop()

# -------------------------------------------------------------
# 2. BARRA LATERAL Y FILTROS INTERACTIVOS
# -------------------------------------------------------------
try:
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Logo_UBO.png/320px-Logo_UBO.png", width=180)
except Exception:
    st.sidebar.markdown("### 🏛️ Universidad Bernardo O'Higgins")

st.sidebar.title("Filtros del Panel")

# Filtro por Cohorte de Matrícula
col_anio = [c for c in df.columns if 'ANIO_MATRICULA' in c or 'ANIO_CENSO' in c or 'ANIO_ING' in c]
if col_anio:
    anios_disp = sorted(df[col_anio[0]].dropna().unique().astype(int))
    anios_sel = st.sidebar.multiselect("Año de Matrícula:", anios_disp, default=anios_disp)
    df_filtrado = df[df[col_anio[0]].isin(anios_sel)].copy() if anios_sel else df.copy()
else:
    df_filtrado = df.copy()

# Filtro por Carrera (Global)
col_carr = [c for c in df.columns if 'CARRERA' in c or 'NOMB_CARRERA' in c]
if col_carr:
    carreras_disp = sorted(df_filtrado[col_carr[0]].dropna().unique().astype(str))
    carreras_sel = st.sidebar.multiselect("Filtrar por Carrera (opcional):", carreras_disp, default=[])
    if carreras_sel:
        df_filtrado = df_filtrado[df_filtrado[col_carr[0]].isin(carreras_sel)]

if df_filtrado.empty:
    st.warning("No hay registros que coincidan con la combinación de filtros seleccionada.")
    st.stop()

# -------------------------------------------------------------
# 3. CABECERA Y METRICAS PRINCIPALES (KPIS)
# -------------------------------------------------------------
st.title("🎓 Dashboard EDA Tesis: Trayectorias y Permanencia UBO")
st.markdown("Diagnóstico demográfico, territorial, vulnerabilidad escolar y dinámicas de retención estudiantil.")

m1, m2, m3, m4 = st.columns(4)
total_est = len(df_filtrado)
m1.metric("Total Estudiantes", f"{total_est:,}")

ret_validos = df_filtrado['RETENCION_NUM'].dropna()
tasa_ret = (ret_validos == 1).mean() * 100 if not ret_validos.empty else 0
m2.metric("Tasa Retención 1° Año", f"{tasa_ret:.1f}%")

col_rbd = [c for c in df_filtrado.columns if 'RBD' in c and 'NOM' not in c and 'COD' not in c]
n_colegios = df_filtrado[col_rbd[0]].nunique() if col_rbd else 0
m3.metric("Colegios Escolares (RBD)", f"{n_colegios:,}")

if 'DISTANCIA_CAMPUS_KM' in df_filtrado.columns:
    dist_media = df_filtrado['DISTANCIA_CAMPUS_KM'].dropna().median()
    m4.metric("Distancia Mediana Campus", f"{dist_media:.1f} km" if pd.notna(dist_media) else "S/I")

st.markdown("---")

# -------------------------------------------------------------
# 4. PESTAÑAS PRINCIPALES DE VISUALIZACIÓN
# -------------------------------------------------------------
tab_carreras, tab_evolucion, tab_avanzado, tab_geo, tab_export = st.tabs([
    "📊 1. Áreas y Carreras",
    "📈 2. Tendencia Temporal por Género",
    "🔬 3. Vulnerabilidad y Retención (Tesis)",
    "🗺️ 4. Densidad Geoespacial",
    "📥 5. Datos y Descarga"
])

# =============================================================
# PESTAÑA 1: ÁREAS Y CARRERAS (DRILL-DOWN INTERACTIVO)
# =============================================================
with tab_carreras:
    st.subheader("Distribución por Área de Conocimiento y Desglose por Género")
    st.markdown("Selecciona un área de conocimiento en el desplegable para ver la proporción entre hombres y mujeres de esa disciplina en específico.")

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
            area_seleccionada = st.selectbox("Selecciona un Área para ver el desglose de género:", areas_disponibles)
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
        st.info("Variables de año o carrera no disponibles para generar la serie histórica.")

# =============================================================
# PESTAÑA 3: VULNERABILIDAD Y RETENCIÓN (GRÁFICOS DE TESIS)
# =============================================================
with tab_avanzado:
    st.subheader("Factores Determinantes de la Permanencia Universitaria")
    st.markdown("Visualizaciones analíticas diseñadas para contrastar los predictores de permanencia y vulnerabilidad escolar.")

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
# PESTAÑA 4: MAPA DE CALOR GEOESPACIAL Y DENSIDAD
# =============================================================
with tab_geo:
    st.subheader("Densidad Geoespacial de Colegios de Procedencia")
    st.markdown("Distribución territorial de los establecimientos escolares de origen vinculados mediante coordenadas oficiales del MINEDUC.")

    if 'LATITUD' in df_filtrado.columns and 'LONGITUD' in df_filtrado.columns:
        df_geo = df_filtrado.dropna(subset=['LATITUD', 'LONGITUD']).copy()
        df_geo = df_geo[(df_geo['LATITUD'] >= -56.0) & (df_geo['LATITUD'] <= -17.5) & 
                        (df_geo['LONGITUD'] >= -76.0) & (df_geo['LONGITUD'] <= -66.0)]

        if not df_geo.empty and col_rbd:
            col_mapa, col_tabla_geo = st.columns([1.6, 1])

            with col_mapa:
                col_nom_c = [c for c in df_geo.columns if 'NOM_RBD' in c or 'NOMBRE_ESTABLE' in c or 'NOMBRE_RBD' in c]
                nom_c_col = col_nom_c[0] if col_nom_c else col_rbd[0]

                colegios_agg = (
                    df_geo.groupby([col_rbd[0], nom_c_col, 'LATITUD', 'LONGITUD'])
                    .size()
                    .reset_index(name='TOTAL_ALUMNOS')
                )

                mapa = folium.Map(location=[-33.4569, -70.6631], zoom_start=10, tiles='CartoDB positron')
                heat_data = [[r['LATITUD'], r['LONGITUD'], r['TOTAL_ALUMNOS']] for _, r in colegios_agg.iterrows()]
                HeatMap(heat_data, radius=13, blur=15, max_zoom=13).add_to(mapa)

                cluster = MarkerCluster(name="Colegios").add_to(mapa)
                for _, r in colegios_agg.sort_values('TOTAL_ALUMNOS', ascending=False).head(400).iterrows():
                    folium.CircleMarker(
                        location=[r['LATITUD'], r['LONGITUD']],
                        radius=4 + min(r['TOTAL_ALUMNOS'], 12),
                        popup=f"<b>RBD:</b> {int(r[col_rbd[0]])}<br><b>Colegio:</b> {r[nom_c_col]}<br><b>Matriculados UBO:</b> {r['TOTAL_ALUMNOS']}",
                        color="#1f77b4",
                        fill=True,
                        fill_opacity=0.65
                    ).add_to(cluster)

                st_folium(mapa, width="100%", height=460)

            with col_tabla_geo:
                col_comuna = [c for c in df_geo.columns if 'NOM_COM_RBD' in c or 'NOM_COM' in c or 'COMUNA' in c]
                if col_comuna:
                    st.write("**Top Comunas con Mayor Concentración**")
                    tabla_com = df_geo[col_comuna[0]].astype(str).str.strip().value_counts().reset_index()
                    tabla_com.columns = ['Comuna', 'Estudiantes']
                    tabla_com['% Total'] = (tabla_com['Estudiantes'] / len(df_geo) * 100).map("{:.2f}%".format)
                    st.dataframe(tabla_com.head(15), use_container_width=True, height=410)
        else:
            st.info("No se encontraron registros georreferenciados válidos para el subconjunto filtrado.")
    else:
        st.warning("El dataset cargado no contiene las columnas de LATITUD y LONGITUD.")

# =============================================================
# PESTAÑA 5: EXPLORADOR DETALLADO Y DESCARGA
# =============================================================
with tab_export:
    st.subheader("Vista de Datos y Exportación")
    st.markdown("Visualiza y descarga el subconjunto de estudiantes resultante de los filtros activos.")
    st.dataframe(df_filtrado.head(100), use_container_width=True)

    csv_bytes = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar datos filtrados en CSV",
        data=csv_bytes,
        file_name="Datos_Filtrados_EDA_UBO.csv",
        mime="text/csv"
    )
