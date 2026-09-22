import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

# Configuración de página
st.set_page_config(
    page_title="EDA Tesis UBO - Retención y Vulnerabilidad",
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

    # Normalización de Retención (1: Permanece, 0: Deserta)
    col_ret = [c for c in df.columns if 'RETIENE' in c or 'RETENCION' in c]
    if col_ret:
        df['RETENCION_TXT'] = df[col_ret[0]].map({1: 'Permanece', 0: 'Deserta'}).fillna('Sin Registro')
        df['RETENCION_NUM'] = pd.to_numeric(df[col_ret[0]], errors='coerce')
    else:
        df['RETENCION_TXT'] = 'Sin Registro'
        df['RETENCION_NUM'] = np.nan

    # Normalización estricta de Carrera (FORZADA A TEXTO CATEGÓRICO)
    col_carr_pos = [c for c in df.columns if 'NOMB_CARRERA' in c or 'NOMBRE_CARRERA' in c or c == 'CARRERA']
    if col_carr_pos:
        df['CARRERA_NOMBRE'] = df[col_carr_pos[0]].astype(str).str.strip().str.upper()
    else:
        df['CARRERA_NOMBRE'] = 'CARRERA S/I'

    # Coordenadas limpias en formato float64 nativo
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
# 2. BARRA LATERAL (LOGO ROBUSTO Y FILTROS)
# -------------------------------------------------------------
st.sidebar.markdown("### 🏛️ **Universidad Bernardo O'Higgins**")
st.sidebar.markdown("**Facultad de Ingeniería, Ciencia y Tecnología**")
st.sidebar.markdown("---")
st.sidebar.title("Filtros del Panel")

# Filtro Cohorte
col_anio = [c for c in df.columns if 'ANIO_MATRICULA' in c or 'ANIO_CENSO' in c or 'ANIO_ING' in c]
col_anio_usada = col_anio[0] if col_anio else None

if col_anio_usada:
    anios_disp = sorted(df[col_anio_usada].dropna().unique().astype(int))
    anios_sel = st.sidebar.multiselect("Año de Matrícula:", anios_disp, default=anios_disp)
    df_filtrado = df[df[col_anio_usada].isin(anios_sel)].copy() if anios_sel else df.copy()
else:
    df_filtrado = df.copy()

# Filtro Carrera Opcional
carreras_disp = sorted([c for c in df_filtrado['CARRERA_NOMBRE'].dropna().unique() if c != 'NAN'])
carreras_sel = st.sidebar.multiselect("Filtrar por Carrera (Opcional):", carreras_disp, default=[])
if carreras_sel:
    df_filtrado = df_filtrado[df_filtrado['CARRERA_NOMBRE'].isin(carreras_sel)]

if df_filtrado.empty:
    st.warning("No hay registros que coincidan con la combinación de filtros seleccionada.")
    st.stop()

# -------------------------------------------------------------
# 3. CABECERA Y METRICAS
# -------------------------------------------------------------
st.title("🎓 Dashboard EDA: Trayectorias y Permanencia Estudiantil UBO")
st.markdown("Diagnóstico de matrícula histórica, origen territorial y factores asociados a la permanencia.")

m1, m2, m3, m4 = st.columns(4)
total_est = len(df_filtrado)
m1.metric("Estudiantes Analizados", f"{total_est:,}")

ret_validos = df_filtrado['RETENCION_NUM'].dropna()
tasa_ret = (ret_validos == 1).mean() * 100 if not ret_validos.empty else 0
m2.metric("Tasa Retención 1° Año", f"{tasa_ret:.1f}%")

col_rbd = [c for c in df_filtrado.columns if 'RBD' in c and 'NOM' not in c and 'COD' not in c]
n_colegios = df_filtrado[col_rbd[0]].nunique() if col_rbd else 0
m3.metric("Colegios de Origen (RBD)", f"{n_colegios:,}")

if 'DISTANCIA_CAMPUS_KM' in df_filtrado.columns:
    dist_med = df_filtrado['DISTANCIA_CAMPUS_KM'].dropna().median()
    m4.metric("Distancia Mediana Campus", f"{dist_med:.1f} km" if pd.notna(dist_med) else "S/I")

st.markdown("---")

# -------------------------------------------------------------
# 4. PESTAÑAS PRINCIPALES
# -------------------------------------------------------------
tab1_carr, tab2_temp, tab3_ret, tab4_geo = st.tabs([
    "📊 1. Áreas y Carreras",
    "📈 2. Evolución 2015-2025",
    "🔬 3. Vulnerabilidad y Permanencia",
    "🗺️ 4. Mapa de Calor Geoespacial"
])

# =============================================================
# PESTAÑA 1: ÁREAS Y CARRERAS (SOLUCIÓN GRÁFICO DE BARRAS)
# =============================================================
with tab1_carr:
    st.subheader("Distribución por Área de Conocimiento y Desglose por Género")
    
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
                title="Distribución por Área del Conocimiento",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("Columna de Área de Conocimiento no detectada.")

    with c2:
        if nombre_col_area:
            areas_disponibles = sorted(df_filtrado[nombre_col_area].dropna().astype(str).unique().tolist())
            area_seleccionada = st.selectbox("Selecciona un Área para ver el desglose por género:", areas_disponibles)
            df_sub_area = df_filtrado[df_filtrado[nombre_col_area].astype(str) == area_seleccionada]
            fig_gen_area = px.pie(
                df_sub_area,
                names='GENERO_TXT',
                title=f"Género en: {area_seleccionada}",
                hole=0.45,
                color='GENERO_TXT',
                color_discrete_map={'Hombre': '#1f77b4', 'Mujer': '#e377c2', 'Sin Info': '#7f7f7f'}
            )
            st.plotly_chart(fig_gen_area, use_container_width=True)

    st.markdown("---")
    st.subheader("Volumen de Estudiantes por Carrera")
    
    # Conteo limpio forzando a texto explícito y ordenando de mayor a menor
    conteo_carr = (
        df_filtrado['CARRERA_NOMBRE']
        .astype(str)
        .value_counts()
        .reset_index()
    )
    conteo_carr.columns = ['Carrera', 'Estudiantes']
    
    # Selector de visualización (Top 15 o todas)
    modo_carr = st.radio("Modo de visualización:", ["Top 15 Carreras", "Todas las Carreras"], horizontal=True)
    if modo_carr == "Top 15 Carreras":
        conteo_carr_plot = conteo_carr.head(15).copy()
        alto_grafico = 500
    else:
        conteo_carr_plot = conteo_carr.copy()
        alto_grafico = max(550, len(conteo_carr_plot) * 22)

    fig_carreras = px.bar(
        conteo_carr_plot,
        x='Estudiantes',
        y='Carrera',
        orientation='h',
        title=f"Cantidad de Estudiantes Matriculados ({len(conteo_carr_plot)} programas)",
        color='Estudiantes',
        color_continuous_scale='Blues',
        text='Estudiantes',
        height=alto_grafico
    )
    fig_carreras.update_layout(
        yaxis={
            'type': 'category',
            'categoryorder': 'total ascending',
            'title': ''
        },
        xaxis={'title': 'Cantidad de Estudiantes'},
        margin=dict(l=10, r=20, t=40, b=20)
    )
    fig_carreras.update_traces(textposition='outside')
    st.plotly_chart(fig_carreras, use_container_width=True)

# =============================================================
# PESTAÑA 2: EVOLUCIÓN HISTÓRICA 2015-2025
# =============================================================
with tab2_temp:
    st.subheader("Evolución Temporal de la Matrícula Universitaria (2015 - 2025)")
    st.markdown("Monitorea la dinámica de crecimiento y el comportamiento de género a lo largo de las cohortes.")

    if col_anio_usada:
        # Gráfico General: Top 8 Carreras a lo largo del tiempo
        top8_nombres = df['CARRERA_NOMBRE'].value_counts().head(8).index.tolist()
        df_top8_tiempo = (
            df[df['CARRERA_NOMBRE'].isin(top8_nombres)]
            .groupby([col_anio_usada, 'CARRERA_NOMBRE'])
            .size()
            .reset_index(name='ESTUDIANTES')
        )
        df_top8_tiempo[col_anio_usada] = df_top8_tiempo[col_anio_usada].astype(int)

        fig_top8 = px.line(
            df_top8_tiempo,
            x=col_anio_usada,
            y='ESTUDIANTES',
            color='CARRERA_NOMBRE',
            markers=True,
            title="Evolución de las 8 Carreras con Mayor Matrícula Histórica (2015-2025)"
        )
        fig_top8.update_layout(
            xaxis_title="Año de Matrícula",
            yaxis_title="Total Estudiantes",
            hovermode="x unified",
            xaxis=dict(tickmode='linear', dtick=1)
        )
        st.plotly_chart(fig_top8, use_container_width=True)

        st.markdown("---")
        st.subheader("Desglose Específico por Carrera y Género")
        
        carreras_lista = sorted(df['CARRERA_NOMBRE'].dropna().unique().tolist())
        carr_sel_ind = st.selectbox("Selecciona la carrera a inspeccionar año a año:", carreras_lista)

        df_carr_ind = df[df['CARRERA_NOMBRE'] == carr_sel_ind].copy()
        df_evo_gen = (
            df_carr_ind.groupby([col_anio_usada, 'GENERO_TXT'])
            .size()
            .reset_index(name='ESTUDIANTES')
        )
        df_evo_gen[col_anio_usada] = df_evo_gen[col_anio_usada].astype(int)

        fig_evo_gen = px.line(
            df_evo_gen,
            x=col_anio_usada,
            y='ESTUDIANTES',
            color='GENERO_TXT',
            markers=True,
            title=f"Evolución Anual por Género: {carr_sel_ind}",
            color_discrete_map={'Hombre': '#1f77b4', 'Mujer': '#e377c2', 'Sin Info': '#7f7f7f'}
        )
        fig_evo_gen.update_layout(
            xaxis_title="Año de Matrícula",
            yaxis_title="Estudiantes Matriculados",
            hovermode="x unified",
            xaxis=dict(tickmode='linear', dtick=1)
        )
        st.plotly_chart(fig_evo_gen, use_container_width=True)
    else:
        st.info("Columna temporal no disponible.")

# =============================================================
# PESTAÑA 3: VULNERABILIDAD Y PERMANENCIA (SIN DEPENDENCIAS EXTERNAS)
# =============================================================
with tab3_ret:
    st.subheader("Análisis de Factores Asociados a la Permanencia")
    st.markdown("Evaluación empírica de variables escolares, territoriales y rezago temporal sobre la condición de permanencia.")

    g1, g2 = st.columns(2)
    with g1:
        if 'BRECHA_REZAGO_ANIOS' in df_filtrado.columns:
            df_b = df_filtrado.dropna(subset=['BRECHA_REZAGO_ANIOS', 'RETENCION_TXT']).copy()
            fig_box = px.box(
                df_b,
                x='RETENCION_TXT',
                y='BRECHA_REZAGO_ANIOS',
                color='RETENCION_TXT',
                title="1. Rezago Escolar Temporal (Años) según Estado de Permanencia",
                labels={'BRECHA_REZAGO_ANIOS': 'Años fuera del sistema formal', 'RETENCION_TXT': 'Estado'},
                color_discrete_map={'Permanece': '#2ca02c', 'Deserta': '#d62728'}
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with g2:
        if 'IVM_ESTABLECIMIENTO' in df_filtrado.columns:
            df_ivm = df_filtrado.dropna(subset=['IVM_ESTABLECIMIENTO', 'RETENCION_TXT']).copy()
            fig_hist_ivm = px.histogram(
                df_ivm,
                x='IVM_ESTABLECIMIENTO',
                color='RETENCION_TXT',
                barmode='overlay',
                marginal='box',
                title="2. Distribución de Vulnerabilidad Escolar (IVM) según Retención",
                color_discrete_map={'Permanece': '#2ca02c', 'Deserta': '#d62728'}
            )
            st.plotly_chart(fig_hist_ivm, use_container_width=True)

    st.markdown("---")
    g3, g4 = st.columns(2)

    with g3:
        if nombre_col_area and 'IVM_TRAMO_ESTABLECIMIENTO' in df_filtrado.columns:
            df_tramo = df_filtrado.dropna(subset=[nombre_col_area, 'IVM_TRAMO_ESTABLECIMIENTO'])
            if not df_tramo.empty:
                tabla_tramo = pd.crosstab(
                    df_tramo[nombre_col_area],
                    df_tramo['IVM_TRAMO_ESTABLECIMIENTO'],
                    normalize='index'
                ) * 100
                fig_tramo = px.imshow(
                    tabla_tramo,
                    text_auto='.1f',
                    aspect='auto',
                    color_continuous_scale='YlOrRd',
                    title="3. Concentración de Vulnerabilidad ROC por Área (%)"
                )
                st.plotly_chart(fig_tramo, use_container_width=True)

    with g4:
        col_via = [c for c in df_filtrado.columns if 'INGRESO' in c or 'VIA' in c]
        if col_via:
            df_v = df_filtrado.dropna(subset=[col_via[0], 'RETENCION_TXT']).copy()
            df_via_agg = df_v.groupby([col_via[0], 'RETENCION_TXT']).size().reset_index(name='TOTAL')
            fig_via = px.bar(
                df_via_agg,
                x=col_via[0],
                y='TOTAL',
                color='RETENCION_TXT',
                barmode='relative',
                title="4. Retención según Vía de Admisión",
                color_discrete_map={'Permanece': '#2ca02c', 'Deserta': '#d62728'}
            )
            st.plotly_chart(fig_via, use_container_width=True)

    st.markdown("---")
    st.subheader("5. Relación entre Distancia al Campus y Retención Comunal")
    
    col_nom_com = [c for c in df_filtrado.columns if 'NOM_COM_RBD' in c or 'NOM_COM' in c or 'COMUNA' in c]
    if 'DISTANCIA_CAMPUS_KM' in df_filtrado.columns and col_nom_com:
        df_scatter_base = df_filtrado.dropna(subset=['DISTANCIA_CAMPUS_KM', 'RETENCION_NUM', col_nom_com[0]]).copy()
        
        if not df_scatter_base.empty:
            df_com_agg = df_scatter_base.groupby(col_nom_com[0]).agg(
                DISTANCIA_MEDIA=('DISTANCIA_CAMPUS_KM', 'median'),
                TASA_RETENCION=('RETENCION_NUM', lambda x: (x == 1).mean() * 100),
                ESTUDIANTES=('RETENCION_NUM', 'count')
            ).reset_index()

            # Umbral mínimo adaptativo para asegurar datos visibles
            umbral = 15 if len(df_com_agg) > 10 else 1
            df_plot_scatter = df_com_agg[df_com_agg['ESTUDIANTES'] >= umbral]

            fig_dist_com = px.scatter(
                df_plot_scatter,
                x='DISTANCIA_MEDIA',
                y='TASA_RETENCION',
                size='ESTUDIANTES',
                hover_name=col_nom_com[0],
                title=f"Distancia Mediana al Campus vs Tasa de Retención por Comuna (Mínimo {umbral} alumnos)",
                labels={'DISTANCIA_MEDIA': 'Distancia al Campus (Km)', 'TASA_RETENCION': 'Tasa de Retención (%)'},
                color='TASA_RETENCION',
                color_continuous_scale='Teal'
            )
            st.plotly_chart(fig_dist_com, use_container_width=True)
        else:
            st.info("No hay suficientes datos válidos para calcular la relación territorial.")

# =============================================================
# PESTAÑA 4: MAPA DE CALOR GEOESPACIAL (ROBUSTO Y COMPATIBLE)
# =============================================================
with tab4_geo:
    st.subheader("Densidad Territorial de Establecimientos Escolares")
    st.markdown("Concentración geoespacial de los colegios de egreso de los estudiantes matriculados en la UBO.")

    if 'LATITUD' in df_filtrado.columns and 'LONGITUD' in df_filtrado.columns:
        # Filtrado estricto de coordenadas reales de Chile continental
        df_geo = df_filtrado.dropna(subset=['LATITUD', 'LONGITUD']).copy()
        df_geo['LATITUD'] = pd.to_numeric(df_geo['LATITUD'], errors='coerce')
        df_geo['LONGITUD'] = pd.to_numeric(df_geo['LONGITUD'], errors='coerce')
        df_geo = df_geo.dropna(subset=['LATITUD', 'LONGITUD'])
        
        df_geo = df_geo[(df_geo['LATITUD'] >= -56.0) & (df_geo['LATITUD'] <= -17.5) &
                        (df_geo['LONGITUD'] >= -76.0) & (df_geo['LONGITUD'] <= -66.0)]

        if not df_geo.empty and col_rbd:
            col_nom_colegio = [c for c in df_geo.columns if 'NOM_RBD' in c or 'NOMBRE_ESTABLE' in c or 'NOMBRE_RBD' in c]
            nom_c_campo = col_nom_colegio[0] if col_nom_colegio else col_rbd[0]

            # Agrupar por coordenadas exactas y colegio
            df_colegios = (
                df_geo.groupby([col_rbd[0], nom_c_campo, 'LATITUD', 'LONGITUD'])
                .size()
                .reset_index(name='TOTAL_ALUMNOS')
            )

            col_map_izq, col_map_der = st.columns([1.7, 1])

            with col_map_izq:
                # Inicializar mapa centrado en Santiago
                mapa = folium.Map(location=[-33.4569, -70.6631], zoom_start=10, tiles='CartoDB positron')

                # Datos para el HeatMap [lat, lon, peso] asegurando tipos float estándar
                heat_puntos = [
                    [float(r['LATITUD']), float(r['LONGITUD']), float(r['TOTAL_ALUMNOS'])]
                    for _, r in df_colegios.iterrows()
                ]
                HeatMap(heat_puntos, radius=14, blur=18, min_opacity=0.3, max_zoom=13).add_to(mapa)

                # Marcadores de establecimientos (Top 300 con mayor masa)
                cluster = MarkerCluster(name="Colegios").add_to(mapa)
                for _, r in df_colegios.sort_values('TOTAL_ALUMNOS', ascending=False).head(300).iterrows():
                    folium.CircleMarker(
                        location=[float(r['LATITUD']), float(r['LONGITUD'])],
                        radius=4 + min(int(r['TOTAL_ALUMNOS']), 12),
                        popup=f"<b>RBD:</b> {int(r[col_rbd[0]])}<br><b>Colegio:</b> {r[nom_c_campo]}<br><b>Matriculados:</b> {int(r['TOTAL_ALUMNOS'])}",
                        color="#1f77b4",
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(cluster)

                st_folium(mapa, width="100%", height=480)

            with col_map_der:
                if col_nom_com:
                    st.write("**Top Comunas con Mayor Concentración Escolar**")
                    tabla_com = df_geo[col_nom_com[0]].astype(str).str.strip().value_counts().reset_index()
                    tabla_com.columns = ['Comuna', 'Estudiantes']
                    tabla_com['% Total'] = (tabla_com['Estudiantes'] / len(df_geo) * 100).map("{:.2f}%".format)
                    st.dataframe(tabla_com.head(15), use_container_width=True, height=430)
        else:
            st.info("No se encontraron coordenadas válidas bajo los filtros seleccionados.")
    else:
        st.warning("El conjunto de datos no posee las columnas LATITUD y LONGITUD.")
