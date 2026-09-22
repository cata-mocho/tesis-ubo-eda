import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

# Configuración de la página
st.set_page_config(
    page_title="EDA Tesis UBO - Retención y Vulnerabilidad",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# CARGA DE DATOS OPTIMIZADA (CACHE)
# -------------------------------------------------------------
@st.cache_data
def cargar_datos():
    # Cargar parquet o excel consolidado
    try:
        df = pd.read_parquet("data/Tablon_Maestro_Tesis_UBO.parquet")
    except Exception:
        df = pd.read_excel("data/Tablon_Maestro_Tesis_UBO.xlsx")
        
    df.columns = df.columns.str.strip().str.upper()
    
    # Coordenadas limpias
    if 'LATITUD' in df.columns and 'LONGITUD' in df.columns:
        df['LATITUD'] = pd.to_numeric(df['LATITUD'].astype(str).str.replace(',', '.'), errors='coerce')
        df['LONGITUD'] = pd.to_numeric(df['LONGITUD'].astype(str).str.replace(',', '.'), errors='coerce')
        
    return df

try:
    df = cargar_datos()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}. Asegúrate de ubicar el archivo en la carpeta 'data/'.")
    st.stop()

# -------------------------------------------------------------
# SIDEBAR: FILTROS INTERACTIVOS
# -------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Logo_UBO.png/320px-Logo_UBO.png", width=180)
st.sidebar.title("Filtros del Panel")

# Filtro por Cohorte/Año de Matrícula
col_anio = [c for c in df.columns if 'ANIO_MATRICULA' in c or 'ANIO_CENSO' in c]
if col_anio:
    anios_disp = sorted(df[col_anio[0]].dropna().unique().astype(int))
    anios_sel = st.sidebar.multiselect("Año de Matrícula:", anios_disp, default=anios_disp)
    df_filtrado = df[df[col_anio[0]].isin(anios_sel)].copy()
else:
    df_filtrado = df.copy()

# Filtro por Carrera
col_carr = [c for c in df.columns if 'CARRERA' in c]
if col_carr:
    carreras_disp = sorted(df_filtrado[col_carr[0]].dropna().unique().astype(str))
    carreras_sel = st.sidebar.multiselect("Carrera:", carreras_disp, default=[])
    if carreras_sel:
        df_filtrado = df_filtrado[df_filtrado[col_carr[0]].isin(carreras_sel)]

# -------------------------------------------------------------
# CABECERA Y METRICAS PRINCIPALES (KPIS)
# -------------------------------------------------------------
st.title("🎓 Análisis Exploratorio de Datos (EDA) - Matrícula y Trayectoria UBO")
st.markdown("Visualización exploratoria de la población estudiantil, origen geoespacial y dimensiones de vulnerabilidad escolar (IVE / IVM).")

m1, m2, m3, m4 = st.columns(4)
total_est = len(df_filtrado)
m1.metric("Total Estudiantes", f"{total_est:,}")

col_ret = [c for c in df_filtrado.columns if 'RETIENE' in c or 'RETENCION' in c]
if col_ret:
    tasa_ret = (df_filtrado[col_ret[0]] == 1).sum() / total_est * 100 if total_est > 0 else 0
    m2.metric("Tasa de Retención (1er Año)", f"{tasa_ret:.1f}%")
else:
    m2.metric("Tasa de Retención", "N/D")

col_rbd = [c for c in df_filtrado.columns if 'RBD' in c and 'NOM' not in c and 'COD' not in c]
if col_rbd:
    n_colegios = df_filtrado[col_rbd[0]].nunique()
    m3.metric("Colegios de Origen (RBD)", f"{n_colegios:,}")

if 'DISTANCIA_CAMPUS_KM' in df_filtrado.columns:
    dist_media = df_filtrado['DISTANCIA_CAMPUS_KM'].median()
    m4.metric("Distancia Mediana al Campus", f"{dist_media:.1f} km")

st.markdown("---")

# -------------------------------------------------------------
# SECCIÓN 1: DEMOGRAFÍA Y PERFIL ACADÉMICO
# -------------------------------------------------------------
st.subheader("1. Perfil Demográfico e Institucional")
tab1, tab2 = st.tabs(["Distribuciones Generales", "Composición de Carreras"])

with tab1:
    col_g1, col_g2 = st.columns(2)
    
    # Gráfico de Torta: Género
    col_gen = [c for c in df_filtrado.columns if 'GEN' in c or 'SEXO' in c]
    if col_gen:
        with col_g1:
            fig_gen = px.pie(
                df_filtrado, 
                names=col_gen[0], 
                title="Distribución por Género",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_gen, use_container_width=True)
            
    # Gráfico de Barras: Rango de Edad
    col_edad = [c for c in df_filtrado.columns if 'EDAD' in c]
    if col_edad:
        with col_g2:
            conteo_edad = df_filtrado[col_edad[0]].value_counts().reset_index()
            conteo_edad.columns = ['Rango', 'Cantidad']
            fig_edad = px.bar(
                conteo_edad, 
                x='Cantidad', 
                y='Rango', 
                orientation='h',
                title="Distribución por Rango de Edad",
                color='Cantidad',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_edad, use_container_width=True)

with tab2:
    col_c1, col_c2 = st.columns(2)
    
    # Top 15 Carreras por Matrícula
    if col_carr:
        with col_c1:
            top_carr = df_filtrado[col_carr[0]].value_counts().head(15).reset_index()
            top_carr.columns = ['Carrera', 'Estudiantes']
            fig_carr = px.bar(
                top_carr, 
                x='Estudiantes', 
                y='Carrera', 
                orientation='h',
                title="Top 15 Carreras con Mayor Matrícula",
                color='Estudiantes',
                color_continuous_scale='Teal'
            )
            fig_carr.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_carr, use_container_width=True)
            
    # Gráfico de Torta: Vía de Admisión / Forma de Ingreso
    col_ing = [c for c in df_filtrado.columns if 'INGRESO' in c or 'VIA' in c]
    if col_ing:
        with col_c2:
            fig_ing = px.pie(
                df_filtrado, 
                names=col_ing[0], 
                title="Distribución según Vía de Admisión",
                hole=0.35,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_ing, use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------
# SECCIÓN 2: MAPA DE CALOR GEOESPACIAL Y DENSIDAD
# -------------------------------------------------------------
st.subheader("2. Distribución Geoespacial y Densidad de Procedencia")
st.markdown("Concentración territorial de los colegios de educación media desde donde ingresaron los estudiantes a la UBO.")

if 'LATITUD' in df_filtrado.columns and 'LONGITUD' in df_filtrado.columns:
    df_geo = df_filtrado.dropna(subset=['LATITUD', 'LONGITUD']).copy()
    # Filtrar coordenadas válidas en Chile
    df_geo = df_geo[(df_geo['LATITUD'] >= -56) & (df_geo['LATITUD'] <= -17) & 
                    (df_geo['LONGITUD'] >= -76) & (df_geo['LONGITUD'] <= -66)]
    
    if not df_geo.empty:
        col_mapa, col_tabla_geo = st.columns([1.6, 1])
        
        with col_mapa:
            # Agrupar por colegio para el mapa
            col_nom_c = [c for c in df_geo.columns if 'NOM_RBD' in c or 'NOMBRE_ESTABLE' in c]
            nom_c_col = col_nom_c[0] if col_nom_c else col_rbd[0]
            
            colegios_agg = df_geo.groupby([col_rbd[0], nom_c_col, 'LATITUD', 'LONGITUD']).size().reset_index(name='TOTAL_ALUMNOS')
            
            # Crear mapa Folium centrado en Santiago
            m = folium.Map(location=[-33.4569, -70.6631], zoom_start=10, tiles='CartoDB positron')
            
            # Capa 1: HeatMap (Densidad)
            heat_data = [[row['LATITUD'], row['LONGITUD'], row['TOTAL_ALUMNOS']] for _, row in colegios_agg.iterrows()]
            HeatMap(heat_data, radius=12, blur=15, max_zoom=13).add_to(m)
            
            # Capa 2: Marcadores Clusterizados
            cluster = MarkerCluster(name="Colegios").add_to(m)
            for _, r in colegios_agg.head(500).iterrows(): # Renderizar top 500 para agilidad
                folium.CircleMarker(
                    location=[r['LATITUD'], r['LONGITUD']],
                    radius=4 + min(r['TOTAL_ALUMNOS'], 10),
                    popup=f"<b>RBD:</b> {int(r[col_rbd[0]])}<br><b>Colegio:</b> {r[nom_c_col]}<br><b>Alumnos:</b> {r['TOTAL_ALUMNOS']}",
                    color="#1f77b4",
                    fill=True,
                    fill_opacity=0.6
                ).add_to(cluster)
                
            st_folium(m, width="100%", height=480)
            
        with col_tabla_geo:
            col_comuna = [c for c in df_geo.columns if 'NOM_COM' in c or 'COMUNA' in c]
            if col_comuna:
                st.write("**Top Comunas de Procedencia**")
                tabla_com = df_geo[col_comuna[0]].value_counts().reset_index()
                tabla_com.columns = ['Comuna', 'Estudiantes']
                tabla_com['% del Total'] = (tabla_com['Estudiantes'] / len(df_geo) * 100).map("{:.2f}%".format)
                st.dataframe(tabla_com.head(15), use_container_width=True, height=430)
    else:
        st.info("No se encontraron registros con coordenadas válidas para los filtros aplicados.")
else:
    st.warning("El dataset no contiene las columnas LATITUD y LONGITUD.")

st.markdown("---")

# -------------------------------------------------------------
# SECCIÓN 3: TABLA DE DATOS DETALLADA Y DESCARGA
# -------------------------------------------------------------
st.subheader("3. Explorador de Datos y Exportación")
with st.expander("Ver tabla completa de datos"):
    st.dataframe(df_filtrado.head(100), use_container_width=True)
    
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar datos filtrados (CSV)",
        data=csv,
        file_name="Datos_Filtrados_EDA_UBO.csv",
        mime="text/csv"
    )
