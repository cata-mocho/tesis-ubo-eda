import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

# Configuración principal de la interfaz
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
    
    archivo_encontrado = None
    for r in candidatos_ruta:
        if os.path.exists(r):
            archivo_encontrado = r
            break
            
    if not archivo_encontrado:
        raise FileNotFoundError("No se encontró 'Tablon_Maestro_Tesis_UBO.parquet' ni en la raíz ni en 'data/'.")
        
    if archivo_encontrado.endswith('.parquet'):
        df = pd.read_parquet(archivo_encontrado)
    else:
        df = pd.read_excel(archivo_encontrado)
        
    df.columns = df.columns.str.strip().str.upper()
    
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

# Filtro por Carrera
col_carr = [c for c in df.columns if 'CARRERA' in c or 'NOMB_CARRERA' in c]
if col_carr:
    carreras_disp = sorted(df_filtrado[col_carr[0]].dropna().unique().astype(str))
    carreras_sel = st.sidebar.multiselect("Filtrar por Carrera (opcional):", carreras_disp, default=[])
    if carreras_sel:
        df_filtrado = df_filtrado[df_filtrado[col_carr[0]].isin(carreras_sel)]

# Validar que no quede vacío el dataset
if df_filtrado.empty:
    st.warning("No hay registros que coincidan con la combinación de filtros seleccionada.")
    st.stop()

# -------------------------------------------------------------
# 3. CABECERA Y METRICAS PRINCIPALES (KPIS)
# -------------------------------------------------------------
st.title("🎓 Análisis Exploratorio de Datos (EDA) - Matrícula y Trayectoria UBO")
st.markdown("Visualización analítica de la población estudiantil, origen geográfico y factores de permanencia.")

m1, m2, m3, m4 = st.columns(4)
total_est = len(df_filtrado)
m1.metric("Total Estudiantes", f"{total_est:,}")

col_ret = [c for c in df_filtrado.columns if 'RETIENE' in c or 'RETENCION' in c]
if col_ret:
    tasa_ret = (df_filtrado[col_ret[0]] == 1).sum() / total_est * 100 if total_est > 0 else 0
    m2.metric("Tasa Retención 1° Año", f"{tasa_ret:.1f}%")
else:
    m2.metric("Tasa de Retención", "No disponible")

col_rbd = [c for c in df_filtrado.columns if 'RBD' in c and 'NOM' not in c and 'COD' not in c]
if col_rbd:
    n_colegios = df_filtrado[col_rbd[0]].nunique()
    m3.metric("Colegios de Origen (RBD)", f"{n_colegios:,}")

if 'DISTANCIA_CAMPUS_KM' in df_filtrado.columns:
    dist_media = df_filtrado['DISTANCIA_CAMPUS_KM'].dropna().median()
    m4.metric("Distancia Mediana Campus", f"{dist_media:.1f} km" if pd.notna(dist_media) else "S/I")

st.markdown("---")

# -------------------------------------------------------------
# 4. SECCIÓN 1: DEMOGRAFÍA Y PERFIL ACADÉMICO
# -------------------------------------------------------------
st.subheader("1. Perfil Demográfico e Institucional")
tab1, tab2 = st.tabs(["Distribuciones Sociodemográficas", "Distribución por Carrera"])

with tab1:
    col_g1, col_g2 = st.columns(2)
    
    # Gráfico de Torta: Género
    col_gen = [c for c in df_filtrado.columns if 'GEN' in c or 'SEXO' in c]
    if col_gen:
        with col_g1:
            df_filtrado['GEN_TXT'] = df_filtrado[col_gen[0]].replace({1: 'Hombre', 2: 'Mujer', '1': 'Hombre', '2': 'Mujer'}).fillna('Sin Info')
            fig_gen = px.pie(
                df_filtrado, 
                names='GEN_TXT', 
                title="Distribución por Género",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_gen, use_container_width=True)
            
    # Gráfico de Barras: Rango de Edad
    col_edad = [c for c in df_filtrado.columns if 'EDAD' in c or 'RANGO_EDAD' in c]
    if col_edad:
        with col_g2:
            conteo_edad = df_filtrado[col_edad[0]].astype(str).str.strip().value_counts().reset_index()
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
            fig_edad.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_edad, use_container_width=True)

with tab2:
    col_c1, col_c2 = st.columns(2)
    
    # Top 15 Carreras
    if col_carr:
        with col_c1:
            top_carr = df_filtrado[col_carr[0]].astype(str).value_counts().head(15).reset_index()
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
            
    # Vía de Admisión / Forma de Ingreso
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
# 5. SECCIÓN 2: MAPA DE CALOR GEOESPACIAL Y DENSIDAD
# -------------------------------------------------------------
st.subheader("2. Densidad Geoespacial de Colegios de Procedencia")
st.markdown("Distribución territorial de los establecimientos escolares de origen vinculados mediante coordenadas oficiales del MINEDUC.")

if 'LATITUD' in df_filtrado.columns and 'LONGITUD' in df_filtrado.columns:
    df_geo = df_filtrado.dropna(subset=['LATITUD', 'LONGITUD']).copy()
    
    # Filtrar coordenadas territoriales chilenas válidas
    df_geo = df_geo[(df_geo['LATITUD'] >= -56.0) & (df_geo['LATITUD'] <= -17.5) & 
                    (df_geo['LONGITUD'] >= -76.0) & (df_geo['LONGITUD'] <= -66.0)]
    
    if not df_geo.empty:
        col_mapa, col_tabla_geo = st.columns([1.6, 1])
        
        with col_mapa:
            col_nom_c = [c for c in df_geo.columns if 'NOM_RBD' in c or 'NOMBRE_ESTABLE' in c or 'NOMBRE_RBD' in c]
            nom_c_col = col_nom_c[0] if col_nom_c else col_rbd[0]
            
            # Agregación por colegio y coordenadas
            colegios_agg = (
                df_geo.groupby([col_rbd[0], nom_c_col, 'LATITUD', 'LONGITUD'])
                .size()
                .reset_index(name='TOTAL_ALUMNOS')
            )
            
            # Centro en Santiago (Campus Rondizzoni)
            mapa = folium.Map(location=[-33.4569, -70.6631], zoom_start=10, tiles='CartoDB positron')
            
            # Capa HeatMap (Densidad)
            heat_data = [[r['LATITUD'], r['LONGITUD'], r['TOTAL_ALUMNOS']] for _, r in colegios_agg.iterrows()]
            HeatMap(heat_data, radius=13, blur=15, max_zoom=13).add_to(mapa)
            
            # Capa MarkerCluster (Top 400 colegios para fluidez gráfica)
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

st.markdown("---")

# -------------------------------------------------------------
# 6. SECCIÓN 3: EXPLORADOR DETALLADO Y DESCARGA
# -------------------------------------------------------------
st.subheader("3. Vista de Datos y Exportación")
with st.expander("🔍 Explorar primeras 100 filas del subconjunto analizado"):
    st.dataframe(df_filtrado.head(100), use_container_width=True)
    
    csv_bytes = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar datos filtrados en CSV",
        data=csv_bytes,
        file_name="Datos_Filtrados_EDA_UBO.csv",
        mime="text/csv"
    )
