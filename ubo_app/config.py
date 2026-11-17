"""Constantes de la plataforma: colores, cortes del modelo y resultados de la tesis.

Todas las cifras de RESULTADOS_TESIS provienen del documento de tesis
(Baeza, 2026). Si el documento cambia, actualízalas aquí: el resto de la
aplicación (textos, contexto del asistente LLM) las lee desde este archivo.
"""

TITULO_APP = "Plataforma Integral de Permanencia Estudiantil — UBO"
SUBTITULO_APP = (
    "Tesis: Modelo predictivo del riesgo de no continuidad institucional de primer año "
    "mediante aprendizaje automático · Catalina Baeza González"
)

# ---------------------------------------------------------------------------
# Paleta (alineada con la presentación institucional UBO)
# ---------------------------------------------------------------------------
AZUL_UBO = "#053380"
AZUL_MEDIO = "#1F5FAD"
NARANJO = "#B5531A"
VERDE = "#2E7D3E"
GRIS = "#8A93A3"

COLORES_GENERO = {"Hombre": AZUL_MEDIO, "Mujer": NARANJO, "Sin información": GRIS}
COLORES_TRAMO = {"Alto": "#C0392B", "Medio": "#E0A800", "Bajo": VERDE}
COLORES_ENTORNO = {"Urbano": AZUL_MEDIO, "Rural": VERDE, "Sin información": GRIS}
ORDEN_TRAMO = ["Alto", "Medio", "Bajo"]
SECUENCIA = [AZUL_UBO, NARANJO, VERDE, "#6A3D9A", "#0E7C86", "#A23B72", GRIS]

# ---------------------------------------------------------------------------
# Cortes del modelo (escala recalibrada de la cohorte 2025, sección 4.10)
# ---------------------------------------------------------------------------
CORTE_TRAMO_MEDIO = 0.102
CORTE_TRAMO_ALTO = 0.150
UMBRAL_ALERTA = 0.140
TASA_ESPERADA_2025 = 0.0928
TASA_REFERENCIA_2022_2024 = 0.0863

# ---------------------------------------------------------------------------
# Resultados de la tesis usados como contexto en textos y en el asistente
# ---------------------------------------------------------------------------
RESULTADOS_TESIS = {
    "poblacion": 12541,
    "tasa_no_continuidad": 12.57,
    "cohortes": "2015–2024",
    "modelo": "XGBoost optimizado",
    "ap_prueba": 0.230,
    "roc_auc_prueba": 0.650,
    "roc_auc_temporal": 0.615,
    "prevalencia": 0.126,
    "umbral": 0.160,
    "recall": 36.7,
    "precision": 21.8,
    "tasa_tramo_alto": 26.9,
    "casos_tramo_bajo": 54.4,
}

TASAS_DESCRIPTIVAS = {
    "jornada": {"Vespertina": 26.1, "Diurna": 11.1},
    "genero": {"Hombre": 15.0, "Mujer": 11.3},
    "edad": {"15 a 19 años": 10.7, "20 a 24 años": 13.8, "25 a 29 años": 19.7},
    "cuartil_ivm": {"Bajo": 10.0, "Medio": 11.4, "Alto": 13.0, "Muy alto": 15.9, "Sin IVM (particular pagado)": 12.7},
}

IMPORTANCIA_SHAP = [
    ("NEM (promedio de notas de enseñanza media)", 17.0, "mayor nota, menor riesgo"),
    ("Rezago entre egreso e ingreso", 10.6, "relación no lineal"),
    ("IVM del establecimiento", 10.5, "mayor IVM, mayor riesgo"),
    ("Jornada", 10.1, "vespertina, mayor riesgo"),
    ("Carrera", 7.6, "categórica"),
    ("IVE-SINAE comunal", 6.9, "ajuste relativo al establecimiento"),
    ("Región del establecimiento", 6.0, "categórica"),
    ("Área del conocimiento", 5.5, "categórica"),
    ("Vía de ingreso", 5.1, "categórica"),
    ("Comuna del establecimiento", 3.9, "categórica"),
]

CONTEXTO_TESIS = f"""
Tesis de Ingeniería en Informática (UBO, 2026): «Modelo predictivo del riesgo de no
continuidad institucional de primer año mediante aprendizaje automático en la
Universidad Bernardo O'Higgins». Metodología CRISP-DM.

- Resultado estudiado: no continuidad institucional = el estudiante de primer año no
  registra matrícula en la UBO el año siguiente. No distingue traslado de abandono.
- Población: {RESULTADOS_TESIS['poblacion']:,} estudiantes, cohortes 2015–2024; tasa de no
  continuidad {RESULTADOS_TESIS['tasa_no_continuidad']} %. La tasa bajó de 15–18 % antes de 2020
  a 7–10 % después.
- Fuentes: matrícula UBO, egreso de enseñanza media (MINEDUC) e índices de
  vulnerabilidad JUNAEB (IVE-SINAE e IVM), enlazados por MRUN y por (RBD, año de egreso).
- Modelo seleccionado: XGBoost optimizado. AP 0,230 (1,8 veces la prevalencia de 0,126) y
  ROC-AUC 0,650 en prueba; ROC-AUC temporal medio 0,615. Sin diferencia significativa con la
  regresión logística (p = 0,470). Capacidad discriminativa moderada: sirve para priorizar,
  no para diagnosticar.
- Umbral 0,160: alerta al 21 % de los estudiantes, detecta el 36,7 % de los casos; el 21,8 %
  de las alertas son casos reales. El tramo alto (10 % de mayor riesgo) tiene 26,9 % de no
  continuidad, pero el 54,4 % de los casos queda en el tramo bajo.
- Tasas descriptivas: vespertina 26,1 % vs diurna 11,1 %; hombres 15,0 % vs mujeres 11,3 %;
  edad 15–19: 10,7 %, 20–24: 13,8 %, 25–29: 19,7 %; cuartil IVM bajo 10,0 % → muy alto 15,9 %.
- SHAP (importancia): NEM 17,0 %, rezago 10,6 %, IVM del establecimiento 10,5 %, jornada
  10,1 %, carrera 7,6 %. Describen el funcionamiento del modelo, no causas.
- Vulnerabilidad: se asocia al riesgo y el modelo la usa, pero retirar el bloque IVE/IVM no
  cambió el desempeño de forma significativa (diferencia de AP −0,015; p = 0,092). Los índices
  son del establecimiento o del territorio, no del estudiante (falacia ecológica).
- Equidad: con umbral único, el 89 % de los estudiantes vespertinos recibe alerta; brechas
  por género y edad; grupos pequeños (rural, agropecuaria) con estimaciones inestables.
- Uso ético: los puntajes orientan apoyos con revisión humana; nunca restringen matrícula,
  beneficios ni acceso.
""".strip()
