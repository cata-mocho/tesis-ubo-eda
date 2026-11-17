# Plataforma Integral de Permanencia Estudiantil — UBO

Aplicación web en Streamlit que acompaña la tesis **«Modelo predictivo del riesgo de no continuidad institucional de primer año mediante aprendizaje automático en la Universidad Bernardo O’Higgins»** (Ingeniería en Informática, UBO, 2026). Autora: Catalina Baeza González. Profesor guía: Luis Ramírez.

La plataforma tiene dos módulos:

1. **Análisis exploratorio (2015–2025).** Catorce vistas, una por gráfico, organizadas en cuatro secciones. Cada vista explica qué muestra, cómo se lee y cómo se relaciona con los resultados de la tesis. Un asistente de IA especializado interpreta el gráfico con los filtros activos.
2. **Sistema de alerta temprana (cohorte 2025).** Resumen de riesgo, listado priorizado y una ficha por estudiante con su **pronóstico**, los factores SHAP que explican su puntaje y un **plan de acompañamiento preventivo** con acciones, responsables, calendario y señales de alerta.

## Resultados de la tesis en una mirada

| Indicador | Valor |
| --- | --- |
| Población de modelado | 12.541 estudiantes de primer año, cohortes 2015–2024 |
| Tasa de no continuidad institucional | 12,57 % |
| Modelo seleccionado | XGBoost optimizado (sin diferencia significativa con la regresión logística) |
| AP en prueba | 0,230 (1,8 veces la prevalencia) |
| ROC-AUC en prueba / temporal | 0,650 / 0,615 |
| Variables más importantes (SHAP) | NEM, rezago egreso-ingreso, IVM del establecimiento, jornada |
| Aporte incremental de IVE/IVM | No detectable (p = 0,092) |

El modelo tiene capacidad discriminativa moderada: **sirve para priorizar el acompañamiento, no para diagnosticar**.

## Estructura del repositorio

```
├── app.py                      # Punto de entrada y navegación
├── requirements.txt
├── ubo_app/
│   ├── config.py               # Colores, cortes del modelo y resultados de la tesis
│   ├── datos.py                # Carga y preparación de las dos bases
│   ├── vistas_eda.py           # Catálogo de las 14 vistas del análisis exploratorio
│   ├── llm.py                  # Asistente de IA especializado en la tesis
│   ├── acompanamiento.py       # Pronóstico y plan de acompañamiento por estudiante
│   └── paginas.py              # Páginas: inicio, exploratorio, alerta temprana, modelo
├── data/
│   └── README.md               # Qué archivos van aquí y con qué columnas
└── .streamlit/
    ├── config.toml             # Tema institucional
    └── secrets.toml.example    # Plantilla de claves (no se sube)
```

## Cómo ejecutarla

```bash
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Copia los archivos de datos a data/ (ver data/README.md)
streamlit run app.py
```

### Asistente de IA (opcional)

Sin clave, la app usa un motor local que ordena los hallazgos del gráfico y los relaciona con la tesis. Para usar IA generativa, copia `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml` y agrega `ANTHROPIC_API_KEY` u `OPENAI_API_KEY`. En Streamlit Community Cloud, pega las claves en **Settings › Secrets**. También se puede escribir una clave temporal en la barra lateral.

Al asistente solo se envían **tablas agregadas** del gráfico o el **perfil anónimo** de un estudiante; nunca el MRUN ni el RBD.

## Despliegue en Streamlit Community Cloud

1. Sube el repositorio a GitHub (de preferencia **privado**, porque contiene datos de estudiantes).
2. En [share.streamlit.io](https://share.streamlit.io), crea una app con `app.py` como archivo principal.
3. Agrega las claves en **Settings › Secrets**.

## Uso ético y privacidad

- Los puntajes orientan apoyos con revisión humana y **nunca** se usan para restringir matrícula, beneficios ni acceso.
- En pantalla se muestra un código seudónimo (`EST-…`) en lugar del MRUN. Para mostrar el MRUN a un equipo autorizado, define `MOSTRAR_MRUN = true` en los secretos.
- Los índices IVE e IVM describen el colegio o el territorio, no al estudiante.
- El tratamiento de datos se rige por la Ley 19.628 y, desde su entrada en vigencia, por la Ley 21.719.

## Limitaciones

- El resultado es la no continuidad **institucional**: no distingue traslado, suspensión o abandono del sistema.
- El IVM se publica desde 2016; para egresados anteriores se usó el valor de 2016.
- Las cifras de la cohorte 2025 son estimaciones: su continuidad aún no se observa.
- El plan de acompañamiento es una propuesta basada en reglas a partir de los factores del modelo; debe validarse con las unidades de acompañamiento.
