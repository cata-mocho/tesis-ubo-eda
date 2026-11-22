"""Catálogo de vistas del análisis exploratorio (Módulo 1).

Cada vista es independiente y devuelve tres cosas:
  - fig: el gráfico Plotly;
  - tabla: los datos agregados que se grafican (se pueden descargar y son lo único
    que se envía al asistente LLM: nunca registros individuales);
  - hallazgos: frases con cifras calculadas sobre los filtros activos.

Además, cada vista declara textos fijos que conectan el gráfico con la tesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
import plotly.express as px

from .config import AZUL_MEDIO, AZUL_UBO, COLORES_ENTORNO, COLORES_GENERO, NARANJO, SECUENCIA


@dataclass
class Vista:
    id: str
    seccion: str
    titulo: str
    pregunta: str
    tipo: str            # tipo de gráfico, en lenguaje simple
    explicacion: str     # texto único para todo público
    construir: Callable[[pd.DataFrame, dict], tuple]
    parametros: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def etiqueta_unidad(unidad: str) -> str:
    return "Estudiantes únicos" if unidad == "estudiantes" else "Registros de matrícula"


def contar(df: pd.DataFrame, por, unidad: str) -> pd.DataFrame:
    g = df.groupby(por, dropna=False)
    serie = g["MRUN"].nunique() if unidad == "estudiantes" else g.size()
    return serie.reset_index(name="N")


def pct(parte, total) -> float:
    return float(parte) / float(total) * 100 if total else 0.0


def fmt(n: float, dec: int = 0) -> str:
    """Formato chileno: punto de miles y coma decimal."""
    txt = f"{n:,.{dec}f}"
    return txt.replace(",", "§").replace(".", ",").replace("§", ".")


def estilo(fig, alto: int = 480):
    fig.update_layout(
        height=alto,
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Arial", size=13, color="#1F2937"),
        legend_title_text="",
        title=None,
        colorway=SECUENCIA,
    )
    return fig


# ---------------------------------------------------------------------------
# A. Composición de la matrícula
# ---------------------------------------------------------------------------
def v_genero(df, p):
    u = p["unidad"]
    t = contar(df, "GEN_ALU", u).rename(columns={"GEN_ALU": "Género"}).sort_values("N", ascending=False)
    t["%"] = t["N"] / t["N"].sum() * 100
    fig = px.pie(t, names="Género", values="N", hole=0.5, color="Género", color_discrete_map=COLORES_GENERO)
    fig.update_traces(textinfo="percent+label")
    h = [f"{r['Género']}: {fmt(r['N'])} ({fmt(r['%'], 1)} %)" for _, r in t.iterrows()]
    return estilo(fig), t, h


def v_evolucion(df, p):
    u = p["unidad"]
    t = contar(df, ["CAT_PERIODO", "GEN_ALU"], u).rename(columns={"CAT_PERIODO": "Año", "GEN_ALU": "Género"})
    t["Año"] = t["Año"].astype("Int64")
    fig = px.line(t, x="Año", y="N", color="Género", markers=True, color_discrete_map=COLORES_GENERO,
                  labels={"N": etiqueta_unidad(u)})
    tot = t.groupby("Año")["N"].sum()
    h = []
    if len(tot) >= 2:
        h.append(f"Total {tot.index.min()}: {fmt(tot.iloc[0])}; total {tot.index.max()}: {fmt(tot.iloc[-1])} "
                 f"({fmt(pct(tot.iloc[-1] - tot.iloc[0], tot.iloc[0]), 1)} % de variación).")
        h.append(f"Año con mayor volumen: {tot.idxmax()} ({fmt(tot.max())}).")
    muj = t[t["Género"] == "Mujer"].set_index("Año")["N"]
    if len(muj) >= 2:
        h.append(f"Participación femenina: {fmt(pct(muj.iloc[0], tot.loc[muj.index[0]]), 1)} % en {muj.index[0]} y "
                 f"{fmt(pct(muj.iloc[-1], tot.loc[muj.index[-1]]), 1)} % en {muj.index[-1]}.")
    return estilo(fig), t, h


def v_area_genero(df, p):
    u = p["unidad"]
    t = contar(df, ["AREA_CONOCIMIENTO", "GEN_ALU"], u).rename(columns={"AREA_CONOCIMIENTO": "Área", "GEN_ALU": "Género"})
    orden = t.groupby("Área")["N"].sum().sort_values().index.tolist()
    fig = px.bar(t, y="Área", x="N", color="Género", orientation="h", color_discrete_map=COLORES_GENERO,
                 category_orders={"Área": orden[::-1]}, labels={"N": etiqueta_unidad(u)})
    ancho = t.pivot_table(index="Área", columns="Género", values="N", aggfunc="sum", fill_value=0)
    ancho["% mujeres"] = ancho.get("Mujer", 0) / ancho.sum(axis=1) * 100
    h = [f"Área con mayor volumen: {orden[-1]} ({fmt(ancho.drop(columns='% mujeres').loc[orden[-1]].sum())})."]
    h.append(f"Mayor proporción de mujeres: {ancho['% mujeres'].idxmax()} ({fmt(ancho['% mujeres'].max(), 1)} %); "
             f"menor: {ancho['% mujeres'].idxmin()} ({fmt(ancho['% mujeres'].min(), 1)} %).")
    return estilo(fig, 500), t, h


def v_edad(df, p):
    u = p["unidad"]
    t = contar(df, ["RANGO_EDAD", "EDAD_MIN"], u).sort_values("EDAD_MIN").rename(columns={"RANGO_EDAD": "Rango de edad"})
    t["%"] = t["N"] / t["N"].sum() * 100
    fig = px.bar(t, x="Rango de edad", y="N", text=t["%"].map(lambda v: f"{fmt(v, 1)} %"),
                 color_discrete_sequence=[AZUL_UBO], labels={"N": etiqueta_unidad(u)})
    dom = t.loc[t["N"].idxmax()]
    adultos = t.loc[t["EDAD_MIN"] >= 25, "N"].sum()
    h = [f"Rango dominante: {dom['Rango de edad']} ({fmt(dom['%'], 1)} %).",
         f"Ingreso con 25 años o más: {fmt(pct(adultos, t['N'].sum()), 1)} %."]
    return estilo(fig), t.drop(columns="EDAD_MIN"), h


# ---------------------------------------------------------------------------
# B. Carreras y jornadas
# ---------------------------------------------------------------------------
def v_top_carreras(df, p):
    u, n = p["unidad"], p.get("top_n", 15)
    t = contar(df, "NOMB_CARRERA", u).rename(columns={"NOMB_CARRERA": "Carrera"}).sort_values("N", ascending=False)
    total = t["N"].sum()
    top = t.head(n).sort_values("N")
    fig = px.bar(top, x="N", y="Carrera", orientation="h", text="N", color_discrete_sequence=[AZUL_UBO],
                 labels={"N": etiqueta_unidad(u)})
    h = [f"Carrera con mayor volumen: {t.iloc[0]['Carrera']} ({fmt(t.iloc[0]['N'])}).",
         f"Las 5 carreras más grandes concentran el {fmt(pct(t.head(5)['N'].sum(), total), 1)} % del total.",
         f"Carreras distintas en el filtro: {len(t)}."]
    return estilo(fig, 560), top.sort_values("N", ascending=False), h


def v_vespertina(df, p):
    u = p["unidad"]
    tot = contar(df, "NOMB_CARRERA", u).rename(columns={"N": "Total"})
    ves = contar(df[df["JORNADA"].str.contains("vespert", case=False, na=False)], "NOMB_CARRERA", u).rename(columns={"N": "Vespertina"})
    t = tot.merge(ves, on="NOMB_CARRERA", how="inner").rename(columns={"NOMB_CARRERA": "Carrera"})
    if t.empty:
        return None, t, ["No hay registros en jornada vespertina con los filtros actuales."]
    t["% vespertina"] = t["Vespertina"] / t["Total"] * 100
    top = t.sort_values("Vespertina", ascending=False).head(10).sort_values("Vespertina")
    fig = px.bar(top, x="Vespertina", y="Carrera", orientation="h", color_discrete_sequence=[NARANJO],
                 text=top["% vespertina"].map(lambda v: f"{fmt(v, 0)} % de la carrera"),
                 labels={"Vespertina": etiqueta_unidad(u)})
    h = [f"Jornada vespertina: {fmt(pct(t['Vespertina'].sum(), tot['Total'].sum()), 1)} % del total filtrado.",
         f"Carrera con más estudiantes vespertinos: {top.iloc[-1]['Carrera']} ({fmt(top.iloc[-1]['Vespertina'])})."]
    return estilo(fig, 520), top.sort_values("Vespertina", ascending=False), h


def v_adultos(df, p):
    u = p["unidad"]
    tot = contar(df, "NOMB_CARRERA", u).rename(columns={"N": "Total"})
    adu = contar(df[df["EDAD_MIN"] >= 25], "NOMB_CARRERA", u).rename(columns={"N": "25 años o más"})
    t = tot.merge(adu, on="NOMB_CARRERA", how="inner").rename(columns={"NOMB_CARRERA": "Carrera"})
    t = t[t["Total"] >= 30]
    if t.empty:
        return None, t, ["No hay carreras con al menos 30 registros y estudiantes de 25 años o más."]
    t["% 25 años o más"] = t["25 años o más"] / t["Total"] * 100
    top = t.sort_values("% 25 años o más", ascending=False).head(10).sort_values("% 25 años o más")
    fig = px.bar(top, x="% 25 años o más", y="Carrera", orientation="h", color_discrete_sequence=[AZUL_MEDIO],
                 text=top["% 25 años o más"].map(lambda v: f"{fmt(v, 0)} %"))
    h = [f"Mayor proporción de ingreso con 25 años o más: {top.iloc[-1]['Carrera']} ({fmt(top.iloc[-1]['% 25 años o más'], 1)} %).",
         "Solo se incluyen carreras con al menos 30 registros."]
    return estilo(fig, 520), top.sort_values("% 25 años o más", ascending=False), h


# ---------------------------------------------------------------------------
# C. Trayectoria escolar (rezago)
# ---------------------------------------------------------------------------
def _rezago(df):
    return df.dropna(subset=["BRECHA_TEMPORAL"]).query("BRECHA_TEMPORAL >= 0")


def v_rezago(df, p):
    d = _rezago(df)
    b = d["BRECHA_TEMPORAL"]
    fig = px.histogram(d[b <= 20], x="BRECHA_TEMPORAL", nbins=21, color_discrete_sequence=[AZUL_UBO],
                       labels={"BRECHA_TEMPORAL": "Años entre el egreso de media y el ingreso"})
    fig.update_yaxes(title="Registros")
    t = pd.DataFrame({
        "Indicador": ["Promedio (años)", "Mediana (años)", "Ingreso directo, 0–1 año (%)", "Rezago mayor a 3 años (%)", "Más de 20 años (%)"],
        "Valor": [b.mean(), b.median(), b.between(0, 1).mean() * 100, (b > 3).mean() * 100, (b > 20).mean() * 100],
    }).round(2)
    h = [f"Promedio {fmt(b.mean(), 2)} años; mediana {fmt(b.median(), 0)} años.",
         f"El {fmt(b.between(0, 1).mean() * 100, 1)} % ingresa dentro de un año desde el egreso.",
         f"El {fmt((b > 3).mean() * 100, 1)} % tiene un rezago mayor a 3 años."]
    return estilo(fig), t, h


def v_rezago_area(df, p):
    d = _rezago(df)
    orden = d.groupby("AREA_CONOCIMIENTO")["BRECHA_TEMPORAL"].median().sort_values(ascending=False).index.tolist()
    fig = px.box(d, x="AREA_CONOCIMIENTO", y="BRECHA_TEMPORAL", points="outliers", color_discrete_sequence=[AZUL_UBO],
                 category_orders={"AREA_CONOCIMIENTO": orden},
                 labels={"AREA_CONOCIMIENTO": "Área", "BRECHA_TEMPORAL": "Rezago (años)"})
    fig.update_xaxes(tickangle=-25)
    t = d.groupby("AREA_CONOCIMIENTO")["BRECHA_TEMPORAL"].describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%", "max"]]
    t = t.rename(columns={"count": "Registros", "50%": "Mediana", "max": "Máximo"}).round(1).reset_index().rename(columns={"AREA_CONOCIMIENTO": "Área"})
    h = [f"Mayor rezago mediano: {orden[0]} ({fmt(t.set_index('Área').loc[orden[0], 'Mediana'], 1)} años).",
         f"Menor rezago mediano: {orden[-1]} ({fmt(t.set_index('Área').loc[orden[-1], 'Mediana'], 1)} años)."]
    return estilo(fig, 520), t, h


def v_rezago_entorno(df, p):
    d = _rezago(df)
    t = d.groupby("ENTORNO_ESCOLAR")["BRECHA_TEMPORAL"].agg(Promedio="mean", Mediana="median", Registros="count").round(2).reset_index()
    t = t.rename(columns={"ENTORNO_ESCOLAR": "Entorno"})
    fig = px.bar(t, x="Entorno", y="Promedio", color="Entorno", color_discrete_map=COLORES_ENTORNO,
                 text=t["Promedio"].map(lambda v: f"{fmt(v, 2)} años"), labels={"Promedio": "Rezago promedio (años)"})
    fig.update_layout(showlegend=False)
    h = [f"{r['Entorno']}: promedio {fmt(r['Promedio'], 2)} años (n = {fmt(r['Registros'])})." for _, r in t.iterrows()]
    return estilo(fig), t, h


# ---------------------------------------------------------------------------
# D. Territorio y vulnerabilidad
# ---------------------------------------------------------------------------
def v_entorno(df, p):
    u = p["unidad"]
    t = contar(df, "ENTORNO_ESCOLAR", u).rename(columns={"ENTORNO_ESCOLAR": "Entorno"})
    t["%"] = t["N"] / t["N"].sum() * 100
    fig = px.pie(t, names="Entorno", values="N", hole=0.5, color="Entorno", color_discrete_map=COLORES_ENTORNO)
    fig.update_traces(textinfo="percent+label")
    h = [f"{r['Entorno']}: {fmt(r['N'])} ({fmt(r['%'], 1)} %)." for _, r in t.iterrows()]
    return estilo(fig), t, h


def v_ivm_region(df, p):
    d = df.dropna(subset=["IVM_REGION"])
    if d.empty:
        return None, d, ["No hay valores de IVM regional con los filtros actuales."]
    t = d.groupby("NOM_REG_RBD_A").agg(IVM_promedio=("IVM_REGION", "mean"), Registros=("IVM_REGION", "size")).round(2)
    t = t.sort_values("IVM_promedio").reset_index().rename(columns={"NOM_REG_RBD_A": "Región", "IVM_promedio": "IVM regional promedio"})
    fig = px.bar(t, x="IVM regional promedio", y="Región", orientation="h", color_discrete_sequence=[AZUL_UBO],
                 text=t["IVM regional promedio"].map(lambda v: fmt(v, 1)), hover_data={"Registros": True})
    fig.update_xaxes(title="IVM regional promedio (puntaje 0–100; mayor = más vulnerable)")
    h = [f"Mayor IVM regional: {t.iloc[-1]['Región']} ({fmt(t.iloc[-1]['IVM regional promedio'], 1)} puntos).",
         f"Menor IVM regional: {t.iloc[0]['Región']} ({fmt(t.iloc[0]['IVM regional promedio'], 1)} puntos).",
         f"Promedio ponderado de los registros: {fmt(d['IVM_REGION'].mean(), 1)} puntos."]
    return estilo(fig, 560), t.sort_values("IVM regional promedio", ascending=False), h


def v_comunas(df, p):
    u = p["unidad"]
    t = contar(df, ["NOM_COM_RBD", "NOM_REG_RBD_A"], u).rename(columns={"NOM_COM_RBD": "Comuna", "NOM_REG_RBD_A": "Región"})
    t = t.sort_values("N", ascending=False)
    total = t["N"].sum()
    top = t.head(15).sort_values("N")
    fig = px.bar(top, x="N", y="Comuna", orientation="h", text="N", color_discrete_sequence=[AZUL_UBO],
                 hover_data={"Región": True}, labels={"N": etiqueta_unidad(u)})
    rm = t[t["Región"].str.contains("METROPOLITANA|RM", case=False, na=False)]["N"].sum()
    h = [f"Comuna con más estudiantes: {t.iloc[0]['Comuna']} ({fmt(t.iloc[0]['N'])}).",
         f"Las 15 primeras comunas suman el {fmt(pct(top['N'].sum(), total), 1)} % del total.",
         f"Región Metropolitana: {fmt(pct(rm, total), 1)} % del total."]
    return estilo(fig, 560), top.sort_values("N", ascending=False), h


def v_mapa(df, p):
    cobertura = p.get("cobertura", "Todo Chile")
    m = df.dropna(subset=["LATITUD", "LONGITUD"])
    m = m[m["LATITUD"].between(-57, -17) & m["LONGITUD"].between(-76, -66)]
    if cobertura == "Región Metropolitana":
        m = m[m["NOM_REG_RBD_A"].str.contains(r"\bRM\b|METROPOLITANA", case=False, na=False, regex=True)]
    if m.empty:
        return None, m, ["No hay coordenadas válidas con los filtros actuales."]
    muestra = m.sample(15000, random_state=42) if len(m) > 15000 else m
    rm = cobertura == "Región Metropolitana"
    kwargs = dict(lat="LATITUD", lon="LONGITUD", radius=10, zoom=8 if rm else 3.5,
                  center={"lat": -33.45, "lon": -70.66} if rm else {"lat": -35.5, "lon": -71.0},
                  hover_name="NOM_RBD", hover_data={"NOM_COM_RBD": True}, color_continuous_scale="Blues")
    if hasattr(px, "density_map"):
        fig = px.density_map(muestra, map_style="open-street-map", **kwargs)
    else:  # plotly < 5.24
        fig = px.density_mapbox(muestra, mapbox_style="open-street-map", **kwargs)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0))
    t = m.groupby(["NOM_COM_RBD", "NOM_REG_RBD_A"]).agg(Establecimientos=("RBD", "nunique"), Registros=("RBD", "size"))
    t = t.sort_values("Registros", ascending=False).head(20).reset_index().rename(columns={"NOM_COM_RBD": "Comuna", "NOM_REG_RBD_A": "Región"})
    h = [f"Registros georreferenciados: {fmt(len(m))}; establecimientos distintos: {fmt(m['RBD'].nunique())}.",
         f"Comuna con mayor concentración: {t.iloc[0]['Comuna']} ({fmt(t.iloc[0]['Registros'])} registros)."]
    return fig, t, h


# ---------------------------------------------------------------------------
# Catálogo
# ---------------------------------------------------------------------------
VISTAS: list[Vista] = [
    Vista("genero", "A. Composición de la matrícula", "Proporción por género",
          "¿Cómo se distribuye la matrícula entre hombres y mujeres?",
          "Gráfico de anillo",
          "El círculo completo representa a todos los estudiantes del filtro actual. Cada porción es la "
          "parte que corresponde a un género: mientras más grande el sector, mayor su participación. "
          "El porcentaje de cada sector aparece escrito sobre él.",
          v_genero),
    Vista("evolucion", "A. Composición de la matrícula", "Evolución anual por género",
          "¿Cómo cambió la matrícula en el tiempo?",
          "Gráfico de líneas",
          "El eje horizontal muestra los años y el vertical, la cantidad de estudiantes. Cada línea es un "
          "género: si sube, la matrícula de ese grupo aumentó ese año; si baja, disminuyó. "
          "Sirve para ver tendencias y comparar el ritmo de cambio entre grupos.",
          v_evolucion),
    Vista("area_genero", "A. Composición de la matrícula", "Matrícula por área y género",
          "¿Qué áreas concentran la matrícula y cómo se compone por género?",
          "Barras horizontales apiladas",
          "Cada barra es un área del conocimiento y su largo total indica cuántos estudiantes tiene. "
          "Los colores dividen cada barra por género, por lo que se ve a la vez qué áreas son más grandes "
          "y cómo se reparte su matrícula entre hombres y mujeres.",
          v_area_genero),
    Vista("edad", "A. Composición de la matrícula", "Rango de edad al ingreso",
          "¿Qué tan tradicional es el perfil etario de quienes ingresan?",
          "Gráfico de barras verticales",
          "Cada barra corresponde a un tramo de edad al momento de ingresar, ordenados de menor a mayor. "
          "Mientras más alta la barra, más estudiantes hay en ese tramo; el porcentaje sobre cada barra "
          "indica qué parte del total representa.",
          v_edad),
    Vista("top_carreras", "B. Carreras y jornadas", "Carreras con mayor matrícula",
          "¿Qué carreras concentran a la mayoría de los estudiantes?",
          "Barras horizontales (ranking)",
          "Las carreras están ordenadas de mayor a menor número de estudiantes. El largo de cada barra es "
          "su matrícula y el número a su lado es el valor exacto. Puedes elegir cuántas carreras mostrar "
          "sobre el gráfico.",
          v_top_carreras, {"top_n": 15}),
    Vista("vespertina", "B. Carreras y jornadas", "Carreras en jornada vespertina",
          "¿Dónde se concentran los estudiantes vespertinos?",
          "Barras horizontales (ranking)",
          "Muestra las 10 carreras con más estudiantes en jornada vespertina. El largo de la barra es la "
          "cantidad de estudiantes vespertinos y la etiqueta indica qué porcentaje de la carrera "
          "representan: una carrera puede tener pocos vespertinos en número, pero ser mayoritaria en esa jornada.",
          v_vespertina),
    Vista("adultos", "B. Carreras y jornadas", "Carreras con ingreso de 25 años o más",
          "¿Qué carreras reciben más estudiantes adultos?",
          "Barras horizontales (ranking por porcentaje)",
          "Cada barra indica qué porcentaje de los estudiantes de una carrera ingresó con 25 años o más. "
          "El orden es por porcentaje y no por cantidad, y solo se incluyen carreras con al menos 30 "
          "registros para evitar porcentajes poco confiables.",
          v_adultos),
    Vista("rezago", "C. Trayectoria escolar", "Distribución del rezago",
          "¿Cuántos años pasan entre el egreso de media y el ingreso a la UBO?",
          "Histograma",
          "El rezago es el tiempo entre terminar la enseñanza media y entrar a la universidad. Cada barra "
          "agrupa a los estudiantes con el mismo número de años; mientras más alta, más estudiantes. "
          "Las barras cercanas a cero corresponden a quienes ingresan de inmediato. El gráfico omite "
          "rezagos sobre 20 años, pero los indicadores sí los consideran.",
          v_rezago),
    Vista("rezago_area", "C. Trayectoria escolar", "Rezago por área del conocimiento",
          "¿Qué áreas reciben estudiantes con más años fuera del sistema?",
          "Diagrama de cajas",
          "Cada caja resume el rezago de un área: la línea del medio es la mediana (el valor central), la "
          "caja contiene al 50 % central de los estudiantes y los puntos aislados son casos extremos. "
          "Una caja más alta indica estudiantes que tardaron más años en ingresar.",
          v_rezago_area),
    Vista("rezago_entorno", "C. Trayectoria escolar", "Rezago según entorno escolar",
          "¿Los egresados de colegios rurales tardan más en ingresar?",
          "Gráfico de barras",
          "Compara el rezago promedio, en años, entre estudiantes de colegios urbanos y rurales. "
          "Conviene revisar la cantidad de casos (n) de cada grupo: cuando uno es muy pequeño, su promedio "
          "puede variar mucho y debe leerse con cautela.",
          v_rezago_entorno),
    Vista("entorno", "D. Territorio y vulnerabilidad", "Entorno del establecimiento de egreso",
          "¿Qué parte de los estudiantes viene de colegios rurales?",
          "Gráfico de anillo",
          "El círculo completo representa a todos los estudiantes del filtro actual y cada sector, la "
          "parte que egresó de un colegio urbano o rural. Mientras más grande el sector, mayor su peso.",
          v_entorno),
    Vista("ivm_region", "D. Territorio y vulnerabilidad", "Vulnerabilidad regional (IVM)",
          "¿Desde qué regiones llegan estudiantes de contextos más vulnerables?",
          "Barras horizontales",
          "El Índice de Vulnerabilidad Multidimensional (IVM) de JUNAEB va de 0 a 100: mientras mayor el "
          "puntaje, más vulnerable es el contexto escolar. Cada barra es el promedio de una región. "
          "Es un indicador del territorio, no una característica de cada estudiante.",
          v_ivm_region),
    Vista("comunas", "D. Territorio y vulnerabilidad", "Comunas de procedencia",
          "¿Cuál es la zona de influencia territorial de la UBO?",
          "Barras horizontales (ranking)",
          "Muestra las 15 comunas donde estudiaron la enseñanza media la mayor cantidad de estudiantes. "
          "El largo de la barra es la cantidad de estudiantes; al pasar el cursor se ve la región de la comuna.",
          v_comunas),
    Vista("mapa", "D. Territorio y vulnerabilidad", "Mapa de establecimientos de egreso",
          "¿Dónde están geográficamente los colegios de egreso?",
          "Mapa de calor",
          "Cada punto es un colegio de egreso. Las zonas con color más intenso concentran más estudiantes "
          "provenientes de colegios cercanos entre sí. Puedes acercar, alejar o mover el mapa, y cambiar "
          "entre Todo Chile y la Región Metropolitana.",
          v_mapa, {"cobertura": "Todo Chile"}),
]


SECCIONES = list(dict.fromkeys(v.seccion for v in VISTAS))
POR_ID = {v.id: v for v in VISTAS}
