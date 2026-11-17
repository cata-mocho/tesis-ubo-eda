"""Carga, limpieza y preparación de las dos bases de la plataforma."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .config import CORTE_TRAMO_ALTO, CORTE_TRAMO_MEDIO

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"

COLUMNAS_EDA = [
    "MRUN", "CAT_PERIODO", "GEN_ALU", "JORNADA", "RANGO_EDAD", "NOMB_CARRERA",
    "AREA_CONOCIMIENTO", "ANIO_EGRESO_MEDIA", "RBD", "NOM_REG_RBD_A", "NOM_COM_RBD",
    "NOM_DEPROV_RBD", "LATITUD", "LONGITUD", "NOM_RBD", "RURAL_RBD", "IVM_REGION",
]
COLUMNAS_SAT = ["MRUN", "NOMB_CARRERA", "JORNADA", "PROB_DESERCION"]


# ---------------------------------------------------------------------------
# Búsqueda de archivos en data/ o en la raíz del repositorio
# ---------------------------------------------------------------------------
def _buscar(patrones: list[str]) -> Path | None:
    for base in (DIR_DATOS, RAIZ):
        if not base.exists():
            continue
        for patron in patrones:
            hits = sorted(base.glob(patron))
            if hits:
                return hits[0]
    return None


def ruta_eda() -> Path | None:
    return _buscar(["*Tablon*.parquet", "*.parquet"])


def ruta_sat() -> Path | None:
    return _buscar(["*Scoring*2025*.csv", "*Scoring*2025*.xlsx", "*Predicciones*2025*.csv"])


# ---------------------------------------------------------------------------
# Módulo 1: base descriptiva de matrícula
# ---------------------------------------------------------------------------
def preparar_eda(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
    faltan = [c for c in COLUMNAS_EDA if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en la base descriptiva: {faltan}")

    for col in ["CAT_PERIODO", "ANIO_EGRESO_MEDIA", "RURAL_RBD", "LATITUD", "LONGITUD", "IVM_REGION"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["GEN_ALU", "JORNADA", "RANGO_EDAD", "NOMB_CARRERA", "AREA_CONOCIMIENTO",
                "NOM_REG_RBD_A", "NOM_COM_RBD", "NOM_DEPROV_RBD", "NOM_RBD"]:
        df[col] = df[col].astype("string").str.strip().fillna("Sin información")

    df["BRECHA_TEMPORAL"] = df["CAT_PERIODO"] - df["ANIO_EGRESO_MEDIA"]
    df["ENTORNO_ESCOLAR"] = df["RURAL_RBD"].map({0: "Urbano", 1: "Rural"}).fillna("Sin información")
    df["EDAD_MIN"] = pd.to_numeric(df["RANGO_EDAD"].str.extract(r"(\d{2})", expand=False), errors="coerce")
    return df


@st.cache_data(show_spinner="Cargando la base descriptiva de matrícula…")
def cargar_eda_ruta(ruta: str) -> pd.DataFrame:
    return preparar_eda(pd.read_parquet(ruta, engine="pyarrow"))


@st.cache_data(show_spinner="Cargando la base descriptiva de matrícula…")
def cargar_eda_bytes(contenido: bytes) -> pd.DataFrame:
    return preparar_eda(pd.read_parquet(io.BytesIO(contenido), engine="pyarrow"))


# ---------------------------------------------------------------------------
# Módulo 2: puntajes de riesgo de la cohorte 2025
# ---------------------------------------------------------------------------
def id_seguro(mrun) -> str:
    """Código estable y no reversible para mostrar en pantalla en lugar del MRUN."""
    return "EST-" + hashlib.sha256(str(mrun).encode()).hexdigest()[:6].upper()


def tramo_desde_prob(p: float) -> str:
    if p >= CORTE_TRAMO_ALTO:
        return "Alto"
    if p >= CORTE_TRAMO_MEDIO:
        return "Medio"
    return "Bajo"


_PATRON_FACTOR = re.compile(r"^\s*(?P<var>[^:]+?)\s*:\s*(?P<valor>.*)$")


def separar_factor(texto) -> tuple[str, str]:
    """'NEM (prom. notas Media): 4.42' -> ('NEM (prom. notas Media)', '4.42')."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return "", ""
    m = _PATRON_FACTOR.match(str(texto))
    if not m:
        return str(texto).strip(), ""
    return m.group("var").strip(), m.group("valor").strip()


def preparar_sat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "PROB_DESERCION" not in df.columns and "PROBABILIDAD_DESERCION" in df.columns:
        df["PROB_DESERCION"] = df["PROBABILIDAD_DESERCION"]
    if "TRAMO_RIESGO" not in df.columns and "NIVEL_RIESGO" in df.columns:
        df["TRAMO_RIESGO"] = df["NIVEL_RIESGO"]
    faltan = [c for c in COLUMNAS_SAT if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el archivo de puntajes: {faltan}")

    df["PROB_DESERCION"] = pd.to_numeric(df["PROB_DESERCION"], errors="coerce")
    if "TRAMO_RIESGO" not in df.columns:
        # Cortes de la tesis en la escala recalibrada de 2025 (no 0,25 / 0,12).
        df["TRAMO_RIESGO"] = df["PROB_DESERCION"].apply(tramo_desde_prob)
    if "PERCENTIL_RIESGO_COHORTE" not in df.columns:
        df["PERCENTIL_RIESGO_COHORTE"] = (df["PROB_DESERCION"].rank(pct=True) * 100).round(1)

    for i in (1, 2, 3):
        col = f"FACTOR_RIESGO_{i}"
        if col not in df.columns:
            df[col] = np.nan
        partes = df[col].apply(separar_factor)
        df[f"FACTOR_{i}_VAR"] = partes.str[0]
        df[f"FACTOR_{i}_VALOR"] = partes.str[1]

    df["ID"] = df["MRUN"].apply(id_seguro)
    df = df.sort_values("PROB_DESERCION", ascending=False).reset_index(drop=True)
    df["PRIORIDAD"] = np.arange(1, len(df) + 1)
    return df


@st.cache_data(show_spinner="Cargando los puntajes de la cohorte 2025…")
def cargar_sat_ruta(ruta: str) -> pd.DataFrame:
    leer = pd.read_excel if ruta.endswith(".xlsx") else pd.read_csv
    return preparar_sat(leer(ruta))


@st.cache_data(show_spinner="Cargando los puntajes de la cohorte 2025…")
def cargar_sat_bytes(contenido: bytes, nombre: str) -> pd.DataFrame:
    buffer = io.BytesIO(contenido)
    df = pd.read_excel(buffer) if nombre.endswith(".xlsx") else pd.read_csv(buffer)
    return preparar_sat(df)


# ---------------------------------------------------------------------------
# Acceso unificado con carga manual opcional desde la barra lateral
# ---------------------------------------------------------------------------
def obtener_eda() -> pd.DataFrame | None:
    with st.sidebar.expander("Fuente de datos", expanded=False):
        subido = st.file_uploader("Reemplazar base descriptiva (.parquet)", type=["parquet"], key="up_eda")
    try:
        if subido is not None:
            return cargar_eda_bytes(subido.getvalue())
        ruta = ruta_eda()
        if ruta is not None:
            return cargar_eda_ruta(str(ruta))
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo leer la base descriptiva: {exc}")
        return None
    st.warning("No se encontró el archivo .parquet en `data/`. Súbelo desde la barra lateral.")
    return None


def obtener_sat() -> pd.DataFrame | None:
    with st.sidebar.expander("Fuente de datos", expanded=False):
        subido = st.file_uploader("Reemplazar puntajes 2025 (.csv/.xlsx)", type=["csv", "xlsx"], key="up_sat")
    try:
        if subido is not None:
            return cargar_sat_bytes(subido.getvalue(), subido.name)
        ruta = ruta_sat()
        if ruta is not None:
            return cargar_sat_ruta(str(ruta))
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo leer el archivo de puntajes: {exc}")
        return None
    st.warning("No se encontró `Scoring_Cohorte2025_Riesgo_Desercion.csv` en `data/`. Súbelo desde la barra lateral.")
    return None
