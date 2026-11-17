"""Pronóstico y plan de acompañamiento preventivo por estudiante (Módulo 2).

El plan se arma con reglas transparentes a partir de:
  - el tramo de riesgo, que define la intensidad del seguimiento;
  - los tres factores SHAP que más elevaron el puntaje, que definen las dimensiones
    de apoyo.
Es una propuesta para revisión humana: el puntaje prioriza, no diagnostica.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import TASA_ESPERADA_2025


@dataclass(frozen=True)
class Dimension:
    nombre: str
    lectura: str
    acciones: tuple[str, ...]
    responsable: str
    indicador: str


ACADEMICA = Dimension(
    "Académica",
    "Antecedentes académicos de ingreso más bajos que el promedio de la cohorte.",
    ("Diagnóstico de entrada en las asignaturas críticas de primer año.",
     "Derivación a tutoría par o ayudantía en las primeras tres semanas.",
     "Taller de hábitos y planificación del estudio."),
    "Unidad de apoyo académico / tutor asignado",
    "Asistencia a tutorías y nota de la primera evaluación de las asignaturas críticas.",
)
TRAYECTORIA = Dimension(
    "Trayectoria y reinserción",
    "Varios años entre el egreso de la enseñanza media y el ingreso.",
    ("Taller de reinserción académica y nivelación de contenidos base.",
     "Plan semanal de estudio acordado con el tutor.",
     "Presentación de los servicios de apoyo (biblioteca, plataforma, horarios de atención)."),
    "Tutor asignado y coordinación de carrera",
    "Cumplimiento del plan semanal y entregas a tiempo en el primer mes.",
)
COMPATIBILIDAD = Dimension(
    "Compatibilidad trabajo, familia y estudio",
    "Jornada vespertina, asociada a mayor carga laboral o familiar.",
    ("Entrevista inicial sobre carga laboral, familiar y tiempos de traslado.",
     "Acceso a material en línea y horarios de atención vespertinos.",
     "Orientación para la carga académica del semestre."),
    "Coordinación de jornada vespertina",
    "Asistencia y conexión a la plataforma en las primeras seis semanas.",
)
NO_TRADICIONAL = Dimension(
    "Estudiante no tradicional",
    "Edad de ingreso mayor que la del perfil tradicional.",
    ("Mentoría con estudiantes de cursos superiores con perfil similar.",
     "Orientación sobre la vida universitaria y los servicios disponibles."),
    "Dirección de Asuntos Estudiantiles (DAE)",
    "Participación en la mentoría durante el primer semestre.",
)
SOCIOECONOMICA = Dimension(
    "Contexto socioeconómico",
    "El colegio o el territorio de origen tiene un índice de vulnerabilidad alto. "
    "Es un dato del contexto, no del estudiante: confirmar en entrevista.",
    ("Entrevista con DAE para revisar beneficios (gratuidad, becas, alimentación, transporte).",
     "Información sobre apoyo psicosocial y redes de la universidad."),
    "Dirección de Asuntos Estudiantiles (DAE)",
    "Beneficios gestionados y continuidad de pagos o becas al cierre del semestre.",
)
TERRITORIAL = Dimension(
    "Territorial",
    "Procede de una región, comuna o colegio rural con menor tasa de continuidad en el modelo.",
    ("Consultar por traslado, alojamiento y redes de apoyo en la ciudad.",
     "Invitación a actividades de integración de primer año."),
    "Dirección de Asuntos Estudiantiles (DAE)",
    "Participación en actividades de integración durante el primer mes.",
)
VOCACIONAL = Dimension(
    "Vocacional y disciplinar",
    "La carrera, el área o la vía de ingreso tienen un riesgo estimado mayor.",
    ("Reunión de bienvenida con la jefatura o coordinación de carrera.",
     "Orientación vocacional temprana si aparecen dudas sobre la elección.",
     "Actividades de vinculación con el campo profesional durante el primer semestre."),
    "Jefatura o coordinación de carrera",
    "Asistencia a la reunión de bienvenida y a las actividades de vinculación.",
)
GENERAL = Dimension(
    "Seguimiento general",
    "Factor demográfico del modelo; no define por sí solo una intervención.",
    ("Incluir en el seguimiento general del tramo, sin acciones diferenciadas.",),
    "Unidad de acompañamiento",
    "Contactos de seguimiento realizados según el calendario.",
)

# Variable del factor SHAP (texto antes de ':') -> dimensión
MAPA_FACTORES: dict[str, Dimension] = {
    "NEM (prom. notas Media)": ACADEMICA,
    "Rezago egreso-ingreso (años)": TRAYECTORIA,
    "Jornada": COMPATIBILIDAD,
    "Rango de edad": NO_TRADICIONAL,
    "IVM establecimiento": SOCIOECONOMICA,
    "IVE-SINAE establecimiento": SOCIOECONOMICA,
    "IVE-SINAE comunal": SOCIOECONOMICA,
    "IVM comuna": SOCIOECONOMICA,
    "IVM provincia": SOCIOECONOMICA,
    "IVM región": SOCIOECONOMICA,
    "Tramo IVM establecimiento": SOCIOECONOMICA,
    "Región establecimiento": TERRITORIAL,
    "Comuna establecimiento": TERRITORIAL,
    "DEPROV establecimiento": TERRITORIAL,
    "Establecimiento rural": TERRITORIAL,
    "Carrera": VOCACIONAL,
    "Área de conocimiento": VOCACIONAL,
    "Vía de ingreso": VOCACIONAL,
    "Género": GENERAL,
}

INTENSIDAD = {
    "Alto": {
        "prioridad": "Prioridad 1",
        "modalidad": "Tutor asignado y seguimiento individual",
        "primer_contacto": "Semana 1 del semestre",
        "frecuencia": "Quincenal durante el primer semestre; mensual en el segundo",
        "revision": "Revisión del caso por el equipo al final del mes 1 y del semestre 1",
    },
    "Medio": {
        "prioridad": "Prioridad 2",
        "modalidad": "Seguimiento grupal con contacto individual",
        "primer_contacto": "Semanas 2 a 4 del semestre",
        "frecuencia": "Mensual durante el primer año",
        "revision": "Revisión del caso al final del semestre 1",
    },
    "Bajo": {
        "prioridad": "Apoyo universal",
        "modalidad": "Acciones generales de primer año",
        "primer_contacto": "Inducción general",
        "frecuencia": "Según las alertas del semestre (inasistencia, notas)",
        "revision": "Solo si aparece una señal de alerta",
    },
}

CALENDARIO = [
    ("Semanas 0–2 · Inducción", "Contacto inicial, entrevista y acuerdo del plan.",
     "No asiste a la inducción ni responde al contacto."),
    ("Semanas 3–6 · Primeras evaluaciones", "Verificar asistencia y resultados de las primeras evaluaciones.",
     "Inasistencia mayor al 25 % o nota bajo 4,0 en una asignatura crítica."),
    ("Semanas 7–10 · Mitad del semestre", "Revisar el avance del plan y ajustar los apoyos.",
     "Retiro de asignaturas o baja participación en tutorías."),
    ("Fin del semestre 1", "Revisar resultados finales y diseñar un plan de recuperación si hay reprobación.",
     "Reprobación de una o más asignaturas críticas."),
    ("Inicio del semestre 2", "Confirmar reinscripción y continuidad de beneficios.",
     "No inscribe asignaturas o pierde un beneficio."),
    ("Cierre del año · rematrícula", "Confirmar la matrícula del año siguiente: es el resultado que mide el modelo.",
     "No se rematricula en el plazo: contacto inmediato."),
]


def dimensiones_estudiante(fila: pd.Series) -> list[tuple[Dimension, str]]:
    """Dimensiones únicas, en el orden de importancia de los factores SHAP."""
    vistas, salida = set(), []
    for i in (1, 2, 3):
        var = str(fila.get(f"FACTOR_{i}_VAR", "") or "")
        valor = str(fila.get(f"FACTOR_{i}_VALOR", "") or "")
        if not var:
            continue
        dim = MAPA_FACTORES.get(var, GENERAL)
        if dim.nombre in vistas:
            continue
        vistas.add(dim.nombre)
        salida.append((dim, f"{var}: {valor}" if valor else var))
    if not salida:
        salida.append((GENERAL, "Sin factores reportados"))
    return salida


def pronostico(fila: pd.Series) -> dict:
    p = float(fila["PROB_DESERCION"])
    tramo = str(fila.get("TRAMO_RIESGO", "Bajo"))
    return {
        "probabilidad": p,
        "de_cada_100": round(p * 100),
        "veces_cohorte": p / TASA_ESPERADA_2025,
        "percentil": float(fila.get("PERCENTIL_RIESGO_COHORTE", float("nan"))),
        "tramo": tramo,
        "alerta": bool(fila.get("ALERTA_UMBRAL_OPTIMO", 0)),
        **INTENSIDAD.get(tramo, INTENSIDAD["Bajo"]),
    }


def _fmt_nem(x) -> str:
    try:
        return f"{float(x):.2f}".replace(".", ",")
    except (TypeError, ValueError):
        return "S/I"


def _fmt_pct(x: float, dec: int = 1) -> str:
    return f"{x * 100:.{dec}f}".replace(".", ",") + " %"


def plan_markdown(fila: pd.Series) -> str:
    """Plan completo en Markdown, para mostrar y descargar."""
    pr = pronostico(fila)
    dims = dimensiones_estudiante(fila)
    lineas = [
        f"# Plan de acompañamiento preventivo · {fila['ID']}",
        "",
        f"**Carrera:** {fila.get('NOMB_CARRERA', 'S/I')} · **Jornada:** {fila.get('JORNADA', 'S/I')} · "
        f"**NEM:** {_fmt_nem(fila.get('PROM_NOTAS_ALU'))}",
        "",
        "## Pronóstico",
        f"- Probabilidad estimada de no continuar en 2026: **{_fmt_pct(pr['probabilidad'])}** "
        f"(de cada 100 estudiantes con un perfil similar, se esperaría que unos {pr['de_cada_100']} no continúen).",
        f"- Equivale a **{pr['veces_cohorte']:.1f}".replace(".", ",") + " veces** la tasa esperada de la cohorte 2025 (9,28 %).",
        f"- Percentil de riesgo en la cohorte: {pr['percentil']:.1f}".replace(".", ",") + f" · Tramo **{pr['tramo']}** · {pr['prioridad']}.",
        f"- Alerta con el umbral del modelo: {'sí' if pr['alerta'] else 'no'}.",
        "",
        "## Intensidad del seguimiento",
        f"- Modalidad: {pr['modalidad']}",
        f"- Primer contacto: {pr['primer_contacto']}",
        f"- Frecuencia: {pr['frecuencia']}",
        f"- Revisión: {pr['revision']}",
        "",
        "## Dimensiones de apoyo (según los factores del modelo)",
    ]
    for dim, factor in dims:
        lineas += [
            f"### {dim.nombre}  ·  _{factor}_",
            dim.lectura,
            *[f"- {a}" for a in dim.acciones],
            f"- **Responsable:** {dim.responsable}",
            f"- **Indicador de seguimiento:** {dim.indicador}",
            "",
        ]
    lineas.append("## Calendario y señales de alerta")
    for etapa, accion, senal in CALENDARIO:
        lineas.append(f"- **{etapa}:** {accion} _Señal:_ {senal}")
    lineas += [
        "",
        "---",
        "_El puntaje es una estimación del modelo de la tesis (capacidad moderada: AP 0,230; ROC-AUC 0,650) "
        "para priorizar el acompañamiento. No es un diagnóstico, requiere revisión humana y nunca se usa para "
        "restringir matrícula ni beneficios._",
    ]
    return "\n".join(lineas)


def perfil_anonimo(fila: pd.Series) -> str:
    """Datos que se envían al asistente LLM: sin MRUN, RBD ni ID."""
    pr = pronostico(fila)
    factores = [f"{i}. {fila.get(f'FACTOR_RIESGO_{i}', '')}" for i in (1, 2, 3) if pd.notna(fila.get(f"FACTOR_RIESGO_{i}"))]
    return (
        f"Carrera: {fila.get('NOMB_CARRERA', 'S/I')} (área {fila.get('AREA_CONOCIMIENTO', 'S/I')})\n"
        f"Jornada: {fila.get('JORNADA', 'S/I')} · Vía de ingreso: {fila.get('FORMA_INGRESO', 'S/I')}\n"
        f"NEM: {fila.get('PROM_NOTAS_ALU', 'S/I')}\n"
        f"Probabilidad estimada: {_fmt_pct(pr['probabilidad'])} · percentil {pr['percentil']:.1f} · tramo {pr['tramo']}\n"
        f"Tasa esperada de la cohorte: 9,28 %\n"
        f"Factores SHAP que más elevaron el puntaje:\n" + "\n".join(factores) +
        f"\nIntensidad definida para el tramo: {pr['modalidad']}; primer contacto {pr['primer_contacto']}; {pr['frecuencia']}."
    )


def tabla_planes(df: pd.DataFrame, tramos=("Alto", "Medio")) -> pd.DataFrame:
    """Resumen de planes para descargar (una fila por estudiante priorizado)."""
    filas = []
    for _, f in df[df["TRAMO_RIESGO"].isin(tramos)].iterrows():
        pr = pronostico(f)
        dims = dimensiones_estudiante(f)
        filas.append({
            "PRIORIDAD": f["PRIORIDAD"],
            "ID": f["ID"],
            "MRUN": f["MRUN"],
            "CARRERA": f.get("NOMB_CARRERA"),
            "JORNADA": f.get("JORNADA"),
            "PROB_DESERCION": round(pr["probabilidad"], 4),
            "TRAMO": pr["tramo"],
            "MODALIDAD": pr["modalidad"],
            "PRIMER_CONTACTO": pr["primer_contacto"],
            "FRECUENCIA": pr["frecuencia"],
            "DIMENSIONES": " | ".join(d.nombre for d, _ in dims),
            "ACCION_PRINCIPAL": dims[0][0].acciones[0],
            "RESPONSABLE_PRINCIPAL": dims[0][0].responsable,
        })
    return pd.DataFrame(filas)
