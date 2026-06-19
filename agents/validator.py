from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
import json
import re

from state import AgentState, Conflict, ValidationReport
from agents.prompts import VALIDATOR_PROMPT
from config.models import VALIDATOR_MODEL, VALIDATOR_TEMPERATURE

from dotenv import load_dotenv

load_dotenv()

def serializar_requisitos(requisitos) -> str:
    """
    Serializa el catálogo completo para que Opus pueda razonar
    sobre todas las relaciones entre requisitos simultáneamente.
    """
    lineas: list[str] = []
    for r in requisitos:
        lineas.append(
            f"{r.id} [{r.requirement_type}/{r.category}] "
            f"actor={r.players} prioridad={r.priority} estado={r.state}\n"
            f"  {r.description}"
        )
    return "\n".join(lineas)

def recuperar_contexto(texto_limpio: str) -> str:
    """
    Recupera de Pinecone patrones de conflictos comunes entre requisitos
    y listas de verificación de consistencia del dominio.
    """
    try:
        from rag.retriever import recuperar_contexto
        docs = recuperar_contexto(
            query  = f"conflictos y consistencia en requisitos: {texto_limpio[:300]}",
            top_k  = 4,
            filtro = {"agente": "validacion"},
        )
        return "\n\n".join(d.page_content for d in docs)
    except Exception as exc:
        print("RAG no disponible para validación: {}".format(exc))
        return ""


def construir_mensaje(state: AgentState, contexto_rag: str) -> str:
    partes: list[str] = []

    if contexto_rag:
        partes.append(
            f"## Patrones de referencia (conflictos comunes, checklists)\n"
            f"{contexto_rag}"
        )

    if state.analisis_report and state.analisis_report.issues:
        issues_str = "\n".join(
            f"  - {i.id_requirement} [{i.dimension}]: {i.description}"
            for i in state.analisis_report.issues
        )
        partes.append(
            f"## Issues detectados en análisis previo (score: {state.quality_score})\n"
            f"{issues_str}"
        )

    partes.append(
        f"## Catálogo completo de requisitos a validar\n"
        f"{serializar_requisitos(state.requirements)}"
    )

    return "\n\n".join(partes)

def extraer_json(contenido: str) -> dict:
    """Extrae JSON de la respuesta del LLM, robusto ante markdown."""
    try:
        return json.loads(contenido.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", contenido, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    print("Validación — no se pudo parsear JSON del LLM")
    return {}

def parsear_reporte(contenido: str) -> ValidationReport:
    """
    Construye un ReporteValidacion desde el JSON del LLM.
    En caso de error de parseo retorna un reporte de rechazo seguro.
    """
    datos = extraer_json(contenido)

    if not datos:
        return ValidationReport(
            approved_validation = False,
            risk_level          = "critico",
            final_opinion       = "rechazado",
            justification       = "Error al parsear respuesta del agente de validación.",
        )

    # Parsear conflictos
    conflictos: list[Conflict] = []
    for item in datos.get("conflictos", []):
        try:
            conflictos.append(Conflict(
                conflict_type         = item.get("tipo", ""),
                requirements_involved = item.get("requisitos_involucrados", []),
                description           = item.get("descripcion", ""),
                impact                = item.get("impacto", "medio"),
                suggested_resolution  = item.get("resolucion_sugerida", ""),
            ))
        except Exception as exc:
            print("Conflicto malformado descartado: {} — {}".format(item, exc))

    # Determinar validacion_aprobada desde el dictamen si no viene explícito
    dictamen            = datos.get("final_opinion", "rechazado")
    validacion_aprobada = datos.get(
        "approved_validation",
        dictamen in ("aprobado", "aprobado_con_observaciones"),
    )

    return ValidationReport(
        approved_validation = validacion_aprobada,
        risk_level          = datos.get("nivel_riesgo", "alto"),
        conflicts           = conflictos,
        dependencies_map    = datos.get("mapa_dependencias", {}),
        detected_gaps       = datos.get("gaps_detectados", []),
        final_opinion       = dictamen,
        justification       = datos.get("justificacion", ""),
    )

def validator_agent(state: AgentState) -> AgentState:
    """
    Nodo de validación de consistencia.

    Analiza el catálogo completo de requisitos en busca de conflictos,
    gaps y dependencias no declaradas. Actualiza state.reporte_validacion.
    """

    llm = ChatOpenAI(
        model=VALIDATOR_MODEL,
        max_tokens=4096,
        temperature=VALIDATOR_TEMPERATURE
    )

    if not state.requirements:
        print("Validación — sin requisitos que validar")
        state.validation_report = ValidationReport(
            approved_validation  = False,
            risk_level           = "critico",
            final_opinion        = "rechazado",
            justification        = "No hay requisitos que validar.",
        )
        state.iterations += 1
        return state

    contexto_rag = recuperar_contexto(state.clean_text)

    mensaje = construir_mensaje(state, contexto_rag)

    prompt = PromptTemplate.from_template(
        VALIDATOR_PROMPT
    )

    chain = (
        {
            "query": RunnablePassthrough(),
            "schema": RunnableLambda(lambda _: json.dumps(AgentState.model_json_schema(), indent=2))
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({
        "query": mensaje
    })

    reporte = parsear_reporte(response)

    state.validation_report = reporte
    state.iterations       += 1

    return state