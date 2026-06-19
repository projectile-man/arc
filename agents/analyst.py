from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
import json

from state import AgentState, IssueAnalysis, AnalysisReport
from utils.utils import recuperar_contexto

from dotenv import load_dotenv

from agents.prompts import ANALYST_PROMPT
from config.models import ANALYST_MODEL, ANALYST_TEMPERATURE
from config.parameters import PENALIZACION

load_dotenv()

import re

def construir_mensaje(requisitos_str: str, contexto_rag: str) -> str:
    partes: list[str] = []

    if contexto_rag:
        partes.append(
            f"## Criterios de referencia (IEEE 830, ejemplos de defectos)\n"
            f"{contexto_rag}"
        )
 
    partes.append(
        f"## Catálogo de requisitos a evaluar\n{requisitos_str}"
    )
 
    return "\n\n".join(partes)

def serializar_requisitos(requisitos) -> str:
    """convierte la lista de Requirement en texto estructurado para el LLM.
    Mas legible que JSON crudo para la evaluacion de calidad"""

    lineas: list[str] = []
    for r in requisitos:
        lineas.append(
            f"{r.id} [{r.requirement_type}] (actor: {r.players}, prioridad: {r.priority})\n"
            f"  {r.description}\n"
            f"  fuente: \"{r.source}\""
        )
    return "\n\n".join(lineas)

def parsear_respuesta(
    contenido: str,
) -> tuple[list[IssueAnalysis], dict[str, int], str]:
    """
    Extrae issues, resumen y recomendación del JSON del LLM.
    Retorna tupla (issues, resumen_dict, recomendacion).
    """
    datos         = extraer_json(contenido)
    issues_raw    = datos.get("issues", [])
    recomendacion = datos.get("recomendacion", "revisar_parcial")
 
    issues: list[IssueAnalysis] = []
    for item in issues_raw:
        try:
            issues.append(IssueAnalysis(
                id_requirement = item.get("id", ""),
                dimension    = item.get("dimension", ""),
                description  = item.get("descripcion", ""),
                suggestion   = item.get("sugerencia", ""),
            ))
        except Exception as exc:
            print("Issue malformado descartado: %s — %s", item, exc)
 
    # Recalcular resumen desde los issues parseados para garantizar consistencia
    resumen: dict[str, int] = {k: 0 for k in PENALIZACION}
    for issue in issues:
        dim = issue.dimension.lower()
        if dim in resumen:
            resumen[dim] += 1
 
    return issues, resumen, recomendacion
 
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
 
    return {}
 
def calcular_score(resumen: dict[str, int]) -> int:
    """
    Aplica las penalizaciones por dimensión y retorna el score final (0-100).
    """
    descuento = sum(
        resumen.get(dim, 0) * pts
        for dim, pts in PENALIZACION.items()
    )
    return max(0, 100 - descuento)

def analyst_agent(state: AgentState) -> AgentState:
    """Nodo de analisis de calidad
    evalua el catalogo de requisitos, calcula el score global
    y actualiza state.AnalysisReport y state.quality_score"""

    llm = ChatOpenAI(
        model=ANALYST_MODEL,
        max_tokens=4096,
        temperature=ANALYST_TEMPERATURE,
    )

    if not state.requirements:
        state.quality_score   = 0
        state.analisis_report = AnalysisReport()
        state.iterations     += 1
        return state
    
    texto_limpio = state.clean_text

    contexto_rag = recuperar_contexto(f"defectos de calidad en requisitos: {texto_limpio[:300]}", 4, "analisis")
       
    requisitos_str = serializar_requisitos(state.requirements)

    prompt = PromptTemplate.from_template(
        ANALYST_PROMPT
    )

    chain = (
        {
            "specs": RunnablePassthrough(),
            "query": RunnablePassthrough(),
            "schema": RunnableLambda(lambda _: json.dumps(AgentState.model_json_schema(), indent=2))
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({
        "specs": contexto_rag,
        "query": requisitos_str
    })

    issues, resumen, recomendacion = parsear_respuesta(response)
    score = calcular_score(resumen)

    state.quality_score    = score
    state.analisis_report = AnalysisReport(
        quality_score  = score,
        issues         = issues,
        issues_summary = resumen,
        recommendation = recomendacion,
    )
    state.iterations += 1
 
    return state


































