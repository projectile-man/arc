from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
import json
from typing import Literal
from dotenv import load_dotenv

from agents.prompts import SUPERVISOR_PROMPT
from config.models import SUPERVISOR_MODEL, SUPERVISOR_TEMPERATURE
from config.parameters import MAX_ITERATIONS, FINISH_SCORE, VALIDATION_SCORE, REGENERATE_SCORE

from state import AgentState

load_dotenv()

def serializar_requisitos(requisitos, validation_report) -> tuple:
    """convierte la lista de Requirement en texto estructurado para el LLM.
    Mas legible que JSON crudo para la evaluacion de calidad"""

    lineas: list[str] = []
    lista_conflictos: list[str] = []
    lista_gaps: list[str] = []

    conflictos = getattr(validation_report, "conflicts", [])
    gaps       = getattr(validation_report, "detected_gaps", [])

    for r in requisitos:
        lineas.append(
            f"{r.id} [{r.requirement_type}] (actor: {r.players}, prioridad: {r.priority})\n"
            f"  {r.description}\n"
            f"  fuente: \"{r.source}\""
        )

    for c in conflictos:
        lista_conflictos.append(
            f"Requerimientos envueltos: {c.requirements_involved}\n" 
            f"Descripcion: {c.description}\n"
            f"Impacto: {c.impact}\n"
        )

    for g in gaps:
        sugerencias = "\n".join(
            f"  - {req}" for req in g["requisitos_sugeridos"]
        )

        lista_gaps.append(
            f"Area: {g['area']}\n"
            f"Descripcion: {g['descripcion']}\n"
            f"Sugerencias:\n{sugerencias}\n"
        )

    requisitos_str = "Los requisitos encontrados son: " + "\n\n".join(lineas)

    if lista_conflictos:
        conflictos_str = "Los conflictos encontrados son: " + "\n\n".join(lista_conflictos)
    else:
        conflictos_str = ""

    if lista_gaps:
        gaps_str       = "Los gaps encontrados son: " + "\n\n".join(lista_gaps)
    else:
        gaps_str = ""

    return requisitos_str, conflictos_str, gaps_str

def tiene_conflictos_criticos(reporte_validacion) -> bool:
    """
    Retorna True si hay al menos un conflicto de impacto alto
    en el reporte de validacion.
    """
    if not reporte_validacion:
        return False
    conflictos = getattr(reporte_validacion, "conflicts", []) or []

    return any(
        getattr(c, "impact", "") == "alto"
        for c in conflictos
    )

def parsear_decision(contenido: str) -> dict:
    """
    Extrae el JSON de decisión de la respuesta del LLM.
    Robusto ante respuestas con texto extra o bloques de código.
    """
    try:
        # Caso ideal: el LLM devolvió JSON puro
        return json.loads(contenido.strip())
    except json.JSONDecodeError:
        pass
 
    # Caso alternativo: el LLM envolvió el JSON en ```json ... ```
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", contenido, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
 
    # Fallback: no se pudo parsear, forzar "human_review" como medida de seguridad
    print(
        "Supervisor: no se pudo parsear la respuesta del LLM. "
        "Forzando decisión 'human_review'. Contenido recibido: {}".format(contenido[:300])
    )
    return {
        "decision":             "human_review",
        "razonamiento":         "Error al parsear respuesta del supervisor.",
        "confianza":            0,
        "mensaje_para_usuario": "El supervisor encontró un error interno. "
                                "Por favor revisa los logs y reintenta.",
    }

def aplicar_clarificaciones(state: AgentState, mensaje_usuario: str) -> AgentState:
    if not mensaje_usuario:
        return state

    cambios = {}
    for linea in mensaje_usuario.split("\n"):
        if ":" in linea:
            req_id, texto = linea.split(":", 1)
            cambios[req_id.strip()] = texto.strip()

    for req in state.requirements or []:
        if req.id in cambios:
            req.description = cambios[req.id]
            req.state = "definido"
        elif getattr(req, "state", "") == "necesita_clarificacion":
            req.state = "definido"

    return state

def supervisor_agent(state: AgentState) -> AgentState:
    """
    Nodo supervisor del grafo LangGraph.
 
    Evalúa el AgentState y devuelve el estado actualizado con la
    decisión tomada en state["decision"].
 
    En modo "asistido", si la decisión es "human_review", ejecuta interrupt()
    para pausar el grafo y esperar respuesta del usuario.
    """

    llm = ChatOpenAI(
        model=SUPERVISOR_MODEL,
        max_tokens=512,
        temperature=SUPERVISOR_TEMPERATURE
    )

    reqs_pendientes = [
        r for r in (state.requirements or [])
        if getattr(r, "state", "") == "necesita_clarificacion"
    ]

    if reqs_pendientes and state.mode == "asistido":
        ids_pendientes = ", ".join(
            getattr(r, "id", "?") for r in reqs_pendientes
        )
        descripciones = "\n".join(
            f"- {getattr(r, 'id', '?')}: "
            f"{getattr(r, 'description', '')[:100]}"
            for r in reqs_pendientes
        )
 
        # print(
        #     "Supervisor — {} requisito(s) necesitan clarificacion: {}\n".format(
        #     len(reqs_pendientes), ids_pendientes,)
        # )

        respuesta_humano = interrupt({
            "pregunta": (
                f"Los siguientes {len(reqs_pendientes)} requisito(s) "
                f"necesitan tu clarificacion:\n\n{descripciones}\n\n"
                f"Por favor aclara cada uno para que el agente pueda "
                f"regenerarlos correctamente."
            ),
            "score_actual":  state.quality_score,
            "iteraciones":   state.iterations,
            "razonamiento":  (
                f"{len(reqs_pendientes)} requisito(s) marcados como "
                f"'necesita_clarificacion': {ids_pendientes}"
            ),
            "requisitos_pendientes": [
                {
                    "id":          getattr(r, "id", ""),
                    "descripcion": getattr(r, "description", ""),
                    "fuente":      getattr(r, "source", ""),
                }
                for r in reqs_pendientes
            ],
        })

        state.user_message = respuesta_humano
        state = aplicar_clarificaciones(state, respuesta_humano)

        for r in reqs_pendientes:
            if r.id in state.requirements:
                continue
            r.state = "definido"

        state.decision = "generator"
        return state

    reporte_val    = state.validation_report
    tiene_criticos = tiene_conflictos_criticos(reporte_val)

    if len(state.requirements) == 0:
        state.analisis_report = {}
        state.validation_report = {}

    validacion_aprobada = (
        getattr(reporte_val, "approved_validation", False)
        if reporte_val else False
    )

    estado_resumen = {
        "project_name":              state.project_name,
        "requisitos_count":          len(state.requirements or []),
        "score_calidad":             state.quality_score,
        "iteraciones":               state.iterations,
        "tiene_reporte_analisis":    bool(state.analisis_report),
        "tiene_reporte_validacion":  bool(reporte_val),
        "validacion_aprobada":       validacion_aprobada,
        "tiene_conflictos_criticos": tiene_criticos,
        "user_message":              state.user_message,
        "modo":                      state.mode,
        "MAX_ITERACIONES":           MAX_ITERATIONS,
        "SCORE_FINALIZAR":           FINISH_SCORE,
        "SCORE_VALIDAR":             VALIDATION_SCORE,
        "SCORE_REGENERAR":           REGENERATE_SCORE,
    }

    prompt = PromptTemplate.from_template(
        SUPERVISOR_PROMPT
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "query": json.dumps(estado_resumen, ensure_ascii=False, indent=2),
        "schema": json.dumps(AgentState.model_json_schema(), indent=2)
    })

    # print(f"\n--El prompt del supervisor es:\n")
    # print(prompt.format(
    #     query=json.dumps(estado_resumen, ensure_ascii=False, indent=2),
    #     schema={}
    # ))

    decision_dict = parsear_decision(response)

    decision    = decision_dict["decision"]
    confianza   = decision_dict.get("confianza", 100)
    razon       = decision_dict.get("razonamiento", "")
    msg_usuario = decision_dict.get("mensaje_para_usuario")
 
    print(
        "\nSupervisor → decisión: '{}' | confianza: {} | razón: {}\n".format(
        decision, confianza, razon
    ))
                                 
    if decision == "human_review" and getattr(state, "mode", "asistido") == "asistido":
        print("Supervisor ejecutando interrupt() — esperando usuario")
 
        respuesta_humano = interrupt({
            "pregunta": msg_usuario or "Revisión manual requerida.",
            "score_actual": state.quality_score,
            "iteraciones":  state.iterations,
            "razonamiento": razon,
            "requisitos_pendientes": [],
        })

        state.decision     = "human_review"
        state.user_message = respuesta_humano

        return state
    
    state.decision     = decision
    state.user_message = None

    if state.decision == "end":
        from utils.utils import indexar_requisitos

        requisitos_str, conflictos_str, gaps_str = serializar_requisitos(state.requirements, state.validation_report)

        indexar_requisitos(requisitos_str, getattr(state, "thread_id", ""), getattr(state, "project_name", ""))
        indexar_requisitos(conflictos_str, getattr(state, "thread_id", ""), getattr(state, "project_name", ""))
        indexar_requisitos(gaps_str, getattr(state, "thread_id", ""), getattr(state, "project_name", ""))

    return state
 
def routing_supervisor(
    state: AgentState,
) -> Literal["extractor", "generator", "analyst", "validator", "human_review", "__end__"]:
    """
    Función pura que LangGraph llama para determinar el siguiente nodo.
    Traduce state["decision"] al nombre del nodo del grafo.
 
    Esta función es el único lugar donde se mapea decision → nodo.
    Separada del nodo supervisor para que LangGraph pueda usarla
    como función de routing en add_conditional_edges().
    """
    mapa = {
        "generator":    "generator",
        "analyst":      "analyst",
        "validator":    "validator",
        "human_review": "human_review",
        "end":          "__end__",
    }
 
    decision = getattr(state, "decision", "generator")
    siguiente = mapa.get(decision, "generator")
 
    print("\nRouting supervisor: '{}' → nodo '{}'\n".format(decision, siguiente))

    return siguiente