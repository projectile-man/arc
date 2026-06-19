from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from config.parameters import KEYWORDS
from config.models import HUMAN_REVIEW_MODEL, HUMAN_REVIEW_TEMPERATURE

def intencion_aprobacion(state: AgentState) -> AgentState:
    """El usuario aprueba el catalogo actual
    Se fuerza el score > 85 para que el nodo supervisor finalice en la 
    siguiente iteracion sin pasar por el analisis o validacion de nuevo"""

    state.quality_score = 90
    state.user_message = None
    state.decision = 'finalizar'

    return state

def intencion_rechazo(state: AgentState) -> AgentState:
    """El usuario rechaza el catalogo y pide empezar de nuevo"""
    state.requisitos         = []
    state.score_calidad      = 0
    state.reporte_analisis   = state.reporte_analisis.__class__()
    state.reporte_validacion = state.reporte_validacion.__class__()
    state.iteraciones        = 0
    state.mensaje_usuario    = None
    state.decision           = "generar"

    return state

def intencion_correccion(state: AgentState, mensaje: str) -> AgentState:
    """EL usuario reporta una correccion o aclaracion especifica
    Se guarda el mensaje para que el agente de generacion lo añada
    en la siguiente iteracion y forzar la regeneracion"""

    state.mensaje_usuario = mensaje
    state.decision        = "generar"

    return state

def construir_resumen_estado(state: AgentState) -> str:
    """
    Construye un resumen legible del estado actual para contexto del LLM
    al responder preguntas del usuario.
    """
    partes: list[str] = [
        f"Score de calidad: {state.score_calidad}/100",
        f"Total de requisitos: {len(state.requisitos)}",
        f"  RF:  {sum(1 for r in state.requisitos if r.tipo == 'RF')}",
        f"  RNF: {sum(1 for r in state.requisitos if r.tipo == 'RNF')}",
        f"  HU:  {sum(1 for r in state.requisitos if r.tipo == 'HU')}",
    ]
 
    if state.reporte_analisis and state.reporte_analisis.issues:
        n = len(state.reporte_analisis.issues)
        partes.append(f"Issues detectados: {n}")
 
    if state.reporte_validacion and state.reporte_validacion.conflictos:
        n = len(state.reporte_validacion.conflictos)
        partes.append(f"Conflictos encontrados: {n}")
 
    if state.requisitos:
        partes.append("\nRequisitos actuales:")
        for r in state.requisitos[:10]:   # máximo 10 para no saturar el contexto
            partes.append(f"  {r.id} [{r.tipo}]: {r.descripcion}")
        if len(state.requisitos) > 10:
            partes.append(f"  ... y {len(state.requisitos) - 10} más")
 
    return "\n".join(partes)

def intencion_pregunta(state: AgentState, mensaje:str, llm: ChatOpenAI) -> AgentState:
    """EL usuario hace una pregunta sobre los requisitos o el proceso, se 
    responde directamete usando el contexto del estado actual sin pasar por los
    agentes de generacion o analisis"""

    resumen_estado = construir_resumen_estado(state)
 
    respuesta = llm.invoke([
        SystemMessage(content=(
            "Eres un asistente de ingeniería de requisitos. "
            "Responde la pregunta del usuario usando el contexto del "
            "catálogo de requisitos actual. Sé conciso y claro."
        )),
        HumanMessage(content=(
            f"Contexto del catálogo actual:\n{resumen_estado}\n\n"
            f"Pregunta del usuario: {mensaje}"
        )),
    ])
 
    # Guardar la respuesta en mensaje_usuario para que Streamlit la muestre
    state.mensaje_usuario = f"[Respuesta del asistente]\n{respuesta.content}"
    state.decision        = "analizar"   # continuar el flujo normal

    return state

def clasificar_intencion(mensaje: str) -> Literal["aprobacion", "rechazo", "pregunta", "correccion", "ambiguo"]:
    """CLasificacion determinista por palabras clave
    retorna ambiguo si no hay match claro, el llm lo resuelve"""

    mensaje = mensaje.lower()

    for intencion, keywords in KEYWORDS.items():
        if any(kw in mensaje for kw in keywords):
            return intencion  # type: ignore[return-value]
 
    # Si el mensaje es largo y no matchea nada, probablemente es una corrección
    if len(mensaje.split()) > 6:
        return "correccion"
 
    return "ambiguo"

def clasificar_con_llm(mensaje: str, llm: ChatOpenAI) -> str:
    """
    Usa haiku para clasificar mensajes ambiguos.
    Solo se llama cuando la clasificación determinista no es suficiente.
    """
    respuesta = llm.invoke([
        SystemMessage(content=(
            "Clasifica el siguiente mensaje de usuario en una de estas "
            "categorías: aprobacion, rechazo, pregunta, correccion. "
            "Responde SOLO con la palabra de la categoría, sin texto adicional."
        )),
        HumanMessage(content=mensaje),
    ])
 
    clasificacion = respuesta.content.strip().lower()
 
    if clasificacion not in ("aprobacion", "rechazo", "pregunta", "correccion"):
        return "correccion"
 
    return clasificacion
    

def human_review(state: AgentState) -> AgentState:
    """Nodo de revision humana, procesa la respuesta del usuario almacenada en state.user_message
    clarifica su intencion y prepara el estado para que el supervisor tome la decision correcta
    en la siguiente iteracion"""

    # llm = ChatAnthropic(
    #     model=HUMAN_REVIEW_MODEL,
    #     max_tokens=4096,
    #     temperature=HUMAN_REVIEW_TEMPERATURE,
    # )

    llm = ChatOpenAI(
        model=HUMAN_REVIEW_MODEL,
        max_tokens=4096,
        temperature=HUMAN_REVIEW_TEMPERATURE,
    )

    mensaje = (state.user_message or "").strip()

    if not mensaje:
        state.decision = 'analizar'

        return state
    
    intencion = clasificar_intencion(mensaje)

    if intencion == 'ambiguo':
        intencion = clasificar_con_llm(mensaje, llm)

    if intencion == "aprobacion":
        return intencion_aprobacion(state)
 
    if intencion == "rechazo":
        return intencion_rechazo(state)
 
    if intencion == "pregunta":
        return intencion_pregunta(state, mensaje, llm)
    
    return intencion_correccion(state, mensaje)