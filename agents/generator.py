import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from agents.prompts import GENERATOR_PROMPT
from config.models import GENERATOR_MODEL, GENERATOR_TEMPERATURE
from state import AgentState, Requirement
from utils.utils import recuperar_contexto

from dotenv import load_dotenv

load_dotenv()

def construir_mensaje(state: AgentState, contexto_rag: str) -> str:
    """Ensambla el mensaje HumanMessage con:
    -Contexto RAG (plantillas y ejemplos)
    -Correccion del usuario si viene de un interrupt
    -texto limpio a analizar
    -Entidades ya detectadas por ingesta"""

    partes: list[str] = []

    if contexto_rag:
        partes.append(
            f"## Contexto de referencia (IEEE 830, ejemplos)\n{contexto_rag}"
        )

    if state.user_message:
        partes.append(
            f"## Corrección del usuario (prioridad máxima)\n{state.user_message}"
        )

    if state.entities and any([
        state.entities.actors,
        state.entities.systems,
        state.entities.actions,
    ]):
        entidades_str = (
            f"Actores: {', '.join(state.entities.actors)}\n"
            f"Sistemas: {', '.join(state.entities.systems)}\n"
            f"Acciones clave: {', '.join(state.entities.actions)}"
        )
        partes.append(f"## Entidades detectadas por ingesta\n{entidades_str}")
 
    partes.append(f"## Texto a analizar\n{state.clean_text}")
 
    return "\n\n".join(partes)

def validar_y_construir(raw:list[dict]) -> list[Requirement]:
    """Convierte cada dict crudo en un objeto Requirement validado por Pydantic
    Descarta elementos malformados y los registra
    """

    requisitos: list[Requirement] = []

    for item in raw:
        try:
            req = Requirement(
                id               = item.get("id", ""),
                requirement_type = item.get("tipo", "RF").upper(),
                category         = item.get("categoria", "funcional"),
                description      = item.get("descripcion", ""),
                players          = item.get("actor", "usuario"),
                priority         = item.get("prioridad", "media"),
                state            = item.get("estado", "definido"),
                source           = item.get("fuente", ""),
            )
            if req.description:          # descartar requisitos vacíos
                requisitos.append(req)
        except Exception as exc:
            print("Requisito malformado descartado: %s — %s", item, exc)
 
    return requisitos


def parsear_requisitos(contenido: str) -> list[Requirement]:
    """Extrae la lista de requisitos del json devuelto por el llm.
    Robusto ante bloques de codigo markdown y texto adicional"""

    try:
        datos = json.loads(contenido.strip())
        return validar_y_construir(datos.get("requirements", []))
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", contenido, re.DOTALL)
    if match:
        try:
            datos = json.loads(match.group(1))
            return validar_y_construir(datos.get("requisitos", []))
        except json.JSONDecodeError:
            pass

def generator_agent(state: AgentState):
    """Nodo de generacion de requisitos
    consulta el RAG, construye el prompt con contexto y correcciones
    del usuario si las hay, llama al LLM y actualiza state.requisito"""

    clean_text = state.clean_text

    contexto_rag = recuperar_contexto(clean_text[:500], 5, "generacion")

    query = construir_mensaje(state, contexto_rag)

    llm = ChatOpenAI(
        model=GENERATOR_MODEL,
        max_tokens=4096,
        temperature=GENERATOR_TEMPERATURE,
    )

    prompt = PromptTemplate.from_template(
        GENERATOR_PROMPT
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
        "query": query
    })

    try:
        requisitos = parsear_requisitos(response)
    except json.JSONDecodeError:
        requisitos = {}

    if state.quality_score == 0:
        state.analisis_report = {}
        state.validation_report = {}

    state.requirements = requisitos
    state.iterations  += 1
    state.user_message = None  

    return state

















