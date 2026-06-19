from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
import json

from state import AgentState, EntitiesDetected
from agents.prompts import EXTRACTOR_PROMPT
from config.models import EXTRACTOR_MODEL, EXTRACTOR_TEMPERATURE
from utils.utils import procesar_documentos, detectar_tipo, indexar

from dotenv import load_dotenv

load_dotenv()

def extraer_entidades(query) -> tuple[dict, str]:

    llm = ChatOpenAI(
        model=EXTRACTOR_MODEL,
        max_tokens=2048,
        temperature=EXTRACTOR_TEMPERATURE
    )

    prompt = PromptTemplate.from_template(
        EXTRACTOR_PROMPT
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
        datos = json.loads(response.strip())
    except json.JSONDecodeError:
        datos = {}

    languaje = datos.pop("languaje", "es")
    datos.pop("summary", None)
    project_name = datos.pop("project_name", "SIN NOMBRE")

    return datos, languaje, project_name


def extractor_agent(state: AgentState):
    archivos: list[dict] = getattr(state, "new_files", [])
    texto_usuario: str = getattr(state, "user_text", "")

    if archivos:
        clean_text = procesar_documentos(archivos)
        entry_type = detectar_tipo(archivos)
    else:
        clean_text = texto_usuario.strip()
        entry_type = 'conversacion'

    if not clean_text:
        return {
            **state,
            "clean_text":"",
            "entities": EntitiesDetected(
                actors=[],
                systems=[],
                actions=[],
                restrictions_mentioned=[],
            ),
            "entry_type": entry_type,
            "languaje": "es"
        }
    
    #extraer entidades
    entities, languaje, project_name = extraer_entidades(clean_text)

    state.project_name = project_name
    state.clean_text   = entities.get("clean_text", "")
    state.entities     = EntitiesDetected(
            actors=entities.get("entities_detected", {}).get("actors", []),
            systems=entities.get("entities_detected", {}).get("systems",[]),
            actions=entities.get("entities_detected", {}).get("actions", []),
            restrictions_mentioned=entities.get("entities_detected", {}).get("restrictions_mentioned", []),
        )
    state.entry_type = entry_type
    state.languaje   = languaje
    state.analisis_report   = {}
    state.validation_report = {}

    indexar(archivos, getattr(state, "thread_id", ""), getattr(state, "project_name", ""))

    return state