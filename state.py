from typing import Any
from pydantic import BaseModel, Field, ConfigDict

class Requirement(BaseModel):
    """
    Es la estructura de un requisito individual generado por el agente Generator
    """
    model_config           = ConfigDict(arbitrary_types_allowed=True)
    id:               Any  = Field(default="", description="RF-01, RNF-02, HU-01")
    requirement_type: Any  = Field(default="", description="RF | RNF | HU")
    category:         Any  = Field(default="", description="funcional | rendimiento | seguridad | ...")
    description:      Any  = Field(default="", description="Texto del requisito")
    players:          Any  = Field(default="", description="Usuario | administrador | sistema")
    priority:         Any  = Field(default="", description="alta | media | baja")
    state:            Any  = Field(default="", description="definido | necesita_clarificacion")
    source:           Any  = Field(default="", description="Fragmento del texto original del que proviene")

class IssueAnalysis(BaseModel):
    """
    Issue detectado por el agente de análisis en un requisito.
    """
    model_config         = ConfigDict(arbitrary_types_allowed=True)
    id_requirement: Any  = Field(default="", description="Por ejemplo el requirimiento RF-01")
    dimension:      Any  = Field(default="", description="Claridad | completitud | verificabilidad | atomicidad")
    description:    Any  = Field(default="", description="Qué está mal")
    suggestion:     Any  = Field(default="", description="Cómo corregirlo")

class Conflict(BaseModel):
    """
    Conflicto detectado por el agente de validación entre requisitos.
    """
    model_config                      = ConfigDict(arbitrary_types_allowed=True)
    conflict_type:               Any  = Field(default="", description="Contradiccion_directa | contradiccion_implicita | ...")
    requirements_involved:       Any  = Field(default_factory=list, description="Por ejemplo, RF-07, RF-12")
    description:                 Any  = Field(default="", description="Descripcion del conflicto")
    impact:                      Any  = Field(default="", description="alto | medio | bajo")
    suggested_resolution:        Any  = Field(default="", description="Cómo resolverlo")

class AnalysisReport(BaseModel):
    """
    Output completo del agente de análisis.
    """
    model_config         = ConfigDict(arbitrary_types_allowed=True)
    quality_score:  Any  = Field(default=0, description="Puntuación global de calidad del catálogo de 0 a 100, calculada por el agente de análisis según las dimensiones IEEE 830 (claridad, completitud, verificabilidad, atomicidad).")
    issues:         Any  = Field(default_factory=list, description="Lista de problemas encontrados")
    issues_summary: Any  = Field(default_factory=dict, description="{claridad: 2, completitud: 1, ...}")
    recommendation: Any  = Field(default="", description="aprobar | revisar_parcial | regenerar")

class ValidationReport(BaseModel):
    """
    Output completo del agente de validación.
    """
    model_config              = ConfigDict(arbitrary_types_allowed=True)
    approved_validation: Any  = Field(default=False, description="Se aprueba o no")
    risk_level:          Any  = Field(default="", description="bajo | medio | alto | critico")
    conflicts:           Any  = Field(default_factory=list, description="Lista de conflictos")
    dependencies_map:    Any  = Field(default_factory=dict, description="Dictioray con las dependencias")
    detected_gaps:       Any  = Field(default_factory=list, description="Huecos encontrados en los requerimientos")
    final_opinion:       Any  = Field(default="", description="aprobado | aprobado_con_observaciones | rechazado")
    justification:       Any  = Field(default="", description="Justificacion breve del resultado")

class EntitiesDetected(BaseModel):
    """
    Entidades extraídas por el nodo de extraction.
    """
    model_config                 = ConfigDict(arbitrary_types_allowed=True)
    actors:                 Any  = Field(default_factory=list, description="Lista de personas o roles involucradas en el levantamiento de los requirimientos")
    systems:                Any  = Field(default_factory=list, description="Lista de sistemas que se involucraran o desarrollaran")
    actions:                Any  = Field(default_factory=list, description="Lista de acciones que se tiene que hacer")
    restrictions_mentioned: Any  = Field(default_factory=list, description="Lista de reestricciones a tomar en cuenta")

# definicion del esquema de estado
class AgentState(BaseModel):
    """
    Estado compartido que fluye entre todos los nodos del grafo.

    Todos los campos son opcionales (total=False) para que SqliteSaver
    pueda serializar estados parciales en cualquier punto del grafo sin
    lanzar errores por campos ausentes.

    Grupos:
      - Entrada        → escritos por ingesta, leídos por los demás
      - Catálogo       → escrito por generación, leído por análisis y validación
      - Análisis       → escrito por el agente de análisis
      - Validación     → escrito por el agente de validación
      - Control        → escritos por el supervisor para orquestar el grafo
      - Infraestructura → identidad de sesión y configuración
      - Comunicación   → canal usuario → grafo vía interrupt()
    """
    model_config            = ConfigDict(arbitrary_types_allowed=True)
    project_name:      Any  = Field(default="SIN NOMBRE", description="Nombre del proyecto actual")
    clean_text:        Any  = Field(default="", description="Texto normalizado producido por el nodo de ingesta.")
    entities:          Any  = Field(default_factory=EntitiesDetected, descripcion="Mapa de entidades extraídas por ingesta: actores, sistemas, acciones, restricciones. Ayuda a generación a identificar quién hace qué sin re-parsear el texto completo.")
    entry_type:        Any  = Field(default="descripcion", description="Origen de la entrada detectado por ingesta. Valores: conversacion | documento | descripcion | email, Default: descripcion")
    languaje:          Any  = Field(default="es", description="Idioma detectado por ingesta. Usado para ajustar el system prompt. Default: es")
    requirements:      Any  = Field(default_factory=list, description="Catálogo completo de requisitos generados. Cada elemento sigue la estructura de Requisito. Se reescribe completo en cada iteración donde el supervisor decide generar")
    quality_score:     Any  = Field(default=0, description="Puntuación global de calidad del catálogo de 0 a 100, calculada por el agente de analyst según las dimensiones IEEE 830 (claridad, completitud, verificabilidad, atomicidad).")
    analisis_report:   Any  = Field(default_factory=AnalysisReport, description="Reporte detallado del agente analyst con los issues encontrados por requisito — dimensión afectada, descripción del problema y sugerencia de corrección")
    validation_report: Any  = Field(default_factory=ValidationReport, description="Resultado del agente de validación con los conflictos entre requisitos, gaps detectados y el mapa de dependencias. Contiene el dictamen final: aprobado, aprobado con observaciones, o rechazado.")
    decision:          Any  = Field(default="", description="La decisión que acaba de tomar el supervisor en esta iteración. Los valores posibles son: generar, analizar, validar, revisar, finalizar.")
    iterations:        Any  = Field(default=1, description="Contador de cuántos ciclos completos ha ejecutado el grafo en esta sesión. El supervisor lo usa como freno de seguridad: si llega a 4 sin alcanzar score mayor a 85, fuerza la decisión revisar en lugar de continuar iterando indefinidamente.")
    thread_id:         Any  = Field(default="", description="Identificador único de la sesión de análisis.")
    mode:              Any  = Field(default="asistido", description="Controla el comportamiento del supervisor ante ambigüedades. En modo asistido el supervisor activa interrupt() cuando la confianza es baja y espera respuesta del usuario — ideal para sesiones iniciales o documentos complejos.")
    user_message:      Any  = Field(default=None, description="La respuesta que el usuario escribe en el panel de interrupt() de Streamlit. Cuando el grafo está pausado este campo es None. Cuando el PO escribe y envía su respuesta, Streamlit escribe aquí el texto y llama a graph.invoke() para reanudar el grafo. El supervisor lo lee, lo incorpora al contexto y decide el siguiente paso. Es el único canal directo de comunicación del usuario con el interior del grafo.")
    new_files:         Any  = Field(default_factory=list, description="Archivos nuevos recibidos del usuario")
    user_text:         Any  = Field(default="", description="Texto del usuario")

def init_state(thread_id: str, mode: str = "asistido") -> AgentState:
    """
    Devuelve un AgentState con todos los campos en sus valores por defecto.
    Usar siempre esta función al iniciar una sesión nueva desde app.py
    para garantizar consistencia.

    Args:
        thread_id: identificador único de sesión (generado con uuid4())
        modo:      "asistido" | "automatico"

    Returns:
        AgentState con valores por defecto listos para el primer nodo (ingesta)

    Ejemplo:
        import uuid
        state = estado_inicial(thread_id=str(uuid.uuid4()), modo="asistido")
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    """

    return AgentState(
        thread_id=thread_id,
        mode=mode
    )