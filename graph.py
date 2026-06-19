from pathlib import Path

from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from state import AgentState
from agents.supervisor import supervisor_agent, routing_supervisor
from agents.extractor import extractor_agent
from agents.generator import generator_agent
from agents.analyst import analyst_agent
from agents.validator import validator_agent
from agents.human_review import human_review

_DB_PATH = Path(__file__).parent / "data" / "checkpoints.db"

def build_graph():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Construyendo grafo LangGraph — db:", _DB_PATH)

    g = StateGraph(AgentState)

    g.add_node("extractor", extractor_agent)
    g.add_node("supervisor", supervisor_agent)
    g.add_node("generator", generator_agent)
    g.add_node("analyst", analyst_agent)
    g.add_node("validator", validator_agent)
    g.add_node("human_review", human_review)

    g.add_edge(START, "extractor")
    g.add_edge("extractor", "supervisor")
    g.add_edge("generator", "supervisor")
    g.add_edge("analyst", "supervisor")
    g.add_edge("validator", "supervisor")
    g.add_edge("human_review", "supervisor")

    g.add_conditional_edges(
        "supervisor",
        routing_supervisor,
        {
            "generator": "generator",
            "analyst": "analyst",
            "validator": "validator",
            "human_review": "human_review",
            "__end__": END,
        },
    )

    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return g.compile(checkpointer=checkpointer)

def get_state(graph, thread_id: str) -> AgentState | None:
    """
    Recupera el AgentState más reciente de una sesión por su thread_id.
    Útil para que la vista del PM monitoree el estado sin reinvocar el grafo.
 
    Args:
        graph:     grafo compilado retornado por build_graph()
        thread_id: identificador de la sesión
 
    Returns:
        AgentState con el estado actual, o None si el thread no existe
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.aget_state(config)
    if snapshot and snapshot.values:
        return snapshot.values
    return None
 
 
def get_history(graph, thread_id: str) -> list[dict]:
    """
    Retorna el historial completo de checkpoints de una sesión.
    Cada entrada contiene el AgentState y los metadatos del paso.
    Útil para el timeline de actividad en la vista del PM.
 
    Args:
        graph:     grafo compilado
        thread_id: identificador de la sesión
 
    Returns:
        Lista de dicts con keys: step, node, state, timestamp
    """
    config   = {"configurable": {"thread_id": thread_id}}
    history  = []
 
    for checkpoint in graph.aget_state_history(config):
        history.append({
            "step":      checkpoint.metadata.get("step", 0),
            "node":      checkpoint.metadata.get("source", "unknown"),
            "state":     checkpoint.values,
            "timestamp": checkpoint.metadata.get("created_at"),
        })
 
    return list(reversed(history))