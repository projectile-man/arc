"""
app.py
─────────────────────────────────────────────────────────────────────────────
Entry point de la aplicacion Streamlit del Asistente de Requisitos.
─────────────────────────────────────────────────────────────────────────────
"""
import logging
import uuid

import streamlit as st
from dotenv import load_dotenv

# Importacion compatible con todas las versiones de LangGraph
try:
    from langgraph.types import NodeInterrupt
except ImportError:
    try:
        from langgraph.errors import GraphInterrupt as NodeInterrupt
    except ImportError:
        from langgraph.errors import NodeInterrupt

import warnings
warnings.filterwarnings('ignore')
# ─────────────────────────────────────────────────────────────────────────────
# Configuracion inicial
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuracion de la pagina Streamlit
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ARC",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .interrupt-banner {
      background: #FAEEDA;
      border: 1.5px solid #BA7517;
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 16px;
  }
  .badge-po  { background:#FAECE7; color:#993C1D; padding:3px 10px;
               border-radius:12px; font-size:12px; font-weight:600; }
  .badge-sm  { background:#EEEDFE; color:#534AB7; padding:3px 10px;
               border-radius:12px; font-size:12px; font-weight:600; }
  .badge-pm  { background:#E1F5EE; color:#0F6E56; padding:3px 10px;
               border-radius:12px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Inicializacion del grafo (singleton por proceso)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def inicializar_grafo():
    """
    Construye y compila el grafo LangGraph una sola vez.
    st.cache_resource garantiza que solo se ejecuta al arrancar la app.
    """
    from graph import build_graph
    return build_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de sesion
# ─────────────────────────────────────────────────────────────────────────────

def _init_session() -> None:
    defaults = {
        "graph":             None,
        "thread_id":         str(uuid.uuid4()),
        "rol":               "Product Owner",
        "mode":              "asistido",
        "state":             None,
        "interrumpido":      False,
        "payload_interrupt": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _nueva_sesion() -> None:
    st.session_state.thread_id         = str(uuid.uuid4())
    st.session_state.state             = None
    st.session_state.interrumpido      = False
    st.session_state.payload_interrupt = None
    st.rerun()


def _config_lg() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


# ─────────────────────────────────────────────────────────────────────────────
# Configuracion de pasos del grafo para mostrar en la UI
# ─────────────────────────────────────────────────────────────────────────────

PASOS_INFO = {
    "ingesta":      ("📥", "Ingesta",      "Procesando documentos y extrayendo entidades..."),
    "supervisor":   ("🧠", "Supervisor",   "Evaluando el estado y decidiendo el siguiente paso..."),
    "generacion":   ("✍️",  "Generacion",  "Generando requisitos RF, RNF e historias de usuario..."),
    "analisis":     ("🔍", "Analisis",     "Analizando calidad: claridad, completitud y verificabilidad..."),
    "validacion":   ("✅", "Validacion",   "Validando consistencia y detectando conflictos..."),
    "human_review": ("👤", "Revision PO",  "Esperando decision del Product Owner..."),
}

def _ejecutar_con_progreso(input_state, config: dict, titulo: str):
    graph = st.session_state.graph

    st.markdown(f"**{titulo}**")
    contenedor_pasos = st.container()
    pasos_completados: list[tuple[str, str, str]] = []

    # stream() detiene la iteracion ante interrupt()
    # NO lanza excepcion — hay que detectar el interrupt
    # consultando el estado del checkpoint despues del stream
    if input_state is None:
        gen = graph.stream(None, config=config)
    else:
        gen = graph.stream(input_state, config=config)

    # for chunk in graph.stream(input_state, config=config, stream_mode="updates"):
    for chunk in gen:
        for nodo, _ in chunk.items():
            if nodo.startswith("__"):
                continue

            emoji, label, detalle = PASOS_INFO.get(
                nodo, ("⚙️", nodo.capitalize(), "Procesando...")
            )
            pasos_completados.append((emoji, label, detalle))

            with contenedor_pasos:
                for e, l, d in pasos_completados[:-1]:
                    st.success(f"{e} **{l}** — completado")
                st.info(f"{emoji} **{label}** — {detalle}")

    # Marcar todos como completados
    with contenedor_pasos:
        for e, l, _ in pasos_completados:
            st.success(f"{e} **{l}** — completado")

    # ── Verificar si el grafo quedo pausado por interrupt() ───────────────
    snapshot = graph.get_state(config)

    if snapshot and snapshot.next:
        # El grafo tiene pasos pendientes — esta pausado por interrupt()
        # Recuperar el payload del interrupt desde las tareas pendientes
        payload = {}
        if snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_obj = task.interrupts[0]
                    payload = getattr(interrupt_obj, "value", {})
                    if isinstance(payload, str):
                        payload = {"pregunta": payload}
                    break

        st.session_state.interrumpido      = True
        st.session_state.payload_interrupt = payload
        logger.info("Interrupt detectado via snapshot — thread: %s",
                    st.session_state.thread_id)
        st.rerun()
        return None

    # Grafo completado normalmente
    st.session_state.interrumpido      = False
    st.session_state.payload_interrupt = None
    return snapshot.values if snapshot else None

# ─────────────────────────────────────────────────────────────────────────────
# Ejecucion del grafo
# ─────────────────────────────────────────────────────────────────────────────

def _invocar_grafo(input_state: dict) -> None:
    """
    Invoca el grafo mostrando el progreso nodo a nodo.
    Maneja NodeInterrupt cuando el supervisor pausa esperando al usuario.
    """
    config = _config_lg()

    try:
        result = _ejecutar_con_progreso(
            input_state,
            config,
            titulo="🚀 Procesando tu solicitud...",
        )
        st.session_state.state             = result
        st.session_state.interrumpido      = False
        st.session_state.payload_interrupt = None
        logger.info("Grafo completado — thread: %s", st.session_state.thread_id)

    except NodeInterrupt as exc:
        st.session_state.interrumpido      = True
        st.session_state.payload_interrupt = getattr(exc, "value", str(exc))
        logger.info("Grafo pausado por interrupt() — thread: %s",
                    st.session_state.thread_id)

    except Exception as exc:
        st.error(f"Error inesperado en el grafo: {exc}")
        logger.error("Error en grafo: %s", exc, exc_info=True)

def _reanudar_grafo(respuesta_usuario: str) -> None:
    config = _config_lg()

    try:
        from langgraph.types import Command

        print("\n\n>>> REANUDANDO CON THREAD:", config)
        print(f">>> RESPUESTA USUARIO: {respuesta_usuario}")

        result = _ejecutar_con_progreso(
            Command(resume=respuesta_usuario),
            config,
            titulo="▶️ Reanudando el analisis...",
        )

        if result:
            st.session_state.state = result

        if not st.session_state.interrumpido:
            st.session_state.interrumpido      = False
            st.session_state.payload_interrupt = None

        logger.info("Grafo reanudado y completado — thread: %s",
                    st.session_state.thread_id)

        st.rerun()

    except Exception as exc:
        st.error(f"Error al reanudar el grafo: {exc}")
        logger.error("Error al reanudar: %s", exc, exc_info=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📋 ARC")
        st.caption("AI Requirements Copilot")
        st.divider()

        roles = ["Product Owner"]
        rol   = st.selectbox("Rol activo", roles,
                             index=roles.index(st.session_state.rol))
        if rol != st.session_state.rol:
            st.session_state.rol = rol
            st.rerun()

        badge_cls = {
            "Product Owner":   "badge-po"
        }[rol]
        st.markdown(f'<span class="{badge_cls}">{rol}</span>',
                    unsafe_allow_html=True)

        st.divider()

        mode = st.radio(
            "Mode de analisis",
            ["asistido"],
            index=0 if st.session_state.mode == "asistido" else 1,
            help=(
                "**Asistido**: pausa ante ambiguedades.\n\n"
            ),
        )
        if mode != st.session_state.mode:
            st.session_state.mode = mode

        st.divider()

        st.caption("Sesion activa")
        st.code(st.session_state.thread_id[:20] + "...", language=None)

        if st.button("🔄 Nueva sesion", use_container_width=True):
            _nueva_sesion()

        st.divider()

        if st.session_state.interrumpido:
            st.error("⏸ Esperando tu respuesta")
        elif st.session_state.state:
            state = st.session_state.state
            score = (state.get("quality_score", 0)
                     if isinstance(state, dict)
                     else getattr(state, "quality_score", 0))
            n_reqs = (len(state.get("requirements", []))
                      if isinstance(state, dict)
                      else len(getattr(state, "requirements", [])))
            st.success(f"✓ Score: {score}/100")
            st.info(f"📄 {n_reqs} requisitos")
        else:
            st.info("Sin sesion activa")

        st.divider()
        st.caption("v1.0 — ARC · LangGraph + Pinecone")

# ─────────────────────────────────────────────────────────────────────────────
# Vistas por rol
# ─────────────────────────────────────────────────────────────────────────────

def _render_po() -> None:
    from views.po_view import render
    render(
        graph         = st.session_state.graph,
        state         = st.session_state.state,
        thread_id     = st.session_state.thread_id,
        mode          = st.session_state.mode,
        invocar_grafo = _invocar_grafo,
        reanudar_grafo  = _reanudar_grafo,
    )


# def _render_sm() -> None:
#     from views.sm_view import render
#     render(state=st.session_state.state)


# def _render_pm() -> None:
#     from views.pm_view import render
#     render(
#         graph     = st.session_state.graph,
#         state     = st.session_state.state,
#         thread_id = st.session_state.thread_id,
#     )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()

    if st.session_state.graph is None:
        with st.spinner("Iniciando ARC..."):
            st.session_state.graph = inicializar_grafo()

    _render_sidebar()

    rol = st.session_state.rol
    if rol == "Product Owner":
        _render_po()
    # elif rol == "Scrum Master":
    #     _render_sm()
    # else:
    #     _render_pm()

if __name__ == "__main__":
    main()