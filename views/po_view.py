"""
po_view.py
─────────────────────────────────────────────────────────────────────────────
Vista del Product Owner en Streamlit.

Responsabilidades:
  - Carga de documentos o texto libre para iniciar el analisis
  - Visualizacion del catalogo de requisitos con estado de aprobacion
  - Panel de metricas de calidad (score, dimensiones, iteraciones)
  - Exportacion del catalogo aprobado
  - Recibe el panel de interrupt() desde app.py (no lo renderiza aqui)

Funcion publica: render(graph, state, thread_id, modo, invocar_grafo)
─────────────────────────────────────────────────────────────────────────────
"""
import tempfile
from pathlib import Path
from typing import Callable
import csv, io
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from state import init_state
from config.models import CHAT_MODEL, CHAT_TEMPERATURE
from agents.prompts import CHAT_PROMPT

def _get(state, campo: str, default=None):
    """Accede a un campo del estado (dict o BaseModel)."""
    if state is None:
        return default
    if isinstance(state, dict):
        return state.get(campo, default)
    return getattr(state, campo, default)

def _score(state) -> int:
    return _get(state, "quality_score", 0)

def _requisitos(state) -> list:
    return _get(state, "requirements", [])

def _reporte_analisis(state):
    return _get(state, "analisis_report", None)

def _reporte_validacion(state):
    return _get(state, "validation_report", None)

def _nombre_proyecto(state):
    return _get(state, "project_name", "")

def _render_chat(thread_id: str) -> None:

    # Inicializar historial en session_state si no existe
    if "chat_historial" not in st.session_state:
        st.session_state.chat_historial = []

    st.subheader("💬 Consulta sobre requisitos")

    # Selector de alcance
    col_alc, col_limpiar = st.columns([3, 1])
    with col_alc:
        alcance = st.radio(
            "Buscar en:",
            ["Proyecto actual", "Todos los proyectos"],
            horizontal=True,
            key="chat_alcance",
        )
    with col_limpiar:
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("🗑️ Limpiar chat", use_container_width=True):
            st.session_state.chat_historial = []
            st.rerun()

    st.divider()

    # ── Input del usuario ─────────────────────────────────────────────────

    # Input principal
    pregunta_input = st.chat_input(
        "Escribe tu pregunta sobre los requisitos...",
        key="chat_input_principal",
    )

    # ── Historial de mensajes ─────────────────────────────────────────────
    contenedor_chat = st.container(height=420)

    with contenedor_chat:
        if not st.session_state.chat_historial:
            st.markdown(
                """
                <div style="text-align:center;color:#888;padding:40px 0;">
                    <div style="font-size:32px">💬</div>
                    <div style="font-size:14px;margin-top:8px">
                        Haz una pregunta sobre los requisitos del proyecto.<br/>
                        Puedes preguntar por requisitos especificos, comparar<br/>
                        proyectos o buscar en el historial.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # for mensaje in st.session_state.chat_historial:
            for mensaje in reversed(st.session_state.chat_historial):
                es_usuario = mensaje["rol"] == "user"
                alineacion = "flex-end" if es_usuario else "flex-start"
                color = "#2563eb" if es_usuario else "#f1f5f9"
                texto = "white" if es_usuario else "#111827"

                st.markdown(
                    f"""
                    <div style="
                        display:flex;
                        justify-content:{alineacion};
                        margin:8px 0;
                    ">
                        <div style="
                            max-width:75%;
                            padding:12px 16px;
                            border-radius:18px;
                            background:{color};
                            color:{texto};
                            box-shadow:0 1px 3px rgba(0,0,0,.15);
                            word-wrap:break-word;
                        ">
                            {mensaje["contenido"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Mostrar fuentes solo para respuestas del assistant
                if not es_usuario and mensaje.get("fuentes"):
                    with st.expander(
                        f"📄 {len(mensaje['fuentes'])} fuente(s) consultadas",
                        expanded=False,
                    ):
                        for i, fuente in enumerate(mensaje["fuentes"], 1):
                            score = fuente.get("score", 0)
                            texto = fuente.get("texto", "")[:200]
                            nombre = fuente.get("nombre", f"Documento {i}")

                            st.markdown(
                                f"**{i}. {nombre}** "
                                f"*(relevancia: {score:.0%})*\n\n"
                                f"> {texto}..."
                            )

    

    # Procesar pregunta — ya sea del input o de una sugerencia
    pregunta_a_procesar = None

    if pregunta_input:
        pregunta_a_procesar = pregunta_input

    if st.session_state.get("chat_pregunta_pendiente"):
        pregunta_a_procesar = st.session_state.pop("chat_pregunta_pendiente")

    if pregunta_a_procesar:
        # Agregar mensaje del usuario al historial
        st.session_state.chat_historial.append({
            "rol":       "user",
            "contenido": pregunta_a_procesar,
            "fuentes":   [],
        })

        # Generar respuesta
        with st.spinner("Buscando en la base de conocimiento..."):
            # respuesta, fuentes = _consultar_rag_chat(
            respuesta = _consultar_rag_chat(
                pregunta    = pregunta_a_procesar,
                thread_id   = thread_id,
                solo_actual = (alcance == "Proyecto actual"),
            )

        # Agregar respuesta al historial
        st.session_state.chat_historial.append({
            "rol":       "assistant",
            "contenido": respuesta,
            # "fuentes":   fuentes,
            "fuentes":   []
        })

        st.rerun()

def _consultar_rag_chat(
    pregunta: str,
    thread_id: str,
    solo_actual: bool,
) -> tuple[str, list[dict]]:
    """
    Recupera contexto de Pinecone y genera una respuesta con el LLM.
    """
    from langchain_openai import ChatOpenAI
    from rag.retriever import recuperar_contexto

    filtro = {"thread_id": thread_id} if solo_actual else {}
    
    try:
        docs = recuperar_contexto(
            query  = pregunta,
            top_k  = 6,
            filtro = filtro,
            umbral = 0.31,
        )
        
    except Exception as exc:
        return f"Error al consultar la base de conocimiento: {exc}"

    if not docs:
        return (
            "No encontré información relevante en la base de conocimiento "
            "para tu pregunta. Intenta reformularla o amplía el alcance "
            "a 'Todos los proyectos'."
        )

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        max_tokens=1024,
        temperature=CHAT_TEMPERATURE,
    )

    try:
        contexto_rag = "\n\n".join(docs)

        prompt = PromptTemplate.from_template(
            CHAT_PROMPT
        )

        chain = (
            {
                "context": RunnablePassthrough(),
                "query": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke({
            "context": contexto_rag,
            "query": pregunta
        })
        
        return response

    except Exception as exc:
        return f"Error al generar la respuesta: {exc}"
    
# ─────────────────────────────────────────────────────────────────────────────
# Render principal
# ─────────────────────────────────────────────────────────────────────────────

def render(
    graph,
    state,
    thread_id: str,
    mode: str,
    invocar_grafo: Callable,
    reanudar_grafo: Callable,
) -> None:
    st.markdown(
        '<span class="badge-po">Product Owner</span>',
        unsafe_allow_html=True,
    )
    st.title("Panel del Product Owner")

    esta_interrumpido = st.session_state.get("interrumpido", False)

    if esta_interrumpido:
        _render_interrupt_po(reanudar_grafo)
        st.divider()

    # ── Metricas superiores ───────────────────────────────────────────────────
    if state:
        _render_metricas(state)
        st.divider()

    # ── Tabs principales ──────────────────────────────────────────────────────
    tab_nueva, tab_catalogo, tab_calidad, tab_chat, tab_export = st.tabs([
        "📥 Nueva entrada",
        "📋 Catalogo de requisitos",
        "📊 Calidad",
        "💬 Chat",
        "📤 Exportar",
    ])

    with tab_nueva:
        _render_nueva_entrada(
            graph, state, thread_id, mode, invocar_grafo
        )

    with tab_catalogo:
        _render_catalogo(state)

    with tab_calidad:
        _render_calidad(state)

    with tab_chat:
        _render_chat(thread_id)

    with tab_export:
        _render_export(state)

def _render_interrupt_po(reanudar_grafo: Callable) -> None:
    """
    Panel de revision humana embebido en la vista del PO.
    Se muestra cuando el supervisor pausa el grafo esperando decision.
    """

    payload  = st.session_state.get("payload_interrupt") or {}
    if isinstance(payload, str):
        payload = {"pregunta": payload}

    pregunta        = payload.get("pregunta", "El sistema necesita tu decision para continuar.")
    score           = payload.get("score_actual", 0)
    iters           = payload.get("iteraciones", 0)
    razon           = payload.get("razonamiento", "")
    reqs_pendientes = payload.get("requisitos_pendientes", [])

    st.markdown(f"""
    <div style="background:#FAEEDA;border:1.5px solid #BA7517;
                border-radius:10px;padding:16px;margin-bottom:8px;">
        <strong>⏸ Necesitas tomar una decision para continuar el analisis</strong><br/>
        Score actual: <strong>{score}/100</strong> &nbsp;|&nbsp;
        Iteraciones: <strong>{iters}</strong>
    </div>
    """, unsafe_allow_html=True)

    if razon:
        st.caption(f"Razonamiento del supervisor: {razon}")

    # CASO 1: requisitos que necesitan clarificacion
    if reqs_pendientes:
        st.warning(
            f"**{len(reqs_pendientes)} requisito(s) necesitan tu "
            f"clarificacion antes de continuar:**"
        )

        st.markdown("**Aclara cada requisito:**")
 
        with st.form("form_clarificacion_po", clear_on_submit=True):
            respuestas: dict[str, str] = {}
            for req in reqs_pendientes:
                req_id   = req.get("id", "?")
                req_desc = req.get("descripcion", "")
                respuestas[req_id] = st.text_area(
                    f"**{req_id}** — {req_desc}",
                    placeholder=(
                        f"Aclara exactamente qué debe hacer {req_id}. "
                        "Ej: por 'gestionar' me refiero a ver y editar, "
                        "no a eliminar..."
                    ),
                    height=80,
                    key=f"clarif_{req_id}",
                )
 
            col1, col2, col3 = st.columns(3)
            with col1:
                enviado   = st.form_submit_button(
                    "✅ Enviar clarificaciones",
                    type="primary",
                    use_container_width=True,
                )
            with col2:
                aprobado  = st.form_submit_button(
                    "👍 Aprobar tal como esta",
                    use_container_width=True,
                )
            with col3:
                rechazado = st.form_submit_button(
                    "🔄 Rechazar y reiniciar",
                    use_container_width=True,
                )
 
        if enviado:
            partes = [
                f"{req_id}: {clarif.strip()}"
                for req_id, clarif in respuestas.items()
                if clarif.strip()
            ]
            if partes:
                mensaje = (
                    "Clarificaciones del Product Owner:\n\n"
                    + "\n".join(partes)
                )
                reanudar_grafo(mensaje)
            else:
                st.warning("Por favor aclara al menos uno de los requisitos.")
 
        if aprobado:
            reanudar_grafo(
                "El Product Owner aprueba el catalogo tal como esta. "
                "Marcar todos los requisitos como definidos y continuar."
            )
 
        if rechazado:
            reanudar_grafo("Rechazar el catalogo y empezar de nuevo.")

    # else:
    #     st.warning(f"**El agente pregunta:** {pregunta}")

    #     with st.form("form_interrupt_po", clear_on_submit=False):

    #         val     = _reporte_validacion(state)
    #         if val:
    #             conflictos = _get(val, "conflicts", []) or []
    #             if conflictos:
    #                 st.subheader(f"Conflictos detectados ({len(conflictos)})")
    #                 for c in conflictos:
    #                     tipo     = _get(c, "conflict_type",         "")
    #                     desc     = _get(c, "description",  "")
    #                     impacto  = _get(c, "impact",      "")
    #                     resol    = _get(c, "suggested_resolution", "")
    #                     reqs_inv = _get(c, "requirements_involved", []) or []
    #                     color    = {
    #                         "alto":  "🔴",
    #                         "medio": "🟡",
    #                         "bajo":  "🟢",
    #                     }.get(impacto, "⚪")
        
    #                     with st.expander(
    #                         f"{color} [{impacto}] {tipo} — {', '.join(reqs_inv)}"
    #                     ):
    #                         st.markdown(f"**Descripcion:** {desc}")
    #                         if resol:
    #                             st.success(f"**Resolucion sugerida:** {resol}")

    #         respuesta = st.text_area(
    #             "Tu respuesta o clarificacion",
    #             placeholder=(
    #                 "Ej: por gestionar el perfil me refiero a que el usuario "
    #                 "pueda editar su nombre y correo, eliminar cuenta queda "
    #                 "fuera del alcance de este sprint..."
    #             ),
    #             height=120,
    #         )
    #         col1, col2, col3 = st.columns(3)
    #         with col1:
    #             enviado   = st.form_submit_button("✅ Enviar y continuar", type="primary")
    #         with col2:
    #             aprobado  = st.form_submit_button("👍 Aprobar tal como esta")
    #         with col3:
    #             rechazado = st.form_submit_button("🔄 Rechazar y reiniciar")

    #     if enviado and respuesta.strip():
    #         reanudar_grafo(respuesta.strip())
    #     elif enviado:
    #         st.warning("Por favor escribe tu respuesta antes de enviar.")

    #     if aprobado:
    #         reanudar_grafo("apruebo el catalogo actual, adelante")
 
    #     if rechazado:
    #         reanudar_grafo("rechazo el catalogo, empezar de nuevo")

def _render_metricas(state) -> None:
    
    st.markdown(
        f"""
        <span style="
            background-color:#1976D2;
            color:white;
            padding:6px 12px;
            border-radius:16px;
            font-size:14px;
            font-weight:600;
        ">
            Proyecto: {_nombre_proyecto(state)}
        </span>
        """,
        unsafe_allow_html=True,
    )

    reqs     = _requisitos(state)
    score    = _score(state)
    iters    = _get(state, "iterations", 0)
    val      = _reporte_validacion(state)
    conflictos   = _get(val, "conflicts", []) or [] if val else []
    n_conflictos = len(conflictos)

    pendientes = sum(
        1 for r in reqs
        if _get(r, "state", "") == "necesita_clarificacion"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
 
    with col1:
        st.metric(
            "Score de calidad", f"{score}/100",
            delta="Listo para exportar" if score >= 85 else None,
        )
    with col2:
        rf  = sum(1 for r in reqs if _get(r, "requirement_type", "") == "RF")
        rnf = sum(1 for r in reqs if _get(r, "requirement_type", "") == "RNF")
        hu  = sum(1 for r in reqs if _get(r, "requirement_type", "") == "HU")
        st.metric(
            "Requisitos", len(reqs),
            help=f"RF: {rf} | RNF: {rnf} | HU: {hu}",
        )
    with col3:
        st.metric("Iteraciones", iters)
    with col4:
        st.metric(
            "Conflictos", n_conflictos,
            delta="Sin conflictos" if n_conflictos == 0 else None,
            delta_color="normal" if n_conflictos == 0 else "inverse",
        )
    with col5:
        st.metric(
            "Pendientes PO", pendientes,
            delta="Requieren clarificacion" if pendientes > 0 else None,
            delta_color="inverse" if pendientes > 0 else "normal",
        )
 
    if pendientes > 0:
        st.warning(
            f"⚠️ Hay **{pendientes} requisito(s)** marcados como ambiguos. "
            f"El agente ha pausado esperando tu clarificacion en el panel superior."
        )

def _render_nueva_entrada(
    graph, state, thread_id, mode, invocar_grafo
) -> None:
    st.subheader("Cargar documentos o describir el requerimiento")

    modo_entrada = st.radio(
        "Tipo de entrada",
        ["Subir archivos"],
        horizontal=True,
    )

    archivos_procesados = []

    uploaded = st.file_uploader(
        "Sube uno o varios archivos",
        type=["pdf", "docx", "txt", "eml", "mp3", "wav", "mp4", "mov"],
        accept_multiple_files=True,
        help="Soporta: PDF, Word, texto, email (.eml), audio y video.",
    )

    texto_usuario   = ""
    archivos_nuevos = []

    if uploaded:
        for f in uploaded:
            # Guardar temporalmente en disco
            sufijo = Path(f.name).suffix.lower().lstrip(".")
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{sufijo}"
            ) as tmp:
                tmp.write(f.getbuffer())
                ruta_tmp = tmp.name

            archivos_nuevos.append({
                "ruta":   ruta_tmp,
                "tipo":   sufijo,
                "nombre": f.name,
            })
            st.success(f"✓ {f.name} listo para procesar")

    st.divider()

    col_btn, col_modo = st.columns([2, 3])
    with col_modo:
        st.caption(
            f"Modo: **{mode}** — "
            + ("el sistema pausara ante ambiguedades." if mode == "asistido"
               else "el sistema decidira automaticamente.")
        )

    with col_btn:
        analizar = st.button(
            "🚀 Analizar con IA",
            type="primary",
            use_container_width=True,
            disabled=(not texto_usuario and not archivos_nuevos),
        )

    if analizar:
        # Construir estado inicial para esta sesion
        state_inicial = init_state(
            thread_id=thread_id,
            mode=mode,
        )
        state_inicial.user_text = texto_usuario
        state_inicial.new_files = archivos_nuevos

        invocar_grafo(state_inicial.model_dump())
        st.rerun()


def _render_catalogo(state) -> None:
    reqs = _requisitos(state)

    if not reqs:
        st.info("Aun no hay requisitos generados. "
                "Ve a 'Nueva entrada' para iniciar el analisis.")
        return

    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filtro_tipo = st.multiselect(
            "Tipo", ["RF", "RNF", "HU"],
            default=["RF", "RNF", "HU"],
        )
    with col_f2:
        filtro_pri = st.multiselect(
            "Prioridad", ["alta", "media", "baja"],
            default=["alta", "media", "baja"],
        )
    with col_f3:
        filtro_estado = st.multiselect(
            "Estado",
            ["definido", "necesita_clarificacion"],
            default=["definido", "necesita_clarificacion"],
        )

    # Aplicar filtros
    reqs_filtrados = [
        r for r in reqs
        if _get(r,"requirement_type","") in filtro_tipo
        and _get(r,"priority","") in filtro_pri
        and _get(r,"state","") in filtro_estado
    ]

    pendientes = [
        r for r in reqs_filtrados
        if _get(r, "estado", "") == "necesita_clarificacion"
    ]
    aprobados = [
        r for r in reqs_filtrados
        if _get(r, "estado", "") != "necesita_clarificacion"
    ]

    st.caption(
        f"Mostrando {len(reqs_filtrados)} de {len(reqs)} requisitos "
        f"({len(pendientes)} pendientes de clarificacion)"
    )
    if pendientes:
        st.markdown("#### ⚠️ Pendientes de clarificacion")
        for req in pendientes:
            _render_req_card(req)
        st.divider()
 
    if aprobados:
        st.markdown("#### ✅ Requisitos definidos")
        for req in aprobados:
            _render_req_card(req)


def _render_req_card(req) -> None:
    req_id   = _get(req, "id", "")
    tipo     = _get(req, "requirement_type", "RF")
    desc     = _get(req, "description",  "")
    actor    = _get(req, "players", "")
    prioridad= _get(req, "priority", "media")
    estado   = _get(req, "state", "definido")
    fuente   = _get(req, "source", "")

    color_tipo = {"RF": "🔵", "RNF": "🟢", "HU": "🟣"}.get(tipo, "⚪")
    color_pri  = {"alta": "🔴", "media": "🟡", "baja": "🟢"}.get(prioridad, "⚪")
    alerta     = "⚠️ " if estado == "necesita_clarificacion" else ""

    with st.expander(
        f"{alerta}{color_tipo} **{req_id}** — {desc[:80]}{'...' if len(desc)>80 else ''}",
        expanded=(estado == "necesita_clarificacion"),
    ):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Tipo:** {tipo}")
        with col2:
            st.markdown(f"**Prioridad:** {color_pri} {prioridad}")
        with col3:
            st.markdown(f"**Actor:** {actor}")

        st.markdown(f"**Descripcion:**\n{desc}")

        if estado == "necesita_clarificacion":
            st.warning("Este requisito necesita clarificacion del Product Owner.")

        if fuente:
            st.caption(f"Fuente: *{fuente[:120]}{'...' if len(fuente)>120 else ''}*")


def _render_calidad(state) -> None:
    score   = _score(state)
    reporte = _reporte_analisis(state)
    val     = _reporte_validacion(state)

    if not reporte and not val:
        st.info("El analisis de calidad se generara automaticamente "
                "despues de analizar los requisitos.")
        return

    # Score visual
    col_score, col_desc = st.columns([1, 3])
    with col_score:
        color = ("green" if score >= 85
                 else "orange" if score >= 70
                 else "red")
        st.markdown(
            f"<h1 style='color:{color};text-align:center'>{score}</h1>"
            f"<p style='text-align:center;color:grey'>de 100</p>",
            unsafe_allow_html=True,
        )
    with col_desc:
        nivel = ("Excelente ✓"                  if score >= 85
                 else "Bueno — validando"       if score >= 70
                 else "Aceptable — reanalizado" if score >= 40
                 else "Insuficiente — regenerando")
        st.markdown(f"### {nivel}")

        if reporte:
            rec = _get(reporte, "suggestion", "")
            if rec:
                st.caption(f"Recomendacion del agente: **{rec}**")

        # Barras de progreso por dimension
        resumen = _get(reporte, "resumen_issues", {}) if reporte else {}
        if not isinstance(resumen, dict):
            resumen = {}
 
        dims = {
            "claridad":        5,
            "completitud":     8,
            "verificabilidad": 10,
            "atomicidad":      3,
        }
        for dim, pts in dims.items():
            n_issues  = resumen.get(dim, 0)
            descuento = n_issues * pts
            pct       = max(0, 100 - descuento) / 100
            col_d, col_b = st.columns([2, 5])
            with col_d:
                st.markdown(f"**{dim.capitalize()}**")
            with col_b:
                st.progress(pct)
 
    st.divider()
 
    if reporte:
        issues = _get(reporte, "issues", []) or []
        if issues:
            st.subheader(f"Issues detectados ({len(issues)})")
            for issue in issues:
                dim    = _get(issue, "dimension",    "")
                desc   = _get(issue, "description",  "")
                sug    = _get(issue, "suggestion",   "")
                req_id = _get(issue, "id_requirement", "")
                icon   = {
                    "claridad":        "💬",
                    "completitud":     "📝",
                    "verificabilidad": "🔍",
                    "atomicidad":      "✂️",
                }.get(dim, "❓")
 
                with st.expander(
                    f"{icon} **{req_id}** [{dim}] — {desc[:60]}"
                ):
                    st.markdown(f"**Problema:** {desc}")
                    if sug:
                        st.success(f"**Sugerencia:** {sug}")
 
    if val:
        st.divider()
        dictamen   = _get(val, "final_opinion",  "")
        riesgo     = _get(val, "risk_level",    "")
        conflictos = _get(val, "conflicts",      []) or []
        gaps       = _get(val, "detected_gaps", []) or []
        just       = _get(val, "justification",   "")
 
        col_d, col_r = st.columns(2)
        with col_d:
            icon_dict = {
                "aprobado":                   "✅",
                "aprobado_con_observaciones": "⚠️",
                "rechazado":                  "❌",
            }.get(dictamen, "❓")
            st.markdown(f"### Dictamen: {icon_dict} {dictamen}")
        with col_r:
            st.markdown(f"### Riesgo: **{riesgo}**")
 
        if just:
            st.info(just)
 
        if conflictos:
            st.subheader(f"Conflictos detectados ({len(conflictos)})")
            for c in conflictos:
                tipo     = _get(c, "conflict_type",         "")
                desc     = _get(c, "description",  "")
                impacto  = _get(c, "impact",      "")
                resol    = _get(c, "suggested_resolution", "")
                reqs_inv = _get(c, "requirements_involved", []) or []
                color    = {
                    "alto":  "🔴",
                    "medio": "🟡",
                    "bajo":  "🟢",
                }.get(impacto, "⚪")
 
                with st.expander(
                    f"{color} [{impacto}] {tipo} — {', '.join(reqs_inv)}"
                ):
                    st.markdown(f"**Descripcion:** {desc}")
                    if resol:
                        st.success(f"**Resolucion sugerida:** {resol}")
 
        if gaps:
            st.subheader(f"Gaps criticos ({len(gaps)})")
            for g in gaps:
                area = _get(g, "area",        "")
                desc = _get(g, "descripcion", "")
                sugs = _get(g, "requisitos_sugeridos", []) or []
                with st.expander(f"⚠️ Gap: {area}"):
                    st.markdown(desc)
                    if sugs:
                        st.markdown("**Requisitos sugeridos:**")
                        for s in sugs:
                            st.markdown(f"- {s}")


def _render_export(state) -> None:
    reqs  = _requisitos(state)
    score = _score(state)
 
    if not reqs:
        st.info("No hay requisitos que exportar todavia.")
        return
 
    st.subheader("Exportar catalogo de requisitos")
 
    if score < 85:
        st.warning(
            f"El score actual es **{score}/100**. "
            "Se recomienda alcanzar **85+** antes de exportar."
        )
 
    pendientes = [
        r for r in reqs
        if _get(r, "estado", "") == "necesita_clarificacion"
    ]
    if pendientes:
        st.error(
            f"⚠️ Hay **{len(pendientes)} requisito(s)** pendientes de "
            f"clarificacion. Responde en el panel superior antes de exportar."
        )
 
    st.divider()
    col1, col2 = st.columns(2)
 
    with col1:
        st.markdown("#### Exportaciones directas")
 
        reqs_dict = [
            r if isinstance(r, dict) else r.model_dump()
            for r in reqs
        ]
 
        buf    = io.StringIO()
        campos = ["id", "requirement_type", "category", "description",
                  "players", "priority", "state", "source"]
        writer = csv.DictWriter(buf, fieldnames=campos)
        writer.writeheader()
        for r in reqs_dict:
            writer.writerow({k: r.get(k, "") for k in campos})
 
        st.download_button(
            label="⬇️ Descargar CSV",
            data=buf.getvalue().encode("utf-8"),
            file_name=f"requisitos_{st.session_state.thread_id[:8]}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.markdown("#### Integraciones- DISPONIBLE EN LA VERSION 2.0 DE ARC")

        if st.button("📝 Exportar a Word (IEEE 830)", use_container_width=True):
            st.info("DISPONIBLE EN LA VERSION 2.0 DE ARC")

        if st.button("📝 Exportar a PDF (IEEE 830)", use_container_width=True):
            st.info("DISPONIBLE EN LA VERSION 2.0 DE ARC")

        if st.button("📋 Exportar a Jira - NO DISPONIBLE AÚN", use_container_width=True):
            st.info("DISPONIBLE EN LA VERSION 2.0 DE ARC")

        if st.button("📄 Exportar a Confluence - NO DISPONIBLE AÚN", use_container_width=True):
            st.info("DISPONIBLE EN LA VERSION 2.0 DE ARC")