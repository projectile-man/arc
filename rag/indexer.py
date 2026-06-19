import hashlib
import uuid
from pathlib import Path
from typing import Optional
import os

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from utils.utils import _extraer_pdf, _extraer_word
from config.parameters import PINECONE_CLOUD, PINECONE_REGION, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_DIM, EMBEDDING_MODEL, PINECONE_INDEX

from dotenv import load_dotenv

load_dotenv()

def get_pc() -> Pinecone:
    return Pinecone(api_key=os.getenv("PINECONE_API_KEY"), ssl_verify=False)

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model = EMBEDDING_MODEL,
        openai_api_key = os.getenv("OPENAI_API_KEY")
    )

def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size         = CHUNK_SIZE,
        chunk_overlap      = CHUNK_OVERLAP,
        separators         = ["\n\n", "\n", ". ", " ", ""],
        length_function    = len,
        is_separator_regex = False,
    )

def asegurar_indice() -> None:
    """
    Crea el índice Pinecone si no existe.
    Idempotente — si ya existe, no hace nada.
    """
    pc             = get_pc()
    indices        = [i.name for i in pc.list_indexes()]
 
    if PINECONE_INDEX not in indices:
        print("Indexer — creando índice '{}'".format(PINECONE_INDEX))
        pc.create_index(
            name      = PINECONE_INDEX,
            dimension = EMBEDDING_DIM,
            metric    = "cosine",
            spec      = ServerlessSpec(
                cloud  = PINECONE_CLOUD,
                region = PINECONE_REGION,
            ),
        )
        print("Indexer — índice '{}' creado correctamente".format(PINECONE_INDEX))
    else:
        print("Indexer — índice '{}' ya existe".format(PINECONE_INDEX))

def indexar_knowledge_base(
    directorio: str | Path,
    agente:     Optional[str] = None,
) -> dict[str, int]:
    asegurar_indice()

    directorio = Path(directorio)
    if not directorio.exists():
        raise FileNotFoundError(f"Directorio KB no encontrado: {directorio}")

    resultados: dict[str, int] = {}

    for archivo in directorio.rglob("*"):
        print("--Archivo a procesar: ", archivo)

        if archivo.suffix.lower() == ".pdf":
            texto = _extraer_pdf(archivo)
        elif archivo.suffix.lower() == ".docx":
            texto = _extraer_word(archivo)
        elif archivo.suffix.lower() == ".txt":  
            texto = archivo.read_text(encoding="utf-8", errors="ignore")
        else:
            continue

        if not texto or not texto.strip():
            continue

        agente_archivo = agente or inferir_agente(archivo, directorio)
        n = indexar_documento(
            texto    = texto,
            metadata = {
                "nombre":   archivo.name,
                "tipo_doc": "knowledge_base",
                "agente":   agente_archivo,
                "fuente":   str(archivo.relative_to(directorio)),
            },
        )
        resultados[archivo.name] = n

    return resultados
 
 
def eliminar_por_sesion(thread_id: str) -> None:
    """
    Elimina de Pinecone todos los vectores asociados a un thread_id.
    Útil para limpiar documentos de sesiones anteriores y no acumular
    vectores huérfanos en el índice.
 
    Args:
        thread_id: identificador de la sesión a limpiar
    """
    try:
        pc    = get_pc()
        index = pc.Index(PINECONE_INDEX)
        index.delete(filter={"thread_id": thread_id})
    except Exception as exc:
        print("Error al eliminar sesion '{}': {}".format(thread_id, exc))
 
 
def generar_id(texto: str, metadata: dict, indice: int) -> str:
    """
    Genera un ID determinista para un chunk basado en su contenido
    y metadata. Si el mismo chunk se re-indexa, Pinecone hace upsert
    en lugar de crear un duplicado.
    """
    clave = f"{metadata.get('nombre', '')}:{metadata.get('thread_id', '')}:{indice}:{texto[:100]}"
    return hashlib.md5(clave.encode()).hexdigest()
 
def inferir_agente(archivo: Path, base: Path) -> str:
    """
    Infiere el agente desde la estructura del directorio.
    data/knowledge_base/generacion/ieee830.txt → "generacion"
    """
    try:
        relativo = archivo.relative_to(base)
        partes   = relativo.parts
        if len(partes) > 1:
            agente = partes[0].lower()
            if agente in ("generacion", "analisis", "validacion"):
                return agente
    except ValueError:
        pass
    return "general"
 
 
def batches(lista: list, size: int):
    """Divide una lista en lotes de tamaño máximo `size`."""
    for i in range(0, len(lista), size):
        yield lista[i : i + size]

def indexar_documento(
    texto:    str,
    metadata: Optional[dict] = None,
) -> int:
    if not texto.strip():
        return 0

    asegurar_indice()
    metadata = metadata or {}

    splitter = get_splitter()
    chunks   = splitter.split_text(texto)

    embeddings = get_embeddings()
    vectores   = embeddings.embed_documents(chunks)

    vectors = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectores)):
        chunk_id = generar_id(texto=chunk, metadata=metadata, indice=i)
        vectors.append({
            "id":     chunk_id,
            "values": vector,
            "metadata": {
                **metadata,
                "texto":       chunk,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "tipo_doc":    metadata.get("tipo_doc", "documento_usuario"),
            },
        })

    pc    = get_pc()
    index = pc.Index(PINECONE_INDEX)

    total_indexados = 0
    for batch in batches(vectors, size=100):
        index.upsert(vectors=batch)
        total_indexados += len(batch)

    return total_indexados