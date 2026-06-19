from functools import lru_cache
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from config.parameters import PINECONE_INDEX, EMBEDDING_MODEL, EMBEDDING_DIM

from dotenv import load_dotenv
import os

load_dotenv()

@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """lru_cache garantiza que solo se crea una vez durante el
    ciclo de vida de la aplicacion para evitar aumentar costo
    por inicializaciones repetidas"""

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

@lru_cache(maxsize=1)
def get_index():
    """Reutiliza la conexion en todas las llamadas del grafo"""

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), ssl_verify=False)
    print("Index pinecone: ", pc.Index(PINECONE_INDEX))
    return pc.Index(PINECONE_INDEX)

def recuperar_contexto(query:str, top_k: int = 5, filtro:Optional[dict] = None, umbral:float=0.31) -> list[Document]:
    """Recupera los documentos mas relevantes desde pinecone"""
    if not query.strip():
        return []
    
    try:
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(query)

        index = get_index()
        
        results = index.query(
            vector          = query_vector,
            top_k           = top_k,
            include_metadata= True,
            filter          = filtro or {},
        )

        # documentos = []

        # for match in results.get("matches", []):
        #     score = match.get("score", 0.0)

        #     if score < umbral:
        #         continue

        #     metadata = match.get("metadata", {})
        #     texto = metadata.get("texto", "")

        #     metadata_limpia = {
        #         k: v
        #         for k, v in metadata.items()
        #         if k != "texto"
        #     }

        #     documentos.append(
        #         Document(
        #             page_content=texto,
        #             metadata={
        #                 **metadata_limpia,
        #                 "score": round(score, 4)
        #             }
        #         )
        #     )

        # print(f"Documentos recuperados: {len(documentos)}")

        # for doc in documentos:
        #     print(doc.metadata["score"], doc.page_content[:100])

        # return documentos
            
        documentos = []

        for match in results.get("matches", []):
            score = match.get("score", 0.0)
            texto = match.get("metadata", {}).get("texto", "")
            if score > umbral:
                documentos.append(texto)

        return documentos
    
    except Exception as e:
        return []

def recuperar_por_sesion(
    query:     str,
    thread_id: str,
    top_k:     int = 5,
) -> list[Document]:
    """
    Recupera documentos indexados en la sesión actual identificada por
    thread_id. Útil para que los agentes consulten solo los documentos
    subidos por el usuario en la sesión en curso, no toda la knowledge base.
    """
    return recuperar_contexto(
        query  = query,
        top_k  = top_k,
        filtro = {"thread_id": thread_id},
        umbral = 0.65, 
    )
 
def recuperar_knowledge_base(
    query:  str,
    agente: str,
    top_k:  int = 4,
) -> list[Document]:
    """
    Recupera documentos de la knowledge base preindexada (IEEE 830,
    BABOK, plantillas, ejemplos de requisitos) filtrados por agente.
    """
    return recuperar_contexto(
        query  = query,
        top_k  = top_k,
        filtro = {"agente": agente, "tipo_doc": "knowledge_base"},
        umbral = 0.72,
    )