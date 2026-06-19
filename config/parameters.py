FINISH_SCORE     = 85
VALIDATION_SCORE = 70
REGENERATE_SCORE = 40
MAX_ITERATIONS   = 6

PENALIZACION = {
    "claridad":        5,
    "completitud":     8,
    "verificabilidad": 10,
    "atomicidad":      3,
}

KEYWORDS: dict[str, list[str]] = {
    "aprobacion": [
        "apruebo", "aprobado", "ok", "listo", "correcto", "perfecto",
        "adelante", "continúa", "continua", "está bien", "de acuerdo",
        "aceptado", "confirmo", "sí", "si",
    ],
    "rechazo": [
        "rechaz", "empezar de nuevo", "reiniciar", "incorrecto",
        "está mal", "no es correcto", "no sirve", "descarta",
        "borra", "elimina todo",
    ],
    "pregunta": [
        "?", "qué significa", "por qué", "cómo", "cuál es",
        "puedes explicar", "no entiendo", "aclárame",
    ],
}

### VECTOR DB CONFIG
PINECONE_INDEX  = "arc-vdb"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536

PINECONE_CLOUD   = "aws"
PINECONE_REGION  = "us-east-1"
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 100