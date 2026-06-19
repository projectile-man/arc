# AI Requirements Copilot (ARC)

AI Requirements Copilot (ARC) es una solución basada en Inteligencia Artificial Generativa diseñada para asistir a Product Owners y equipos de desarrollo en la generación, análisis y validación de requisitos funcionales, requisitos no funcionales e historias de usuario a partir de información no estructurada.

## Requisitos previos

Antes de ejecutar el proyecto, es necesario contar con:

* Python 3.13
* Cuenta en OpenAI con saldo disponible para consumo de API
* Cuenta en Pinecone
* Git (Opcional, puedes descargar el proyecto directamente en GitHub)
* uv (gestor de entornos y dependencias)

---

# 1. Configuración de OpenAI

## 1.1 Crear una cuenta

Accede a:

```text
https://platform.openai.com
```

Inicia sesión con tu cuenta de ChatGPT o crea una nueva cuenta.

## 1.2 Configurar facturación

1. Accede a **Organization Settings**
2. Selecciona **Billing**
3. Añade crédito mediante **Add to credit balance**

> Nota: Es necesario disponer de una tarjeta bancaria para realizar recargas.

## 1.3 Crear API Key

1. Accede a **API Keys**
2. Haz clic en **Create new secret key**
3. Introduce un nombre para la clave
4. Selecciona **Create secret key**
5. Guarda la clave generada, ya que será necesaria posteriormente

---

# 2. Configuración de Pinecone

## 2.1 Crear una cuenta

Accede a la plataforma de Pinecone y crea una cuenta o inicia sesión.

## 2.2 Crear un índice vectorial

Navega a:

```text
Database → Indexes → Create Index
```

Configura los siguientes parámetros:

| Parámetro       | Valor                  |
| --------------- | ---------------------- |
| Nombre          | arc-vs                 |
| Embedding Model | text-embedding-3-small |
| Dimensión       | 1536                   |

Mantén el resto de opciones por defecto y selecciona **Create Index**.

## 2.3 Crear API Key

1. Accede a **API Keys**
2. Selecciona **+ API Key**
3. Asigna un nombre
4. Haz clic en **Create Key**

Guarda la clave generada para los siguientes pasos.

---

# 3. Clonar el repositorio

Clona el repositorio en tu equipo:

```bash
git clone (https://github.com/projectile-man/arc.git)
```

Accede al directorio del proyecto:

```bash
cd arc
```

---

# 4. Instalación y configuración del entorno

## 4.1 Instalar uv

ARC utiliza **uv** como gestor de dependencias y entornos virtuales debido a su rendimiento y simplicidad.

Instalación:

```bash
pip install uv
```

## 4.2 Crear el entorno virtual

El proyecto fue desarrollado utilizando Python 3.13.

```bash
uv venv --python 3.13
```

## 4.3 Activar el entorno virtual

Windows:

```bash
.venv\Scripts\activate
```

## 4.4 Instalar dependencias

```bash
uv pip install -r requirements.txt
```

---

# 5. Configuración de variables de entorno

Renombra el archivo:

```text
.env.example
```

a:

```text
.env
```

Configura las siguientes variables:

```env
OPENAI_API_KEY=tu_api_key_openai
PINECONE_API_KEY=tu_api_key_pinecone
```

---

# 6. Configuración de Pinecone

Abre el archivo:

```text
config/parameters.py
```

Configura los parámetros del índice creado anteriormente:

```python
PINECONE_INDEX_NAME = "arc-vs"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
```

---

# 7. Indexación de la base de conocimiento

Antes de utilizar ARC es necesario indexar los documentos de buenas prácticas almacenados en:

```text
data/knowledge_base
```

Ejecuta:

```bash
python -m scripts.seed_kb
```

Este proceso:

* Genera embeddings de los documentos.
* Los almacena en Pinecone.
* Construye la base de conocimiento utilizada por los agentes del sistema.

---

# 8. Ejecutar la aplicación

Desde la raíz del proyecto:

```bash
streamlit run app.py
```

La aplicación estará disponible en:

```text
http://localhost:8501
```

---

# Arquitectura general

ARC está compuesto por los siguientes componentes principales:

* Ingestión y procesamiento documental
* Base de datos vectorial Pinecone
* Arquitectura RAG (Retrieval-Augmented Generation)
* Sistema multiagente basado en LangGraph
* Generación de requisitos funcionales y no funcionales
* Generación de historias de usuario
* Detección de ambigüedades e inconsistencias
* Interfaz conversacional desarrollada con Streamlit

---

# Consideraciones

ARC implementa un enfoque **Human-in-the-Loop (HITL)**, donde los requisitos generados por la inteligencia artificial deben ser revisados y validados por el Product Owner o responsable funcional antes de su utilización dentro del ciclo de desarrollo de software.

---

# Licencia

Proyecto desarrollado como Trabajo Fin de Máster (TFM).
