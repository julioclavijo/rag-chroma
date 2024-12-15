# RAG usando solo lmstudio y chromadb tanto embeddings y chat

import requests
import chromadb

# Configuración de ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("knowledge_base")

# Constante para la URL del endpoint de LM Studio. Facilita el cambio si es necesario.
LM_STUDIO_ENDPOINT = "http://localhost:1234/v1/embeddings"

# Función para obtener embeddings desde LM Studio, con manejo de errores
def get_embeddings_from_lm_studio(texts, model="text-embedding-nomic-embed-text-v1.5"):
    embeddings = []
    for text in texts:
        text = text.replace("\n", " ")  # Eliminar saltos de línea
        try:
            response = requests.post(
                LM_STUDIO_ENDPOINT,
                json={"input": [text], "model": model},
                timeout=10  # Agregar un timeout para evitar bloqueos indefinidos
            )
            response.raise_for_status()  # Lanzar una excepción si hay un error HTTP
            embeddings.append(response.json()["data"][0]["embedding"])
        except requests.exceptions.RequestException as e:
            print(f"Error al obtener embeddings para el texto '{text}': {e}")
            return None  # Devolver None en caso de error
    return embeddings

# Paso 1: Configurar los documentos
documents = [
    {"id": "doc1", "text": "Pinebook es libro de recetas que nos va a hacer descubrir alimentos típicamente veganos, como el tofu.", "source": "pinebook.pdf"},
    {"id": "doc2", "text": "Python es un lenguaje de programación ampliamente utilizado por su simplicidad y versatilidad.", "source": "PythonGuide.pdf"}
]

# Generar embeddings para los documentos y almacenarlos en ChromaDB
texts = [doc["text"] for doc in documents]
doc_embeddings = get_embeddings_from_lm_studio(texts)

# Verificar si hubo errores al obtener los embeddings
if doc_embeddings is None:
    print("Error al obtener embeddings. Saliendo del programa.")
    exit(1)  # Salir con código de error

# Agregar documentos a ChromaDB en una sola operación (más eficiente)
collection.add(
    embeddings=doc_embeddings,
    metadatas=[{"source": doc["source"]} for doc in documents],
    ids=[doc["id"] for doc in documents]
)


# Paso 2: Consultar ChromaDB
query = "¿Qué es Pinebook?"
query_embedding = get_embeddings_from_lm_studio([query])

# Verificar si hubo errores al obtener el embedding de la consulta
if query_embedding is None:
    print("Error al obtener el embedding de la consulta. Saliendo del programa.")
    exit(1)

results = collection.query(
    query_embeddings=query_embedding,
    n_results=1
)

print(results)


# Obtener los documentos originales basándonos en los IDs devueltos por ChromaDB
retrieved_docs = []
for doc_id in results['ids'][0]: # iteramos por si n_results > 1
    retrieved_doc = next((doc for doc in documents if doc['id'] == doc_id), None)
    if retrieved_doc:
        retrieved_docs.append(retrieved_doc)
    else:
        print(f"Advertencia: Documento con ID '{doc_id}' no encontrado en la lista original.")



retrieved_text = "\n".join([  # Unimos los textos de los documentos recuperados
    f"Fuente: {doc['source']}\nContenido: {doc['text']}"
    for doc in retrieved_docs
])



messages = [
    {"role": "system", "content": "Eres un asistente útil y formal.  Debes responder a la pregunta del usuario utilizando el contexto proporcionado.  Tu respuesta debe ser positiva, sin tuteos, y en un tono profesional."},
    {"role": "user", "content": f"Contexto:\n{retrieved_text}\n\nPregunta: {query}"}
]

response = requests.post(
    LM_STUDIO_ENDPOINT.replace("embeddings", "chat/completions"),
    json={
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.2
    }
)

if response.status_code == 200:
    print("Respuesta del modelo:")
    print(response.json()["choices"][0]["message"]["content"])
else:
    print(f"Error al conectarse con LM Studio: {response.status_code}")
    print(response.text)