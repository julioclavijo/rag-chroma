# Ejemplod de rag usando lm studio y chromadb
import requests
import chromadb

# Configuración de ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma")
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
    {"id": "doc2", "text": "  es un lenguaje de programación ampliamente utilizado por su simplicidad y versatilidad.", "source": "PythonGuide.pdf"}
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

collection.peek()