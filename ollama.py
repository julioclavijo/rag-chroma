# Prueba de chromadb usando ollama desde docker
import requests
import chromadb
from rag import get_embeddings_from_lm_studio
import json
import time


chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_collection(name="knowledge_base")

collection.peek()

# Paso 2: Consultar ChromaDB
query = input('Cual es la consulta?')
query_embedding = get_embeddings_from_lm_studio([query])

# Verificar si hubo errores al obtener el embedding de la consulta
if query_embedding is None:
    print("Error al obtener el embedding de la consulta. Saliendo del programa.")
    exit(1)

results = collection.query(
    query_embeddings=query_embedding,
    n_results=1
)

documents = [
    {"id": "doc1", "text": "Pinebook es libro de recetas que nos va a hacer descubrir alimentos típicamente veganos, como el tofu.", "source": "pinebook.pdf"},
    {"id": "doc2", "text": "Python es un lenguaje de programación ampliamente utilizado por su simplicidad y versatilidad.", "source": "PythonGuide.pdf"}
]

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
    "http://localhost:11434/api/chat",
    json={
        "model": "tinyllama",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.2
    }
)

if response.status_code == 200:
    print("Procesando respuesta del modelo:\n")
    
    try:
        # Leer línea por línea en el flujo de respuesta
        for line in response.iter_lines():
            if line:  # Ignorar líneas vacías
                # Procesar cada línea como JSON
                parsed_line = json.loads(line.decode('utf-8'))
                
                # Extraer contenido si existe en 'message'
                if "message" in parsed_line and "content" in parsed_line["message"]:
                    content = parsed_line["message"]["content"]
                    
                    # Mostrar el fragmento recibido con un relay
                    print(content, end="", flush=True)
                    time.sleep(0.05)  # Relay de 100 ms

        print("\n\nRespuesta completa procesada.")
    except json.JSONDecodeError as e:
        print(f"\nError al procesar JSON: {e}")
else:
    print(f"Error al conectarse con LM Studio: {response.status_code}")
    print(response.text)