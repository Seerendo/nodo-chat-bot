from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import (
    TextLoader, 
    PyMuPDFLoader, 
    UnstructuredWordDocumentLoader, 
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.cache import InMemoryCache
import os
import langchain

langchain.llm_cache = InMemoryCache()

# ======= CONFIGURACIÓN GENERAL =======
DOCS_DIR = "corpus" # DIRECTORIO LOCAL DONDE ESTÁN TUS DOCUMENTOS DE TEXTO
EMBED_MODEL = "all-MiniLM-L6-v2" # MODELO DE EMBEDDINGS
EMBED_MODEL_PATH = "models/all-MiniLM-L6-v2" # MODELO DE EMBEDDINGS LOCAL
LLM_OLLAMA_MODEL = "llama3.2" # MODELO EN OLLAMA
CHUNK_SIZE = 500 # TAMAÑO DE LOS "CHUNKS" (fragmentos de texto) EN CARACTERES
CHUNK_OVERLAP = 50 # CANTIDAD DE CARACTERES DE SOLAPAMIENTO ENTRE CHUNKS

# ======= CADENA PROMPT (PREGUNTAS Y RESPUESTAS) =======
prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": """
        Eres NodoBot, un asistente virtual de NODO Cía. Ltda. Responde siempre de manera amable y concisa.
        
        Directrices importantes:
        1. Responde ÚNICAMENTE a preguntas relacionadas con la empresa y sus servicios
        2. La página web oficial de la empresa es: https://nodo.com.ec/home
        3. La página de la academia es: https://nodo.com.ec/academia
        4. Cuando te pregunten por información general de la empresa o cursos, usa SIEMPRE los enlaces generales
        5. Usa enlaces específicos de inscripción SOLO cuando el usuario pregunte específicamente por un curso en particular
        6. Enlaces de inscripción específicos solo se proporcionan cuando se solicita información detallada sobre un curso concreto
        """
    },
    {
        "role": "system",
        "content": "Aquí el contexto de la conversación: {context}"
    },
    {
        "role": "user",
        "content": "El usuario ha formulado la siguiente pregunta: {question}"
    },
    {
        "role": "assistant",
        "content": "Responda en español (responda sólo a las preguntas relacionadas con el contexto de la empresa y en español):"
    }
])

# ======= CARGA Y PROCESAMIENTO DE DOCUMENTOS =======
def load_documents(DOCS_DIR):
    documents = []
    for filename in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, filename)
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            elif filename.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith(".xlsx"):
                loader = UnstructuredExcelLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error al cargar {filename}: {e}")
    return documents

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    documents = load_documents(DOCS_DIR)

    # ======= CREACIÓN DE VECTORSTORE (FAISS) Y EMBEDDINGS =======
    if os.path.exists("vectorstore"):
        vectorstore = FAISS.load_local("vectorstore", HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH), allow_dangerous_deserialization=True)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        split_docs = text_splitter.split_documents(documents)

        embedding_model = SentenceTransformer(EMBED_MODEL)
        embedding_model.save(EMBED_MODEL_PATH)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local("vectorstore")

    # ======= CARGA DEL MODELO OLLAMA (LLM) =======
    llm = OllamaLLM(model=LLM_OLLAMA_MODEL)

    # ======= MEMORIA DE CONVERSACIÓN =======
    memory = ConversationBufferWindowMemory(
        llm=llm,
        k=10,
        return_messages=True,
        memory_key="chat_history",
        input_key="question"
    )

    # ======= CADENA RAG (RETRIEVAL QA) =======
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": min(len(documents), 5), "fetch_k": 8}
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,
        chain_type="stuff"
    )

    return chain

rag_chain = get_rag_chain()

def chat():
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        result = rag_chain.invoke({"question": user_input})
        print("Bot: " + result["answer"])

if __name__ == "__main__":
    chat()