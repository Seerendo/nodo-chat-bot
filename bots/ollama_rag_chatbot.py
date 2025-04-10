from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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
import os

# ======= CONFIGURACIÓN GENERAL =======
DOCS_DIR = "corpus" # DIRECTORIO LOCAL DONDE ESTÁN TUS DOCUMENTOS DE TEXTO
EMBED_MODEL = "all-MiniLM-L6-v2" # MODELO DE EMBEDDINGS
EMBED_MODEL_PATH = "models/all-MiniLM-L6-v2" # MODELO DE EMBEDDINGS LOCAL
LLM_OLLAMA_MODEL = "llama3.2" # MODELO EN OLLAMA
CHUNK_SIZE = 200 # TAMAÑO DE LOS "CHUNKS" (fragmentos de texto) EN CARACTERES
CHUNK_OVERLAP = 20 # CANTIDAD DE CARACTERES DE SOLAPAMIENTO ENTRE CHUNKS

# ======= CADENA PROMPT (PREGUNTAS Y RESPUESTAS) =======
prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": """
        You are an employee of the company NODO Cía. Ltda, your name is NODOBot. Always respond in a friendly and polite manner, and make sure to return greetings and say goodbye if asked.

        **Important:** You should only answer questions related to the company and its context. If a question is asked about something outside the company's context, kindly inform the user that you don't have information about that topic and encourage them to ask questions about the company instead.
        If you mention links, make sure they correspond to the specific course or context.
        """
    },
    {
        "role": "system",
        "content": "Here is the context of the conversation: {context}"
    },
    {
        "role": "user",
        "content": "The user has asked the following question: {question}"
    },
    {
        "role": "assistant",
        "content": "Answer (only answer questions that relate to the company context and Spanish):"
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

    # ======= CADENA RAG (RETRIEVAL QA) =======
    retriever = vectorstore.as_retriever(search_kwargs={"k": min(len(documents), 3)})    

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

rag_chain = get_rag_chain()

def chat():
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        result = rag_chain.invoke(user_input)
        print("Bot: " + result)

if __name__ == "__main__":
    chat()