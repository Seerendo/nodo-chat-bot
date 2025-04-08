from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, PDFMinerLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# ======= CONFIGURACIÓN GENERAL =======
DOCS_DIR = "corpus" # DIRECTORIO LOCAL DONDE ESTÁN TUS DOCUMENTOS DE TEXTO
EMBED_MODEL = "all-MiniLM-L6-v2" # MODELO DE EMBEDDINGS
EMBED_MODEL_PATH = "models/all-MiniLM-L6-v2" # MODELO DE EMBEDDINGS LOCAL
LLM_OLLAMA_MODEL = "gemma2:2b" # MODELO EN OLLAMA
CHUNK_SIZE = 300 # TAMAÑO DE LOS "CHUNKS" (fragmentos de texto) EN CARACTERES
CHUNK_OVERLAP = 30 # CANTIDAD DE CARACTERES DE SOLAPAMIENTO ENTRE CHUNKS

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
                loader = PDFMinerLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith(".xlsx"):
                loader = UnstructuredExcelLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error al cargar {filename}: {e}")
    return documents

def initialize_rag_chain():
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

    # ======= CADENA PROMPT (PREGUNTAS Y RESPUESTAS) =======
    prompt = ChatPromptTemplate.from_template("""
    Answer the question below in Spanish.       
    You are an employee of the company NODO Cía. Ltda, your name is NODOBot. Always respond in a friendly and polite manner, and make sure to return greetings and say goodbye if asked.

    **Important:** You should only answer questions related to the company and its context. If a question is asked about something outside the company's context, kindly inform the user that you don't have information about that topic and encourage them to ask questions about the company instead.

    Here is the context of the conversation:
    {context}

    Here is the conversation history:                            
    {chat_history}

    The user has asked the following question:
    {question}

    Answer (only answer questions that relate to the company context):

    """)

    # ======= CONFIGURACIÓN DE MEMORIA =======
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    # ======= CARGA DEL MODELO OLLAMA (LLM) =======
    llm = OllamaLLM(model=LLM_OLLAMA_MODEL)

    # ======= CADENA RAG (RETRIEVAL QA) =======
    retriever = vectorstore.as_retriever(search_kwargs={"k": min(len(documents), 3)})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": prompt},
        get_chat_history=lambda h: h,
        verbose=False
    )

    return chain

rag_chain = initialize_rag_chain()