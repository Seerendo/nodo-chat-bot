from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from bots.ollama_chatbot import chain
from bots.ollama_rag_chatbot import rag_chain
""" from business_info import info """
from func.load_business_info import info
from func.update_context import update_context

app = FastAPI(title="API del Chatbot RAG")

context_store = {}

class ChatRequest(BaseModel):
    user_id: str
    question: str

@app.post("/chat")
def ask_question(data: ChatRequest):
    try:
        user_context = context_store.get(data.user_id, "")       
        
        result = chain.invoke({
            "business_info": info,
            "context": user_context,
            "question": data.question
        })        
        
        new_context = update_context(user_context, data.question, result)
        context_store[data.user_id] = new_context

        """ allowed_keywords = ["misión", "visión", "servicios", "contactos", "empresa"]
        if any(keyword in data.question.lower() for keyword in allowed_keywords):
            new_context = update_context(user_context, data.question, result)
            context_store[data.user_id] = new_context """

        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)