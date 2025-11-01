# File: backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

# Import the compiled LangGraph app from your agent.py file
from backend.agent import agent_app

# Initialize FastAPI app
app = FastAPI()

# --- API Data Models ---

# This defines the JSON structure that Streamlit must send
class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] # e.g., [{"role": "user", "content": "hi"}]

# This defines the JSON structure FastAPI will send back
class ChatResponse(BaseModel):
    answer: str

# --- API Endpoints ---
@app.get("/")
async def root():
    """A simple endpoint to check if the backend is running."""
    return {"status": "Backend is running!"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    The main chat endpoint. It receives a query and chat history,
    runs the LangGraph agent, and returns the final answer.
    """
    print(f"Received query: {request.query}")
    print(f"Received history: {request.chat_history}")
    
    # This is the initial state for your LangGraph
    # "collected_data" is always empty at the start of a new user turn
    inputs = {
        "user_query": request.query,
        "chat_history": request.chat_history,
        "collected_data": []
    }
    
    try:
        # Run the agent. The recursion limit is important for loops.
        result = agent_app.invoke(inputs, {"recursion_limit": 25})
        
        # The final answer is in the 'final_response' key
        final_answer = result.get('final_response', "Sorry, I couldn't process that.")
        
        return {"answer": final_answer}
    
    except Exception as e:
        print(f"Error during agent invocation: {e}")
        return {"answer": f"An error occurred: {e}"}