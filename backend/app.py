"""
🚀 FastAPI Backend for Multi-Agent Chatbot
===========================================
Serves the LangGraph agents via REST API
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Multi-Agent Chatbot API",
    description="A LangGraph-based multi-agent system with supervisor pattern",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    agent_used: str
    response: str


# Lazy import to avoid loading LangGraph at module level
_agent_graph = None

def get_chat_function():
    """Lazy load the agent graph"""
    global _agent_graph
    if _agent_graph is None:
        from langgraph_agents import chat_with_agents
        _agent_graph = chat_with_agents
    return _agent_graph


@app.get("/")
def root():
    return {
        "message": "🤖 Multi-Agent Chatbot API",
        "agents": ["🔬 Scientist", "🎨 Creative", "💻 Coder"],
        "docs": "/docs"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Send a message to the multi-agent system.
    The supervisor will route it to the appropriate specialist.
    """
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        chat_with_agents = get_chat_function()
        result = chat_with_agents(request.message)
        return ChatResponse(
            agent_used=result["agent_used"],
            response=result["response"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

