from fastapi import APIRouter, HTTPException
from ..models import ChatRequest, ChatResponse
from ..services.graph_service import KnowledgeAssistant

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, knowledge_assistant: KnowledgeAssistant):
    """Process a chat message and return a response."""
    try:
        response = await knowledge_assistant.process_question(request.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
