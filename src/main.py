from fastapi import FastAPI, Depends, UploadFile, HTTPException
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .models import ChatRequest, ChatResponse, UploadResponse, UploadResult
from .services.rag_service import RAGService
from .services.graph_service import KnowledgeAssistant

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Mini Knowledge Assistant",
    description="A RAG-based knowledge assistant using LangGraph",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rag_service = RAGService()
knowledge_assistant = KnowledgeAssistant(rag_service)

# Dependency injection
def get_rag_service() -> RAGService:
    return rag_service

def get_knowledge_assistant() -> KnowledgeAssistant:
    return knowledge_assistant

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Mini Knowledge Assistant is running!"}

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile],
    rag_service: RAGService = Depends(get_rag_service)
):
    """Upload and process multiple documents."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    successful_uploads = []
    failed_uploads = []
    
    for file in files:
        try:
            # Validate file type
            from .utils.file_loader import FileLoader
            if not FileLoader.validate_file_type(file.filename):
                raise ValueError(f"Unsupported file type: {file.filename}")
            
            content = await file.read()
            await rag_service.process_document(content, file.filename)
            
            result = UploadResult(
                filename=file.filename,
                status="success"
            )
            successful_uploads.append(result)
            results.append(result)
            
        except Exception as e:
            result = UploadResult(
                filename=file.filename,
                status="failed",
                error=str(e)
            )
            failed_uploads.append(result)
            results.append(result)
    
    # Prepare response
    total_processed = len(files)
    success_count = len(successful_uploads)
    failure_count = len(failed_uploads)
    
    if failure_count == 0:
        message = f"Successfully processed all {success_count} documents."
        return UploadResponse(
            message=message,
            successful_uploads=successful_uploads,
            total_processed=total_processed,
            success_count=success_count,
            failure_count=failure_count
        )
    elif success_count > 0:
        message = f"Processed {success_count} documents successfully, {failure_count} failed."
        return UploadResponse(
            message=message,
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads,
            total_processed=total_processed,
            success_count=success_count,
            failure_count=failure_count
        )
    else:
        # All uploads failed
        error_details = [f"{f.filename}: {f.error}" for f in failed_uploads]
        raise HTTPException(
            status_code=500, 
            detail=f"All {failure_count} document uploads failed: {'; '.join(error_details)}"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    knowledge_assistant: KnowledgeAssistant = Depends(get_knowledge_assistant)
):
    """Process a chat message and return a response."""
    try:
        detailed_response = await knowledge_assistant.process_question(request.message)
        
        # Convert execution log to the proper format
        execution_log = [
            {
                "step": log["step"],
                "status": log["status"],
                "details": log["details"],
                "timestamp": log["timestamp"]
            }
            for log in detailed_response.get("execution_log", [])
        ]
        
        # Convert sources to the proper format
        sources_used = [
            {
                "source": source["source"],
                "content_preview": source["content_preview"],
                "relevance_score": source.get("relevance_score")
            }
            for source in detailed_response.get("sources_used", [])
        ]
        
        return ChatResponse(
            response=detailed_response["answer"],
            execution_log=execution_log,
            sources_used=sources_used,
            workflow_path=detailed_response.get("workflow_path", []),
            total_execution_time=detailed_response.get("total_execution_time", 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))