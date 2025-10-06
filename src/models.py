from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    message: str

class ExecutionStep(BaseModel):
    step: str
    status: str
    details: Dict[str, Any]
    timestamp: float

class SourceInfo(BaseModel):
    source: str
    content_preview: str
    relevance_score: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    execution_log: List[ExecutionStep]
    sources_used: List[SourceInfo]
    workflow_path: List[str]
    total_execution_time: float

class UploadResult(BaseModel):
    filename: str
    status: str
    error: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    successful_uploads: List[UploadResult]
    failed_uploads: Optional[List[UploadResult]] = None
    total_processed: int
    success_count: int
    failure_count: int