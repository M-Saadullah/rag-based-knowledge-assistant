from fastapi import APIRouter, UploadFile, HTTPException
from ..services.rag_service import RAGService

router = APIRouter()

@router.post("/upload")
async def upload_documents(files: list[UploadFile], rag_service: RAGService):
    """Upload and process a document."""
    results = []
    for file in files:
        try:
            content = await file.read()
            await rag_service.process_document(content, file.filename)
            results.append({"filename": file.filename, "status": "success"})
        except Exception as e:
            results.append({"filename": file.filename, "status": "failed", "error": str(e)})
    
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "failed"]

    if failures:
        detail_message = f"Successfully processed {len(successes)} documents. Failed to process {len(failures)} documents: {failures}"
        if successes:
            return {"message": detail_message, "successful_uploads": successes, "failed_uploads": failures}
        else:
            raise HTTPException(status_code=500, detail=detail_message)
    else:
        return {"message": f"Successfully processed all {len(successes)} documents.", "successful_uploads": successes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
