import os
import io
import base64
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4
from fastapi import FastAPI, Request, Form, HTTPException, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import logging

from src.processors import PDFProcessor, ProcessingManager
from src.models import (
    PDFDocument,
    SplitConfiguration,
    PageBatch,
    ProcessingTask,
    BatchStatus,
    ProcessingStage,
)
from src.prompts import build_manual_prompt, get_default_manual_prompt

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Processing Application", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize PDF processor with environment variables
pdf_processor = PDFProcessor(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_region=os.getenv('AWS_REGION', 'us-east-1'),
    s3_bucket_name=os.getenv('S3_BUCKET_NAME')
)

# Initialize Processing Manager for pipeline
processing_manager = ProcessingManager(max_workers=4)


@dataclass
class ManualSession:
    session_id: str
    document: PDFDocument
    source_path: str
    temp_file: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    batches: List[Dict[str, Any]] = field(default_factory=list)
    display_name: Optional[str] = None
    processing_task: Optional[ProcessingTask] = None
    processing_prompt: Optional[str] = None


manual_sessions: Dict[str, ManualSession] = {}


def _get_manual_session(session_id: str) -> ManualSession:
    session = manual_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Manual session not found or has expired")
    return session


def _build_manual_batches_payload(session: ManualSession) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for batch_info in session.batches:
        batch_obj: PageBatch = batch_info["page_batch"]
        payload.append(
            {
                "batch_id": batch_obj.batch_id,
                "name": batch_info.get("name", f"Batch {batch_obj.batch_number}"),
                "page_numbers": batch_info.get("page_numbers", batch_obj.page_numbers),
                "page_count": len(batch_info.get("page_numbers", batch_obj.page_numbers)),
                "status": batch_obj.status.value,
                "processed_at": batch_obj.processed_at.isoformat() if batch_obj.processed_at else None,
                "error_message": batch_obj.error_message,
            }
        )
    return payload


def _calculate_unassigned_pages(session: ManualSession) -> int:
    assigned = sum(len(batch_info.get("page_numbers", [])) for batch_info in session.batches)
    return max(session.document.total_pages - assigned, 0)


def _build_manual_session_response(session: ManualSession) -> Dict[str, Any]:
    batches_payload = _build_manual_batches_payload(session)
    task_payload = session.processing_task.to_dict() if session.processing_task else None

    return {
        "success": True,
        "session_id": session.session_id,
        "task_id": session.processing_task.task_id if session.processing_task else None,
        "total_batches": len(batches_payload),
        "total_pages": session.document.total_pages,
        "unassigned_pages": _calculate_unassigned_pages(session),
        "default_prompt": get_default_manual_prompt(),
        "processing_prompt": session.processing_prompt,
        "task": task_payload,
        "batches": batches_payload,
    }


def _process_manual_batches_job(session_id: str, prompt_template: str) -> None:
    session = manual_sessions.get(session_id)
    if not session or not session.processing_task:
        logger.error("Manual processing job received invalid session: %s", session_id)
        return

    task = session.processing_task
    document = session.document

    try:
        if not session.batches:
            task.status = ProcessingStage.ERROR
            task.error_message = "No batches available for processing"
            task.completed_at = datetime.utcnow()
            return

        processing_manager.context_manager.reset_context(document.file_path)
        lmm_processor = processing_manager.lmm_processor

        task.status = ProcessingStage.LMM_PROCESSING
        task.started_at = datetime.utcnow()
        task.completed_at = None
        task.error_message = None
        task.progress = 0.0

        total_batches = len(session.batches)

        for index, batch_info in enumerate(session.batches, start=1):
            batch = batch_info["page_batch"]
            try:
                batch.status = BatchStatus.PROCESSING
                batch.error_message = None
                batch.processing_time = None
                batch.processed_at = None

                context_payload = processing_manager.context_manager.build_context_payload(
                    document.file_path
                )
                prompt_for_batch = build_manual_prompt(prompt_template, context_payload)

                lmm_result = lmm_processor.process_batch(
                    batch,
                    context=context_payload,
                    prompt=prompt_for_batch,
                    model=task.model or lmm_processor.DEFAULT_MODEL,
                    temperature=task.temperature,
                )

                batch.lmm_output = lmm_result.get("raw_output")
                batch.chunk_summary = lmm_result.get("chunks", [])
                batch.context_snapshot = lmm_result.get("context", {})
                batch.prompt_used = lmm_result.get("prompt_used")
                batch.processing_time = lmm_result.get("processing_time")
                batch.processed_at = datetime.utcnow()
                batch.status = BatchStatus.COMPLETED

                processing_manager.context_manager.update_context(
                    document.file_path,
                    last_chunks=lmm_result.get("last_chunks"),
                    heading_hierarchy=lmm_result.get("heading_hierarchy"),
                    continuation_metadata=lmm_result.get("continuation_metadata"),
                )

                task.progress = (index / total_batches) * 100.0
            except Exception as batch_error:
                batch.status = BatchStatus.ERROR
                batch.error_message = str(batch_error)
                logger.error(
                    "Manual processing error for session %s batch %s: %s",
                    session_id,
                    batch.batch_id,
                    batch_error,
                )
                task.status = ProcessingStage.ERROR
                task.error_message = str(batch_error)
                task.completed_at = datetime.utcnow()
                return

        task.status = ProcessingStage.COMPLETED
        task.completed_at = datetime.utcnow()
        task.context_state = processing_manager.context_manager.get_context(document.file_path)
    except Exception as unexpected_error:
        logger.exception(
            "Unexpected error while processing manual session %s: %s",
            session_id,
            unexpected_error,
        )
        task.status = ProcessingStage.ERROR
        task.error_message = str(unexpected_error)
        task.completed_at = datetime.utcnow()
    finally:
        if session.temp_file and os.path.exists(session.temp_file):
            try:
                os.remove(session.temp_file)
                session.temp_file = None
            except OSError:
                logger.warning("Unable to remove temporary file %s", session.temp_file)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with upload interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_dashboard(request: Request):
    """Pipeline dashboard for multi-document processing"""
    return templates.TemplateResponse("pipeline_v2.html", {"request": request})

@app.post("/process-pdf")
async def process_pdf(
    request: Request,
    file_path: Optional[str] = Form(None),
    s3_url: Optional[str] = Form(None),
    process_mode: str = Form("auto"),
    file: Optional[UploadFile] = File(None)
):
    """Process PDF from local path, S3 URL, or uploaded file"""
    try:
        document = None
        temp_uploaded_path = None
        is_human_loop = process_mode == "human"
        display_name: Optional[str] = None

        if file and file.filename:
            # Handle uploaded file
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="File must be a PDF")

            # Save uploaded file temporarily
            temp_path = f"/tmp/{file.filename}"
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            document = pdf_processor.load_from_local(temp_path)
            if is_human_loop:
                temp_uploaded_path = temp_path
                display_name = file.filename
            else:
                # Clean up temp file
                os.remove(temp_path)

        elif s3_url:
            # Handle S3 URL
            document = pdf_processor.load_from_url(s3_url)
            display_name = os.path.basename(s3_url)

        elif file_path:
            # Handle local file path
            document = pdf_processor.load_from_local(file_path)
            display_name = os.path.basename(file_path)

        else:
            raise HTTPException(status_code=400, detail="Please provide a file, file path, or S3 URL")

        # Get document info
        doc_info = pdf_processor.get_document_info(document)

        # Get all pages as images with page numbers
        page_images = pdf_processor.get_pages_as_images(document, add_page_numbers=True, max_size=(400, 600))

        # Convert all images to base64 for display
        pages_base64 = []
        for i, image in enumerate(page_images):
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            pages_base64.append({
                "page_number": i + 1,
                "image": img_base64
            })

        if is_human_loop:
            session_id = uuid4().hex
            source_path = file_path or s3_url or document.file_path
            manual_sessions[session_id] = ManualSession(
                session_id=session_id,
                document=document,
                source_path=source_path,
                temp_file=temp_uploaded_path,
                display_name=display_name or (os.path.basename(source_path) if source_path else None)
            )

            return templates.TemplateResponse("manual_processing.html", {
                "request": request,
                "document_info": doc_info,
                "page_images": pages_base64,
                "session_id": session_id
            })

        return templates.TemplateResponse("result.html", {
            "request": request,
            "document_info": doc_info,
            "page_images": pages_base64,
            "success": True
        })

    except FileNotFoundError as e:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": f"File not found: {str(e)}",
            "success": False
        })
    except ValueError as e:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": f"Invalid input: {str(e)}",
            "success": False
        })
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": f"Processing error: {str(e)}",
            "success": False
        })


@app.post("/api/manual/{session_id}/create-batches")
async def create_manual_batches(session_id: str, request: Request):
    """Create custom batches for a manual processing session."""

    session = _get_manual_session(session_id)
    payload = await request.json()
    batches_payload = payload.get("batches", []) if isinstance(payload, dict) else []

    if not isinstance(batches_payload, list) or not batches_payload:
        raise HTTPException(status_code=400, detail="No batches provided")

    document = session.document
    page_lookup = {page.page_number: page for page in document.pages}

    created_batches: List[Dict[str, Any]] = []
    covered_pages: set[int] = set()
    largest_batch_size = 0

    for index, batch_data in enumerate(batches_payload, start=1):
        if not isinstance(batch_data, dict):
            raise HTTPException(status_code=400, detail="Invalid batch format")

        raw_pages = batch_data.get("pages", [])
        if not isinstance(raw_pages, list) or not raw_pages:
            raise HTTPException(status_code=400, detail=f"Batch {index} must include at least one page")

        if not all(isinstance(page, int) for page in raw_pages):
            raise HTTPException(status_code=400, detail="Page numbers must be integers")

        unique_pages = sorted(set(raw_pages))
        for page_number in unique_pages:
            if page_number < 1 or page_number > document.total_pages:
                raise HTTPException(status_code=400, detail=f"Page {page_number} is out of range")
            if page_number not in page_lookup:
                raise HTTPException(status_code=400, detail=f"Page {page_number} is not available")

        batch_name = (batch_data.get("name") or f"Batch {index}").strip()
        batch_pages = [page_lookup[p] for p in unique_pages]

        source_basename = os.path.splitext(os.path.basename(session.source_path or "manual_document"))[0]
        page_batch = PageBatch(
            batch_id=f"{source_basename}_manual_{index}_{uuid4().hex[:8]}",
            batch_number=index,
            pages=batch_pages,
            start_page=unique_pages[0],
            end_page=unique_pages[-1],
            total_pages=len(batch_pages),
            document_id=session.source_path or document.file_path
        )

        covered_pages.update(unique_pages)
        largest_batch_size = max(largest_batch_size, len(batch_pages))
        created_batches.append({
            "name": batch_name,
            "page_numbers": unique_pages,
            "page_batch": page_batch
        })

    session.batches = created_batches

    manual_config = SplitConfiguration(
        batch_size=largest_batch_size or 1,
        overlap_pages=0
    )

    default_prompt = get_default_manual_prompt()

    manual_task = ProcessingTask(
        task_id=f"manual_{uuid4().hex[:12]}",
        document=document,
        config=manual_config,
        status=ProcessingStage.SPLITTING,
        progress=60.0,
        started_at=None,
        batches=[batch_info["page_batch"] for batch_info in created_batches],
        prompt=default_prompt,
        model=processing_manager.lmm_processor.DEFAULT_MODEL,
        temperature=processing_manager.lmm_processor.temperature,
    )

    manual_task.document.file_path = (
        session.display_name or session.source_path or manual_task.document.file_path
    )
    manual_task.document.source_type = "manual"
    metadata = dict(manual_task.document.metadata or {})
    metadata["processing_mode"] = "human_loop"
    if session.display_name:
        metadata.setdefault("display_name", session.display_name)
    if session.source_path:
        metadata.setdefault("source_path", session.source_path)
    manual_task.document.metadata = metadata

    session.processing_task = manual_task
    session.processing_prompt = default_prompt

    return JSONResponse(content=_build_manual_session_response(session))


@app.get("/api/manual/{session_id}/status")
async def get_manual_session_status(session_id: str):
    """Return the latest status for a manual processing session."""

    session = _get_manual_session(session_id)
    return JSONResponse(content=_build_manual_session_response(session))


@app.post("/api/manual/{session_id}/process-batches")
async def process_manual_batches(session_id: str, request: Request):
    """Trigger LLM processing for the batches defined in a manual session."""

    session = _get_manual_session(session_id)

    if not session.batches:
        raise HTTPException(status_code=400, detail="No batches defined for this session")

    task = session.processing_task
    if not task:
        raise HTTPException(status_code=400, detail="Manual session is not ready for processing")

    if task.status == ProcessingStage.LMM_PROCESSING:
        raise HTTPException(status_code=400, detail="Processing already in progress")

    payload = await request.json()
    instructions = ""
    if isinstance(payload, dict):
        instructions = (payload.get("instructions") or "").strip()

    prompt_template = instructions or get_default_manual_prompt()
    session.processing_prompt = prompt_template
    task.prompt = prompt_template
    task.error_message = None
    task.started_at = None
    task.completed_at = None
    task.progress = 0.0
    task.status = ProcessingStage.LMM_PROCESSING

    for batch_info in session.batches:
        batch_obj = batch_info["page_batch"]
        batch_obj.status = BatchStatus.PENDING
        batch_obj.error_message = None
        batch_obj.processing_time = None
        batch_obj.processed_at = None
        batch_obj.lmm_output = None
        batch_obj.chunk_summary = []
        batch_obj.context_snapshot = {}
        batch_obj.prompt_used = None

    processing_manager.executor.submit(_process_manual_batches_job, session_id, prompt_template)

    return JSONResponse(
        content={
            "success": True,
            "session_id": session_id,
            "task_id": task.task_id,
        }
    )


@app.get("/api/document-info/{source_type}")
async def get_document_info_api(
    source_type: str,
    file_path: Optional[str] = None,
    s3_url: Optional[str] = None
):
    """API endpoint to get document information"""
    try:
        if source_type == "local" and file_path:
            document = pdf_processor.load_from_local(file_path)
        elif source_type == "s3" and s3_url:
            document = pdf_processor.load_from_url(s3_url)
        else:
            raise HTTPException(status_code=400, detail="Invalid parameters")

        doc_info = pdf_processor.get_document_info(document)
        return JSONResponse(content=doc_info)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/page-preview/{source_type}/{page_number}")
async def get_page_preview_api(
    source_type: str,
    page_number: int,
    file_path: Optional[str] = None,
    s3_url: Optional[str] = None
):
    """API endpoint to get page preview"""
    try:
        if source_type == "local" and file_path:
            document = pdf_processor.load_from_local(file_path)
        elif source_type == "s3" and s3_url:
            document = pdf_processor.load_from_url(s3_url)
        else:
            raise HTTPException(status_code=400, detail="Invalid parameters")

        preview_image = pdf_processor.get_page_preview(document, page_number)

        # Convert to base64
        img_buffer = io.BytesIO()
        preview_image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        return JSONResponse(content={"image": img_base64, "page_number": page_number})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pipeline API Endpoints
@app.post("/api/pipeline/submit")
async def submit_pipeline_task(
    file: UploadFile = File(...),
    batch_size: int = Form(4),
    overlap_pages: int = Form(0),
    prompt: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None)
):
    """Submit a file for pipeline processing"""
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Create configuration
        config = SplitConfiguration(
            batch_size=batch_size,
            overlap_pages=overlap_pages
        )

        # Submit task to processing manager
        task_id = processing_manager.submit_task(
            file_path=temp_path,
            source_type="upload",
            config=config,
            prompt=prompt,
            model=model,
            temperature=temperature
        )

        return JSONResponse(content={
            "task_id": task_id,
            "filename": file.filename,
            "config": config.to_dict(),
            "prompt": prompt,
            "model": model,
            "temperature": temperature
        })

    except Exception as e:
        logger.error(f"Error submitting pipeline task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pipeline/tasks")
async def get_all_pipeline_tasks():
    """Get all pipeline tasks"""
    try:
        tasks = processing_manager.get_all_tasks()
        return JSONResponse(content=tasks)
    except Exception as e:
        logger.error(f"Error getting pipeline tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pipeline/tasks/{task_id}")
async def get_pipeline_task(task_id: str):
    """Get specific pipeline task"""
    try:
        task = processing_manager.get_task_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return JSONResponse(content=task)
    except Exception as e:
        logger.error(f"Error getting pipeline task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/tasks/{task_id}/cancel")
async def cancel_pipeline_task(task_id: str):
    """Cancel a pipeline task"""
    try:
        success = processing_manager.cancel_task(task_id)
        if not success:
            raise HTTPException(status_code=400, detail="Task cannot be cancelled")
        return JSONResponse(content={"success": True, "message": "Task cancelled"})
    except Exception as e:
        logger.error(f"Error cancelling pipeline task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/clear-completed")
async def clear_completed_tasks():
    """Clear all completed tasks"""
    try:
        processing_manager.clear_completed_tasks()
        return JSONResponse(content={"success": True, "message": "Completed tasks cleared"})
    except Exception as e:
        logger.error(f"Error clearing completed tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Get pipeline queue status"""
    try:
        status = processing_manager.get_queue_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pipeline/tasks/{task_id}/images")
async def get_task_processed_images(task_id: str):
    """Get processed PDF images for a task with page numbers"""
    try:
        # Get the actual task object from the processing manager
        with processing_manager._lock:
            task_obj = processing_manager.active_tasks.get(task_id) or processing_manager.completed_tasks.get(task_id)
        
        if not task_obj:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if not task_obj.document or task_obj.document.total_pages == 0:
            raise HTTPException(status_code=400, detail="Document not processed yet")
        
        # Get the document from the task object
        document = task_obj.document
        
        # Get all pages as images with page numbers
        page_images = pdf_processor.get_pages_as_images(document, add_page_numbers=True, max_size=(800, 1200))
        
        # Convert all images to base64 for display
        pages_base64 = []
        for i, image in enumerate(page_images):
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            pages_base64.append({
                "page_number": i + 1,
                "image": img_base64
            })
        
        return JSONResponse(content={
            "task_id": task_id,
            "total_pages": len(pages_base64),
            "pages": pages_base64
        })
        
    except Exception as e:
        logger.error(f"Error getting processed images for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pipeline/tasks/{task_id}/batches/{batch_id}/images")
async def get_batch_images(task_id: str, batch_id: str):
    """Get images for a specific batch"""
    try:
        # Get the actual task object from the processing manager
        with processing_manager._lock:
            task_obj = processing_manager.active_tasks.get(task_id) or processing_manager.completed_tasks.get(task_id)
        
        if not task_obj:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Find the specific batch
        batch_obj = None
        for b in task_obj.batches:
            if b.batch_id == batch_id:
                batch_obj = b
                break
        
        if not batch_obj:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        if not task_obj.document or task_obj.document.total_pages == 0:
            raise HTTPException(status_code=400, detail="Document not processed yet")
        
        # Get the document from the task object
        document = task_obj.document
        
        # Get images for the specific page range in this batch
        batch_pages = []
        start_page = batch_obj.start_page
        end_page = batch_obj.end_page
        
        for page_num in range(start_page, end_page + 1):
            if page_num <= len(document.pages):
                page = document.pages[page_num - 1]
                if page.image:
                    # Create a copy and resize for preview
                    image = page.image.copy()
                    image.thumbnail((600, 800), Image.Resampling.LANCZOS)
                    
                    # Add page number overlay
                    image = pdf_processor.add_page_number_to_image(image, page_num)
                    
                    # Convert to base64
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    
                    batch_pages.append({
                        "page_number": page_num,
                        "image": img_base64
                    })
        
        return JSONResponse(content={
            "task_id": task_id,
            "batch_id": batch_id,
            "batch_number": batch_obj.batch_number,
            "start_page": start_page,
            "end_page": end_page,
            "pages": batch_pages
        })
        
    except Exception as e:
        logger.error(f"Error getting batch images for task {task_id}, batch {batch_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)