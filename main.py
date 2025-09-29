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

from src.processors import PDFProcessor, ProcessingManager, DatabaseWriter
from src.models import (
    PDFDocument,
    SplitConfiguration,
    PageBatch,
    ProcessingTask,
    BatchStatus,
    ProcessingStage,
)
from src.models.chunk_schema import BatchProcessingResult, ChunkOutput
from src.prompts import build_manual_prompt, get_default_manual_prompt
from src.managers import DocumentLifecycleManager
from src.database.document_lifecycle_models import ProcessingMode, ProcessingStatus
from src.recovery import RestartManager

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

# Initialize Database Writer for chunk storage
try:
    db_writer = DatabaseWriter()
    logger.info("Database writer initialized successfully")
    logger.info(f"Database writer connection status: {db_writer.get_connection_status()}")
except Exception as e:
    logger.error(f"Database writer initialization failed: {e}")
    import traceback
    logger.error(f"Database writer error details: {traceback.format_exc()}")
    db_writer = None

# Initialize Document Lifecycle Manager for comprehensive tracking
try:
    lifecycle_manager = DocumentLifecycleManager()
    logger.info("Document lifecycle manager initialized successfully")
    logger.info(f"Lifecycle manager connection status: {lifecycle_manager.get_connection_status()}")
except Exception as e:
    logger.error(f"Document lifecycle manager initialization failed: {e}")
    import traceback
    logger.error(f"Document lifecycle manager error details: {traceback.format_exc()}")
    lifecycle_manager = None

# Initialize Restart Manager for recovery
try:
    restart_manager = RestartManager(lifecycle_manager)
    logger.info("Restart manager initialized successfully")

    # Run startup recovery check
    recovery_result = restart_manager.run_recovery_check(max_age_hours=24, auto_resume=True)
    if recovery_result.get("total_incomplete_documents", 0) > 0:
        logger.info(f"Startup recovery: {recovery_result.get('successful_resumes', 0)}/{recovery_result.get('total_processed', 0)} documents resumed")
    else:
        logger.info("Startup recovery: No incomplete documents found")
except Exception as e:
    logger.error(f"Restart manager initialization failed: {e}")
    import traceback
    logger.error(f"Restart manager error details: {traceback.format_exc()}")
    restart_manager = None


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
    document_id: Optional[str] = None  # Document ID from lifecycle manager
    s3_pdf_url: Optional[str] = None  # S3 URL of uploaded PDF


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
                "warnings": batch_obj.warnings,
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
    lifecycle_tracking_enabled = lifecycle_manager is not None and session.document_id
    lifecycle_stage_started = False
    lifecycle_stage_name = "manual_llm_processing"
    total_chunks_generated = 0
    total_batches = 0

    try:
        if not session.batches:
            task.status = ProcessingStage.ERROR
            task.error_message = "No batches available for processing"
            task.completed_at = datetime.utcnow()
            return

        total_batches = len(session.batches)

        if lifecycle_tracking_enabled:
            try:
                lifecycle_manager.update_document_status(
                    session.document_id,
                    ProcessingStatus.PROCESSING,
                    summary_data={"total_batches": total_batches}
                )
                lifecycle_manager.track_stage_start(
                    document_id=session.document_id,
                    stage_name=lifecycle_stage_name,
                    stage_data={
                        "total_batches": total_batches,
                        "prompt_template_preview": prompt_template[:500],
                        "processing_mode": ProcessingMode.HUMAN_LOOP.value
                    }
                )
                lifecycle_stage_started = True
            except Exception as lifecycle_error:
                logger.error(
                    "Manual processing: Failed to update lifecycle status to processing for document %s: %s",
                    session.document_id,
                    lifecycle_error,
                )

        processing_manager.context_manager.reset_context(document.file_path)
        lmm_processor = processing_manager.lmm_processor

        task.status = ProcessingStage.LMM_PROCESSING
        task.started_at = datetime.utcnow()
        task.completed_at = None
        task.error_message = None
        task.progress = 0.0

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

                # Process batch with enhanced structured output
                lmm_result: BatchProcessingResult = lmm_processor.process_batch(
                    batch,
                    context=context_payload,
                    prompt=prompt_for_batch,
                    model=task.model or lmm_processor.DEFAULT_MODEL,
                    temperature=task.temperature,
                )

                chunk_count = len(lmm_result.chunks)
                total_chunks_generated += chunk_count

                # Store structured output data
                batch.lmm_output = lmm_result.raw_output
                batch.chunk_summary = [chunk.to_dict() for chunk in lmm_result.chunks]
                batch.context_snapshot = {
                    "continuation_context": lmm_result.continuation_context,
                    "processing_metadata": lmm_result.processing_metadata,
                    "structured_chunks": [chunk.to_dict() for chunk in lmm_result.chunks],
                    "last_chunk": lmm_result.last_chunk.to_dict() if lmm_result.last_chunk else None,
                }
                batch.prompt_used = lmm_result.processing_metadata.get("prompt_used")
                batch.processing_time = lmm_result.processing_metadata.get("processing_time")
                batch.warnings = lmm_result.processing_metadata.get("warnings", [])
                batch.processed_at = datetime.utcnow()
                batch.status = BatchStatus.COMPLETED

                # Update context using new structured method
                processing_manager.context_manager.update_context_from_batch_result(
                    document.file_path,
                    lmm_result,
                    index,
                )

                # Save chunks to enhanced lifecycle database if available
                logger.info(f"Manual processing: lifecycle_manager available: {lifecycle_manager is not None}")
                if lifecycle_manager and session.document_id:
                    try:
                        logger.info(f"Manual processing: Attempting to save enhanced chunks for batch {batch.batch_id}")

                        # Create batch record if it doesn't exist
                        page_images_info = []
                        for page_num in range(batch.start_page, batch.end_page + 1):
                            if page_num <= len(session.document.pages):
                                page = session.document.pages[page_num - 1]
                                if hasattr(page, 's3_url') and page.s3_url:
                                    from src.database.document_lifecycle_models import PageImageInfo
                                    page_image_info = PageImageInfo(
                                        page_number=page_num,
                                        s3_original_url=page.s3_url,
                                        s3_thumbnail_url=getattr(page, 's3_thumbnail_url', None),
                                        image_dimensions={"width": page.image.width, "height": page.image.height} if page.image else {},
                                        file_size_bytes=0
                                    )
                                    page_images_info.append(page_image_info)

                        # Create batch record
                        batch_id = lifecycle_manager.create_batch_record(
                            document_id=session.document_id,
                            batch_number=batch.batch_number,
                            page_range={
                                "start_page": batch.start_page,
                                "end_page": batch.end_page,
                                "total_pages": batch.total_pages
                            },
                            page_images=page_images_info
                        )

                        # Update batch processing result
                        processing_metadata = {
                            "model": task.model,
                            "temperature": task.temperature,
                            "prompt_used": prompt_for_batch,
                            "processing_time": batch.processing_time,
                            "tokens_used": lmm_result.processing_metadata.get("tokens_used", 0)
                        }
                        lifecycle_manager.update_batch_processing_result(
                            batch_id=batch_id,
                            batch_result=lmm_result,
                            processing_metadata=processing_metadata
                        )

                        # Save enhanced chunks
                        chunk_ids = lifecycle_manager.save_chunks(
                            document_id=session.document_id,
                            batch_id=batch_id,
                            batch_result=lmm_result,
                            processing_task=task
                        )
                        logger.info(f"Manual processing: Successfully saved {len(chunk_ids)} enhanced chunks for batch {batch.batch_id}")

                    except Exception as lifecycle_error:
                        logger.error(f"Manual processing: Lifecycle manager error for batch {batch.batch_id}: {lifecycle_error}")
                        import traceback
                        logger.error(f"Manual processing: Lifecycle manager traceback: {traceback.format_exc()}")

                # Save chunks to legacy database if db_writer is available (for backwards compatibility)
                logger.info(f"Manual processing: db_writer available: {db_writer is not None}")
                if db_writer:
                    try:
                        logger.info(f"Manual processing: Attempting to save chunks for batch {batch.batch_id}")
                        write_result = db_writer.write_batch(task, batch, lmm_result)
                        if write_result["success"]:
                            logger.info(f"Manual processing: Successfully saved {write_result['chunks_saved']} chunks for batch {batch.batch_id}")
                        else:
                            logger.error(f"Manual processing: Failed to save chunks for batch {batch.batch_id}: {write_result.get('error')}")
                    except Exception as db_error:
                        logger.error(f"Manual processing: Database write error for batch {batch.batch_id}: {db_error}")
                        import traceback
                        logger.error(f"Manual processing: Database write traceback: {traceback.format_exc()}")
                else:
                    logger.warning(f"Manual processing: db_writer is None, cannot save chunks for batch {batch.batch_id}")

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

                if lifecycle_tracking_enabled:
                    error_stage_data = {
                        "failed_batch_number": batch.batch_number,
                        "batch_id": batch.batch_id,
                        "page_range": {
                            "start_page": batch.start_page,
                            "end_page": batch.end_page
                        },
                        "processed_batches": max(index - 1, 0),
                        "total_batches": total_batches
                    }
                    try:
                        lifecycle_manager.track_stage_error(
                            document_id=session.document_id,
                            stage_name=lifecycle_stage_name,
                            error_message=str(batch_error),
                            stage_data=error_stage_data
                        )
                    except Exception as lifecycle_error:
                        logger.error(
                            "Manual processing: Failed to record lifecycle stage error for document %s: %s",
                            session.document_id,
                            lifecycle_error,
                        )
                    try:
                        lifecycle_manager.update_document_status(
                            session.document_id,
                            ProcessingStatus.ERROR,
                            summary_data={
                                "total_batches": total_batches,
                                "total_chunks": total_chunks_generated
                            }
                        )
                    except Exception as lifecycle_error:
                        logger.error(
                            "Manual processing: Failed to set document %s status to error: %s",
                            session.document_id,
                            lifecycle_error,
                        )

                # Register the failed task with processing_manager so it appears in the pipeline dashboard
                with processing_manager._lock:
                    processing_manager.completed_tasks[task.task_id] = task
                return

        task.status = ProcessingStage.COMPLETED
        task.completed_at = datetime.utcnow()
        task.context_state = processing_manager.context_manager.get_context(document.file_path)

        if lifecycle_tracking_enabled:
            processing_duration = None
            if task.started_at and task.completed_at:
                processing_duration = (task.completed_at - task.started_at).total_seconds()

            warnings_detected = 0
            try:
                warnings_detected = sum(
                    len(getattr(batch_info.get("page_batch"), "warnings", []) or [])
                    for batch_info in session.batches
                )
            except Exception:
                warnings_detected = 0

            stage_completion_data = {
                "total_batches": total_batches,
                "processed_batches": total_batches,
                "total_chunks_generated": total_chunks_generated,
                "processing_duration_seconds": processing_duration,
                "warnings_detected": warnings_detected,
                "prompt_template_preview": prompt_template[:500]
            }

            try:
                if lifecycle_stage_started:
                    lifecycle_manager.track_stage_completion(
                        document_id=session.document_id,
                        stage_name=lifecycle_stage_name,
                        stage_data=stage_completion_data
                    )
                lifecycle_manager.update_document_status(
                    session.document_id,
                    ProcessingStatus.COMPLETED,
                    summary_data={
                        "total_batches": total_batches,
                        "total_chunks": total_chunks_generated
                    }
                )
            except Exception as lifecycle_error:
                logger.error(
                    "Manual processing: Failed to record lifecycle completion for document %s: %s",
                    session.document_id,
                    lifecycle_error,
                )

        # Register the completed task with processing_manager so it appears in the pipeline dashboard
        with processing_manager._lock:
            processing_manager.completed_tasks[task.task_id] = task
    except Exception as unexpected_error:
        logger.exception(
            "Unexpected error while processing manual session %s: %s",
            session_id,
            unexpected_error,
        )
        task.status = ProcessingStage.ERROR
        task.error_message = str(unexpected_error)
        task.completed_at = datetime.utcnow()

        if lifecycle_tracking_enabled:
            processed_batches = 0
            try:
                processed_batches = sum(
                    1 for batch_info in session.batches
                    if batch_info.get("page_batch") and batch_info["page_batch"].status == BatchStatus.COMPLETED
                )
            except Exception:
                processed_batches = 0

            error_stage_data = {
                "processed_batches": processed_batches,
                "total_batches": total_batches,
                "context": "unexpected_error"
            }

            try:
                lifecycle_manager.track_stage_error(
                    document_id=session.document_id,
                    stage_name=lifecycle_stage_name,
                    error_message=str(unexpected_error),
                    stage_data=error_stage_data
                )
            except Exception as lifecycle_error:
                logger.error(
                    "Manual processing: Failed to record unexpected lifecycle error for document %s: %s",
                    session.document_id,
                    lifecycle_error,
                )

            try:
                lifecycle_manager.update_document_status(
                    session.document_id,
                    ProcessingStatus.ERROR,
                    summary_data={
                        "total_batches": total_batches,
                        "total_chunks": total_chunks_generated
                    }
                )
            except Exception as lifecycle_error:
                logger.error(
                    "Manual processing: Failed to set document %s status to error after unexpected failure: %s",
                    session.document_id,
                    lifecycle_error,
                )

        # Register the failed task with processing_manager so it appears in the pipeline dashboard
        with processing_manager._lock:
            processing_manager.completed_tasks[task.task_id] = task
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
        document_id = None
        s3_pdf_url = None

        if file and file.filename:
            # Handle uploaded file
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="File must be a PDF")

            # Save uploaded file temporarily
            temp_path = f"/tmp/{file.filename}"
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Register document with lifecycle manager and upload to S3
            if lifecycle_manager:
                try:
                    processing_mode = ProcessingMode.HUMAN_LOOP if is_human_loop else ProcessingMode.AUTO
                    document_id, s3_pdf_url = lifecycle_manager.register_document(
                        pdf_path=temp_path,
                        original_filename=file.filename,
                        processing_mode=processing_mode,
                        metadata={"upload_source": "web_interface", "process_mode": process_mode}
                    )
                    logger.info(f"Registered document {document_id} with S3 URL {s3_pdf_url}")
                except Exception as e:
                    logger.error(f"Failed to register document with lifecycle manager: {e}")

            document = pdf_processor.load_from_local(temp_path)
            if is_human_loop:
                temp_uploaded_path = temp_path
                display_name = file.filename
            else:
                # Clean up temp file after registration
                os.remove(temp_path)

        elif s3_url:
            # Handle S3 URL
            document = pdf_processor.load_from_url(s3_url)
            display_name = os.path.basename(s3_url)

            # Register document with lifecycle manager if it's not already tracked
            if lifecycle_manager:
                try:
                    # For S3 URLs, we assume the PDF is already uploaded, so we create a minimal record
                    processing_mode = ProcessingMode.HUMAN_LOOP if is_human_loop else ProcessingMode.AUTO
                    # Note: For existing S3 URLs, we would need to download temporarily to register
                    logger.info(f"Processing existing S3 URL: {s3_url}")
                except Exception as e:
                    logger.error(f"Failed to process S3 URL with lifecycle manager: {e}")

        elif file_path:
            # Handle local file path
            document = pdf_processor.load_from_local(file_path)
            display_name = os.path.basename(file_path)

            # Register document with lifecycle manager and upload to S3
            if lifecycle_manager:
                try:
                    processing_mode = ProcessingMode.HUMAN_LOOP if is_human_loop else ProcessingMode.AUTO
                    document_id, s3_pdf_url = lifecycle_manager.register_document(
                        pdf_path=file_path,
                        original_filename=os.path.basename(file_path),
                        processing_mode=processing_mode,
                        metadata={"upload_source": "local_file", "process_mode": process_mode, "original_path": file_path}
                    )
                    logger.info(f"Registered local document {document_id} with S3 URL {s3_pdf_url}")
                except Exception as e:
                    logger.error(f"Failed to register local document with lifecycle manager: {e}")

        else:
            raise HTTPException(status_code=400, detail="Please provide a file, file path, or S3 URL")

        # Get document info
        doc_info = pdf_processor.get_document_info(document)

        # Get all pages as images with page numbers
        page_images = pdf_processor.get_pages_as_images(document, add_page_numbers=True, max_size=(400, 600))

        # Save images to S3 if lifecycle manager is available and document was registered
        if lifecycle_manager and document_id and is_human_loop:
            try:
                # Get original size images for S3 storage
                original_images = pdf_processor.get_pages_as_images(document, add_page_numbers=False, max_size=(2000, 3000))
                page_images_info = pdf_processor.save_images_with_lifecycle_manager(document, lifecycle_manager, document_id)
                logger.info(f"Saved {len(page_images_info)} images to S3 for manual session document {document_id}")
            except Exception as e:
                logger.error(f"Failed to save images to S3 for manual session: {e}")
                # Continue without S3 images

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
                display_name=display_name or (os.path.basename(source_path) if source_path else None),
                document_id=document_id,
                s3_pdf_url=s3_pdf_url
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
    overlap_pages = 0
    if isinstance(payload, dict) and "overlap_pages" in payload:
        overlap_candidate = payload.get("overlap_pages")
        try:
            overlap_pages = int(overlap_candidate)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Overlap pages must be a whole number")
        if overlap_pages < 0:
            raise HTTPException(status_code=400, detail="Overlap pages cannot be negative")

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

    if overlap_pages and largest_batch_size and overlap_pages >= largest_batch_size:
        logger.warning(
            "Manual batching: overlap_pages=%s trimmed to fit largest batch size=%s",
            overlap_pages,
            largest_batch_size,
        )
        overlap_pages = max(largest_batch_size - 1, 0)

    manual_config = SplitConfiguration(
        batch_size=largest_batch_size or 1,
        overlap_pages=overlap_pages
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
        lifecycle_document_id=session.document_id,
    )

    manual_task.document.file_path = (
        session.display_name or session.source_path or manual_task.document.file_path
    )
    manual_task.document.source_type = "manual"
    metadata = dict(manual_task.document.metadata or {})
    if session.document_id:
        metadata.setdefault("document_id", session.document_id)
    metadata["processing_mode"] = "human_loop"
    metadata["overlap_pages"] = overlap_pages
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

@app.get("/api/database/status")
async def get_database_status():
    """Get database connection status for debugging"""
    try:
        main_db_status = {
            "main_db_writer_initialized": db_writer is not None,
            "main_db_writer_status": db_writer.get_connection_status() if db_writer else None
        }

        processing_db_status = {
            "processing_manager_db_writer_initialized": processing_manager.db_writer is not None,
            "processing_manager_db_writer_status": processing_manager.db_writer.get_connection_status() if processing_manager.db_writer else None
        }

        lifecycle_status = {
            "lifecycle_manager_initialized": lifecycle_manager is not None,
            "lifecycle_manager_status": lifecycle_manager.get_connection_status() if lifecycle_manager else None
        }

        return JSONResponse(content={
            "main": main_db_status,
            "processing_manager": processing_db_status,
            "lifecycle_manager": lifecycle_status
        })
    except Exception as e:
        logger.error(f"Error getting database status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recovery/status")
async def get_recovery_status():
    """Get recovery system status and statistics"""
    try:
        if not restart_manager:
            raise HTTPException(status_code=503, detail="Recovery system not available")

        stats = restart_manager.get_recovery_statistics(max_age_hours=24)
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting recovery status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recovery/run")
async def run_recovery_check():
    """Run a recovery check and optionally auto-resume incomplete documents"""
    try:
        if not restart_manager:
            raise HTTPException(status_code=503, detail="Recovery system not available")

        # Run recovery check with auto-resume enabled
        result = restart_manager.run_recovery_check(max_age_hours=24, auto_resume=True)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error running recovery check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recovery/document/{document_id}/resume")
async def resume_document_processing_api(document_id: str):
    """Resume processing for a specific document"""
    try:
        if not restart_manager:
            raise HTTPException(status_code=503, detail="Recovery system not available")

        result = restart_manager.resume_document_processing(document_id)
        if result["success"]:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Resume failed"))
    except Exception as e:
        logger.error(f"Error resuming document {document_id}: {e}")
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


@app.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request):
    """Documents history page"""
    return templates.TemplateResponse("documents.html", {"request": request})


@app.get("/api/documents")
async def get_documents():
    """Get all processed documents from database"""
    if not lifecycle_manager:
        # Return mock data for testing when database is not available
        return JSONResponse(content={
            "documents": [
                {
                    "document_id": "doc_sample_001",
                    "original_filename": "sample_document.pdf",
                    "status": "completed",
                    "processing_mode": "auto",
                    "created_at": "2024-09-29T10:00:00Z",
                    "updated_at": "2024-09-29T10:05:00Z",
                    "file_size_bytes": 1024000,
                    "total_pages": 10,
                    "s3_pdf_url": "s3://bucket/sample.pdf",
                    "metadata": {"source": "test"}
                },
                {
                    "document_id": "doc_sample_002",
                    "original_filename": "another_document.pdf",
                    "status": "processing",
                    "processing_mode": "human_loop",
                    "created_at": "2024-09-29T11:00:00Z",
                    "updated_at": "2024-09-29T11:02:00Z",
                    "file_size_bytes": 2048000,
                    "total_pages": 25,
                    "s3_pdf_url": "s3://bucket/another.pdf",
                    "metadata": {"source": "web_upload"}
                }
            ],
            "total_count": 2
        })

    try:
        documents = lifecycle_manager.mongodb_manager.get_all_documents()

        # Format the response for UI
        document_list = []
        for doc in documents:
            document_list.append({
                "document_id": doc.document_id,
                "original_filename": doc.original_filename,
                "status": doc.current_status.value,
                "processing_mode": doc.processing_mode.value if doc.processing_mode else "unknown",
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "file_size_bytes": doc.file_size_bytes,
                "total_pages": getattr(doc, 'total_pages', 0),
                "s3_pdf_url": doc.s3_pdf_url,
                "metadata": getattr(doc, 'metadata', {})
            })

        return JSONResponse(content={
            "documents": document_list,
            "total_count": len(document_list)
        })

    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{document_id}")
async def get_document_details(document_id: str):
    """Get detailed information about a specific document"""
    if not lifecycle_manager:
        # Return mock data for testing when database is not available
        if document_id in ["doc_sample_001", "doc_sample_002"]:
            return JSONResponse(content={
                "document": {
                    "document_id": document_id,
                    "original_filename": "sample_document.pdf" if document_id == "doc_sample_001" else "another_document.pdf",
                    "status": "completed" if document_id == "doc_sample_001" else "processing",
                    "processing_mode": "auto" if document_id == "doc_sample_001" else "human_loop",
                    "created_at": "2024-09-29T10:00:00Z",
                    "updated_at": "2024-09-29T10:05:00Z",
                    "file_size_bytes": 1024000,
                    "total_pages": 10,
                    "s3_pdf_url": "s3://bucket/sample.pdf",
                    "s3_images_folder": "s3://bucket/images/",
                    "metadata": {"source": "test"}
                },
                "stages": [
                    {
                        "stage_id": "stage_001",
                        "stage_name": "PDF Upload",
                        "status": "completed",
                        "started_at": "2024-09-29T10:00:00Z",
                        "completed_at": "2024-09-29T10:01:00Z",
                        "error_message": None,
                        "metrics": {"duration": 60}
                    },
                    {
                        "stage_id": "stage_002",
                        "stage_name": "Image Conversion",
                        "status": "completed",
                        "started_at": "2024-09-29T10:01:00Z",
                        "completed_at": "2024-09-29T10:03:00Z",
                        "error_message": None,
                        "metrics": {"images_created": 10}
                    }
                ],
                "batches": [
                    {
                        "batch_id": "batch_001",
                        "batch_number": 1,
                        "start_page": 1,
                        "end_page": 5,
                        "status": "completed",
                        "created_at": "2024-09-29T10:02:00Z",
                        "processed_at": "2024-09-29T10:04:00Z",
                        "chunk_count": 3,
                        "s3_images_url": "s3://bucket/images/batch1/"
                    },
                    {
                        "batch_id": "batch_002",
                        "batch_number": 2,
                        "start_page": 6,
                        "end_page": 10,
                        "status": "completed",
                        "created_at": "2024-09-29T10:02:00Z",
                        "processed_at": "2024-09-29T10:05:00Z",
                        "chunk_count": 2,
                        "s3_images_url": "s3://bucket/images/batch2/"
                    }
                ],
                "chunks": [
                    {
                        "chunk_id": "chunk_001",
                        "batch_id": "batch_001",
                        "chunk_number": 1,
                        "page_numbers": [1, 2],
                        "content": "This is the content of the first chunk from pages 1-2. It contains important information about the document structure and...",
                        "full_content_length": 1250,
                        "metadata": {"headings": ["Introduction", "Overview"]}
                    },
                    {
                        "chunk_id": "chunk_002",
                        "batch_id": "batch_001",
                        "chunk_number": 2,
                        "page_numbers": [3, 4, 5],
                        "content": "This is the second chunk containing details from pages 3-5. The content includes technical specifications and procedures...",
                        "full_content_length": 2100,
                        "metadata": {"headings": ["Technical Details", "Procedures"]}
                    }
                ],
                "images": [
                    {
                        "page_number": 1,
                        "s3_image_url": "s3://bucket/images/page_1.png",
                        "image_size": "800x1200",
                        "created_at": "2024-09-29T10:01:30Z"
                    },
                    {
                        "page_number": 2,
                        "s3_image_url": "s3://bucket/images/page_2.png",
                        "image_size": "800x1200",
                        "created_at": "2024-09-29T10:01:35Z"
                    }
                ]
            })
        else:
            raise HTTPException(status_code=404, detail="Document not found")

    try:
        # Get document record
        doc = lifecycle_manager.mongodb_manager.get_document_by_id(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get processing stages
        stages = lifecycle_manager.mongodb_manager.get_stages_by_document_id(document_id)

        # Get batches
        batches = lifecycle_manager.mongodb_manager.get_batches_by_document_id(document_id)

        # Get chunks
        chunks = lifecycle_manager.mongodb_manager.get_chunks_by_document_id(document_id)

        # Get page images if available
        images = lifecycle_manager.mongodb_manager.get_page_images_by_document_id(document_id)

        # Format the response
        document_details = {
            "document": {
                "document_id": doc.document_id,
                "original_filename": doc.original_filename,
                "status": doc.current_status.value,
                "processing_mode": doc.processing_mode.value if doc.processing_mode else "unknown",
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "file_size_bytes": doc.file_size_bytes,
                "total_pages": getattr(doc, 'total_pages', 0),
                "s3_pdf_url": doc.s3_pdf_url,
                "s3_images_folder": doc.s3_images_folder,
                "metadata": getattr(doc, 'metadata', {})
            },
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "stage_name": stage.stage_name,
                    "status": stage.status.value,
                    "started_at": stage.started_at.isoformat() if stage.started_at else None,
                    "completed_at": stage.completed_at.isoformat() if stage.completed_at else None,
                    "error_message": stage.error_message,
                    "metrics": stage.metrics or {}
                }
                for stage in stages
            ],
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "batch_number": batch.batch_number,
                    "start_page": batch.start_page,
                    "end_page": batch.end_page,
                    "status": batch.status.value if batch.status else "unknown",
                    "created_at": batch.created_at.isoformat() if batch.created_at else None,
                    "processed_at": batch.processed_at.isoformat() if batch.processed_at else None,
                    "chunk_count": batch.chunk_count or 0,
                    "s3_images_url": batch.s3_images_url
                }
                for batch in batches
            ],
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "batch_id": chunk.batch_id,
                    "chunk_number": chunk.chunk_number,
                    "page_numbers": chunk.page_numbers or [],
                    "content": chunk.content[:500] + "..." if chunk.content and len(chunk.content) > 500 else chunk.content,  # Truncate for listing
                    "full_content_length": len(chunk.content) if chunk.content else 0,
                    "metadata": chunk.metadata or {}
                }
                for chunk in chunks
            ],
            "images": [
                {
                    "page_number": img.page_number,
                    "s3_image_url": img.s3_image_url,
                    "image_size": img.image_size,
                    "created_at": img.created_at.isoformat() if img.created_at else None
                }
                for img in images
            ]
        }

        return JSONResponse(content=document_details)

    except Exception as e:
        logger.error(f"Error fetching document details for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{document_id}/chunks/{chunk_id}")
async def get_chunk_full_content(document_id: str, chunk_id: str):
    """Get full content of a specific chunk"""
    if not lifecycle_manager:
        # Return mock data for testing when database is not available
        if chunk_id in ["chunk_001", "chunk_002"]:
            chunk_content = {
                "chunk_001": """This is the complete content of the first chunk from pages 1-2.

# Introduction

This document provides a comprehensive overview of the PDF processing system. The system is designed to handle large documents efficiently by breaking them down into manageable chunks while preserving the semantic structure and context.

## Key Features
- Automatic page batching
- Intelligent chunk splitting
- S3 storage integration
- MongoDB persistence
- Human-in-the-loop processing

The introduction section covers the fundamental concepts and architecture decisions that drive the system's design.""",
                "chunk_002": """This is the complete content of the second chunk from pages 3-5.

# Technical Details

The technical implementation relies on several key components:

## PDF Processing
- PyMuPDF for PDF manipulation
- PIL for image processing
- Custom batch management

## Storage Layer
- S3 for file storage
- MongoDB for metadata
- Redis for caching (optional)

## Processing Pipeline
1. PDF Upload and Validation
2. Page Extraction and Conversion
3. Batch Creation and Management
4. Chunk Generation and Storage
5. Quality Assurance and Review

### Procedures

The standard operating procedure for document processing includes:
- Initial upload validation
- Content analysis and categorization
- Batch processing configuration
- Quality control checkpoints
- Final review and approval"""
            }

            return JSONResponse(content={
                "chunk_id": chunk_id,
                "document_id": document_id,
                "batch_id": "batch_001",
                "chunk_number": 1 if chunk_id == "chunk_001" else 2,
                "content": chunk_content[chunk_id],
                "page_numbers": [1, 2] if chunk_id == "chunk_001" else [3, 4, 5],
                "metadata": {"headings": ["Introduction", "Overview"] if chunk_id == "chunk_001" else ["Technical Details", "Procedures"]}
            })
        else:
            raise HTTPException(status_code=404, detail="Chunk not found")

    try:
        chunk = lifecycle_manager.mongodb_manager.get_chunk_by_id(chunk_id)
        if not chunk or chunk.document_id != document_id:
            raise HTTPException(status_code=404, detail="Chunk not found")

        return JSONResponse(content={
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "batch_id": chunk.batch_id,
            "chunk_number": chunk.chunk_number,
            "content": chunk.content,
            "page_numbers": chunk.page_numbers or [],
            "metadata": chunk.metadata or {}
        })

    except Exception as e:
        logger.error(f"Error fetching chunk {chunk_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
