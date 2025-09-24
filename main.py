import os
import io
import base64
import tempfile
from typing import Optional
from fastapi import FastAPI, Request, Form, HTTPException, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
import logging

from src.processors import PDFProcessor, ProcessingManager
from src.models import PDFDocument, SplitConfiguration

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
    file: Optional[UploadFile] = File(None)
):
    """Process PDF from local path, S3 URL, or uploaded file"""
    try:
        document = None

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
            # Clean up temp file
            os.remove(temp_path)

        elif s3_url:
            # Handle S3 URL
            document = pdf_processor.load_from_url(s3_url)

        elif file_path:
            # Handle local file path
            document = pdf_processor.load_from_local(file_path)

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
    overlap_pages: int = Form(0)
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
            config=config
        )

        return JSONResponse(content={
            "task_id": task_id,
            "filename": file.filename,
            "config": config.to_dict()
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