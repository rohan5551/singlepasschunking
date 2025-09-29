import logging
import os
import uuid
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty

from ..models.pdf_document import PDFDocument
from ..models.batch_models import (
    ProcessingTask, SplitConfiguration, BatchingResult,
    ProcessingStage, BatchStatus
)
from ..models.chunk_schema import BatchProcessingResult
from .pdf_processor import PDFProcessor
from .pdf_splitter import PDFSplitter
from .lmm_processor import LMMProcessor
from .context_manager import ContextManager
from .db_writer import DatabaseWriter
from ..prompts.manual import build_manual_prompt
from ..managers import DocumentLifecycleManager
from ..database.document_lifecycle_models import ProcessingMode, ProcessingStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingManager:
    """
    Manages parallel processing of multiple PDF documents through the pipeline:
    Upload → PDF Processing → Splitting → Batching

    Features:
    - Parallel processing of multiple documents
    - Real-time progress tracking
    - Queue management
    - Error handling and recovery
    - WebSocket notifications for UI updates
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize ProcessingManager

        Args:
            max_workers: Maximum number of concurrent processing threads
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Task management
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingTask] = {}
        self.task_queue = Queue()

        # Processing components
        self.pdf_processor = PDFProcessor()
        self.pdf_splitter = PDFSplitter()
        self.lmm_processor = LMMProcessor()
        self.context_manager = ContextManager()

        # Initialize Database Writer
        try:
            self.db_writer = DatabaseWriter()
            logger.info("ProcessingManager: Database writer initialized successfully")
            logger.info(f"ProcessingManager: Database writer connection status: {self.db_writer.get_connection_status()}")
        except Exception as e:
            logger.error(f"ProcessingManager: Database writer initialization failed: {e}")
            import traceback
            logger.error(f"ProcessingManager: Database writer error details: {traceback.format_exc()}")
            self.db_writer = None

        # Initialize Document Lifecycle Manager
        try:
            self.lifecycle_manager = DocumentLifecycleManager()
            logger.info("ProcessingManager: Document lifecycle manager initialized successfully")
            logger.info(f"ProcessingManager: Lifecycle manager connection status: {self.lifecycle_manager.get_connection_status()}")
        except Exception as e:
            logger.error(f"ProcessingManager: Document lifecycle manager initialization failed: {e}")
            import traceback
            logger.error(f"ProcessingManager: Document lifecycle manager error details: {traceback.format_exc()}")
            self.lifecycle_manager = None

        # Event callbacks for UI updates
        self.progress_callbacks: List[Callable[[str, float, str], None]] = []
        self.status_callbacks: List[Callable[[str, ProcessingStage], None]] = []
        self.completion_callbacks: List[Callable[[str, ProcessingTask], None]] = []

        # Thread safety
        self._lock = threading.Lock()

        # Start processing thread
        self._running = True
        self._processor_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._processor_thread.start()

        logger.info(f"ProcessingManager initialized with {max_workers} workers")

    def add_progress_callback(self, callback: Callable[[str, float, str], None]):
        """Add callback for progress updates"""
        self.progress_callbacks.append(callback)

    def add_status_callback(self, callback: Callable[[str, ProcessingStage], None]):
        """Add callback for status updates"""
        self.status_callbacks.append(callback)

    def add_completion_callback(self, callback: Callable[[str, ProcessingTask], None]):
        """Add callback for task completion"""
        self.completion_callbacks.append(callback)

    def submit_task(self, file_path: str, source_type: str = "local",
                   config: Optional[SplitConfiguration] = None,
                   s3_url: Optional[str] = None,
                   prompt: Optional[str] = None,
                   model: Optional[str] = None,
                   temperature: Optional[float] = None,
                   document_id: Optional[str] = None) -> str:
        """
        Submit a new processing task

        Args:
            file_path: Path to PDF file or S3 URL
            source_type: "local", "s3", or "upload"
            config: Split configuration
            s3_url: S3 URL if source_type is "s3"
            prompt: Processing prompt
            model: LLM model to use
            temperature: Model temperature
            document_id: Optional existing document ID from lifecycle manager

        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        split_config = config or SplitConfiguration()

        # Register document with lifecycle manager if not already registered
        lifecycle_document_id = document_id
        if self.lifecycle_manager and not document_id:
            try:
                filename = os.path.basename(file_path) if file_path else f"bulk_upload_{task_id}.pdf"
                lifecycle_document_id, s3_pdf_url = self.lifecycle_manager.register_document(
                    pdf_path=file_path,
                    original_filename=filename,
                    processing_mode=ProcessingMode.BULK,
                    metadata={
                        "task_id": task_id,
                        "source_type": source_type,
                        "prompt": prompt,
                        "model": model,
                        "temperature": temperature,
                        "config": split_config.to_dict()
                    }
                )
                logger.info(f"Registered document {lifecycle_document_id} for task {task_id}")
            except Exception as e:
                logger.error(f"Failed to register document for task {task_id}: {e}")
                lifecycle_document_id = None

        # Create placeholder document (will be populated during processing)
        placeholder_document = PDFDocument(
            file_path=file_path,
            pages=[],
            total_pages=0,
            metadata={"document_id": lifecycle_document_id} if lifecycle_document_id else {},
            source_type=source_type
        )

        task = ProcessingTask(
            task_id=task_id,
            document=placeholder_document,
            config=split_config,
            status=ProcessingStage.UPLOAD,
            prompt=prompt,
            model=model or self.lmm_processor.DEFAULT_MODEL,
            temperature=temperature if temperature is not None else self.lmm_processor.temperature
        )

        with self._lock:
            self.active_tasks[task_id] = task

        # Queue for processing
        self.task_queue.put((
            task_id,
            file_path,
            source_type,
            s3_url,
            split_config,
            prompt,
            model,
            temperature,
            lifecycle_document_id,
        ))

        logger.info(f"Submitted task {task_id} for {file_path}")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a task"""
        with self._lock:
            task = self.active_tasks.get(task_id) or self.completed_tasks.get(task_id)

        if task:
            return task.to_dict()
        return None

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tasks"""
        with self._lock:
            all_tasks = {}

            # Active tasks
            for task_id, task in self.active_tasks.items():
                all_tasks[task_id] = task.to_dict()

            # Completed tasks
            for task_id, task in self.completed_tasks.items():
                all_tasks[task_id] = task.to_dict()

        return all_tasks

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if it's still in queue or processing"""
        with self._lock:
            task = self.active_tasks.get(task_id)
            if task and task.status in [ProcessingStage.UPLOAD, ProcessingStage.PDF_PROCESSING]:
                task.status = ProcessingStage.ERROR
                task.error_message = "Cancelled by user"
                self._notify_status_change(task_id, ProcessingStage.ERROR)
                return True
        return False

    def clear_completed_tasks(self):
        """Clear all completed tasks"""
        with self._lock:
            self.completed_tasks.clear()
        logger.info("Cleared completed tasks")

    def _process_queue(self):
        """Background thread to process the task queue"""
        while self._running:
            try:
                # Get next task from queue (blocks for up to 1 second)
                try:
                    task_data = self.task_queue.get(timeout=1.0)
                except Empty:
                    continue

                task_id, file_path, source_type, s3_url, config, prompt, model, temperature, lifecycle_document_id = task_data

                # Submit to thread pool
                future = self.executor.submit(
                    self._process_single_task,
                    task_id, file_path, source_type, s3_url, config,
                    prompt, model, temperature, lifecycle_document_id
                )

                # Don't block on individual tasks
                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"Error in queue processing: {e}")

    def _process_single_task(self, task_id: str, file_path: str,
                           source_type: str, s3_url: Optional[str],
                           config: SplitConfiguration,
                           prompt: Optional[str], model: Optional[str],
                           temperature: Optional[float],
                           lifecycle_document_id: Optional[str]):
        """Process a single task through the pipeline"""
        temp_file_to_cleanup = None

        try:
            with self._lock:
                task = self.active_tasks.get(task_id)

            if not task:
                logger.error(f"Task {task_id} not found")
                return

            task.started_at = datetime.now()

            # Stage 1: PDF Processing
            self._update_task_progress(task_id, ProcessingStage.PDF_PROCESSING, 10.0)

            # Remember temp file path for cleanup if it's an upload
            if source_type == "upload":
                temp_file_to_cleanup = file_path

            if source_type == "s3" or (file_path.startswith("s3://")):
                document = self.pdf_processor.load_from_url(file_path)
            elif source_type in ["local", "upload"]:
                document = self.pdf_processor.load_from_local(file_path)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            # Update task with real document
            task.document = document
            task.model = model or self.lmm_processor.DEFAULT_MODEL
            task.prompt = prompt
            task.temperature = temperature if temperature is not None else self.lmm_processor.temperature
            self.context_manager.reset_context(document.file_path)

            # Save images to S3 if lifecycle manager is available
            if self.lifecycle_manager and lifecycle_document_id:
                try:
                    page_images_info = self.pdf_processor.save_images_with_lifecycle_manager(
                        document, self.lifecycle_manager, lifecycle_document_id
                    )
                    logger.info(f"Saved {len(page_images_info)} images to S3 for document {lifecycle_document_id}")
                except Exception as e:
                    logger.error(f"Failed to save images to S3 for document {lifecycle_document_id}: {e}")

            self._update_task_progress(task_id, ProcessingStage.PDF_PROCESSING, 40.0)

            # Stage 2: Splitting
            self._update_task_progress(task_id, ProcessingStage.SPLITTING, 50.0)

            # Track splitting stage start
            if self.lifecycle_manager and lifecycle_document_id:
                self.lifecycle_manager.track_stage_start(
                    document_id=lifecycle_document_id,
                    stage_name="splitting",
                    stage_data={"total_pages": document.total_pages, "batch_size": config.batch_size}
                )

            batching_result = self.pdf_splitter.split_document(document, config)
            task.batches = batching_result.batches

            # Create batch records in lifecycle manager
            if self.lifecycle_manager and lifecycle_document_id:
                try:
                    for i, batch in enumerate(task.batches):
                        # Get page images for this batch
                        batch_page_images = []
                        for page_num in range(batch.start_page, batch.end_page + 1):
                            if page_num <= len(document.pages):
                                page = document.pages[page_num - 1]
                                if hasattr(page, 's3_url') and page.s3_url:
                                    from ..database.document_lifecycle_models import PageImageInfo
                                    page_image_info = PageImageInfo(
                                        page_number=page_num,
                                        s3_original_url=page.s3_url,
                                        s3_thumbnail_url=getattr(page, 's3_thumbnail_url', None),
                                        image_dimensions={"width": page.image.width, "height": page.image.height} if page.image else {},
                                        file_size_bytes=0
                                    )
                                    batch_page_images.append(page_image_info)

                        batch_id = self.lifecycle_manager.create_batch_record(
                            document_id=lifecycle_document_id,
                            batch_number=batch.batch_number,
                            page_range={
                                "start_page": batch.start_page,
                                "end_page": batch.end_page,
                                "total_pages": batch.total_pages
                            },
                            page_images=batch_page_images
                        )
                        logger.info(f"Created batch record {batch_id} for batch {batch.batch_number}")

                    # Track splitting stage completion
                    self.lifecycle_manager.track_stage_completion(
                        document_id=lifecycle_document_id,
                        stage_name="splitting",
                        stage_data={
                            "batches_created": len(task.batches),
                            "batch_ids": [batch.batch_id for batch in task.batches]
                        }
                    )

                    # Save batch configurations to S3
                    try:
                        batch_configs = {
                            "document_info": {
                                "document_id": lifecycle_document_id,
                                "original_filename": task.document.filename,
                                "total_pages": task.document.total_pages,
                                "processing_timestamp": datetime.utcnow().isoformat()
                            },
                            "split_configuration": {
                                "batch_size": config.batch_size,
                                "overlap_pages": config.overlap_pages,
                                "min_batch_size": config.min_batch_size
                            },
                            "batches": [
                                {
                                    "batch_number": batch.batch_number,
                                    "batch_id": getattr(batch, 'batch_id', f"batch_{batch.batch_number}"),
                                    "start_page": batch.start_page,
                                    "end_page": batch.end_page,
                                    "total_pages": batch.total_pages,
                                    "page_count": batch.end_page - batch.start_page + 1
                                }
                                for batch in task.batches
                            ]
                        }

                        self.lifecycle_manager.save_processing_artifacts(
                            document_id=lifecycle_document_id,
                            artifacts={"batch_configs.json": batch_configs}
                        )
                        logger.info(f"Saved batch_configs.json for document {lifecycle_document_id}")
                    except Exception as e:
                        logger.error(f"Failed to save batch configurations for document {lifecycle_document_id}: {e}")
                except Exception as e:
                    logger.error(f"Failed to create batch records for document {lifecycle_document_id}: {e}")

            self._update_task_progress(task_id, ProcessingStage.SPLITTING, 80.0)

            # Stage 3: LMM Processing
            if task.batches:
                # Track LMM processing stage start
                if self.lifecycle_manager and lifecycle_document_id:
                    self.lifecycle_manager.track_stage_start(
                        document_id=lifecycle_document_id,
                        stage_name="lmm_processing",
                        stage_data={
                            "total_batches": len(task.batches),
                            "model": task.model,
                            "temperature": task.temperature
                        }
                    )

                total_batches = len(task.batches)
                for index, batch in enumerate(task.batches, start=1):
                    try:
                        batch.status = BatchStatus.PROCESSING
                        context_payload = self.context_manager.build_context_payload(document.file_path)

                        # Build prompt with context like manual processing
                        prompt_for_batch = build_manual_prompt(task.prompt, context_payload)

                        # Process batch with enhanced structured output
                        lmm_result: BatchProcessingResult = self.lmm_processor.process_batch(
                            batch,
                            context=context_payload,
                            prompt=prompt_for_batch,
                            model=task.model,
                            temperature=task.temperature,
                        )

                        # Store structured output data using BatchProcessingResult properties
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
                        batch.processed_at = datetime.now()
                        batch.status = BatchStatus.COMPLETED

                        # Update context using new structured method
                        self.context_manager.update_context_from_batch_result(
                            document.file_path,
                            lmm_result,
                            index,
                        )

                        # Update batch processing result in lifecycle manager
                        if self.lifecycle_manager and lifecycle_document_id:
                            try:
                                processing_metadata = {
                                    "model": task.model,
                                    "temperature": task.temperature,
                                    "prompt_used": prompt_for_batch,
                                    "processing_time": batch.processing_time,
                                    "tokens_used": lmm_result.processing_metadata.get("tokens_used", 0)
                                }
                                self.lifecycle_manager.update_batch_processing_result(
                                    batch_id=batch.batch_id,
                                    batch_result=lmm_result,
                                    processing_metadata=processing_metadata
                                )
                                logger.info(f"Updated batch processing result for {batch.batch_id}")
                            except Exception as e:
                                logger.error(f"Failed to update batch processing result for {batch.batch_id}: {e}")

                        # Save enhanced chunks using lifecycle manager
                        if self.lifecycle_manager and lifecycle_document_id:
                            try:
                                chunk_ids = self.lifecycle_manager.save_chunks(
                                    document_id=lifecycle_document_id,
                                    batch_id=batch.batch_id,
                                    batch_result=lmm_result,
                                    processing_task=task
                                )
                                logger.info(f"Saved {len(chunk_ids)} enhanced chunks for batch {batch.batch_id}")
                            except Exception as e:
                                logger.error(f"Failed to save enhanced chunks for batch {batch.batch_id}: {e}")

                        # Save chunks to legacy database if db_writer is available (for backwards compatibility)
                        logger.info(f"Bulk processing: db_writer available: {self.db_writer is not None}")
                        if self.db_writer:
                            try:
                                logger.info(f"Bulk processing: Attempting to save chunks for batch {batch.batch_id}")
                                write_result = self.db_writer.write_batch(task, batch, lmm_result)
                                if write_result["success"]:
                                    logger.info(f"Bulk processing: Successfully saved {write_result['chunks_saved']} chunks for batch {batch.batch_id}")
                                else:
                                    logger.error(f"Bulk processing: Failed to save chunks for batch {batch.batch_id}: {write_result.get('error')}")
                            except Exception as db_error:
                                logger.error(f"Bulk processing: Database write error for batch {batch.batch_id}: {db_error}")
                                import traceback
                                logger.error(f"Bulk processing: Database write traceback: {traceback.format_exc()}")
                        else:
                            logger.warning(f"Bulk processing: db_writer is None, cannot save chunks for batch {batch.batch_id}")

                        progress = 80.0 + (index / total_batches) * 15.0
                        self._update_task_progress(task_id, ProcessingStage.LMM_PROCESSING, progress)
                    except Exception as batch_error:
                        batch.status = BatchStatus.ERROR
                        batch.error_message = str(batch_error)
                        logger.error(
                            "Error processing batch %s for task %s: %s",
                            batch.batch_id,
                            task_id,
                            batch_error,
                        )
                        raise

                # Track LMM processing stage completion
                if self.lifecycle_manager and lifecycle_document_id:
                    total_chunks = sum(len(batch.chunk_summary) for batch in task.batches)
                    self.lifecycle_manager.track_stage_completion(
                        document_id=lifecycle_document_id,
                        stage_name="lmm_processing",
                        stage_data={
                            "batches_processed": len(task.batches),
                            "total_chunks_generated": total_chunks
                        }
                    )

                task.context_state = self.context_manager.get_context(document.file_path)
            else:
                self._update_task_progress(task_id, ProcessingStage.LMM_PROCESSING, 90.0)

            # Stage 4: Database Writing
            if self.db_writer:
                self._update_task_progress(task_id, ProcessingStage.DB_WRITING, 92.0)
                try:
                    # Write any remaining chunks if needed (for cleanup or final aggregation)
                    db_summary = self.db_writer.write_task_chunks(task)
                    if db_summary["success"]:
                        logger.info(f"Database writing completed for task {task_id}: {db_summary['total_chunks']} total chunks saved")
                    else:
                        logger.warning(f"Database writing had issues for task {task_id}: {len(db_summary['errors'])} errors")
                except Exception as db_error:
                    logger.error(f"Database writing stage failed for task {task_id}: {db_error}")

            # Stage 5: Chunking completion
            self._update_task_progress(task_id, ProcessingStage.CHUNKING, 95.0)

            # Track chunking stage completion
            if self.lifecycle_manager and lifecycle_document_id:
                total_chunks = sum(len(batch.chunk_summary) for batch in task.batches if hasattr(batch, 'chunk_summary'))
                self.lifecycle_manager.track_stage_completion(
                    document_id=lifecycle_document_id,
                    stage_name="chunking",
                    stage_data={"total_chunks_saved": total_chunks}
                )

            # Stage 6: Completion
            self._update_task_progress(task_id, ProcessingStage.COMPLETED, 100.0)
            task.completed_at = datetime.now()

            # Update document status to completed
            if self.lifecycle_manager and lifecycle_document_id:
                total_chunks = sum(len(batch.chunk_summary) for batch in task.batches if hasattr(batch, 'chunk_summary'))
                self.lifecycle_manager.update_document_status(
                    document_id=lifecycle_document_id,
                    status=ProcessingStatus.COMPLETED,
                    summary_data={
                        "total_batches": len(task.batches),
                        "total_chunks": total_chunks
                    }
                )
                logger.info(f"Document {lifecycle_document_id} processing completed with {total_chunks} chunks")

            # Move to completed tasks
            with self._lock:
                self.completed_tasks[task_id] = task
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]

            # Notify completion
            for callback in self.completion_callbacks:
                try:
                    callback(task_id, task)
                except Exception as e:
                    logger.error(f"Error in completion callback: {e}")

            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            self._handle_task_error(task_id, str(e))

        finally:
            # Cleanup temporary file if it was an upload
            if temp_file_to_cleanup and os.path.exists(temp_file_to_cleanup):
                try:
                    os.remove(temp_file_to_cleanup)
                    logger.debug(f"Cleaned up temporary file: {temp_file_to_cleanup}")
                except Exception as e:
                    logger.error(f"Error cleaning up temp file {temp_file_to_cleanup}: {e}")
            if 'document' in locals():
                self.context_manager.drop_context(document.file_path)

    def _update_task_progress(self, task_id: str, stage: ProcessingStage, progress: float):
        """Update task progress and notify callbacks"""
        with self._lock:
            task = self.active_tasks.get(task_id) or self.completed_tasks.get(task_id)

        if task:
            task.status = stage
            task.progress = progress

            # Notify callbacks
            for callback in self.progress_callbacks:
                try:
                    callback(task_id, progress, stage.value)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")

            self._notify_status_change(task_id, stage)

    def _notify_status_change(self, task_id: str, stage: ProcessingStage):
        """Notify status change callbacks"""
        for callback in self.status_callbacks:
            try:
                callback(task_id, stage)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def _handle_task_error(self, task_id: str, error_message: str):
        """Handle task error and update status"""
        with self._lock:
            task = self.active_tasks.get(task_id)

        if task:
            task.status = ProcessingStage.ERROR
            task.error_message = error_message
            task.completed_at = datetime.now()

            # Move to completed tasks
            with self._lock:
                self.completed_tasks[task_id] = task
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]

            self._notify_status_change(task_id, ProcessingStage.ERROR)

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        with self._lock:
            active_count = len(self.active_tasks)
            completed_count = len(self.completed_tasks)

        return {
            "active_tasks": active_count,
            "completed_tasks": completed_count,
            "queue_size": self.task_queue.qsize(),
            "max_workers": self.max_workers,
            "is_running": self._running
        }

    def shutdown(self):
        """Shutdown the processing manager"""
        logger.info("Shutting down ProcessingManager")
        self._running = False

        if hasattr(self, '_processor_thread'):
            self._processor_thread.join(timeout=5.0)

        self.executor.shutdown(wait=True)
        logger.info("ProcessingManager shutdown complete")