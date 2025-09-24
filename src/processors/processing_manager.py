import asyncio
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
from .pdf_processor import PDFProcessor
from .pdf_splitter import PDFSplitter

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
                   s3_url: Optional[str] = None) -> str:
        """
        Submit a new processing task

        Args:
            file_path: Path to PDF file or S3 URL
            source_type: "local", "s3", or "upload"
            config: Split configuration
            s3_url: S3 URL if source_type is "s3"

        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        split_config = config or SplitConfiguration()

        # Create placeholder document (will be populated during processing)
        placeholder_document = PDFDocument(
            file_path=file_path,
            pages=[],
            total_pages=0,
            metadata={},
            source_type=source_type
        )

        task = ProcessingTask(
            task_id=task_id,
            document=placeholder_document,
            config=split_config,
            status=ProcessingStage.UPLOAD
        )

        with self._lock:
            self.active_tasks[task_id] = task

        # Queue for processing
        self.task_queue.put((task_id, file_path, source_type, s3_url, split_config))

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

                task_id, file_path, source_type, s3_url, config = task_data

                # Submit to thread pool
                future = self.executor.submit(
                    self._process_single_task,
                    task_id, file_path, source_type, s3_url, config
                )

                # Don't block on individual tasks
                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"Error in queue processing: {e}")

    def _process_single_task(self, task_id: str, file_path: str,
                           source_type: str, s3_url: Optional[str],
                           config: SplitConfiguration):
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
            self._update_task_progress(task_id, ProcessingStage.PDF_PROCESSING, 40.0)

            # Stage 2: Splitting
            self._update_task_progress(task_id, ProcessingStage.SPLITTING, 50.0)

            batching_result = self.pdf_splitter.split_document(document, config)
            task.batches = batching_result.batches
            self._update_task_progress(task_id, ProcessingStage.SPLITTING, 80.0)

            # Stage 3: Completion
            self._update_task_progress(task_id, ProcessingStage.COMPLETED, 100.0)
            task.completed_at = datetime.now()

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