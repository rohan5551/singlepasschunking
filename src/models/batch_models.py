from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import uuid
from datetime import datetime

from .pdf_document import PDFDocument, PDFPage

class BatchStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class ProcessingStage(Enum):
    UPLOAD = "upload"
    PDF_PROCESSING = "pdf_processing"
    SPLITTING = "splitting"
    LMM_PROCESSING = "lmm_processing"
    CHUNKING = "chunking"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class PageBatch:
    """Represents a batch of PDF pages"""
    batch_id: str
    batch_number: int
    pages: List[PDFPage]
    start_page: int
    end_page: int
    total_pages: int
    document_id: str
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    lmm_output: Optional[str] = None
    chunk_summary: List[str] = field(default_factory=list)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    prompt_used: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def page_numbers(self) -> List[int]:
        return [page.page_number for page in self.pages]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "batch_number": self.batch_number,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "total_pages": self.total_pages,
            "page_count": self.page_count,
            "page_numbers": self.page_numbers,
            "document_id": self.document_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "error_message": self.error_message,
            "processing_time": self.processing_time,
            "lmm_output": self.lmm_output,
            "chunk_summary": self.chunk_summary,
            "context_snapshot": self.context_snapshot,
            "prompt_used": self.prompt_used,
            "warnings": self.warnings,
        }

@dataclass
class SplitConfiguration:
    """Configuration for PDF splitting"""
    batch_size: int = 4
    overlap_pages: int = 0
    min_batch_size: int = 1
    max_batch_size: int = 10

    def __post_init__(self):
        if self.batch_size < self.min_batch_size:
            self.batch_size = self.min_batch_size
        if self.batch_size > self.max_batch_size:
            self.batch_size = self.max_batch_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "overlap_pages": self.overlap_pages,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size
        }

@dataclass
class ProcessingTask:
    """Represents a document processing task"""
    task_id: str
    document: PDFDocument
    config: SplitConfiguration
    status: ProcessingStage = ProcessingStage.UPLOAD
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    batches: List[PageBatch] = field(default_factory=list)
    prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.1
    context_state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())

    @property
    def total_batches(self) -> int:
        return len(self.batches)

    @property
    def completed_batches(self) -> int:
        return sum(1 for batch in self.batches if batch.status == BatchStatus.COMPLETED)

    @property
    def processing_time(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_completed(self) -> bool:
        return self.status == ProcessingStage.COMPLETED

    @property
    def has_error(self) -> bool:
        return self.status == ProcessingStage.ERROR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "total_batches": self.total_batches,
            "completed_batches": self.completed_batches,
            "processing_time": self.processing_time,
            "document_info": {
                "file_path": self.document.file_path,
                "total_pages": self.document.total_pages,
                "source_type": self.document.source_type,
                "metadata": self.document.metadata
            },
            "config": self.config.to_dict(),
            "batches": [batch.to_dict() for batch in self.batches],
            "prompt": self.prompt,
            "model": self.model,
            "temperature": self.temperature,
            "context_state": self.context_state
        }

@dataclass
class BatchingResult:
    """Result of the batching operation"""
    document_id: str
    batches: List[PageBatch]
    config: SplitConfiguration
    total_pages: int
    total_batches: int
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def batch_distribution(self) -> Dict[str, int]:
        """Get distribution of batch sizes"""
        distribution = {}
        for batch in self.batches:
            size = batch.page_count
            distribution[f"{size}_pages"] = distribution.get(f"{size}_pages", 0) + 1
        return distribution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "total_pages": self.total_pages,
            "total_batches": self.total_batches,
            "batch_distribution": self.batch_distribution,
            "created_at": self.created_at.isoformat(),
            "config": self.config.to_dict(),
            "batches": [batch.to_dict() for batch in self.batches]
        }