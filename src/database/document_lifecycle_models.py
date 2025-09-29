"""MongoDB document models for complete document lifecycle tracking."""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class ProcessingStatus(Enum):
    """Processing status enumeration."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class StageStatus(Enum):
    """Stage processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class ProcessingMode(Enum):
    """Processing mode enumeration."""
    HUMAN_LOOP = "human_loop"
    BULK = "bulk"
    AUTO = "auto"


@dataclass
class DocumentRecord:
    """Master document registry record."""

    # Identifiers
    document_id: str
    session_id: Optional[str] = None

    # File Information
    original_filename: str = ""
    file_hash: Optional[str] = None
    file_size_bytes: int = 0

    # S3 Storage
    s3_pdf_url: Optional[str] = None
    s3_images_folder: Optional[str] = None

    # PDF Metadata
    pdf_metadata: Dict[str, Any] = field(default_factory=dict)

    # Processing Configuration
    processing_config: Dict[str, Any] = field(default_factory=dict)

    # Status Tracking
    current_status: ProcessingStatus = ProcessingStatus.UPLOADED
    processing_mode: ProcessingMode = ProcessingMode.AUTO

    # Relationships
    total_batches: int = 0
    total_chunks: int = 0
    processing_session_id: Optional[str] = None

    # Timestamps
    upload_timestamp: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Error Handling
    error_message: Optional[str] = None
    retry_count: int = 0

    # Tags and Categories
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None

    # Access Control
    uploaded_by_user: Optional[str] = None
    organization: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        data = asdict(self)

        # Convert enums to strings
        data['current_status'] = self.current_status.value
        data['processing_mode'] = self.processing_mode.value

        # Convert datetimes to ISO format
        for field_name in ['upload_timestamp', 'created_at', 'updated_at', 'completed_at']:
            if data[field_name]:
                data[field_name] = data[field_name].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentRecord':
        """Create from dictionary retrieved from MongoDB."""
        # Convert string enums back to enum objects
        if 'current_status' in data:
            data['current_status'] = ProcessingStatus(data['current_status'])
        if 'processing_mode' in data:
            data['processing_mode'] = ProcessingMode(data['processing_mode'])

        # Convert ISO strings back to datetime objects
        for field_name in ['upload_timestamp', 'created_at', 'updated_at', 'completed_at']:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])

        return cls(**data)


@dataclass
class ProcessingStageRecord:
    """Pipeline stage tracking record."""

    # Identifiers
    document_id: str
    stage_id: str

    # Stage Information
    stage_name: str  # upload, pdf_processing, splitting, lmm_processing, chunking, db_writing, completed
    stage_status: StageStatus = StageStatus.PENDING
    stage_order: int = 0

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Stage-Specific Data
    stage_data: Dict[str, Any] = field(default_factory=dict)

    # Progress Tracking
    progress_percentage: float = 0.0
    items_processed: int = 0
    items_total: int = 0

    # Error Handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    retry_attempts: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        data = asdict(self)

        # Convert enums to strings
        data['stage_status'] = self.stage_status.value

        # Convert datetimes to ISO format
        for field_name in ['started_at', 'completed_at', 'created_at', 'updated_at']:
            if data[field_name]:
                data[field_name] = data[field_name].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingStageRecord':
        """Create from dictionary retrieved from MongoDB."""
        # Convert string enums back to enum objects
        if 'stage_status' in data:
            data['stage_status'] = StageStatus(data['stage_status'])

        # Convert ISO strings back to datetime objects
        for field_name in ['started_at', 'completed_at', 'created_at', 'updated_at']:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])

        return cls(**data)


@dataclass
class PageImageInfo:
    """Information about a page image."""
    page_number: int
    s3_original_url: str
    s3_thumbnail_url: Optional[str] = None
    image_dimensions: Dict[str, int] = field(default_factory=dict)  # {width, height}
    file_size_bytes: int = 0


@dataclass
class BatchRecord:
    """Batch processing details record."""

    # Identifiers
    batch_id: str
    document_id: str

    # Batch Configuration
    batch_number: int
    page_range: Dict[str, int] = field(default_factory=dict)  # {start_page, end_page, total_pages}

    # Page Images
    page_images: List[PageImageInfo] = field(default_factory=list)

    # Processing Status
    status: StageStatus = StageStatus.PENDING
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    processing_duration_seconds: Optional[float] = None

    # LLM Processing Details
    llm_processing: Dict[str, Any] = field(default_factory=dict)

    # Chunks Generated
    chunks_generated: int = 0
    chunk_ids: List[str] = field(default_factory=list)

    # Context Management
    context_from_previous: Dict[str, Any] = field(default_factory=dict)
    context_for_next: Dict[str, Any] = field(default_factory=dict)

    # Quality Metrics
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    # Error Handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    retry_count: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        data = asdict(self)

        # Convert enums to strings
        data['status'] = self.status.value

        # Convert PageImageInfo objects to dicts
        data['page_images'] = [asdict(img) for img in self.page_images]

        # Convert datetimes to ISO format
        for field_name in ['processing_started_at', 'processing_completed_at', 'created_at', 'updated_at']:
            if data[field_name]:
                data[field_name] = data[field_name].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchRecord':
        """Create from dictionary retrieved from MongoDB."""
        # Convert string enums back to enum objects
        if 'status' in data:
            data['status'] = StageStatus(data['status'])

        # Convert page images back to PageImageInfo objects
        if 'page_images' in data:
            data['page_images'] = [PageImageInfo(**img) for img in data['page_images']]

        # Convert ISO strings back to datetime objects
        for field_name in ['processing_started_at', 'processing_completed_at', 'created_at', 'updated_at']:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])

        return cls(**data)


@dataclass
class EnhancedChunkRecord:
    """Enhanced chunk storage with complete traceability."""

    # Identifiers
    chunk_id: str
    document_id: str
    batch_id: str

    # Chunk Content (preserving current structure)
    content: str
    table: Optional[str] = None  # markdown table

    # Page Information
    start_page: int = 0
    end_page: int = 0
    source_images: List[str] = field(default_factory=list)

    # Hierarchical Structure
    level_1: Optional[str] = None
    level_2: Optional[str] = None
    level_3: Optional[str] = None

    # Continuation Flags
    continues_from_previous: bool = False
    continues_to_next: bool = False

    # Enhanced Metadata
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)

    # File References
    source_pdf: Dict[str, str] = field(default_factory=dict)  # {s3_url, filename}

    # Processing Information
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    # Search and Indexing
    search_keywords: List[str] = field(default_factory=list)
    semantic_tags: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Quality and Validation
    validation_status: str = "pending"
    quality_score: Optional[float] = None
    manual_review_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        data = asdict(self)

        # Convert datetimes to ISO format
        for field_name in ['created_at', 'updated_at']:
            if data[field_name]:
                data[field_name] = data[field_name].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedChunkRecord':
        """Create from dictionary retrieved from MongoDB."""
        # Convert ISO strings back to datetime objects
        for field_name in ['created_at', 'updated_at']:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])

        return cls(**data)


@dataclass
class ProcessingSession:
    """Processing session management record."""

    # Identifiers
    session_id: str

    # Session Information
    session_type: ProcessingMode = ProcessingMode.AUTO
    initiated_by: Optional[str] = None
    initiated_at: datetime = field(default_factory=datetime.utcnow)

    # Documents in Session
    documents: List[Dict[str, Any]] = field(default_factory=list)

    # Session Status
    status: ProcessingStatus = ProcessingStatus.PROCESSING
    completed_at: Optional[datetime] = None

    # Processing Summary
    summary: Dict[str, Any] = field(default_factory=dict)

    # Resource Usage
    resource_usage: Dict[str, Any] = field(default_factory=dict)

    # Configuration Used
    default_config: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        data = asdict(self)

        # Convert enums to strings
        data['session_type'] = self.session_type.value
        data['status'] = self.status.value

        # Convert datetimes to ISO format
        for field_name in ['initiated_at', 'completed_at', 'created_at', 'updated_at']:
            if data[field_name]:
                data[field_name] = data[field_name].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingSession':
        """Create from dictionary retrieved from MongoDB."""
        # Convert string enums back to enum objects
        if 'session_type' in data:
            data['session_type'] = ProcessingMode(data['session_type'])
        if 'status' in data:
            data['status'] = ProcessingStatus(data['status'])

        # Convert ISO strings back to datetime objects
        for field_name in ['initiated_at', 'completed_at', 'created_at', 'updated_at']:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])

        return cls(**data)


def generate_document_id(filename: str, timestamp: Optional[datetime] = None) -> str:
    """Generate unique document ID with timestamp and hash."""
    import hashlib

    if timestamp is None:
        timestamp = datetime.utcnow()

    # Create hash from filename and timestamp
    content = f"{filename}_{timestamp.isoformat()}"
    hash_obj = hashlib.sha256(content.encode())
    hash_short = hash_obj.hexdigest()[:8]

    # Generate sequential number (simplified)
    seq_num = timestamp.hour * 100 + timestamp.minute

    return f"doc_{timestamp.strftime('%Y_%m%d')}_{seq_num:03d}_{hash_short}"


def generate_batch_id(document_id: str, batch_number: int) -> str:
    """Generate unique batch ID."""
    import hashlib

    content = f"{document_id}_{batch_number}_{datetime.utcnow().isoformat()}"
    hash_obj = hashlib.sha256(content.encode())
    hash_short = hash_obj.hexdigest()[:8]

    return f"batch_{document_id}_{batch_number:03d}_{hash_short}"


def generate_chunk_id(batch_id: str, chunk_number: int) -> str:
    """Generate unique chunk ID."""
    return f"chunk_{batch_id}_{chunk_number:03d}"


def generate_stage_id(document_id: str, stage_name: str) -> str:
    """Generate unique stage ID."""
    import hashlib

    content = f"{document_id}_{stage_name}_{datetime.utcnow().isoformat()}"
    hash_obj = hashlib.sha256(content.encode())
    hash_short = hash_obj.hexdigest()[:8]

    return f"stage_{stage_name}_{hash_short}"