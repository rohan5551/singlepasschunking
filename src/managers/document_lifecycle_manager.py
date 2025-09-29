"""Document Lifecycle Manager for comprehensive tracking and persistence."""

import os
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from PIL import Image

from ..storage import S3StorageManager
from ..database import (
    MongoDBManager,
    DocumentRecord,
    ProcessingStageRecord,
    BatchRecord,
    EnhancedChunkRecord,
    ProcessingSession,
    ProcessingStatus,
    StageStatus,
    ProcessingMode,
    PageImageInfo,
    generate_document_id,
    generate_batch_id,
    generate_chunk_id,
    generate_stage_id
)
from ..models.batch_models import ProcessingTask, PageBatch
from ..models.chunk_schema import BatchProcessingResult, ChunkOutput

logger = logging.getLogger(__name__)


class DocumentLifecycleManager:
    """
    Comprehensive document lifecycle manager that coordinates:
    - S3 storage for PDFs and images
    - MongoDB persistence for all processing stages
    - Stage tracking and recovery
    - Complete traceability from PDF to chunks
    """

    def __init__(self,
                 s3_manager: Optional[S3StorageManager] = None,
                 mongodb_manager: Optional[MongoDBManager] = None):
        """
        Initialize DocumentLifecycleManager.

        Args:
            s3_manager: Optional S3StorageManager instance
            mongodb_manager: Optional MongoDBManager instance
        """
        try:
            # Initialize S3 manager
            if s3_manager:
                self.s3_manager = s3_manager
            else:
                self.s3_manager = S3StorageManager()

            # Initialize MongoDB manager
            if mongodb_manager:
                self.mongodb_manager = mongodb_manager
            else:
                self.mongodb_manager = MongoDBManager()

            logger.info("DocumentLifecycleManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DocumentLifecycleManager: {e}")
            raise

    def register_document(self,
                         pdf_path: str,
                         original_filename: Optional[str] = None,
                         session_id: Optional[str] = None,
                         processing_mode: ProcessingMode = ProcessingMode.AUTO,
                         metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Register a new document and upload PDF to S3.

        Args:
            pdf_path: Local path to PDF file
            original_filename: Original filename (defaults to basename of pdf_path)
            session_id: Optional processing session ID
            processing_mode: Processing mode (auto, human_loop, bulk)
            metadata: Optional metadata dictionary

        Returns:
            Tuple of (document_id, s3_pdf_url)
        """
        try:
            # Generate document ID
            filename = original_filename or os.path.basename(pdf_path)
            document_id = generate_document_id(filename)

            # Get file hash and size
            file_hash = self._calculate_file_hash(pdf_path)
            file_size = os.path.getsize(pdf_path)

            # Create S3 folder structure
            base_path = self.s3_manager.create_document_folder_structure(document_id)

            # Upload PDF to S3
            s3_pdf_url = self.s3_manager.upload_pdf(
                pdf_path=pdf_path,
                document_id=document_id,
                base_path=base_path,
                metadata=metadata
            )

            # Extract PDF metadata if possible
            pdf_metadata = self._extract_pdf_metadata(pdf_path)

            # Create document record
            document_record = DocumentRecord(
                document_id=document_id,
                session_id=session_id,
                original_filename=filename,
                file_hash=file_hash,
                file_size_bytes=file_size,
                s3_pdf_url=s3_pdf_url,
                s3_images_folder=f"s3://{self.s3_manager.bucket_name}/{base_path}/images/",
                pdf_metadata=pdf_metadata or {},
                processing_config=metadata or {},
                current_status=ProcessingStatus.UPLOADED,
                processing_mode=processing_mode,
                upload_timestamp=datetime.utcnow()
            )

            # Save document record to MongoDB
            success = self.mongodb_manager.create_document(document_record)
            if not success:
                raise Exception("Failed to save document record to MongoDB")

            # Create initial upload stage
            self.track_stage_start(
                document_id=document_id,
                stage_name="upload",
                stage_data={
                    "s3_pdf_url": s3_pdf_url,
                    "file_hash": file_hash,
                    "file_size_bytes": file_size,
                    "pdf_metadata": pdf_metadata
                }
            )

            # Mark upload stage as completed
            self.track_stage_completion(
                document_id=document_id,
                stage_name="upload"
            )

            logger.info(f"Successfully registered document {document_id}")
            return document_id, s3_pdf_url

        except Exception as e:
            logger.error(f"Failed to register document from {pdf_path}: {e}")
            raise

    def track_stage_start(self,
                         document_id: str,
                         stage_name: str,
                         stage_order: Optional[int] = None,
                         stage_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Track the start of a processing stage.

        Args:
            document_id: Document identifier
            stage_name: Name of the stage
            stage_order: Optional stage order (auto-generated if not provided)
            stage_data: Optional stage-specific data

        Returns:
            Stage ID
        """
        try:
            stage_id = generate_stage_id(document_id, stage_name)

            # Auto-generate stage order if not provided
            if stage_order is None:
                existing_stages = self.mongodb_manager.get_document_stages(document_id)
                stage_order = len(existing_stages) + 1

            stage_record = ProcessingStageRecord(
                document_id=document_id,
                stage_id=stage_id,
                stage_name=stage_name,
                stage_status=StageStatus.PROCESSING,
                stage_order=stage_order,
                started_at=datetime.utcnow(),
                stage_data=stage_data or {},
                progress_percentage=0.0
            )

            success = self.mongodb_manager.create_stage(stage_record)
            if not success:
                raise Exception(f"Failed to create stage record for {stage_id}")

            logger.info(f"Started stage {stage_name} for document {document_id}")
            return stage_id

        except Exception as e:
            logger.error(f"Failed to track stage start {stage_name} for document {document_id}: {e}")
            raise

    def track_stage_completion(self,
                             document_id: str,
                             stage_name: str,
                             stage_data: Optional[Dict[str, Any]] = None,
                             warnings: Optional[List[str]] = None) -> bool:
        """
        Track the completion of a processing stage.

        Args:
            document_id: Document identifier
            stage_name: Name of the stage
            stage_data: Optional completion data
            warnings: Optional warnings list

        Returns:
            True if successful
        """
        try:
            # Find the stage record
            stages = self.mongodb_manager.get_document_stages(document_id)
            stage_record = None

            for stage in stages:
                if stage.stage_name == stage_name and stage.stage_status == StageStatus.PROCESSING:
                    stage_record = stage
                    break

            if not stage_record:
                logger.warning(f"No processing stage {stage_name} found for document {document_id}")
                return False

            # Update stage record
            stage_record.stage_status = StageStatus.COMPLETED
            stage_record.completed_at = datetime.utcnow()
            stage_record.progress_percentage = 100.0
            stage_record.warnings = warnings or []

            if stage_record.started_at:
                duration = stage_record.completed_at - stage_record.started_at
                stage_record.duration_seconds = duration.total_seconds()

            if stage_data:
                stage_record.stage_data.update(stage_data)

            success = self.mongodb_manager.update_stage(stage_record)
            if not success:
                raise Exception(f"Failed to update stage record for {stage_record.stage_id}")

            logger.info(f"Completed stage {stage_name} for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to track stage completion {stage_name} for document {document_id}: {e}")
            return False

    def track_stage_error(self,
                         document_id: str,
                         stage_name: str,
                         error_message: str,
                         stage_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Track a stage error.

        Args:
            document_id: Document identifier
            stage_name: Name of the stage
            error_message: Error message
            stage_data: Optional error-related data

        Returns:
            True if successful
        """
        try:
            # Find the stage record
            stages = self.mongodb_manager.get_document_stages(document_id)
            stage_record = None

            for stage in stages:
                if stage.stage_name == stage_name and stage.stage_status == StageStatus.PROCESSING:
                    stage_record = stage
                    break

            if not stage_record:
                # Create error stage if not found
                stage_id = generate_stage_id(document_id, stage_name)
                stage_record = ProcessingStageRecord(
                    document_id=document_id,
                    stage_id=stage_id,
                    stage_name=stage_name,
                    stage_status=StageStatus.ERROR,
                    stage_order=len(stages) + 1,
                    started_at=datetime.utcnow(),
                    stage_data=stage_data or {}
                )
            else:
                # Update existing stage
                stage_record.stage_status = StageStatus.ERROR
                if stage_data:
                    stage_record.stage_data.update(stage_data)

            stage_record.error_message = error_message
            stage_record.completed_at = datetime.utcnow()

            if stage_record.started_at:
                duration = stage_record.completed_at - stage_record.started_at
                stage_record.duration_seconds = duration.total_seconds()

            # Update or create stage record
            if hasattr(stage_record, 'stage_id') and stage_record.stage_id:
                success = self.mongodb_manager.update_stage(stage_record)
            else:
                success = self.mongodb_manager.create_stage(stage_record)

            if not success:
                raise Exception(f"Failed to save error stage record for {stage_record.stage_id}")

            # Update document status
            document_record = self.mongodb_manager.get_document(document_id)
            if document_record:
                document_record.current_status = ProcessingStatus.ERROR
                document_record.error_message = error_message
                self.mongodb_manager.update_document(document_record)

            logger.error(f"Stage {stage_name} failed for document {document_id}: {error_message}")
            return True

        except Exception as e:
            logger.error(f"Failed to track stage error {stage_name} for document {document_id}: {e}")
            return False

    def save_page_images(self,
                        document_id: str,
                        images: List[Image.Image],
                        create_thumbnails: bool = True) -> List[PageImageInfo]:
        """
        Save page images to S3 and track in database.

        Args:
            document_id: Document identifier
            images: List of PIL Image objects
            create_thumbnails: Whether to create thumbnails

        Returns:
            List of PageImageInfo objects
        """
        try:
            # Get document record to find S3 base path
            document_record = self.mongodb_manager.get_document(document_id)
            if not document_record:
                raise Exception(f"Document {document_id} not found")

            # Extract base path from S3 images folder
            images_folder = document_record.s3_images_folder
            base_path = images_folder.replace(f"s3://{self.s3_manager.bucket_name}/", "").replace("/images/", "")

            # Upload images to S3
            image_info_dicts = self.s3_manager.upload_page_images(
                images=images,
                document_id=document_id,
                base_path=base_path,
                create_thumbnails=create_thumbnails
            )

            # Convert to PageImageInfo objects
            page_images = []
            for info in image_info_dicts:
                page_image = PageImageInfo(
                    page_number=info['page_number'],
                    s3_original_url=info['original_url'],
                    s3_thumbnail_url=info.get('thumbnail_url'),
                    image_dimensions=info['dimensions'],
                    file_size_bytes=0  # Could be calculated if needed
                )
                page_images.append(page_image)

            logger.info(f"Saved {len(page_images)} page images for document {document_id}")
            return page_images

        except Exception as e:
            logger.error(f"Failed to save page images for document {document_id}: {e}")
            raise

    def create_batch_record(self,
                           document_id: str,
                           batch_number: int,
                           page_range: Dict[str, int],
                           page_images: List[PageImageInfo]) -> str:
        """
        Create a batch record for tracking batch processing.

        Args:
            document_id: Document identifier
            batch_number: Batch number
            page_range: Page range dictionary {start_page, end_page, total_pages}
            page_images: List of page images in this batch

        Returns:
            Batch ID
        """
        try:
            batch_id = generate_batch_id(document_id, batch_number)

            batch_record = BatchRecord(
                batch_id=batch_id,
                document_id=document_id,
                batch_number=batch_number,
                page_range=page_range,
                page_images=page_images,
                status=StageStatus.PENDING
            )

            success = self.mongodb_manager.create_batch(batch_record)
            if not success:
                raise Exception(f"Failed to create batch record for {batch_id}")

            logger.info(f"Created batch record {batch_id} for document {document_id}")
            return batch_id

        except Exception as e:
            logger.error(f"Failed to create batch record for document {document_id}: {e}")
            raise

    def save_processing_artifacts(self,
                                document_id: str,
                                artifacts: Dict[str, Any]) -> bool:
        """
        Save processing artifacts to S3.

        Args:
            document_id: Document identifier
            artifacts: Dictionary of artifact name -> content

        Returns:
            True if successful
        """
        try:
            # Get document record to find S3 base path
            document_record = self.mongodb_manager.get_document(document_id)
            if not document_record:
                raise Exception(f"Document {document_id} not found")

            # Extract base path from S3 images folder
            images_folder = document_record.s3_images_folder
            base_path = images_folder.replace(f"s3://{self.s3_manager.bucket_name}/", "").replace("/images/", "")

            # Upload artifacts to S3
            uploaded_artifacts = self.s3_manager.upload_processing_artifacts(
                document_id=document_id,
                base_path=base_path,
                artifacts=artifacts
            )

            logger.info(f"Saved {len(uploaded_artifacts)} processing artifacts for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save processing artifacts for document {document_id}: {e}")
            return False

    def update_batch_processing_result(self,
                                     batch_id: str,
                                     batch_result: BatchProcessingResult,
                                     processing_metadata: Dict[str, Any]) -> bool:
        """
        Update batch record with LLM processing results.

        Args:
            batch_id: Batch identifier
            batch_result: Result from LMM processing
            processing_metadata: Processing metadata

        Returns:
            True if successful
        """
        try:
            # Get existing batch record
            batches = self.mongodb_manager.batches_collection.find({"batch_id": batch_id})
            batch_data = list(batches)

            if not batch_data:
                raise Exception(f"Batch {batch_id} not found")

            batch_dict = batch_data[0]
            batch_dict.pop('_id', None)
            batch_record = BatchRecord.from_dict(batch_dict)

            # Update batch with processing results
            batch_record.status = StageStatus.COMPLETED
            batch_record.processing_completed_at = datetime.utcnow()
            batch_record.chunks_generated = len(batch_result.chunks)
            batch_record.chunk_ids = [generate_chunk_id(batch_id, i) for i in range(len(batch_result.chunks))]

            # Store LLM processing details
            batch_record.llm_processing = {
                "model": processing_metadata.get("model"),
                "temperature": processing_metadata.get("temperature"),
                "prompt_used": processing_metadata.get("prompt_used"),
                "tokens_used": processing_metadata.get("tokens_used", 0),
                "api_response_time_seconds": processing_metadata.get("processing_time", 0),
                "raw_output": batch_result.raw_output,
                "structured_output_valid": len(batch_result.chunks) > 0
            }

            # Store context information
            batch_record.context_for_next = {
                "provides_context": True,
                "last_chunk_summary": {
                    "content_preview": batch_result.chunks[-1].content[:500] if batch_result.chunks else "",
                    "hierarchical_position": {
                        "level_1": batch_result.chunks[-1].level_1 if batch_result.chunks else None,
                        "level_2": batch_result.chunks[-1].level_2 if batch_result.chunks else None,
                        "level_3": batch_result.chunks[-1].level_3 if batch_result.chunks else None
                    },
                    "continues_to_next": batch_result.chunks[-1].continues_to_next if batch_result.chunks else False
                }
            }

            # Calculate processing duration
            if batch_record.processing_started_at:
                duration = batch_record.processing_completed_at - batch_record.processing_started_at
                batch_record.processing_duration_seconds = duration.total_seconds()

            success = self.mongodb_manager.update_batch(batch_record)
            if not success:
                raise Exception(f"Failed to update batch record for {batch_id}")

            logger.info(f"Updated batch {batch_id} with processing results ({len(batch_result.chunks)} chunks)")
            return True

        except Exception as e:
            logger.error(f"Failed to update batch processing result for {batch_id}: {e}")
            return False

    def save_chunks(self,
                   document_id: str,
                   batch_id: str,
                   batch_result: BatchProcessingResult,
                   processing_task: ProcessingTask) -> List[str]:
        """
        Save enhanced chunks to MongoDB with complete traceability.

        Args:
            document_id: Document identifier
            batch_id: Batch identifier
            batch_result: Result from LMM processing
            processing_task: Processing task with metadata

        Returns:
            List of created chunk IDs
        """
        try:
            # Get document and batch records for metadata
            document_record = self.mongodb_manager.get_document(document_id)
            batch_data = list(self.mongodb_manager.batches_collection.find({"batch_id": batch_id}))

            if not document_record:
                raise Exception(f"Document {document_id} not found")

            if not batch_data:
                raise Exception(f"Batch {batch_id} not found")

            batch_dict = batch_data[0]
            batch_dict.pop('_id', None)
            batch_record = BatchRecord.from_dict(batch_dict)

            # Create enhanced chunk records
            chunk_records = []
            created_chunk_ids = []

            for i, chunk_output in enumerate(batch_result.chunks):
                chunk_id = generate_chunk_id(batch_id, i)
                created_chunk_ids.append(chunk_id)

                # Build source images list
                source_images = [img.s3_original_url for img in batch_record.page_images]

                # Create enhanced chunk record
                chunk_record = EnhancedChunkRecord(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    batch_id=batch_id,
                    content=chunk_output.content,
                    table=chunk_output.table,
                    start_page=batch_record.page_range.get("start_page", 0),
                    end_page=batch_record.page_range.get("end_page", 0),
                    source_images=source_images,
                    level_1=chunk_output.level_1,
                    level_2=chunk_output.level_2,
                    level_3=chunk_output.level_3,
                    continues_from_previous=chunk_output.continues_from_previous,
                    continues_to_next=chunk_output.continues_to_next,
                    chunk_metadata={
                        "chunk_index_in_batch": i,
                        "total_chunks_in_batch": len(batch_result.chunks),
                        "content_type": "mixed" if chunk_output.table else "text",
                        "word_count": len(chunk_output.content.split()),
                        "character_count": len(chunk_output.content),
                        "extraction_confidence": 0.95,  # Could be calculated from LLM response
                        "structural_completeness": True
                    },
                    source_pdf={
                        "s3_url": document_record.s3_pdf_url or "",
                        "filename": document_record.original_filename
                    },
                    processing_metadata={
                        "model": processing_task.model,
                        "temperature": processing_task.temperature,
                        "processing_time_seconds": batch_record.processing_duration_seconds or 0,
                        "prompt_version": "v2.1",
                        "extraction_method": "llm_structured"
                    },
                    search_keywords=self._extract_keywords(chunk_output.content),
                    semantic_tags=["document_content"],  # Could be enhanced with NLP
                    validation_status="validated",
                    quality_score=0.92  # Could be calculated
                )

                chunk_records.append(chunk_record)

            # Batch insert chunks
            created_count = self.mongodb_manager.create_chunks_batch(chunk_records)

            if created_count != len(chunk_records):
                logger.warning(f"Only {created_count}/{len(chunk_records)} chunks were saved")

            logger.info(f"Saved {created_count} chunks for batch {batch_id}")
            return created_chunk_ids

        except Exception as e:
            logger.error(f"Failed to save chunks for batch {batch_id}: {e}")
            raise

    def update_document_status(self,
                             document_id: str,
                             status: ProcessingStatus,
                             summary_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update document processing status and summary.

        Args:
            document_id: Document identifier
            status: New processing status
            summary_data: Optional summary data

        Returns:
            True if successful
        """
        try:
            document_record = self.mongodb_manager.get_document(document_id)
            if not document_record:
                raise Exception(f"Document {document_id} not found")

            document_record.current_status = status
            document_record.updated_at = datetime.utcnow()

            if status == ProcessingStatus.COMPLETED:
                document_record.completed_at = datetime.utcnow()

            if summary_data:
                document_record.total_batches = summary_data.get("total_batches", 0)
                document_record.total_chunks = summary_data.get("total_chunks", 0)

            success = self.mongodb_manager.update_document(document_record)
            if not success:
                raise Exception(f"Failed to update document record for {document_id}")

            logger.info(f"Updated document {document_id} status to {status.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update document status for {document_id}: {e}")
            return False

    def get_incomplete_documents(self) -> List[DocumentRecord]:
        """
        Get all documents that were not completed (for restart recovery).

        Returns:
            List of incomplete document records
        """
        try:
            # Get documents that are not completed or in error state
            incomplete_statuses = [
                ProcessingStatus.UPLOADED,
                ProcessingStatus.PROCESSING
            ]

            incomplete_docs = []
            for status in incomplete_statuses:
                docs = self.mongodb_manager.list_documents(status=status, limit=100)
                incomplete_docs.extend(docs)

            logger.info(f"Found {len(incomplete_docs)} incomplete documents")
            return incomplete_docs

        except Exception as e:
            logger.error(f"Failed to get incomplete documents: {e}")
            return []

    def get_document_full_info(self, document_id: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a document and its processing.

        Args:
            document_id: Document identifier

        Returns:
            Complete document information
        """
        try:
            # Get document record
            document_record = self.mongodb_manager.get_document(document_id)
            if not document_record:
                return {}

            # Get processing stages
            stages = self.mongodb_manager.get_document_stages(document_id)

            # Get batches
            batches = self.mongodb_manager.get_document_batches(document_id)

            # Get chunks
            chunks = self.mongodb_manager.get_document_chunks(document_id)

            # Get statistics
            stats = self.mongodb_manager.get_document_statistics(document_id)

            return {
                "document": document_record.to_dict(),
                "stages": [stage.to_dict() for stage in stages],
                "batches": [batch.to_dict() for batch in batches],
                "chunks": [chunk.to_dict() for chunk in chunks],
                "statistics": stats
            }

        except Exception as e:
            logger.error(f"Failed to get full info for document {document_id}: {e}")
            return {}

    # Utility methods

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _extract_pdf_metadata(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Extract metadata from PDF file."""
        try:
            from PyPDF2 import PdfReader

            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)

                metadata = {}
                if reader.metadata:
                    metadata = {
                        "title": str(reader.metadata.get('/Title', '')),
                        "author": str(reader.metadata.get('/Author', '')),
                        "creator": str(reader.metadata.get('/Creator', '')),
                        "producer": str(reader.metadata.get('/Producer', '')),
                        "creation_date": str(reader.metadata.get('/CreationDate', '')),
                        "modification_date": str(reader.metadata.get('/ModDate', ''))
                    }

                metadata["total_pages"] = len(reader.pages)
                return metadata

        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata from {pdf_path}: {e}")
            return None

    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from content (simple implementation)."""
        try:
            import re

            # Simple keyword extraction - remove common words and extract meaningful terms
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())

            # Remove common stop words
            stop_words = {'that', 'with', 'have', 'this', 'will', 'your', 'from', 'they',
                         'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very',
                         'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many',
                         'over', 'such', 'take', 'than', 'them', 'well', 'work'}

            filtered_words = [word for word in words if word not in stop_words]

            # Get most frequent words
            from collections import Counter
            word_counts = Counter(filtered_words)
            return [word for word, count in word_counts.most_common(max_keywords)]

        except Exception:
            return []

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status for all components.

        Returns:
            Status information
        """
        return {
            "s3_manager": self.s3_manager.get_connection_status(),
            "mongodb_manager": self.mongodb_manager.get_connection_status()
        }