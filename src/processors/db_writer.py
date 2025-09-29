"""Database writer stage for the processing pipeline."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from ..models.batch_models import PageBatch, ProcessingTask
from ..models.chunk_schema import BatchProcessingResult
from .chunk_manager import ChunkManager

logger = logging.getLogger(__name__)


class DatabaseWriter:
    """
    Database writer stage for storing processed chunks.

    This stage is responsible for:
    - Storing chunks to MongoDB after LMM processing
    - Managing document and batch metadata
    - Providing feedback on storage success/failure
    - Maintaining data integrity
    """

    def __init__(self, mongodb_uri: Optional[str] = None, collection_name: Optional[str] = None):
        """
        Initialize DatabaseWriter.

        Args:
            mongodb_uri: MongoDB connection string (defaults to env var)
            collection_name: Collection name (defaults to env var)
        """
        try:
            self.chunk_manager = ChunkManager(mongodb_uri, collection_name)
            self.is_connected = True
            logger.info("DatabaseWriter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseWriter: {e}")
            self.is_connected = False
            raise

    def write_batch(self,
                   task: ProcessingTask,
                   batch: PageBatch,
                   batch_result: BatchProcessingResult) -> Dict[str, Any]:
        """
        Write a processed batch to the database.

        Args:
            task: Processing task containing document information
            batch: Batch that was processed
            batch_result: Result from LMM processing

        Returns:
            Dictionary with write results and metadata
        """
        if not self.is_connected:
            raise RuntimeError("DatabaseWriter not connected to MongoDB")

        try:
            # Generate document ID from task
            document_id = self._generate_document_id(task)

            # Prepare metadata for storage
            metadata = {
                "task_id": task.task_id,
                "document_path": task.document.file_path,
                "source_type": task.document.source_type,
                "batch_number": batch.batch_number,
                "total_pages": batch.total_pages,
                "processing_model": task.model,
                "processing_temperature": task.temperature,
                "processing_prompt": task.prompt,
                "processed_at": batch.processed_at.isoformat() if batch.processed_at else None,
                "processing_time": batch.processing_time,
                "document_metadata": task.document.metadata,
            }

            # Save chunks to database
            start_time = datetime.utcnow()
            chunk_ids = self.chunk_manager.save_batch_chunks(
                document_id=document_id,
                batch_id=batch.batch_id,
                batch_result=batch_result,
                metadata=metadata
            )
            write_time = (datetime.utcnow() - start_time).total_seconds()

            # Store write information in batch for pipeline tracking
            batch.db_write_result = {
                "success": True,
                "chunk_ids": chunk_ids,
                "chunks_saved": len(chunk_ids),
                "document_id": document_id,
                "write_time": write_time,
                "written_at": datetime.utcnow().isoformat(),
            }

            logger.info(f"Successfully wrote {len(chunk_ids)} chunks for batch {batch.batch_id}")

            return {
                "success": True,
                "document_id": document_id,
                "batch_id": batch.batch_id,
                "chunk_ids": chunk_ids,
                "chunks_saved": len(chunk_ids),
                "write_time": write_time,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Failed to write batch {batch.batch_id}: {e}")

            # Store error information in batch
            batch.db_write_result = {
                "success": False,
                "error": str(e),
                "written_at": datetime.utcnow().isoformat(),
            }

            return {
                "success": False,
                "error": str(e),
                "batch_id": batch.batch_id,
                "document_id": self._generate_document_id(task),
            }

    def write_task_chunks(self, task: ProcessingTask) -> Dict[str, Any]:
        """
        Write all processed chunks from a task to the database.

        Args:
            task: Completed processing task

        Returns:
            Summary of write operations
        """
        if not self.is_connected:
            raise RuntimeError("DatabaseWriter not connected to MongoDB")

        results = {
            "success": True,
            "document_id": self._generate_document_id(task),
            "total_batches": len(task.batches),
            "successful_writes": 0,
            "failed_writes": 0,
            "total_chunks": 0,
            "batch_results": [],
            "errors": [],
        }

        for batch in task.batches:
            # Skip batches that haven't been processed or don't have chunk data
            if not hasattr(batch, 'chunk_summary') or not batch.chunk_summary:
                logger.warning(f"Batch {batch.batch_id} has no chunk data, skipping")
                continue

            try:
                # Reconstruct BatchProcessingResult from stored data
                batch_result = self._reconstruct_batch_result(batch)

                # Write batch to database
                write_result = self.write_batch(task, batch, batch_result)

                if write_result["success"]:
                    results["successful_writes"] += 1
                    results["total_chunks"] += write_result["chunks_saved"]
                else:
                    results["failed_writes"] += 1
                    results["errors"].append({
                        "batch_id": batch.batch_id,
                        "error": write_result["error"]
                    })

                results["batch_results"].append(write_result)

            except Exception as e:
                logger.error(f"Error processing batch {batch.batch_id}: {e}")
                results["failed_writes"] += 1
                results["errors"].append({
                    "batch_id": batch.batch_id,
                    "error": str(e)
                })

        # Overall success depends on having at least one successful write and no failures
        results["success"] = results["successful_writes"] > 0 and results["failed_writes"] == 0

        if not results["success"]:
            logger.error(f"Task {task.task_id} had {results['failed_writes']} failed writes")

        logger.info(f"Task {task.task_id}: {results['successful_writes']}/{results['total_batches']} batches written successfully")

        return results

    def _generate_document_id(self, task: ProcessingTask) -> str:
        """
        Generate a consistent document ID for a task.

        Args:
            task: Processing task

        Returns:
            Document ID string
        """
        # Use task ID as document ID to ensure uniqueness and consistency
        return f"{task.task_id}_{task.document.source_type}"

    def _reconstruct_batch_result(self, batch: PageBatch) -> BatchProcessingResult:
        """
        Reconstruct BatchProcessingResult from stored batch data.

        Args:
            batch: Batch with stored processing results

        Returns:
            Reconstructed BatchProcessingResult
        """
        from ..models.chunk_schema import ChunkOutput

        # Reconstruct chunks from stored data
        chunks = []
        if hasattr(batch, 'chunk_summary') and batch.chunk_summary:
            for chunk_data in batch.chunk_summary:
                chunk = ChunkOutput.from_dict(chunk_data)
                chunks.append(chunk)

        # Get context snapshot data
        context_snapshot = getattr(batch, 'context_snapshot', {})

        # Create BatchProcessingResult
        batch_result = BatchProcessingResult(
            chunks=chunks,
            raw_output=getattr(batch, 'lmm_output', ''),
            last_chunk=chunks[-1] if chunks else None,
            continuation_context=context_snapshot.get('continuation_context'),
            processing_metadata=context_snapshot.get('processing_metadata', {})
        )

        return batch_result

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a stored document.

        Args:
            document_id: Document identifier

        Returns:
            Document information or None if not found
        """
        if not self.is_connected:
            return None

        try:
            stats = self.chunk_manager.get_document_statistics(document_id)
            chunks = self.chunk_manager.get_document_chunks(document_id)

            if not chunks:
                return None

            # Extract metadata from first chunk (should be consistent across all chunks)
            first_chunk = chunks[0]
            metadata = first_chunk.get('metadata', {})

            return {
                "document_id": document_id,
                "statistics": stats,
                "metadata": metadata,
                "first_chunk_created": first_chunk.get('created_at'),
                "total_chunks": len(chunks),
            }

        except Exception as e:
            logger.error(f"Failed to get document info for {document_id}: {e}")
            return None

    def search_document_chunks(self,
                              document_id: str,
                              query: str,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search chunks within a specific document.

        Args:
            document_id: Document identifier
            query: Search query
            limit: Maximum results

        Returns:
            List of matching chunks
        """
        if not self.is_connected:
            return []

        try:
            return self.chunk_manager.search_chunks(query, document_id, limit)
        except Exception as e:
            logger.error(f"Failed to search chunks in document {document_id}: {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document identifier

        Returns:
            True if successful
        """
        if not self.is_connected:
            return False

        try:
            deleted_count = self.chunk_manager.delete_document_chunks(document_id)
            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get database connection status.

        Returns:
            Connection status information
        """
        return {
            "connected": self.is_connected,
            "collection_name": getattr(self.chunk_manager, 'collection_name', None) if self.is_connected else None,
            "mongodb_uri_configured": bool(getattr(self.chunk_manager, 'mongodb_uri', None)) if self.is_connected else False,
        }

    def close(self):
        """Close database connections."""
        if hasattr(self, 'chunk_manager'):
            self.chunk_manager.close()
            logger.info("DatabaseWriter closed")