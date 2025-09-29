"""MongoDB manager for document lifecycle tracking."""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv

from .document_lifecycle_models import (
    DocumentRecord,
    ProcessingStageRecord,
    BatchRecord,
    EnhancedChunkRecord,
    ProcessingSession,
    ProcessingStatus,
    StageStatus
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MongoDBManager:
    """
    MongoDB manager for complete document lifecycle tracking.

    Manages 5 collections:
    - documents: Master document registry
    - processing_stages: Stage tracking for each document
    - batches: Batch processing details
    - chunks: Enhanced chunk storage
    - processing_sessions: Session management
    """

    def __init__(self, mongodb_uri: Optional[str] = None, database_name: Optional[str] = None):
        """
        Initialize MongoDB manager.

        Args:
            mongodb_uri: MongoDB connection string (defaults to env var)
            database_name: Database name (extracted from URI or defaults)
        """
        self.mongodb_uri = mongodb_uri or os.getenv('MONGODB_URI')
        if not self.mongodb_uri:
            raise ValueError("MONGODB_URI must be provided or set as environment variable")

        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)

            # Extract database name from URI or use provided name
            if database_name:
                self.db_name = database_name
            else:
                # Extract from URI (format: mongodb://user:pass@host:port/database?options)
                db_part = self.mongodb_uri.split('/')[-1].split('?')[0]
                self.db_name = db_part if db_part else 'document_processing'

            self.db = self.client[self.db_name]

            # Test connection
            self.client.admin.command('ping')

            # Initialize collections
            self.documents_collection = self.db['documents']
            self.stages_collection = self.db['processing_stages']
            self.batches_collection = self.db['batches']
            self.chunks_collection = self.db['chunks']
            self.sessions_collection = self.db['processing_sessions']

            # Create indexes
            self._create_indexes()

            logger.info(f"MongoDBManager initialized with database: {self.db_name}")

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MongoDBManager: {e}")
            raise

    def _create_indexes(self):
        """Create necessary indexes for efficient querying."""
        try:
            # Documents collection indexes
            self.documents_collection.create_index([("document_id", ASCENDING)], unique=True)
            self.documents_collection.create_index([("session_id", ASCENDING)])
            self.documents_collection.create_index([("current_status", ASCENDING)])
            self.documents_collection.create_index([("created_at", DESCENDING)])
            self.documents_collection.create_index([("original_filename", ASCENDING)])

            # Processing stages collection indexes
            self.stages_collection.create_index([("document_id", ASCENDING)])
            self.stages_collection.create_index([("stage_id", ASCENDING)], unique=True)
            self.stages_collection.create_index([("document_id", ASCENDING), ("stage_order", ASCENDING)])
            self.stages_collection.create_index([("stage_status", ASCENDING)])

            # Batches collection indexes
            self.batches_collection.create_index([("batch_id", ASCENDING)], unique=True)
            self.batches_collection.create_index([("document_id", ASCENDING)])
            self.batches_collection.create_index([("document_id", ASCENDING), ("batch_number", ASCENDING)])
            self.batches_collection.create_index([("status", ASCENDING)])

            # Chunks collection indexes
            self.chunks_collection.create_index([("chunk_id", ASCENDING)], unique=True)
            self.chunks_collection.create_index([("document_id", ASCENDING)])
            self.chunks_collection.create_index([("batch_id", ASCENDING)])
            self.chunks_collection.create_index([("document_id", ASCENDING), ("start_page", ASCENDING)])
            self.chunks_collection.create_index([("search_keywords", ASCENDING)])
            self.chunks_collection.create_index([("created_at", DESCENDING)])

            # Processing sessions collection indexes
            self.sessions_collection.create_index([("session_id", ASCENDING)], unique=True)
            self.sessions_collection.create_index([("status", ASCENDING)])
            self.sessions_collection.create_index([("created_at", DESCENDING)])

            # Text indexes for search
            self.chunks_collection.create_index([("content", "text"), ("search_keywords", "text")])

            logger.info("MongoDB indexes created successfully")

        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")

    # Document Management Methods

    def create_document(self, document_record: DocumentRecord) -> bool:
        """
        Create a new document record.

        Args:
            document_record: Document record to create

        Returns:
            True if successful
        """
        try:
            document_record.created_at = datetime.utcnow()
            document_record.updated_at = datetime.utcnow()

            self.documents_collection.insert_one(document_record.to_dict())
            logger.info(f"Created document record: {document_record.document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create document {document_record.document_id}: {e}")
            return False

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """
        Get document record by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document record or None if not found
        """
        try:
            doc_data = self.documents_collection.find_one({"document_id": document_id})
            if doc_data:
                # Remove MongoDB _id field
                doc_data.pop('_id', None)
                return DocumentRecord.from_dict(doc_data)
            return None

        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None

    def update_document(self, document_record: DocumentRecord) -> bool:
        """
        Update document record.

        Args:
            document_record: Updated document record

        Returns:
            True if successful
        """
        try:
            document_record.updated_at = datetime.utcnow()

            result = self.documents_collection.update_one(
                {"document_id": document_record.document_id},
                {"$set": document_record.to_dict()}
            )

            if result.modified_count > 0:
                logger.info(f"Updated document: {document_record.document_id}")
                return True
            else:
                logger.warning(f"No document found to update: {document_record.document_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to update document {document_record.document_id}: {e}")
            return False

    def list_documents(self,
                      status: Optional[ProcessingStatus] = None,
                      limit: int = 50) -> List[DocumentRecord]:
        """
        List documents with optional status filter.

        Args:
            status: Optional status filter
            limit: Maximum results

        Returns:
            List of document records
        """
        try:
            query = {}
            if status:
                query["current_status"] = status.value

            cursor = self.documents_collection.find(query).sort("created_at", DESCENDING).limit(limit)

            documents = []
            for doc_data in cursor:
                doc_data.pop('_id', None)
                documents.append(DocumentRecord.from_dict(doc_data))

            return documents

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    # Stage Management Methods

    def create_stage(self, stage_record: ProcessingStageRecord) -> bool:
        """
        Create a new processing stage record.

        Args:
            stage_record: Stage record to create

        Returns:
            True if successful
        """
        try:
            stage_record.created_at = datetime.utcnow()
            stage_record.updated_at = datetime.utcnow()

            self.stages_collection.insert_one(stage_record.to_dict())
            logger.info(f"Created stage record: {stage_record.stage_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create stage {stage_record.stage_id}: {e}")
            return False

    def update_stage(self, stage_record: ProcessingStageRecord) -> bool:
        """
        Update processing stage record.

        Args:
            stage_record: Updated stage record

        Returns:
            True if successful
        """
        try:
            stage_record.updated_at = datetime.utcnow()

            result = self.stages_collection.update_one(
                {"stage_id": stage_record.stage_id},
                {"$set": stage_record.to_dict()}
            )

            if result.modified_count > 0:
                logger.info(f"Updated stage: {stage_record.stage_id}")
                return True
            else:
                logger.warning(f"No stage found to update: {stage_record.stage_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to update stage {stage_record.stage_id}: {e}")
            return False

    def get_document_stages(self, document_id: str) -> List[ProcessingStageRecord]:
        """
        Get all stages for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of stage records ordered by stage_order
        """
        try:
            cursor = self.stages_collection.find(
                {"document_id": document_id}
            ).sort("stage_order", ASCENDING)

            stages = []
            for stage_data in cursor:
                stage_data.pop('_id', None)
                stages.append(ProcessingStageRecord.from_dict(stage_data))

            return stages

        except Exception as e:
            logger.error(f"Failed to get stages for document {document_id}: {e}")
            return []

    # Batch Management Methods

    def create_batch(self, batch_record: BatchRecord) -> bool:
        """
        Create a new batch record.

        Args:
            batch_record: Batch record to create

        Returns:
            True if successful
        """
        try:
            batch_record.created_at = datetime.utcnow()
            batch_record.updated_at = datetime.utcnow()

            self.batches_collection.insert_one(batch_record.to_dict())
            logger.info(f"Created batch record: {batch_record.batch_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create batch {batch_record.batch_id}: {e}")
            return False

    def update_batch(self, batch_record: BatchRecord) -> bool:
        """
        Update batch record.

        Args:
            batch_record: Updated batch record

        Returns:
            True if successful
        """
        try:
            batch_record.updated_at = datetime.utcnow()

            result = self.batches_collection.update_one(
                {"batch_id": batch_record.batch_id},
                {"$set": batch_record.to_dict()}
            )

            if result.modified_count > 0:
                logger.info(f"Updated batch: {batch_record.batch_id}")
                return True
            else:
                logger.warning(f"No batch found to update: {batch_record.batch_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to update batch {batch_record.batch_id}: {e}")
            return False

    def get_document_batches(self, document_id: str) -> List[BatchRecord]:
        """
        Get all batches for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of batch records ordered by batch_number
        """
        try:
            cursor = self.batches_collection.find(
                {"document_id": document_id}
            ).sort("batch_number", ASCENDING)

            batches = []
            for batch_data in cursor:
                batch_data.pop('_id', None)
                batches.append(BatchRecord.from_dict(batch_data))

            return batches

        except Exception as e:
            logger.error(f"Failed to get batches for document {document_id}: {e}")
            return []

    # Chunk Management Methods

    def create_chunk(self, chunk_record: EnhancedChunkRecord) -> bool:
        """
        Create a new enhanced chunk record.

        Args:
            chunk_record: Chunk record to create

        Returns:
            True if successful
        """
        try:
            chunk_record.created_at = datetime.utcnow()
            chunk_record.updated_at = datetime.utcnow()

            self.chunks_collection.insert_one(chunk_record.to_dict())
            logger.info(f"Created chunk record: {chunk_record.chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create chunk {chunk_record.chunk_id}: {e}")
            return False

    def create_chunks_batch(self, chunk_records: List[EnhancedChunkRecord]) -> int:
        """
        Create multiple chunk records in a batch.

        Args:
            chunk_records: List of chunk records to create

        Returns:
            Number of successfully created chunks
        """
        try:
            if not chunk_records:
                return 0

            now = datetime.utcnow()
            chunk_dicts = []

            for chunk_record in chunk_records:
                chunk_record.created_at = now
                chunk_record.updated_at = now
                chunk_dicts.append(chunk_record.to_dict())

            result = self.chunks_collection.insert_many(chunk_dicts)
            created_count = len(result.inserted_ids)

            logger.info(f"Created {created_count} chunk records in batch")
            return created_count

        except Exception as e:
            logger.error(f"Failed to create chunk batch: {e}")
            return 0

    def get_document_chunks(self, document_id: str) -> List[EnhancedChunkRecord]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of chunk records ordered by start_page
        """
        try:
            cursor = self.chunks_collection.find(
                {"document_id": document_id}
            ).sort("start_page", ASCENDING)

            chunks = []
            for chunk_data in cursor:
                chunk_data.pop('_id', None)
                chunks.append(EnhancedChunkRecord.from_dict(chunk_data))

            return chunks

        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            return []

    def search_chunks(self,
                     query: str,
                     document_id: Optional[str] = None,
                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search chunks using text search.

        Args:
            query: Search query
            document_id: Optional document ID filter
            limit: Maximum results

        Returns:
            List of matching chunks with scores
        """
        try:
            search_filter = {"$text": {"$search": query}}
            if document_id:
                search_filter["document_id"] = document_id

            cursor = self.chunks_collection.find(
                search_filter,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)

            results = []
            for chunk_data in cursor:
                chunk_data.pop('_id', None)
                results.append(chunk_data)

            return results

        except Exception as e:
            logger.error(f"Failed to search chunks: {e}")
            return []

    # Session Management Methods

    def create_session(self, session_record: ProcessingSession) -> bool:
        """
        Create a new processing session record.

        Args:
            session_record: Session record to create

        Returns:
            True if successful
        """
        try:
            session_record.created_at = datetime.utcnow()
            session_record.updated_at = datetime.utcnow()

            self.sessions_collection.insert_one(session_record.to_dict())
            logger.info(f"Created session record: {session_record.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create session {session_record.session_id}: {e}")
            return False

    def update_session(self, session_record: ProcessingSession) -> bool:
        """
        Update processing session record.

        Args:
            session_record: Updated session record

        Returns:
            True if successful
        """
        try:
            session_record.updated_at = datetime.utcnow()

            result = self.sessions_collection.update_one(
                {"session_id": session_record.session_id},
                {"$set": session_record.to_dict()}
            )

            if result.modified_count > 0:
                logger.info(f"Updated session: {session_record.session_id}")
                return True
            else:
                logger.warning(f"No session found to update: {session_record.session_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to update session {session_record.session_id}: {e}")
            return False

    # Utility Methods

    def get_document_statistics(self, document_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a document.

        Args:
            document_id: Document identifier

        Returns:
            Statistics dictionary
        """
        try:
            stats = {
                "document_id": document_id,
                "stages_count": self.stages_collection.count_documents({"document_id": document_id}),
                "batches_count": self.batches_collection.count_documents({"document_id": document_id}),
                "chunks_count": self.chunks_collection.count_documents({"document_id": document_id}),
                "stages_completed": self.stages_collection.count_documents({
                    "document_id": document_id,
                    "stage_status": StageStatus.COMPLETED.value
                }),
                "batches_completed": self.batches_collection.count_documents({
                    "document_id": document_id,
                    "status": StageStatus.COMPLETED.value
                })
            }

            # Get processing duration if completed
            stages = self.get_document_stages(document_id)
            if stages:
                first_stage = min(stages, key=lambda s: s.started_at or datetime.utcnow())
                last_stage = max(stages, key=lambda s: s.completed_at or datetime.min)

                if first_stage.started_at and last_stage.completed_at:
                    duration = last_stage.completed_at - first_stage.started_at
                    stats["total_processing_time_seconds"] = duration.total_seconds()

            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics for document {document_id}: {e}")
            return {"document_id": document_id}

    def delete_document_data(self, document_id: str) -> bool:
        """
        Delete all data for a document from all collections.

        Args:
            document_id: Document identifier

        Returns:
            True if successful
        """
        try:
            # Delete from all collections
            self.chunks_collection.delete_many({"document_id": document_id})
            self.batches_collection.delete_many({"document_id": document_id})
            self.stages_collection.delete_many({"document_id": document_id})
            self.documents_collection.delete_one({"document_id": document_id})

            logger.info(f"Deleted all data for document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document data {document_id}: {e}")
            return False

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get MongoDB connection status.

        Returns:
            Connection status information
        """
        try:
            self.client.admin.command('ping')
            return {
                "connected": True,
                "database": self.db_name,
                "collections": {
                    "documents": self.documents_collection.count_documents({}),
                    "stages": self.stages_collection.count_documents({}),
                    "batches": self.batches_collection.count_documents({}),
                    "chunks": self.chunks_collection.count_documents({}),
                    "sessions": self.sessions_collection.count_documents({})
                }
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }

    def close(self):
        """Close database connection."""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB connection closed")