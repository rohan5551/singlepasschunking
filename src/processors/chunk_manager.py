"""MongoDB chunk manager for storing processed document chunks."""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

from ..models.chunk_schema import ChunkOutput, BatchProcessingResult

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ChunkManager:
    """
    Manages storage and retrieval of processed chunks in MongoDB.

    Features:
    - Store individual chunks with metadata
    - Batch operations for efficient storage
    - Document-level chunk management
    - Search and retrieval capabilities
    """

    def __init__(self, mongodb_uri: Optional[str] = None, collection_name: Optional[str] = None):
        """
        Initialize ChunkManager with MongoDB connection.

        Args:
            mongodb_uri: MongoDB connection string (defaults to env var)
            collection_name: Collection name (defaults to env var)
        """
        self.mongodb_uri = mongodb_uri or os.getenv('MONGODB_URI')
        self.collection_name = collection_name or os.getenv('COLLECTION', 'singlepasschunking')

        if not self.mongodb_uri:
            raise ValueError("MongoDB URI not provided. Set MONGODB_URI environment variable.")

        try:
            # Parse database name from URI more reliably
            self.client = MongoClient(self.mongodb_uri)

            # Extract database name from URI
            if '/' in self.mongodb_uri and '?' in self.mongodb_uri:
                # Format: mongodb://user:pass@host:port/dbname?options
                db_part = self.mongodb_uri.split('/')[-1]
                db_name = db_part.split('?')[0]
            elif '/' in self.mongodb_uri:
                # Format: mongodb://user:pass@host:port/dbname
                db_name = self.mongodb_uri.split('/')[-1]
            else:
                # Default database name
                db_name = "one-sea"

            logger.info(f"Extracted database name: {db_name}")
            self.db = self.client[db_name]
            self.collection: Collection = self.db[self.collection_name]

            # Test connection with timeout
            self.client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB database '{db_name}', collection '{self.collection_name}'")

            # Test write permissions by attempting to create indexes
            self._ensure_indexes()

        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.error(f"MongoDB URI: {self.mongodb_uri[:20]}...")  # Log partial URI for debugging
            logger.error(f"Collection name: {self.collection_name}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            raise

    def _ensure_indexes(self):
        """Create necessary indexes for efficient queries."""
        try:
            # Index for document-based queries
            self.collection.create_index([("document_id", 1), ("batch_id", 1)])

            # Index for content search
            self.collection.create_index([("content", "text")])

            # Index for hierarchical structure queries
            self.collection.create_index([
                ("level_1", 1),
                ("level_2", 1),
                ("level_3", 1)
            ])

            # Index for page-based queries
            self.collection.create_index([("start_page", 1), ("end_page", 1)])

            # Index for timestamp queries
            self.collection.create_index([("created_at", -1)])

        except PyMongoError as e:
            logger.warning(f"Failed to create indexes: {e}")

    def save_batch_chunks(self,
                         document_id: str,
                         batch_id: str,
                         batch_result: BatchProcessingResult,
                         metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Save all chunks from a batch processing result.

        Args:
            document_id: Unique identifier for the source document
            batch_id: Batch identifier
            batch_result: Result containing processed chunks
            metadata: Additional metadata to store with chunks

        Returns:
            List of inserted chunk IDs
        """
        if not batch_result.chunks:
            logger.warning(f"No chunks to save for batch {batch_id}")
            return []

        documents_to_insert = []
        current_time = datetime.utcnow()

        for i, chunk in enumerate(batch_result.chunks):
            chunk_doc = {
                # Chunk content and structure
                "content": chunk.content,
                "table": chunk.table,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "level_1": chunk.level_1,
                "level_2": chunk.level_2,
                "level_3": chunk.level_3,
                "continues_from_previous": chunk.continues_from_previous,
                "continues_to_next": chunk.continues_to_next,

                # Processing metadata
                "document_id": document_id,
                "batch_id": batch_id,
                "chunk_index": i,
                "total_chunks_in_batch": len(batch_result.chunks),

                # Processing context
                "processing_metadata": batch_result.processing_metadata,
                "continuation_context": batch_result.continuation_context,

                # Timestamps
                "created_at": current_time,
                "updated_at": current_time,

                # Additional metadata
                "metadata": metadata or {},
            }

            # Add last chunk context if this is the final chunk
            if i == len(batch_result.chunks) - 1 and batch_result.last_chunk:
                chunk_doc["is_last_chunk"] = True
                chunk_doc["last_chunk_context"] = batch_result.get_last_chunk_for_context()
            else:
                chunk_doc["is_last_chunk"] = False

            documents_to_insert.append(chunk_doc)

        try:
            logger.info(f"Attempting to save {len(documents_to_insert)} chunks for batch {batch_id}")
            logger.debug(f"Sample chunk data keys: {list(documents_to_insert[0].keys()) if documents_to_insert else 'No chunks'}")

            result = self.collection.insert_many(documents_to_insert)
            inserted_ids = [str(id) for id in result.inserted_ids]

            logger.info(f"Successfully saved {len(inserted_ids)} chunks for batch {batch_id} to collection '{self.collection_name}'")
            return inserted_ids

        except PyMongoError as e:
            logger.error(f"MongoDB error saving chunks for batch {batch_id}: {e}")
            logger.error(f"Collection: {self.collection_name}")
            logger.error(f"Database: {self.db.name}")
            logger.error(f"Chunks to save: {len(documents_to_insert)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving chunks for batch {batch_id}: {e}")
            raise

    def save_single_chunk(self,
                         document_id: str,
                         batch_id: str,
                         chunk: ChunkOutput,
                         chunk_index: int = 0,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a single chunk to MongoDB.

        Args:
            document_id: Unique identifier for the source document
            batch_id: Batch identifier
            chunk: Chunk to save
            chunk_index: Index of chunk within batch
            metadata: Additional metadata

        Returns:
            Inserted chunk ID
        """
        current_time = datetime.utcnow()

        chunk_doc = {
            # Chunk content
            **chunk.to_dict(),

            # Processing metadata
            "document_id": document_id,
            "batch_id": batch_id,
            "chunk_index": chunk_index,

            # Timestamps
            "created_at": current_time,
            "updated_at": current_time,

            # Additional metadata
            "metadata": metadata or {},
            "is_last_chunk": False,
        }

        try:
            result = self.collection.insert_one(chunk_doc)
            inserted_id = str(result.inserted_id)

            logger.info(f"Saved single chunk {inserted_id} for batch {batch_id}")
            return inserted_id

        except PyMongoError as e:
            logger.error(f"Failed to save chunk for batch {batch_id}: {e}")
            raise

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of chunk documents
        """
        try:
            chunks = list(self.collection.find(
                {"document_id": document_id}
            ).sort([("batch_id", 1), ("chunk_index", 1)]))

            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks

        except PyMongoError as e:
            logger.error(f"Failed to retrieve chunks for document {document_id}: {e}")
            raise

    def get_batch_chunks(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific batch.

        Args:
            batch_id: Batch identifier

        Returns:
            List of chunk documents
        """
        try:
            chunks = list(self.collection.find(
                {"batch_id": batch_id}
            ).sort([("chunk_index", 1)]))

            logger.info(f"Retrieved {len(chunks)} chunks for batch {batch_id}")
            return chunks

        except PyMongoError as e:
            logger.error(f"Failed to retrieve chunks for batch {batch_id}: {e}")
            raise

    def search_chunks(self,
                     query: str,
                     document_id: Optional[str] = None,
                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search chunks by content.

        Args:
            query: Text search query
            document_id: Optional document filter
            limit: Maximum results to return

        Returns:
            List of matching chunk documents
        """
        search_filter = {"$text": {"$search": query}}

        if document_id:
            search_filter["document_id"] = document_id

        try:
            chunks = list(self.collection.find(
                search_filter,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit))

            logger.info(f"Found {len(chunks)} chunks matching query: {query}")
            return chunks

        except PyMongoError as e:
            logger.error(f"Failed to search chunks: {e}")
            raise

    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document identifier

        Returns:
            Number of deleted chunks
        """
        try:
            result = self.collection.delete_many({"document_id": document_id})
            deleted_count = result.deleted_count

            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count

        except PyMongoError as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {e}")
            raise

    def delete_batch_chunks(self, batch_id: str) -> int:
        """
        Delete all chunks for a batch.

        Args:
            batch_id: Batch identifier

        Returns:
            Number of deleted chunks
        """
        try:
            result = self.collection.delete_many({"batch_id": batch_id})
            deleted_count = result.deleted_count

            logger.info(f"Deleted {deleted_count} chunks for batch {batch_id}")
            return deleted_count

        except PyMongoError as e:
            logger.error(f"Failed to delete chunks for batch {batch_id}: {e}")
            raise

    def get_document_statistics(self, document_id: str) -> Dict[str, Any]:
        """
        Get statistics for a document's chunks.

        Args:
            document_id: Document identifier

        Returns:
            Statistics dictionary
        """
        try:
            pipeline = [
                {"$match": {"document_id": document_id}},
                {"$group": {
                    "_id": None,
                    "total_chunks": {"$sum": 1},
                    "total_batches": {"$addToSet": "$batch_id"},
                    "page_range": {
                        "$push": {
                            "start": "$start_page",
                            "end": "$end_page"
                        }
                    },
                    "content_length": {"$sum": {"$strLenCP": "$content"}},
                    "levels": {
                        "$addToSet": {
                            "level_1": "$level_1",
                            "level_2": "$level_2",
                            "level_3": "$level_3"
                        }
                    }
                }}
            ]

            result = list(self.collection.aggregate(pipeline))

            if result:
                stats = result[0]
                stats["total_batches"] = len(stats["total_batches"])

                # Calculate page statistics
                if stats["page_range"]:
                    pages = stats["page_range"]
                    min_page = min(p["start"] for p in pages)
                    max_page = max(p["end"] for p in pages)
                    stats["page_range"] = {"min": min_page, "max": max_page}

                del stats["_id"]
                return stats

            return {
                "total_chunks": 0,
                "total_batches": 0,
                "page_range": {"min": 0, "max": 0},
                "content_length": 0,
                "levels": []
            }

        except PyMongoError as e:
            logger.error(f"Failed to get statistics for document {document_id}: {e}")
            raise

    def close(self):
        """Close MongoDB connection."""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB connection closed")