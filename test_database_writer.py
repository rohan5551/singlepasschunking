#!/usr/bin/env python3
"""
Test script to verify DatabaseWriter functionality.
"""

import os
import logging
from dotenv import load_dotenv
from datetime import datetime
from uuid import uuid4

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_writer():
    """Test DatabaseWriter initialization and basic operations."""

    logger.info("Testing DatabaseWriter initialization...")

    try:
        from src.processors.db_writer import DatabaseWriter
        from src.models.chunk_schema import ChunkOutput, BatchProcessingResult
        from src.models.batch_models import ProcessingTask, SplitConfiguration, PageBatch, BatchStatus
        from src.models.pdf_document import PDFDocument

        # Initialize DatabaseWriter
        db_writer = DatabaseWriter()
        logger.info("✓ DatabaseWriter initialized successfully")

        # Create test data
        test_document = PDFDocument(
            file_path="/test/document.pdf",
            pages=[],
            total_pages=5,
            metadata={"test": True},
            source_type="test"
        )

        test_task = ProcessingTask(
            task_id=f"test_task_{uuid4().hex[:8]}",
            document=test_document,
            config=SplitConfiguration(),
            status=None,
            prompt="Test prompt",
            lifecycle_document_id=None
        )

        test_batch = PageBatch(
            batch_id=f"test_batch_{uuid4().hex[:8]}",
            batch_number=1,
            pages=[],
            start_page=1,
            end_page=2,
            total_pages=2,
            document_id=test_task.task_id,
            status=BatchStatus.COMPLETED,
            processed_at=datetime.utcnow()
        )

        # Create test chunks
        test_chunks = [
            ChunkOutput(
                content="This is test chunk 1 content with some text.",
                table=None,
                start_page=1,
                end_page=1,
                level_1="Test Document",
                level_2="Test Section",
                level_3="Test Subsection 1",
                continues_from_previous=False,
                continues_to_next=True
            ),
            ChunkOutput(
                content="This is test chunk 2 content with more text.",
                table=None,
                start_page=2,
                end_page=2,
                level_1="Test Document",
                level_2="Test Section",
                level_3="Test Subsection 2",
                continues_from_previous=True,
                continues_to_next=False
            )
        ]

        test_batch_result = BatchProcessingResult(
            chunks=test_chunks,
            raw_output="Test raw output from LLM",
            last_chunk=test_chunks[-1],
            continuation_context={"test_context": "value"},
            processing_metadata={"processing_time": 1.5, "model": "test-model"}
        )

        # Test write_batch method
        logger.info("Testing write_batch method...")
        write_result = db_writer.write_batch(test_task, test_batch, test_batch_result)

        if write_result["success"]:
            logger.info(f"✓ write_batch successful - Saved {write_result['chunks_saved']} chunks")
            logger.info(f"  Document ID: {write_result['document_id']}")
            logger.info(f"  Batch ID: {write_result['batch_id']}")
        else:
            logger.error(f"✗ write_batch failed: {write_result.get('error')}")
            return False

        # Test connection status
        status = db_writer.get_connection_status()
        logger.info(f"✓ Connection status: {status}")

        return True

    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ DatabaseWriter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("DatabaseWriter Test")
    logger.info("=" * 50)

    success = test_database_writer()

    if success:
        logger.info("\n✅ DatabaseWriter test passed!")
    else:
        logger.error("\n❌ DatabaseWriter test failed.")