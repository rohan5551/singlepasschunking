import math
import logging
import os
import uuid
from typing import List, Optional, Tuple
from datetime import datetime

from ..models.pdf_document import PDFDocument
from ..models.batch_models import (
    PageBatch, SplitConfiguration, BatchingResult,
    BatchStatus, ProcessingTask, ProcessingStage
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFSplitter:
    """
    Split PDF documents into configurable page batches for processing

    This class handles the splitting logic for the Vision-Guided Chunking System,
    creating batches that can be processed independently while maintaining
    contextual relationships between adjacent batches.
    """

    def __init__(self, config: Optional[SplitConfiguration] = None):
        """
        Initialize PDFSplitter with configuration

        Args:
            config: SplitConfiguration object with batch size and overlap settings
        """
        self.config = config or SplitConfiguration()
        logger.info(f"PDFSplitter initialized with batch_size={self.config.batch_size}, "
                   f"overlap_pages={self.config.overlap_pages}")

    def split_document(self, document: PDFDocument,
                      config: Optional[SplitConfiguration] = None) -> BatchingResult:
        """
        Split a PDF document into page batches

        Args:
            document: PDFDocument to split
            config: Optional configuration to override default

        Returns:
            BatchingResult containing all created batches
        """
        split_config = config or self.config
        logger.info(f"Splitting document: {document.file_path} "
                   f"({document.total_pages} pages) into batches of {split_config.batch_size}")

        try:
            # Calculate number of batches needed: k = ⌈n/b⌉
            total_pages = document.total_pages
            batch_size = split_config.batch_size
            overlap = split_config.overlap_pages

            # Calculate effective batch size considering overlap
            effective_batch_size = batch_size - overlap if overlap > 0 else batch_size

            # Calculate total number of batches
            if overlap > 0:
                # With overlap, we need to calculate differently
                total_batches = self._calculate_batches_with_overlap(
                    total_pages, batch_size, overlap
                )
            else:
                # Without overlap: simple ceiling division
                total_batches = math.ceil(total_pages / batch_size)

            logger.info(f"Creating {total_batches} batches for {total_pages} pages")

            # Create batches
            batches = []
            for batch_num in range(1, total_batches + 1):
                batch = self._create_batch(
                    document, batch_num, split_config, total_batches
                )
                batches.append(batch)

            # Create result
            result = BatchingResult(
                document_id=document.file_path,
                batches=batches,
                config=split_config,
                total_pages=total_pages,
                total_batches=total_batches
            )

            logger.info(f"Successfully created {len(batches)} batches")
            return result

        except Exception as e:
            logger.error(f"Error splitting document: {e}")
            raise

    def _calculate_batches_with_overlap(self, total_pages: int,
                                      batch_size: int, overlap: int) -> int:
        """
        Calculate number of batches needed when using overlap

        Args:
            total_pages: Total number of pages in document
            batch_size: Size of each batch
            overlap: Number of overlapping pages between batches

        Returns:
            Number of batches needed
        """
        if overlap >= batch_size:
            raise ValueError("Overlap cannot be greater than or equal to batch size")

        effective_step = batch_size - overlap
        if effective_step <= 0:
            raise ValueError("Effective step size must be positive")

        # Calculate how many batches we need
        remaining_pages = total_pages
        batches = 0

        while remaining_pages > 0:
            batches += 1
            if batches == 1:
                # First batch takes full batch_size
                remaining_pages -= batch_size
            else:
                # Subsequent batches take effective_step pages
                remaining_pages -= effective_step

        return batches

    def _create_batch(self, document: PDFDocument, batch_number: int,
                     config: SplitConfiguration, total_batches: int) -> PageBatch:
        """
        Create a single batch with page references

        Args:
            document: Source document
            batch_number: Current batch number (1-indexed)
            config: Split configuration
            total_batches: Total number of batches

        Returns:
            PageBatch object
        """
        batch_size = config.batch_size
        overlap = config.overlap_pages

        # Calculate page range for this batch
        if overlap > 0:
            start_page, end_page = self._calculate_batch_range_with_overlap(
                batch_number, batch_size, overlap, document.total_pages
            )
        else:
            start_page, end_page = self._calculate_batch_range_simple(
                batch_number, batch_size, document.total_pages
            )

        # Get pages for this batch
        batch_pages = []
        for page_num in range(start_page, min(end_page + 1, document.total_pages + 1)):
            if page_num <= len(document.pages):
                page = document.pages[page_num - 1]  # Convert to 0-indexed
                batch_pages.append(page)

        # Create batch object
        # Create a URL-safe batch id so it can be used in API routes without
        # breaking the path structure (e.g. temporary files that live in /tmp)
        safe_doc_id = os.path.splitext(os.path.basename(document.file_path))[0]
        batch_identifier = f"{safe_doc_id}_batch_{batch_number}_{uuid.uuid4().hex[:8]}"

        batch = PageBatch(
            batch_id=batch_identifier,
            batch_number=batch_number,
            pages=batch_pages,
            start_page=start_page,
            end_page=min(end_page, document.total_pages),
            total_pages=len(batch_pages),
            document_id=document.file_path,
            status=BatchStatus.PENDING
        )

        logger.debug(f"Created batch {batch_number}: pages {start_page}-{end_page} "
                    f"({len(batch_pages)} pages)")

        return batch

    def _calculate_batch_range_simple(self, batch_number: int, batch_size: int,
                                    total_pages: int) -> Tuple[int, int]:
        """
        Calculate page range for a batch without overlap

        Args:
            batch_number: Current batch number (1-indexed)
            batch_size: Size of each batch
            total_pages: Total pages in document

        Returns:
            Tuple of (start_page, end_page) both 1-indexed
        """
        start_page = (batch_number - 1) * batch_size + 1
        end_page = min(batch_number * batch_size, total_pages)

        return start_page, end_page

    def _calculate_batch_range_with_overlap(self, batch_number: int, batch_size: int,
                                          overlap: int, total_pages: int) -> Tuple[int, int]:
        """
        Calculate page range for a batch with overlap

        Args:
            batch_number: Current batch number (1-indexed)
            batch_size: Size of each batch
            overlap: Number of overlapping pages
            total_pages: Total pages in document

        Returns:
            Tuple of (start_page, end_page) both 1-indexed
        """
        if batch_number == 1:
            # First batch starts at page 1
            start_page = 1
            end_page = min(batch_size, total_pages)
        else:
            # Subsequent batches have overlap
            effective_step = batch_size - overlap
            start_page = (batch_number - 1) * effective_step + 1
            end_page = min(start_page + batch_size - 1, total_pages)

        return start_page, end_page

    def get_batch_summary(self, result: BatchingResult) -> dict:
        """
        Get a summary of the batching result

        Args:
            result: BatchingResult to summarize

        Returns:
            Dictionary with batch statistics
        """
        batches = result.batches

        summary = {
            "total_pages": result.total_pages,
            "total_batches": result.total_batches,
            "batch_size_config": result.config.batch_size,
            "overlap_pages": result.config.overlap_pages,
            "batch_distribution": result.batch_distribution,
            "average_batch_size": sum(b.page_count for b in batches) / len(batches) if batches else 0,
            "min_batch_size": min(b.page_count for b in batches) if batches else 0,
            "max_batch_size": max(b.page_count for b in batches) if batches else 0,
            "batches_detail": [
                {
                    "batch_number": b.batch_number,
                    "pages": f"{b.start_page}-{b.end_page}",
                    "page_count": b.page_count,
                    "status": b.status.value
                }
                for b in batches
            ]
        }

        return summary

    def reconfigure_batches(self, document: PDFDocument,
                           new_config: SplitConfiguration) -> BatchingResult:
        """
        Reconfigure batches with new settings

        Args:
            document: Document to re-split
            new_config: New configuration to apply

        Returns:
            New BatchingResult with updated configuration
        """
        logger.info(f"Reconfiguring batches for {document.file_path} "
                   f"with new batch_size={new_config.batch_size}")

        self.config = new_config
        return self.split_document(document, new_config)

    def validate_configuration(self, config: SplitConfiguration,
                             document: PDFDocument) -> List[str]:
        """
        Validate configuration against document constraints

        Args:
            config: Configuration to validate
            document: Document to validate against

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if config.batch_size <= 0:
            errors.append("Batch size must be greater than 0")

        if config.overlap_pages < 0:
            errors.append("Overlap pages cannot be negative")

        if config.overlap_pages >= config.batch_size:
            errors.append("Overlap pages must be less than batch size")

        if config.batch_size > document.total_pages:
            errors.append(f"Batch size ({config.batch_size}) cannot be larger "
                         f"than document pages ({document.total_pages})")

        if config.min_batch_size <= 0:
            errors.append("Minimum batch size must be greater than 0")

        if config.max_batch_size < config.min_batch_size:
            errors.append("Maximum batch size must be >= minimum batch size")

        return errors