"""Utility for maintaining multimodal chunking context across batches."""

from __future__ import annotations

import copy
import threading
from typing import Dict, List, Optional, Any
from ..models.chunk_schema import BatchProcessingResult, ChunkOutput


class ContextManager:
    """Track contextual state for each document across batches.

    The context object exposed to the LMM aligns with the structure defined in the
    product specification::

        context = {
            "last_chunks": [...],
            "heading_hierarchy": {
                "level1": "Document Title",
                "level2": "Section",
                "level3": "Subsection"
            },
            "continuation_metadata": {...}
        }

    A dedicated context is maintained per document identifier to support parallel
    processing.
    """

    def __init__(self) -> None:
        self._contexts: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Context lifecycle helpers
    # ------------------------------------------------------------------
    def _empty_context(self) -> Dict[str, Any]:
        return {
            "last_chunks": [],
            "heading_hierarchy": {},
            "continuation_metadata": {},
            "continuation_context": None,
            "processing_metadata": {},
        }

    def reset_context(self, document_id: str) -> None:
        """Reset stored context for a document."""

        with self._lock:
            self._contexts[document_id] = self._empty_context()

    def drop_context(self, document_id: str) -> None:
        """Remove all stored information for a document."""

        with self._lock:
            self._contexts.pop(document_id, None)

    # ------------------------------------------------------------------
    # Context accessors
    # ------------------------------------------------------------------
    def get_context(self, document_id: str) -> Dict[str, Any]:
        """Return a copy of the current context for a document."""

        with self._lock:
            context = self._contexts.setdefault(document_id, self._empty_context())
            return copy.deepcopy(context)

    def build_context_payload(self, document_id: str) -> Dict[str, Any]:
        """Return a context object ready to be injected into a prompt."""

        return self.get_context(document_id)

    # ------------------------------------------------------------------
    # Context mutation helpers
    # ------------------------------------------------------------------
    def update_context(
        self,
        document_id: str,
        *,
        last_chunks: Optional[List[str]] = None,
        heading_hierarchy: Optional[Dict[str, str]] = None,
        continuation_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge new context details for a document.

        Args:
            document_id: Unique document identifier.
            last_chunks: Recent chunk summaries produced by the LMM.
            heading_hierarchy: Detected headings for the latest batch.
            continuation_metadata: Arbitrary metadata describing continuation
                requirements for the next batch.

        Returns:
            Updated context dictionary.
        """

        with self._lock:
            context = self._contexts.setdefault(document_id, self._empty_context())

            if last_chunks is not None:
                context["last_chunks"] = last_chunks

            if heading_hierarchy is not None:
                context["heading_hierarchy"] = heading_hierarchy

            if continuation_metadata is not None:
                context["continuation_metadata"] = continuation_metadata

            return copy.deepcopy(context)

    def update_context_from_batch_result(
        self,
        document_id: str,
        batch_result: BatchProcessingResult,
        batch_number: int,
    ) -> Dict[str, Any]:
        """Update context using structured batch processing result.

        Args:
            document_id: Unique document identifier.
            batch_result: Result from LMM batch processing.
            batch_number: Current batch number for tracking.

        Returns:
            Updated context dictionary.
        """
        with self._lock:
            context = self._contexts.setdefault(document_id, self._empty_context())

            # Update with structured data from batch result
            context["continuation_context"] = batch_result.continuation_context
            context["processing_metadata"] = {
                **batch_result.processing_metadata,
                "batch_number": batch_number,
            }

            # Maintain legacy fields for backward compatibility
            if batch_result.chunks:
                # Create legacy last_chunks format
                context["last_chunks"] = [
                    f"{chunk.level_3}: {chunk.content[:150]}..." if len(chunk.content) > 150 else f"{chunk.level_3}: {chunk.content}"
                    for chunk in batch_result.chunks[-2:]
                ]

                # Create heading hierarchy from last chunk
                last_chunk = batch_result.chunks[-1]
                context["heading_hierarchy"] = {
                    "level1": last_chunk.level_1,
                    "level2": last_chunk.level_2,
                    "level3": last_chunk.level_3,
                }

            # Legacy continuation metadata
            context["continuation_metadata"] = {
                "chunk_count": len(batch_result.chunks),
                "has_continuation": batch_result.last_chunk and batch_result.last_chunk.continues_to_next,
                "batch_number": batch_number,
            }

            return copy.deepcopy(context)


__all__ = ["ContextManager"]
