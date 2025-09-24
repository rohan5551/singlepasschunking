"""Utility for maintaining multimodal chunking context across batches."""

from __future__ import annotations

import copy
import threading
from typing import Dict, List, Optional, Any


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
            "continuation_metadata": {}
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


__all__ = ["ContextManager"]
