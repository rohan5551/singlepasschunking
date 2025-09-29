"""Structured output schema for LLM chunk processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json


@dataclass
class ChunkOutput:
    """Represents a single chunk from LLM processing."""

    content: str
    table: Optional[str]
    start_page: int
    end_page: int
    level_1: str
    level_2: str
    level_3: str
    continues_from_previous: bool = False
    continues_to_next: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "table": self.table,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "level_1": self.level_1,
            "level_2": self.level_2,
            "level_3": self.level_3,
            "continues_from_previous": self.continues_from_previous,
            "continues_to_next": self.continues_to_next,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChunkOutput:
        """Create ChunkOutput from dictionary."""
        return cls(
            content=data["content"],
            table=data.get("table"),
            start_page=data["start_page"],
            end_page=data["end_page"],
            level_1=data["level_1"],
            level_2=data["level_2"],
            level_3=data["level_3"],
            continues_from_previous=data.get("continues_from_previous", False),
            continues_to_next=data.get("continues_to_next", False),
        )


@dataclass
class BatchProcessingResult:
    """Enhanced result from batch processing with structured chunks."""

    chunks: List[ChunkOutput]
    raw_output: str
    last_chunk: Optional[ChunkOutput]
    continuation_context: Optional[Dict[str, Any]]
    processing_metadata: Dict[str, Any]

    def get_last_chunk_for_context(self) -> Optional[Dict[str, Any]]:
        """Get the last chunk formatted for next batch context."""
        if not self.last_chunk:
            return None

        return {
            "content": self.last_chunk.content[-500:] if len(self.last_chunk.content) > 500 else self.last_chunk.content,
            "level_1": self.last_chunk.level_1,
            "level_2": self.last_chunk.level_2,
            "level_3": self.last_chunk.level_3,
            "continues_to_next": self.last_chunk.continues_to_next,
            "end_page": self.last_chunk.end_page,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "raw_output": self.raw_output,
            "last_chunk": self.last_chunk.to_dict() if self.last_chunk else None,
            "continuation_context": self.continuation_context,
            "processing_metadata": self.processing_metadata,
        }


def parse_structured_llm_output(raw_output: str) -> List[ChunkOutput]:
    """Parse structured JSON output from LLM into ChunkOutput objects."""
    try:
        # Try to extract JSON from the raw output
        json_start = raw_output.find('[')
        json_end = raw_output.rfind(']') + 1

        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON array found in output")

        json_str = raw_output[json_start:json_end]
        data = json.loads(json_str)

        if not isinstance(data, list):
            raise ValueError("Expected JSON array of chunks")

        chunks = []
        for chunk_data in data:
            try:
                chunk = ChunkOutput.from_dict(chunk_data)
                chunks.append(chunk)
            except KeyError as e:
                raise ValueError(f"Missing required field in chunk: {e}")

        return chunks

    except (json.JSONDecodeError, ValueError) as e:
        # Fallback: try to parse as raw text chunks
        return _fallback_parse_chunks(raw_output)


def _fallback_parse_chunks(raw_output: str) -> List[ChunkOutput]:
    """Fallback parsing for non-structured output."""
    chunks = []
    current_content = []

    for line in raw_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("###") or stripped.lower().startswith("chunk"):
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    # Create a basic chunk with minimal structure
                    chunk = ChunkOutput(
                        content=content,
                        table=None,
                        start_page=1,  # Default values
                        end_page=1,
                        level_1="Unknown Document",
                        level_2="Unknown Section",
                        level_3=stripped,
                        continues_from_previous=False,
                        continues_to_next=False,
                    )
                    chunks.append(chunk)
                current_content = []
        current_content.append(line)

    # Handle final chunk
    if current_content:
        content = "\n".join(current_content).strip()
        if content:
            chunk = ChunkOutput(
                content=content,
                table=None,
                start_page=1,
                end_page=1,
                level_1="Unknown Document",
                level_2="Unknown Section",
                level_3="Final Chunk",
                continues_from_previous=False,
                continues_to_next=False,
            )
            chunks.append(chunk)

    return chunks


__all__ = [
    "ChunkOutput",
    "BatchProcessingResult",
    "parse_structured_llm_output",
]