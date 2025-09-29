"""Prompt templates for human-in-the-loop LLM processing."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

DEFAULT_MANUAL_PROMPT_TEMPLATE = """You are a document analysis system. Process the provided PDF pages and extract structured chunks for a RAG system.


## CRITICAL REQUIREMENTS:

### 1. EXTRACTION RULES:
- Extract ALL text content from EVERY page in the batch
- Process pages sequentially - DO NOT skip any page
- Preserve exact text without paraphrasing or summarizing
- Identify logical sections and semantic boundaries

### 2. MANDATORY 3-LEVEL HEADING STRUCTURE:
Every chunk MUST have exactly 3 heading levels:
- Level 1: Document/Product title (full name with context)
- Level 2: Major section (Features, Procedures, Specifications, etc.)
- Level 3: Specific subtopic (Step 1, Subsection details, etc.)

If headings are missing, infer from content and context.

### 3. CHUNKING RULES (HIGHEST PRIORITY):
- **NEVER split numbered steps or procedures** - keep ALL steps together
- **NEVER split bulleted/numbered lists** - keep ALL items together
- **Tables**: Create ONE chunk per table row, include headers in each
- **Minimum chunk size**: 3 lines (merge smaller content)
- **Multi-page content**: Track if content continues across pages

### 4. SKIP THESE ELEMENTS:
- Table of contents
- Page numbers, headers, footers
- Index pages
- Copyright notices (unless specifically relevant)

### 5. CONTINUATION FLAGS:
Tag each chunk with continuation status:
- [CONTINUES]True[/CONTINUES] - Content continues from previous
- [CONTINUES]False[/CONTINUES] - New independent content
- [CONTINUES]Partial[/CONTINUES] - Uncertain relationship

### 6. SPECIAL HANDLING:

**For Tables:**
- Keep complete tables together in a single chunk
- Preserve entire table structure including all headers and rows
- If table spans multiple pages, maintain as one coherent chunk
- Include all column headers, alignment, and formatting
- Only split if table is exceptionally large (rare case)

**For Steps/Procedures:**
- Keep ALL related steps in ONE chunk
- Even if spanning multiple pages
- Include all sub-steps and notes

**For Multi-page content:**
- Note where content is incomplete
- Flag for continuation in next batch

### 7. OUTPUT FORMAT:
Return a JSON structure with the extracted chunks following the exact schema below.

## CONTEXT FROM PREVIOUS BATCH:
{context_placeholder}

## OUTPUT REQUIREMENTS:
Provide response in valid JSON format only. No additional text or explanations.

required schema 

{
  "schema": {
    "type": "array",
    "description": "List of extracted chunks from the batch",
    "items": {
      "type": "object",
      "properties": {
        "content": {
          "type": "string",
          "description": "The extracted text content from the document"
        },
        "table": {
          "type": "string",
          "description": "Markdown formatted table if chunk contains a table, null otherwise"
        },
        "start_page": {
          "type": "integer",
          "description": "Starting page number as shown in red on top right corner of PDF"
        },
        "end_page": {
          "type": "integer",
          "description": "Ending page number as shown in red on top right corner of PDF"
        },
        "level_1": {
          "type": "string",
          "description": "First level heading - Document or product title"
        },
        "level_2": {
          "type": "string",
          "description": "Second level heading - Major section"
        },
        "level_3": {
          "type": "string",
          "description": "Third level heading - Specific subtopic"
        },
        "continues_from_previous": {
          "type": "boolean",
          "description": "True if this chunk continues from previous chunk"
        },
        "continues_to_next": {
          "type": "boolean",
          "description": "True if this chunk continues to next chunk"
        }
      },
      "required": ["content", "start_page", "end_page", "level_1", "level_2", "level_3"]
    }
  }
}
"""


def get_default_manual_prompt() -> str:
    """Return the default manual processing prompt template."""

    return DEFAULT_MANUAL_PROMPT_TEMPLATE


def build_manual_prompt(
    template: Optional[str],
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Inject contextual information into a manual processing prompt."""

    prompt_template = template or DEFAULT_MANUAL_PROMPT_TEMPLATE
    context_dict: Dict[str, Any] = context or {}

    # Enhanced context formatting for better continuity
    context_text = _format_enhanced_context(context_dict)

    if "{context_placeholder}" in prompt_template:
        return prompt_template.replace("{context_placeholder}", context_text)

    return (
        f"{prompt_template}\n\n## CONTEXT FROM PREVIOUS BATCH:\n{context_text}"
    )


def _format_enhanced_context(context: Dict[str, Any]) -> str:
    """Format context with enhanced information for better continuity."""
    if not context:
        return "No previous context available. This is the first batch."

    parts = []

    # Previous batch continuation info
    if "continuation_context" in context:
        continuation = context["continuation_context"]
        if continuation and continuation.get("expects_continuation"):
            parts.append("⚠️ **CONTINUATION REQUIRED**: The previous batch ended with incomplete content that continues into this batch.")

            # Add previous chunk summary
            if "previous_chunk_summary" in continuation:
                parts.append(f"**Previous content ended with:**\n{continuation['previous_chunk_summary']}")

            # Add previous headings for context
            if "previous_headings" in continuation:
                headings = continuation["previous_headings"]
                parts.append(f"**Previous section structure:**")
                parts.append(f"- Level 1: {headings.get('level_1', 'Unknown')}")
                parts.append(f"- Level 2: {headings.get('level_2', 'Unknown')}")
                parts.append(f"- Level 3: {headings.get('level_3', 'Unknown')}")

            if "previous_end_page" in continuation:
                parts.append(f"**Previous batch ended on page:** {continuation['previous_end_page']}")

    # Legacy context support
    last_chunks = context.get("last_chunks", [])
    if last_chunks and not context.get("continuation_context"):
        parts.append("**Recent chunks from previous batch:**")
        for i, chunk in enumerate(last_chunks[-2:], 1):
            if isinstance(chunk, str):
                preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                parts.append(f"{i}. {preview}")

    # Heading hierarchy
    heading_hierarchy = context.get("heading_hierarchy", {})
    if heading_hierarchy:
        parts.append("**Document structure context:**")
        for level, title in heading_hierarchy.items():
            parts.append(f"- {level}: {title}")

    # Processing metadata
    if "processing_metadata" in context:
        metadata = context["processing_metadata"]
        if "batch_number" in metadata:
            parts.append(f"**Current batch number:** {metadata['batch_number']}")

    return "\n\n".join(parts) if parts else "No specific context from previous batch."


__all__ = [
    "DEFAULT_MANUAL_PROMPT_TEMPLATE",
    "build_manual_prompt",
    "get_default_manual_prompt",
]
