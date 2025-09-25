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
    context_json = json.dumps(context_dict, ensure_ascii=False, indent=2)

    if "{context_placeholder}" in prompt_template:
        return prompt_template.replace("{context_placeholder}", context_json)

    return (
        f"{prompt_template}\n\n## CONTEXT FROM PREVIOUS BATCH:\n{context_json}"
    )


__all__ = [
    "DEFAULT_MANUAL_PROMPT_TEMPLATE",
    "build_manual_prompt",
    "get_default_manual_prompt",
]
