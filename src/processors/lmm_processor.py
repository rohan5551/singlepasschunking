"""Large Multimodal Model (LMM) processor for batch chunking."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from ..models.batch_models import PageBatch

logger = logging.getLogger(__name__)


class LMMProcessor:
    """Interface with the OpenRouter API to process page batches."""

    DEFAULT_MODEL = "google/gemini-2.5-pro-exp"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: int = 60,
        default_prompt: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.default_prompt = default_prompt or (
            "You are a meticulous chunking assistant. Break the provided document "
            "pages into semantically coherent chunks, returning well formatted "
            "markdown with headings and short summaries for each chunk."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_batch(
        self,
        batch: PageBatch,
        *,
        context: Dict[str, Any],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process a single batch of pages with the LMM."""

        model_to_use = model or self.DEFAULT_MODEL
        prompt_to_use = prompt.strip() if prompt else self.default_prompt
        temperature_to_use = temperature if temperature is not None else self.temperature

        payload = self._build_payload(
            batch,
            context=context,
            prompt=prompt_to_use,
            model=model_to_use,
            temperature=temperature_to_use,
        )

        start_time = time.time()

        if not self.api_key:
            logger.warning(
                "OPENROUTER_API_KEY not set. Falling back to mock LMM output for batch %s.",
                batch.batch_id,
            )
            raw_output = self._build_mock_output(batch, prompt_to_use, context)
        else:
            response_json = self._send_request(payload)
            raw_output = self._extract_output(response_json)

        processing_time = time.time() - start_time

        parsed_chunks = self._parse_chunks(raw_output)
        heading_hierarchy = self._derive_heading_hierarchy(raw_output)
        continuation_metadata = self._derive_continuation_metadata(parsed_chunks, raw_output)
        last_chunks = parsed_chunks[-2:] if parsed_chunks else []

        context_snapshot = {
            "previous_context": context,
            "last_chunks": last_chunks,
            "heading_hierarchy": heading_hierarchy,
            "continuation_metadata": continuation_metadata,
        }

        return {
            "raw_output": raw_output,
            "chunks": parsed_chunks,
            "heading_hierarchy": heading_hierarchy,
            "continuation_metadata": continuation_metadata,
            "last_chunks": last_chunks,
            "context": context_snapshot,
            "prompt_used": prompt_to_use,
            "model": model_to_use,
            "processing_time": processing_time,
        }

    # ------------------------------------------------------------------
    # Payload preparation helpers
    # ------------------------------------------------------------------
    def _build_payload(
        self,
        batch: PageBatch,
        *,
        context: Dict[str, Any],
        prompt: str,
        model: str,
        temperature: float,
    ) -> Dict[str, Any]:
        message_context = self._format_context_for_prompt(context)
        batch_metadata = self._format_batch_metadata(batch)
        images = self._prepare_image_payloads(batch)

        user_text = (
            "## Chunking Instructions\n"
            "You will receive consecutive PDF page images. "
            "Apply the provided prompt to segment the content into high quality "
            "chunks with headings, summaries, and any relevant metadata."
            "\n\n"
            f"### User Prompt\n{prompt}\n\n"
            f"### Batch Metadata\n{batch_metadata}\n\n"
            f"### Previous Context\n{message_context}\n\n"
            "Return the response in markdown. Use the following guidelines:\n"
            "- Mark each chunk with a level-3 heading (###).\n"
            "- Provide a one sentence summary and bullet highlights when relevant.\n"
            "- Note if the chunk continues into the next batch."
        )

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        if images:
            user_content.extend(images)
        else:
            user_content.append(
                {
                    "type": "text",
                    "text": "[Image content unavailable – proceed using prior context only]",
                }
            )

        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a large multimodal model specialised in PDF chunking. "
                                "Preserve document structure, keep headings accurate, and "
                                "annotate continuation requirements clearly."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        }

        return payload

    def _prepare_image_payloads(self, batch: PageBatch) -> List[Dict[str, Any]]:
        images: List[Dict[str, Any]] = []

        for page in batch.pages:
            if not page.image:
                continue

            buffer = io.BytesIO()
            page.image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            data_url = f"data:image/png;base64,{encoded}"
            images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )

        return images

    def _format_batch_metadata(self, batch: PageBatch) -> str:
        return (
            f"Batch ID: {batch.batch_id}\n"
            f"Batch Number: {batch.batch_number}\n"
            f"Page Range: {batch.start_page}-{batch.end_page}\n"
            f"Page Count: {batch.page_count}"
        )

    def _format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        if not context:
            return "No previous context provided."

        parts: List[str] = []

        last_chunks = context.get("last_chunks") or []
        if last_chunks:
            chunks_text = "\n".join(f"- {chunk}" for chunk in last_chunks)
            parts.append(f"Recent chunks:\n{chunks_text}")

        heading_hierarchy = context.get("heading_hierarchy") or {}
        if heading_hierarchy:
            headings_text = "\n".join(
                f"  {level}: {title}" for level, title in heading_hierarchy.items()
            )
            parts.append(f"Heading hierarchy:\n{headings_text}")

        continuation_metadata = context.get("continuation_metadata") or {}
        if continuation_metadata:
            continuation_text = json.dumps(continuation_metadata, indent=2)
            parts.append(f"Continuation metadata:\n{continuation_text}")

        return "\n\n".join(parts) if parts else "No previous context provided."

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------
    def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self.max_retries + 1):
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 429:
                wait_time = min(2 ** attempt, 10)
                logger.warning(
                    "Rate limited by OpenRouter (attempt %s/%s). Retrying in %ss.",
                    attempt,
                    self.max_retries,
                    wait_time,
                )
                time.sleep(wait_time)
                continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"OpenRouter API error {response.status_code}: {response.text}"
                )

            return response.json()

        raise RuntimeError("Exceeded maximum retries when calling OpenRouter API")

    def _extract_output(self, response_json: Dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            raise ValueError("No choices returned from LMM response")

        message = choices[0].get("message", {})
        content = message.get("content")

        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(part for part in text_parts if part).strip()

        if isinstance(content, str):
            return content.strip()

        return ""

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parse_chunks(self, text: str) -> List[str]:
        if not text:
            return []

        chunks: List[str] = []
        current: List[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("###") or stripped.lower().startswith("chunk"):
                if current:
                    chunk_text = "\n".join(current).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    current = []
            current.append(line)

        if current:
            chunk_text = "\n".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

        if not chunks:
            single_chunk = text.strip()
            return [single_chunk] if single_chunk else []

        return chunks

    def _derive_heading_hierarchy(self, text: str) -> Dict[str, str]:
        hierarchy: Dict[str, str] = {}

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and "level1" not in hierarchy:
                hierarchy["level1"] = stripped[2:].strip()
            elif stripped.startswith("## ") and "level2" not in hierarchy:
                hierarchy["level2"] = stripped[3:].strip()
            elif stripped.startswith("### ") and "level3" not in hierarchy:
                hierarchy["level3"] = stripped[4:].strip()

            if len(hierarchy) == 3:
                break

        return hierarchy

    def _derive_continuation_metadata(
        self, chunks: List[str], raw_text: str
    ) -> Dict[str, Any]:
        continuation = {}
        if chunks:
            continuation["chunk_count"] = len(chunks)
            continuation["last_chunk_length"] = len(chunks[-1])

        if raw_text.strip().endswith("…") or raw_text.strip().endswith("..."):
            continuation["continues"] = True

        return continuation

    # ------------------------------------------------------------------
    # Mock helpers
    # ------------------------------------------------------------------
    def _build_mock_output(
        self, batch: PageBatch, prompt: str, context: Dict[str, Any]
    ) -> str:
        parts = [
            f"# Mock Chunking Output for Batch {batch.batch_number}",
            f"Prompt Summary: {prompt[:120]}{'...' if len(prompt) > 120 else ''}",
        ]

        if context.get("heading_hierarchy"):
            parts.append("## Estimated Heading Context")
            for level, heading in context["heading_hierarchy"].items():
                parts.append(f"- {level}: {heading}")

        for idx, page in enumerate(batch.page_numbers, start=1):
            parts.append(f"### Chunk {idx}: Pages {page}")
            parts.append(
                "This is a placeholder chunk generated because no API key was provided. "
                "Replace with real output by configuring OPENROUTER_API_KEY."
            )

        return "\n".join(parts)


__all__ = ["LMMProcessor"]
