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
from ..models.chunk_schema import ChunkOutput, BatchProcessingResult, parse_structured_llm_output
from ..prompts.manual import DEFAULT_MANUAL_PROMPT_TEMPLATE, build_manual_prompt

logger = logging.getLogger(__name__)


class LMMProcessor:
    """Interface with the OpenRouter API to process page batches."""

    DEFAULT_MODEL = "google/gemini-2.5-pro"

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
        # Read and normalize the API key once during initialization. Some
        # execution paths (like background threads started before
        # ``load_dotenv`` runs) may not have the environment variable loaded
        # yet, so we store the explicit value if provided and fall back to
        # resolving it dynamically during request execution.
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        self.base_url = base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.default_prompt = default_prompt or DEFAULT_MANUAL_PROMPT_TEMPLATE

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
    ) -> BatchProcessingResult:
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

        warnings: List[str] = []

        api_key = self._resolve_api_key()

        if not api_key:
            logger.warning(
                "OPENROUTER_API_KEY not set. Falling back to mock LMM output for batch %s.",
                batch.batch_id,
            )
            fallback_reason = (
                "OpenRouter API key not configured. Returned mock output instead of "
                "calling the live service."
            )
            warnings.append(fallback_reason)
            raw_output = self._build_mock_output(
                batch,
                prompt_to_use,
                context,
                reason=fallback_reason,
            )
        else:
            try:
                response_json = self._send_request(payload, api_key=api_key)
                raw_output = self._extract_output(response_json)
            except RuntimeError as api_error:
                if self._is_authentication_error(api_error):
                    fallback_reason = (
                        "OpenRouter authentication failed. Verify your OPENROUTER_API_KEY "
                        "and account status. Returning mock output so processing can "
                        "continue."
                    )
                    logger.warning(
                        "Authentication error when calling OpenRouter for batch %s: %s",
                        batch.batch_id,
                        api_error,
                    )
                    warnings.append(fallback_reason)
                    raw_output = self._build_mock_output(
                        batch,
                        prompt_to_use,
                        context,
                        reason=fallback_reason,
                    )
                else:
                    raise

        processing_time = time.time() - start_time

        # Parse structured output using new schema
        try:
            structured_chunks = parse_structured_llm_output(raw_output)
        except Exception as e:
            logger.warning(f"Failed to parse structured output for batch {batch.batch_id}: {e}")
            # Fallback to basic parsing
            structured_chunks = parse_structured_llm_output(raw_output)

        # Get last chunk for context continuity
        last_chunk = structured_chunks[-1] if structured_chunks else None

        # Build continuation context for next batch
        continuation_context = None
        if last_chunk and last_chunk.continues_to_next:
            continuation_context = {
                "previous_chunk_summary": last_chunk.content[-300:] if len(last_chunk.content) > 300 else last_chunk.content,
                "previous_headings": {
                    "level_1": last_chunk.level_1,
                    "level_2": last_chunk.level_2,
                    "level_3": last_chunk.level_3,
                },
                "previous_end_page": last_chunk.end_page,
                "expects_continuation": True,
            }

        # Build processing metadata
        processing_metadata = {
            "prompt_used": prompt_to_use,
            "model": model_to_use,
            "processing_time": processing_time,
            "warnings": warnings,
            "batch_id": batch.batch_id,
            "batch_number": batch.batch_number,
            "chunk_count": len(structured_chunks),
        }

        return BatchProcessingResult(
            chunks=structured_chunks,
            raw_output=raw_output,
            last_chunk=last_chunk,
            continuation_context=continuation_context,
            processing_metadata=processing_metadata,
        )

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
    def _is_authentication_error(self, error: Exception) -> bool:
        message = str(error).lower()
        return any(keyword in message for keyword in ["401", "403", "unauthorized", "forbidden", "authentication"])

    def _resolve_api_key(self) -> Optional[str]:
        """Return a normalized API key, refreshing from the environment."""

        if self.api_key:
            return self.api_key

        env_value = os.getenv("OPENROUTER_API_KEY")
        if env_value:
            self.api_key = env_value.strip()
            if self.api_key:
                return self.api_key

        return None

    def _send_request(self, payload: Dict[str, Any], *, api_key: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise RuntimeError(f"Failed to reach OpenRouter API: {exc}") from exc

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
                error_text = response.text.strip()
                if response.status_code in (401, 403):
                    raise RuntimeError(
                        f"OpenRouter API authentication error {response.status_code}: {error_text or 'Authentication failed'}"
                    )
                raise RuntimeError(
                    f"OpenRouter API error {response.status_code}: {error_text or 'Unknown error'}"
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
        self,
        batch: PageBatch,
        prompt: str,
        context: Dict[str, Any],
        *,
        reason: Optional[str] = None,
    ) -> str:
        parts = [
            f"# Mock Chunking Output for Batch {batch.batch_number}",
            f"Prompt Summary: {prompt[:120]}{'...' if len(prompt) > 120 else ''}",
        ]

        if reason:
            parts.append(f"⚠️ {reason}")

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
