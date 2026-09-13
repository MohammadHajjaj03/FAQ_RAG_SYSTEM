"""The end-to-end RAG pipeline shared by the CLI, the API, and the eval harness."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator

from .config import Settings, get_settings
from .generate import AnswerGenerator, clean_answer_text, parse_sources, to_citations
from .lang import detect_language
from .llm import build_llm
from .logging_utils import build_trace, log_retrieval_summary, write_trace
from .models import QueryResponse, RetrievalResult, StreamEvent
from .prompts import INSUFFICIENT_CONTEXT_MARKER, casual_reply
from .retriever import HybridRetriever

logger = logging.getLogger(__name__)
_STREAM_SOURCES_MARKER = "SOURCES:"


class _SpeechChunkBuffer:
    """Hold token fragments until they form a clean sentence for TTS."""

    def __init__(self, max_chars: int = 180) -> None:
        self._pending = ""
        self._max_chars = max_chars
        self._emitted = False
        self._separator_pending = False

    def _clean_chunk(self, text: str) -> str:
        chunk = clean_answer_text(text)
        if not chunk:
            self._separator_pending |= bool(text)
            return ""
        if self._emitted and (self._separator_pending or text[:1].isspace()):
            chunk = " " + chunk
        self._emitted = True
        self._separator_pending = text[-1:].isspace()
        return chunk

    def add(self, text: str) -> list[str]:
        self._pending += text
        completed: list[str] = []
        while True:
            match = re.search(r"[.!?\u061f](?:\s|$)", self._pending)
            if match:
                completed.append(self._clean_chunk(self._pending[: match.end()]))
                self._pending = self._pending[match.end() :]
                continue
            if len(self._pending) >= self._max_chars:
                split_at = self._pending.rfind(" ", 0, self._max_chars)
                if split_at <= 0:
                    split_at = self._max_chars
                completed.append(self._clean_chunk(self._pending[:split_at]))
                self._pending = self._pending[split_at:]
                continue
            break
        return [chunk for chunk in completed if chunk]

    def flush(self) -> list[str]:
        chunk = self._clean_chunk(self._pending)
        self._pending = ""
        return [chunk] if chunk else []


class RagPipeline:
    """Retrieval plus grounded generation, with per-query tracing.

    Construct once and reuse: building it loads the index and the embedding
    client, which should not happen per request.
    """

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.generator = generator

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RagPipeline":
        """Build a pipeline from configuration, loading the persisted index."""
        settings = settings or get_settings()
        retriever = HybridRetriever.from_settings(settings)
        generator = AnswerGenerator(settings, build_llm(settings))
        return cls(settings, retriever, generator)

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        """Run retrieval only, without generating an answer."""
        return self.retriever.retrieve(question, top_k)

    def _casual_response(self, question: str, started: float) -> QueryResponse | None:
        """Reply to standalone social messages without invoking RAG or an LLM."""
        language = detect_language(question)
        answer = casual_reply(question, language)
        if answer is None:
            return None
        return QueryResponse(
            answer=answer,
            confidence=1.0,
            confident=True,
            language=language,
            query=question,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
        )

    def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        """Answer ``question`` from the FAQ corpus.

        Args:
            question: The user's question in Arabic or English.
            top_k: Chunks to retrieve; defaults to ``settings.top_k``.

        Returns:
            A :class:`QueryResponse` with the answer, cited FAQ ids, the source
            chunks, a confidence score, and the detected language. When nothing
            clears the relevance threshold, ``confident`` is ``False`` and the
            answer says so rather than guessing.
        """
        started = time.perf_counter()
        casual = self._casual_response(question, started)
        if casual is not None:
            return casual
        result = self.retriever.retrieve(question, top_k)
        log_retrieval_summary(logger, result)

        answer_text, cited_ids = self.generator.generate(result)
        latency_ms = (time.perf_counter() - started) * 1000.0

        confidence = max((c.relevance or 0.0 for c in result.chunks), default=0.0)
        response = QueryResponse(
            answer=answer_text,
            cited_faq_ids=cited_ids,
            sources=to_citations(result.chunks, cited_ids) if result.confident else [],
            # Always reported, including on a refusal: seeing what retrieval
            # surfaced is exactly what you need to debug a wrong refusal.
            retrieved=to_citations(result.chunks, []),
            confidence=round(confidence, 4),
            confident=result.confident,
            language=result.query_lang,
            query=question,
            reranked=result.reranked,
            cross_lingual_fallback=result.cross_lingual_fallback,
            latency_ms=round(latency_ms, 1),
        )

        if self.settings.log_retrieval_traces:
            write_trace(
                self.settings.log_dir,
                build_trace(
                    result,
                    extra={
                        "answer": answer_text,
                        "cited_faq_ids": cited_ids,
                        "confidence": response.confidence,
                        "latency_ms": response.latency_ms,
                    },
                ),
            )
        return response

    def stream_answer(self, question: str, top_k: int | None = None) -> Iterator[StreamEvent]:
        """Yield structured events for low-latency HTTP streaming."""
        started = time.perf_counter()
        casual = self._casual_response(question, started)
        if casual is not None:
            yield StreamEvent(
                event="metadata",
                data={
                    "query": question,
                    "language": casual.language,
                    "confident": True,
                    "confidence": 1.0,
                    "reranked": False,
                    "cross_lingual_fallback": False,
                    "social": True,
                },
            )
            yield StreamEvent(event="delta", data={"text": casual.answer})
            yield StreamEvent(event="final", data=casual.model_dump())
            return
        result = self.retriever.retrieve(question, top_k)
        log_retrieval_summary(logger, result)

        confidence = max((c.relevance or 0.0 for c in result.chunks), default=0.0)
        yield StreamEvent(
            event="metadata",
            data={
                "query": question,
                "language": result.query_lang,
                "confident": result.confident,
                "confidence": round(confidence, 4),
                "reranked": result.reranked,
                "cross_lingual_fallback": result.cross_lingual_fallback,
            },
        )

        parts: list[str] = []
        emitted_raw_length = 0
        speech_buffer = _SpeechChunkBuffer()
        for chunk in self.generator.stream_generate(result):
            if not chunk:
                continue
            parts.append(chunk)
            current = "".join(parts)
            marker_index = current.upper().find(_STREAM_SOURCES_MARKER)
            # Retain a small tail so a marker split between provider tokens can
            # never reach the client or be spoken by the voice engine.
            visible_end = marker_index if marker_index != -1 else max(0, len(current) - 7)
            newly_visible = current[emitted_raw_length:visible_end]
            emitted_raw_length = visible_end
            for delta_text in speech_buffer.add(newly_visible):
                yield StreamEvent(event="delta", data={"text": delta_text})

        # Release the marker look-behind when the model finishes without sources.
        current = "".join(parts)
        marker_index = current.upper().find(_STREAM_SOURCES_MARKER)
        visible_end = marker_index if marker_index != -1 else len(current)
        for delta_text in speech_buffer.add(current[emitted_raw_length:visible_end]):
            yield StreamEvent(event="delta", data={"text": delta_text})
        for delta_text in speech_buffer.flush():
            yield StreamEvent(event="delta", data={"text": delta_text})

        raw_answer = "".join(parts).strip()
        allowed = {item.chunk.faq_id for item in result.chunks}
        answer_text, cited_ids = parse_sources(raw_answer, allowed)
        if INSUFFICIENT_CONTEXT_MARKER in answer_text:
            answer_text = self.generator._no_match(result.query_lang)
            cited_ids = []
        elif not answer_text:
            answer_text, cited_ids = self.generator.generate(result)

        latency_ms = (time.perf_counter() - started) * 1000.0
        response = QueryResponse(
            answer=answer_text,
            cited_faq_ids=cited_ids,
            sources=to_citations(result.chunks, cited_ids) if result.confident else [],
            retrieved=to_citations(result.chunks, []),
            confidence=round(confidence, 4),
            confident=result.confident,
            language=result.query_lang,
            query=question,
            reranked=result.reranked,
            cross_lingual_fallback=result.cross_lingual_fallback,
            latency_ms=round(latency_ms, 1),
        )

        if self.settings.log_retrieval_traces:
            write_trace(
                self.settings.log_dir,
                build_trace(
                    result,
                    extra={
                        "answer": answer_text,
                        "cited_faq_ids": cited_ids,
                        "confidence": response.confidence,
                        "latency_ms": response.latency_ms,
                    },
                ),
            )

        yield StreamEvent(event="final", data=response.model_dump())
