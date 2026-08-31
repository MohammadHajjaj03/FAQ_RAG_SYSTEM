"""The end-to-end RAG pipeline shared by the CLI, the API, and the eval harness."""

from __future__ import annotations

import logging
import time
from typing import Iterator

from .config import Settings, get_settings
from .generate import (
    AnswerGenerator,
    StreamDelta,
    StreamFinal,
    StreamReset,
    to_citations,
)
from .llm import build_llm
from .logging_utils import build_trace, log_retrieval_summary, write_trace
from .models import QueryResponse, RetrievalResult
from .retriever import HybridRetriever

logger = logging.getLogger(__name__)


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
        result = self.retriever.retrieve(question, top_k)
        log_retrieval_summary(logger, result)

        answer_text, cited_ids = self.generator.generate(result)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._build_response(question, result, answer_text, cited_ids, latency_ms)

    def answer_stream(
        self, question: str, top_k: int | None = None
    ) -> Iterator[StreamDelta | StreamReset | QueryResponse]:
        """Answer ``question``, yielding answer text as the model produces it.

        Retrieval is unchanged and still happens up front -- only generation is
        incremental, so the first delta arrives after retrieval and reranking
        have completed, not immediately.

        Yields:
            :class:`StreamDelta` for each piece of answer text, possibly a
            :class:`StreamReset` instructing the consumer to discard what it has
            shown, and finally exactly one :class:`QueryResponse` -- the same
            object :meth:`answer` returns, so both paths agree on citations,
            confidence, and the trace written to disk.
        """
        started = time.perf_counter()
        result = self.retriever.retrieve(question, top_k)
        log_retrieval_summary(logger, result)

        answer_text = ""
        cited_ids: list[str] = []
        for event in self.generator.generate_stream(result):
            if isinstance(event, StreamFinal):
                answer_text, cited_ids = event.answer, event.cited_faq_ids
            else:
                yield event

        latency_ms = (time.perf_counter() - started) * 1000.0
        yield self._build_response(question, result, answer_text, cited_ids, latency_ms)

    def _build_response(
        self,
        question: str,
        result: RetrievalResult,
        answer_text: str,
        cited_ids: list[str],
        latency_ms: float,
    ) -> QueryResponse:
        """Assemble the wire response and write the retrieval trace.

        Shared by the blocking and streaming paths so the two cannot diverge in
        what they report.
        """
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
