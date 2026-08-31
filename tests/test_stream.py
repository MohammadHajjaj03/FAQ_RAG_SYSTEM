"""Tests for incremental answer generation.

Streaming re-derives, delta by delta, what the blocking path computes in one
pass. The risk is therefore divergence: text reaching the user that
``parse_sources`` would have removed, or a refusal leaking out as answer text
because the decision arrives after the first token is already on the wire.

Every test here pins the streaming path to the blocking path's output rather
than to a hand-written expectation, so the two cannot drift apart.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from faqrag.config import Settings
from faqrag.generate import (
    AnswerGenerator,
    StreamDelta,
    StreamFinal,
    StreamReset,
    _SourcesLineFilter,
    parse_sources,
)
from faqrag.llm import LLMClient, LLMError
from faqrag.models import Chunk, RetrievalResult, ScoredChunk
from faqrag.prompts import INSUFFICIENT_CONTEXT_MARKER, no_match_message


def make_scored(faq_id: str, lang: str = "en") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"{faq_id}::{lang}",
            faq_id=faq_id,
            category="Test",
            lang=lang,  # type: ignore[arg-type]
            question=f"Question {faq_id}?",
            answer=f"Answer for {faq_id}.",
            keywords=["kw"],
        ),
        score=0.9,
        relevance=0.9,
    )


def make_result(*faq_ids: str, confident: bool = True, lang: str = "en") -> RetrievalResult:
    return RetrievalResult(
        query="test query",
        query_lang=lang,  # type: ignore[arg-type]
        chunks=[make_scored(fid, lang) for fid in faq_ids],
        confident=confident,
        threshold=0.45,
    )


class StreamingStubLLM(LLMClient):
    """Yields a canned reply in fixed-size pieces, optionally failing part-way.

    ``chunk_size`` matters: a provider chooses its own delta boundaries, so any
    logic that inspects partial text has to hold for every possible split.
    """

    def __init__(
        self,
        reply: str = "",
        chunk_size: int = 3,
        fail_after: int | None = None,
    ) -> None:
        self.reply = reply
        self.chunk_size = chunk_size
        self.fail_after = fail_after
        self.stream_calls = 0

    def complete(self, system, user, temperature=None, max_tokens=None) -> str:
        return self.reply

    def stream(self, system, user, temperature=None, max_tokens=None) -> Iterator[str]:
        self.stream_calls += 1
        for index in range(0, len(self.reply), self.chunk_size):
            if self.fail_after is not None and index >= self.fail_after:
                raise LLMError("stub failure mid-stream")
            yield self.reply[index : index + self.chunk_size]


def collect(events) -> tuple[str, StreamFinal, list[StreamReset]]:
    """Drain a generate_stream generator into (streamed_text, final, resets)."""
    text: list[str] = []
    resets: list[StreamReset] = []
    final: StreamFinal | None = None
    for event in events:
        if isinstance(event, StreamDelta):
            text.append(event.text)
        elif isinstance(event, StreamReset):
            resets.append(event)
            # A reset invalidates everything shown so far, exactly as a client
            # is required to do.
            text.clear()
        else:
            final = event
    assert final is not None, "generate_stream must always end with a StreamFinal"
    return "".join(text), final, resets


REPLY_WITH_SOURCES = """You can pay with Tabby, Tamara, and mada.
SOURCES: 011"""

REPLY_SOURCES_MID_LINE = """Our sources: are listed on the platform.
SOURCES: 011"""

REPLY_MULTI_PARAGRAPH = """First paragraph here.

Second paragraph here.
SOURCES: 011, 012"""

REPLY_NO_SOURCES_LINE = "A bare answer with no citation line."

REPLY_INDENTED_SOURCES = """Answer body.
   SOURCES: 011"""

ALL_REPLIES = [
    REPLY_WITH_SOURCES,
    REPLY_SOURCES_MID_LINE,
    REPLY_MULTI_PARAGRAPH,
    REPLY_NO_SOURCES_LINE,
    REPLY_INDENTED_SOURCES,
    "Sorry, booking is required first.\nSOURCES: 011",
    "تقدر تدفع بتابي وتمارا.\nSOURCES: 011",
]


class TestSourcesLineFilter:
    """The filter must be indistinguishable from parse_sources, at any split."""

    @pytest.mark.parametrize("reply", ALL_REPLIES)
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 500])
    def test_matches_the_blocking_parser(self, reply: str, chunk_size: int) -> None:
        """Whatever the delta boundaries, the streamed text equals the parsed body.

        Pinning to ``parse_sources`` rather than a literal keeps the two
        implementations honest: if either changes, this fails.
        """
        expected, _ = parse_sources(reply, {"011", "012"})

        filt = _SourcesLineFilter()
        streamed = ""
        for index in range(0, len(reply), chunk_size):
            streamed += filt.feed(reply[index : index + chunk_size])
        streamed += filt.flush()

        assert streamed.strip() == expected.strip()

    @pytest.mark.parametrize("chunk_size", [1, 4])
    def test_never_emits_the_citation_line(self, chunk_size: int) -> None:
        filt = _SourcesLineFilter()
        streamed = ""
        for index in range(0, len(REPLY_WITH_SOURCES), chunk_size):
            streamed += filt.feed(REPLY_WITH_SOURCES[index : index + chunk_size])
        streamed += filt.flush()
        assert "SOURCES" not in streamed.upper()

    def test_keeps_a_sentence_that_merely_contains_sources(self) -> None:
        """Regression guard: an early version tested each newline-split fragment as
        if it began a line, which silently swallowed "Our sources: ..." mid-answer.
        """
        filt = _SourcesLineFilter()
        streamed = ""
        for char in REPLY_SOURCES_MID_LINE:
            streamed += filt.feed(char)
        streamed += filt.flush()
        assert "Our sources: are listed on the platform." in streamed


class TestGenerateStream:
    """Streaming generation, with the provider stubbed."""

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings()

    @pytest.mark.parametrize("chunk_size", [1, 3, 500])
    def test_deltas_concatenate_to_the_final_answer(
        self, settings: Settings, chunk_size: int
    ) -> None:
        """The core contract: rendering deltas then replacing them with the final
        answer must never make the text change."""
        stub = StreamingStubLLM(REPLY_WITH_SOURCES, chunk_size=chunk_size)
        text, final, resets = collect(
            AnswerGenerator(settings, stub).generate_stream(make_result("011"))
        )
        assert resets == []
        assert text.strip() == final.answer
        assert final.cited_faq_ids == ["011"]
        assert "SOURCES" not in text.upper()

    def test_agrees_with_the_blocking_path(self, settings: Settings) -> None:
        """Both paths answer the same question the same way."""
        result = make_result("011", "012")
        blocking_answer, blocking_cited = AnswerGenerator(
            settings, StreamingStubLLM(REPLY_MULTI_PARAGRAPH)
        ).generate(result)
        _, final, _ = collect(
            AnswerGenerator(settings, StreamingStubLLM(REPLY_MULTI_PARAGRAPH)
                            ).generate_stream(result)
        )
        assert final.answer == blocking_answer
        assert final.cited_faq_ids == blocking_cited

    def test_declines_without_calling_the_model(self, settings: Settings) -> None:
        """A weak match is refused before generation, streaming or not."""
        stub = StreamingStubLLM(REPLY_WITH_SOURCES)
        text, final, _ = collect(
            AnswerGenerator(settings, stub).generate_stream(
                make_result("011", confident=False)
            )
        )
        assert stub.stream_calls == 0, "the LLM must not be called on a weak match"
        assert final.cited_faq_ids == []
        assert text == no_match_message("en", settings.answer_style)

    def test_refusal_marker_leaks_no_answer_text(self, settings: Settings) -> None:
        """When the model declines, the user must see only the refusal.

        The marker arrives as the first thing the model writes, so the opening is
        held back until it can be ruled out -- otherwise a partial refusal would
        already be rendered as though it were an answer.
        """
        stub = StreamingStubLLM(INSUFFICIENT_CONTEXT_MARKER, chunk_size=2)
        text, final, resets = collect(
            AnswerGenerator(settings, stub).generate_stream(make_result("011"))
        )
        assert resets == []
        assert final.cited_faq_ids == []
        assert text == no_match_message("en", settings.answer_style)
        assert INSUFFICIENT_CONTEXT_MARKER not in text

    def test_late_refusal_retracts_what_was_streamed(self, settings: Settings) -> None:
        """A marker after real text cannot be un-sent, so it must be retracted."""
        reply = "Here is a long enough opening to be released.\n" + INSUFFICIENT_CONTEXT_MARKER
        stub = StreamingStubLLM(reply, chunk_size=5)
        text, final, resets = collect(
            AnswerGenerator(settings, stub).generate_stream(make_result("011"))
        )
        assert len(resets) == 1, "a retraction must be signalled"
        assert final.cited_faq_ids == []
        assert text == no_match_message("en", settings.answer_style)

    def test_mid_stream_failure_falls_back_to_retrieved_text(
        self, settings: Settings
    ) -> None:
        """An LLM outage degrades to the FAQ text, never to invented content.

        The failure lands after text has been released, so the fallback answer
        replaces what the user was already reading -- hence the retraction.
        """
        stub = StreamingStubLLM(REPLY_WITH_SOURCES, chunk_size=5, fail_after=30)
        text, final, resets = collect(
            AnswerGenerator(settings, stub).generate_stream(make_result("011"))
        )
        assert len(resets) == 1, "text was already shown, so it must be retracted"
        assert final.answer == "Answer for 011."
        assert final.cited_faq_ids == ["011"]
        assert text == "Answer for 011."

    def test_early_failure_needs_no_retraction(self, settings: Settings) -> None:
        """Failing before any text is released is the quiet case.

        Nothing reached the user, so emitting a reset would tell a client to
        discard something it never displayed.
        """
        stub = StreamingStubLLM(REPLY_WITH_SOURCES, chunk_size=5, fail_after=10)
        text, final, resets = collect(
            AnswerGenerator(settings, stub).generate_stream(make_result("011"))
        )
        assert resets == [], "nothing was streamed, so nothing to retract"
        assert final.answer == "Answer for 011."
        assert text == "Answer for 011."

    def test_extractive_mode_streams_the_faq_answer(self, settings: Settings) -> None:
        """With no LLM configured the top FAQ answer is streamed verbatim."""
        text, final, resets = collect(
            AnswerGenerator(settings, None).generate_stream(make_result("011"))
        )
        assert resets == []
        assert text == "Answer for 011."
        assert final.cited_faq_ids == ["011"]

    def test_missing_citation_line_attributes_to_the_top_chunk(
        self, settings: Settings
    ) -> None:
        """An uncited answer is still attributed, exactly as in generate()."""
        stub = StreamingStubLLM(REPLY_NO_SOURCES_LINE)
        text, final, _ = collect(
            AnswerGenerator(settings, stub).generate_stream(make_result("011", "012"))
        )
        assert final.cited_faq_ids == ["011"]
        assert text.strip() == REPLY_NO_SOURCES_LINE


class TestDefaultStreamFallback:
    """A provider with no streaming endpoint still satisfies the interface."""

    def test_non_streaming_provider_yields_one_chunk(self) -> None:
        class BlockingOnly(LLMClient):
            def complete(self, system, user, temperature=None, max_tokens=None) -> str:
                return "One shot answer."

        assert list(BlockingOnly().stream("sys", "user")) == ["One shot answer."]
