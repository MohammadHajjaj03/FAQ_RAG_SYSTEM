"""Grounded answer generation over retrieved FAQ chunks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

from .config import Settings
from .llm import LLMClient, LLMError, strip_reasoning
from .models import Language, RetrievalResult, ScoredChunk, SourceCitation
from .prompts import (
    INSUFFICIENT_CONTEXT_MARKER,
    build_answer_prompt,
    build_answer_system_prompt,
    no_match_message,
)

logger = logging.getLogger(__name__)

# Matches the trailing "SOURCES: 001, 007" line the system prompt mandates.
_SOURCES_RE = re.compile(r"^\s*SOURCES\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_FAQ_ID_RE = re.compile(r"[0-9A-Za-z_-]+")


def parse_sources(text: str, allowed_ids: set[str]) -> tuple[str, list[str]]:
    """Split a model reply into its answer body and cited FAQ ids.

    Citations are intersected with ``allowed_ids`` -- the FAQs actually placed in
    the context -- so a model that invents an id cannot produce a citation
    pointing at content it never saw.

    Args:
        text: The raw model reply.
        allowed_ids: FAQ ids that were supplied as context.

    Returns:
        ``(answer_without_sources_line, cited_ids)``, preserving citation order
        and dropping duplicates.
    """
    cited: list[str] = []
    for match in _SOURCES_RE.finditer(text):
        for token in _FAQ_ID_RE.findall(match.group(1)):
            if token in allowed_ids and token not in cited:
                cited.append(token)

    body = _SOURCES_RE.sub("", text).strip()
    return body, cited


@dataclass
class StreamDelta:
    """A piece of answer text, safe to append to what the user already sees."""

    text: str


@dataclass
class StreamReset:
    """Discard everything streamed so far and show ``answer`` instead.

    Emitted only when the model reveals *after* it has started writing that the
    context was insufficient. Streamed text cannot be recalled, so the contract
    has to let the server retract it explicitly rather than leaving a client
    displaying an answer the pipeline has already disowned.
    """

    reason: str


@dataclass
class StreamFinal:
    """End of generation: the settled answer text and its citations."""

    answer: str
    cited_faq_ids: list[str] = field(default_factory=list)


#: Anything the model could still turn into the mandated trailing SOURCES line.
_SOURCES_LINE_START = "sources:"


class _SourcesLineFilter:
    """Streams answer text while withholding the trailing ``SOURCES:`` line.

    The system prompt requires the model to end its reply with ``SOURCES: 011``,
    and :func:`parse_sources` strips that line from the final answer. A naive
    stream would show it to the user mid-typing before it could be removed, so
    text is released only once it cannot begin that line.

    Two properties keep this faithful to the non-streaming path:

    * Only a **line start** can begin a SOURCES line. Once ordinary text has
      been released for the current line, the rest of that line streams freely --
      otherwise a sentence such as "Our sources: are listed below" would be
      silently dropped, which ``parse_sources`` would never do.
    * The hold-back is bounded by the length of ``"sources:"``, so prose is
      released as it arrives and only a line genuinely opening "s", "so",
      "sou"... waits for the next delta to disambiguate.
    """

    def __init__(self) -> None:
        self._buffer = ""
        # Whether the next character would land at the start of a fresh line.
        # Only then can a SOURCES line begin.
        self._at_line_start = True

    @staticmethod
    def _may_be_sources(partial: str) -> bool:
        """True while ``partial`` could still grow into a SOURCES line."""
        candidate = partial.lstrip().lower()
        if not candidate:
            # Leading whitespace alone tells us nothing yet. Holding it costs a
            # single delta and avoids emitting the indent of a dropped line.
            return True
        return _SOURCES_LINE_START.startswith(candidate) or candidate.startswith(
            _SOURCES_LINE_START
        )

    def feed(self, delta: str) -> str:
        """Absorb ``delta`` and return the text that is now safe to emit."""
        self._buffer += delta
        out: list[str] = []

        while True:
            newline = self._buffer.find("\n")
            if newline >= 0:
                line = self._buffer[:newline]
                self._buffer = self._buffer[newline + 1 :]
                # A citation line is only a citation line when it opens one.
                if not (self._at_line_start and _SOURCES_RE.match(line)):
                    out.append(line + "\n")
                self._at_line_start = True
                continue

            if self._buffer and not (
                self._at_line_start and self._may_be_sources(self._buffer)
            ):
                out.append(self._buffer)
                self._buffer = ""
                # Text has been shown for this line, so nothing later in it can
                # turn out to have been a SOURCES line.
                self._at_line_start = False
            break

        return "".join(out)

    def flush(self) -> str:
        """Return what is left once the model stops, minus a trailing SOURCES line."""
        tail, self._buffer = self._buffer, ""
        if self._at_line_start and _SOURCES_RE.match(tail):
            return ""
        return tail

def to_citations(chunks: list[ScoredChunk], faq_ids: list[str]) -> list[SourceCitation]:
    """Build citation objects for ``faq_ids``, in the order the model cited them.

    Falls back to every retrieved chunk when the model cited nothing, so a UI
    always has something to render alongside the answer.
    """
    by_faq_id: dict[str, ScoredChunk] = {}
    for item in chunks:
        by_faq_id.setdefault(item.chunk.faq_id, item)

    selected = [by_faq_id[fid] for fid in faq_ids if fid in by_faq_id] or list(chunks)
    return [
        SourceCitation(
            faq_id=item.chunk.faq_id,
            category=item.chunk.category,
            lang=item.chunk.lang,
            question=item.chunk.question,
            answer=item.chunk.answer,
            score=round(item.relevance if item.relevance is not None else item.score, 4),
        )
        for item in selected
    ]


class AnswerGenerator:
    """Turns a :class:`RetrievalResult` into a grounded, cited answer.

    With ``llm_provider='extractive'`` (``client=None``) the top-ranked FAQ
    answer is returned verbatim. That mode cannot hallucinate by construction,
    at the cost of not synthesising across FAQs or matching the user's phrasing.
    """

    def __init__(self, settings: Settings, client: LLMClient | None) -> None:
        self.settings = settings
        self.client = client
        # Built once: the system prompt is fixed for the life of the generator.
        self.system_prompt = build_answer_system_prompt(settings.answer_style)

    def _no_match(self, lang: Language) -> str:
        """Return the refusal message in the configured voice."""
        return no_match_message(lang, self.settings.answer_style)

    def _extractive(self, result: RetrievalResult) -> tuple[str, list[str]]:
        top = result.chunks[0]
        return top.chunk.answer, [top.chunk.faq_id]

    def generate(self, result: RetrievalResult) -> tuple[str, list[str]]:
        """Generate an answer for ``result``.

        Returns:
            ``(answer_text, cited_faq_ids)``. When retrieval was not confident,
            or the model reports insufficient context, the answer is a fixed
            "I don't know" message in the query language and no ids are cited --
            the system declines rather than answering from a weak match.
        """
        # Refusing here, before the model ever sees a weak context, is what keeps
        # a low-relevance match from being dressed up as a confident answer.
        if not result.confident or not result.chunks:
            logger.info("no confident match for %r; declining to answer", result.query)
            return self._no_match(result.query_lang), []

        if self.client is None:
            return self._extractive(result)

        prompt = build_answer_prompt(result.query, result.chunks, result.query_lang)
        try:
            raw = self.client.complete(self.system_prompt, prompt)
        except LLMError as exc:
            # Falling back to the retrieved text keeps the system useful during
            # an LLM outage without ever inventing content.
            logger.error("generation failed (%s); falling back to extractive answer", exc)
            return self._extractive(result)

        allowed = {item.chunk.faq_id for item in result.chunks}
        answer, cited = parse_sources(raw, allowed)

        if INSUFFICIENT_CONTEXT_MARKER in answer:
            logger.info("model reported insufficient context for %r", result.query)
            return self._no_match(result.query_lang), []

        if not answer:
            logger.warning("model returned only a sources line; using extractive fallback")
            return self._extractive(result)

        if not cited:
            # The model answered but skipped the citation line. Attribute to the
            # top chunk rather than returning an uncited answer.
            logger.warning("model omitted the SOURCES line; citing the top chunk")
            cited = [result.chunks[0].chunk.faq_id]

        return answer, cited

    def generate_stream(
        self, result: RetrievalResult
    ) -> Iterator[StreamDelta | StreamReset | StreamFinal]:
        """Generate an answer for ``result``, yielding text as the model writes it.

        Emits zero or more :class:`StreamDelta` events, then exactly one
        :class:`StreamFinal`. A :class:`StreamReset` in between means the text
        already sent must be discarded -- see below.

        The settled answer in :class:`StreamFinal` is authoritative. Deltas are
        the same text arriving early, so a client that renders deltas and then
        replaces them with the final answer is always correct.

        Three things make streaming here more than a loop over the provider:

        * The mandated trailing ``SOURCES: 011`` line must never reach the user,
          so it is withheld by :class:`_SourcesLineFilter`.
        * A refusal must never appear as a streamed answer. The model emits
          ``INSUFFICIENT_CONTEXT`` alone when the context does not support an
          answer, so the opening of the reply is held back just long enough to
          rule that marker out before any text is released.
        * Should the model reveal insufficient context *after* it has begun
          writing -- or fail mid-stream -- text is already on the wire and cannot
          be recalled. That is what :class:`StreamReset` is for.
        """
        # Same refusal-before-generation rule as generate(): a weak context is
        # declined without the model ever seeing it.
        if not result.confident or not result.chunks:
            logger.info("no confident match for %r; declining to answer", result.query)
            answer = self._no_match(result.query_lang)
            yield StreamDelta(answer)
            yield StreamFinal(answer, [])
            return

        if self.client is None:
            answer, cited = self._extractive(result)
            yield StreamDelta(answer)
            yield StreamFinal(answer, cited)
            return

        prompt = build_answer_prompt(result.query, result.chunks, result.query_lang)
        allowed = {item.chunk.faq_id for item in result.chunks}
        sources_filter = _SourcesLineFilter()
        raw_parts: list[str] = []
        probe = ""
        probe_done = False
        streamed_anything = False

        def release(text: str) -> Iterator[StreamDelta]:
            """Emit filtered text, if any survived the filter."""
            if text:
                yield StreamDelta(text)

        try:
            for delta in self.client.stream(self.system_prompt, prompt):
                raw_parts.append(delta)

                if not probe_done:
                    probe += delta
                    if INSUFFICIENT_CONTEXT_MARKER in probe:
                        # The expected shape of a decline: nothing has been sent,
                        # so the refusal is the only text the client ever sees.
                        logger.info(
                            "model reported insufficient context for %r", result.query
                        )
                        answer = self._no_match(result.query_lang)
                        yield StreamDelta(answer)
                        yield StreamFinal(answer, [])
                        return
                    if len(probe) < len(INSUFFICIENT_CONTEXT_MARKER):
                        continue  # not yet enough to rule the marker out
                    probe_done = True
                    pending = sources_filter.feed(probe)
                else:
                    pending = sources_filter.feed(delta)

                for event in release(pending):
                    streamed_anything = True
                    yield event

            if not probe_done:
                # The whole reply was shorter than the marker, so it was never
                # released above.
                for event in release(sources_filter.feed(probe)):
                    streamed_anything = True
                    yield event

            for event in release(sources_filter.flush()):
                streamed_anything = True
                yield event

        except LLMError as exc:
            # Same policy as generate(): fall back to retrieved text rather than
            # inventing content or failing the request outright.
            logger.error("streaming generation failed (%s); falling back", exc)
            answer, cited = self._extractive(result)
            if streamed_anything:
                yield StreamReset("generation failed mid-stream")
            yield StreamDelta(answer)
            yield StreamFinal(answer, cited)
            return

        raw = strip_reasoning("".join(raw_parts))
        answer, cited = parse_sources(raw, allowed)

        if INSUFFICIENT_CONTEXT_MARKER in answer:
            logger.info("model reported insufficient context for %r", result.query)
            answer = self._no_match(result.query_lang)
            if streamed_anything:
                yield StreamReset("model reported insufficient context")
                yield StreamDelta(answer)
            yield StreamFinal(answer, [])
            return

        if not answer:
            logger.warning("model returned only a sources line; using extractive fallback")
            answer, cited = self._extractive(result)
            if streamed_anything:
                yield StreamReset("model returned only a citation line")
            yield StreamDelta(answer)
            yield StreamFinal(answer, cited)
            return

        if not cited:
            logger.warning("model omitted the SOURCES line; citing the top chunk")
            cited = [result.chunks[0].chunk.faq_id]

        yield StreamFinal(answer, cited)
