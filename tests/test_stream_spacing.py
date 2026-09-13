import json
from contextlib import contextmanager

import httpx
import pytest

from faqrag.config import Settings
from faqrag.generate import AnswerGenerator
from faqrag.llm import AnthropicLLM, OllamaLLM, OpenAILLM
from faqrag.pipeline import RagPipeline, _SpeechChunkBuffer
from test_pipeline_stream import StubRetriever, make_result


@pytest.mark.parametrize("provider", [OllamaLLM, OpenAILLM, AnthropicLLM])
@pytest.mark.parametrize("with_sources", [False, True])
def test_provider_stream_preserves_word_boundaries(monkeypatch, provider, with_sources):
    fragments = [
        "Mw", "faq is Saudi Arabia's", " leading digital platform for auto",
        "mating and simplifying mandatory medical examinations.",
        " It operates through a unified national network of", " ",
        "certified medical providers.",
    ]
    expected = "".join(fragments)
    if with_sources:
        fragments += ["\nSOU", "RCES: 011"]
    lines = []
    for fragment in fragments:
        if provider is OllamaLLM:
            lines.append(json.dumps({"message": {"content": fragment}}))
        elif provider is OpenAILLM:
            lines.append("data: " + json.dumps({"choices": [{"delta": {"content": fragment}}]}))
        else:
            lines.append("data: " + json.dumps({"type": "content_block_delta", "delta": {"text": fragment}}))

    @contextmanager
    def mock_stream(*args, **kwargs):
        yield httpx.Response(200, text="\n".join(lines), request=httpx.Request("POST", "https://example.test"))

    monkeypatch.setattr(httpx, "stream", mock_stream)
    settings = Settings(log_retrieval_traces=False, openai_api_key="test", anthropic_api_key="test")
    client = provider(settings)
    assert "".join(client.stream_complete("system", "user")) == "".join(fragments)
    pipeline = RagPipeline(settings, StubRetriever(make_result()), AnswerGenerator(settings, client))
    events = list(pipeline.stream_answer("What is Mwfaq?"))
    assert "".join(event.data["text"] for event in events if event.event == "delta") == expected
    assert events[-1].data["answer"] == expected


@pytest.mark.parametrize("text", ["First sentence. Second sentence.", "Saudi Arabia's leading platform", "automating"])
def test_speech_chunks_concatenate_without_merging_or_splitting_words(text):
    buffer = _SpeechChunkBuffer(max_chars=8)
    chunks = []
    for character in text:
        chunks.extend(buffer.add(character))
    chunks.extend(buffer.flush())
    assert "".join(chunks) == text
