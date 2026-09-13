"""LLM providers behind a common interface.

The pipeline only ever calls :meth:`LLMClient.complete`, so switching between a
local Ollama model, an OpenAI-compatible endpoint, or Anthropic is a config
change.
"""

from __future__ import annotations

import logging
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator

import httpx

from .config import Settings
from .http_utils import describe_http_error

logger = logging.getLogger(__name__)

# Reasoning models sometimes inline their scratchpad in the visible content.
# Strip it so chain-of-thought never leaks into a user-facing answer.
_THINK_TAG_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    """Raised when a generation backend fails."""


def strip_reasoning(text: str, *, trim: bool = True) -> str:
    """Remove ``<think>``-style reasoning blocks and surrounding whitespace."""
    text = _THINK_TAG_RE.sub("", text)
    # Stream fragments may carry word separators or only part of a word.
    return text.strip() if trim else text


class LLMClient(ABC):
    """Generates a completion from a system and user prompt."""

    name: str = "abstract"

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's response text.

        Args:
            system: System prompt establishing the grounding rules.
            user: User message carrying the query and retrieved context.
            temperature: Overrides the configured temperature when given.
            max_tokens: Overrides the configured token budget when given.
        """

    def stream_complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield incremental text fragments as the model produces them."""
        yield self.complete(system, user, temperature=temperature, max_tokens=max_tokens)


class OllamaLLM(LLMClient):
    """Chat completions from a local or Ollama-cloud model via ``/api/chat``.

    Reasoning models (``deepseek-v4-flash:cloud`` and friends) return their
    scratchpad in a separate ``thinking`` field, which is logged but never
    returned as the answer.
    """

    def __init__(self, settings: Settings, model: str | None = None) -> None:
        self._model = model or settings.llm_model
        self._url = settings.ollama_base_url.rstrip("/") + "/api/chat"
        self._timeout = settings.ollama_timeout
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self.name = f"ollama:{self._model}"

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a two-message chat request and return the assistant content."""
        budget = self._max_tokens if max_tokens is None else max_tokens
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self._temperature if temperature is None else temperature,
                "num_predict": budget,
            },
        }
        try:
            response = httpx.post(self._url, json=payload, timeout=self._timeout)
            response.raise_for_status()
            body = response.json()
            message = body.get("message", {})
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Ollama chat request to {self._model} failed ({exc}). Is the server "
                f"running at {self._url}?"
            ) from exc

        thinking = message.get("thinking") or ""
        if thinking:
            logger.debug("model reasoning (%d chars) suppressed", len(thinking))

        content = strip_reasoning(message.get("content", ""))
        if not content:
            # On a reasoning model the token budget covers the hidden reasoning,
            # so exhausting it leaves nothing for the visible answer. Name that
            # cause explicitly -- "empty completion" alone sends you hunting the
            # prompt for a fault that is really a budget setting.
            if body.get("done_reason") == "length":
                raise LLMError(
                    f"{self._model} exhausted its {budget}-token budget while "
                    f"reasoning ({len(thinking)} chars) and produced no answer. "
                    f"Raise FAQRAG_LLM_MAX_TOKENS or FAQRAG_RERANK_MAX_TOKENS."
                )
            raise LLMError(f"{self._model} returned an empty completion")
        return content

    def stream_complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        budget = self._max_tokens if max_tokens is None else max_tokens
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "options": {
                "temperature": self._temperature if temperature is None else temperature,
                "num_predict": budget,
            },
        }
        try:
            with httpx.stream("POST", self._url, json=payload, timeout=self._timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    body = json.loads(line)
                    message = body.get("message", {})
                    content = strip_reasoning(message.get("content", ""), trim=False)
                    if content:
                        yield content
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Ollama chat stream to {self._model} failed ({exc}). Is the server "
                f"running at {self._url}?"
            ) from exc


class OpenAILLM(LLMClient):
    """Chat completions from an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(self, settings: Settings, model: str | None = None) -> None:
        if not settings.openai_api_key:
            raise LLMError("llm_provider='openai' requires FAQRAG_OPENAI_API_KEY to be set")
        self._model = model or settings.llm_model
        self._url = settings.openai_base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self.name = f"openai:{self._model}"

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant content."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
        }
        try:
            response = httpx.post(self._url, json=payload, headers=self._headers, timeout=120.0)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            raise LLMError(
                f"OpenAI chat request failed: {describe_http_error(exc)}"
            ) from exc
        return strip_reasoning(content or "")

    def stream_complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }
        try:
            with httpx.stream(
                "POST", self._url, json=payload, headers=self._headers, timeout=120.0
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        break
                    body = json.loads(raw)
                    delta = body["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield strip_reasoning(delta, trim=False)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"OpenAI chat stream failed: {describe_http_error(exc)}"
            ) from exc


class AnthropicLLM(LLMClient):
    """Completions from the Anthropic Messages API."""

    def __init__(self, settings: Settings, model: str | None = None) -> None:
        if not settings.anthropic_api_key:
            raise LLMError("llm_provider='anthropic' requires FAQRAG_ANTHROPIC_API_KEY to be set")
        self._model = model or settings.llm_model
        self._url = settings.anthropic_base_url.rstrip("/") + "/messages"
        self._headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": settings.anthropic_version,
            "content-type": "application/json",
        }
        self._timeout = settings.anthropic_timeout
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self.name = f"anthropic:{self._model}"

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a Messages API request and return the concatenated text blocks."""
        payload = {
            "model": self._model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "temperature": self._temperature if temperature is None else temperature,
        }
        try:
            response = httpx.post(
                self._url, json=payload, headers=self._headers, timeout=self._timeout
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Anthropic request to {self._model} failed: {describe_http_error(exc)}"
            ) from exc

        blocks = body.get("content", [])
        content = strip_reasoning(
            "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        )
        if not content:
            # A reply can carry no text block at all: the budget ran out before
            # the answer, or a safety classifier declined the request.
            if body.get("stop_reason") == "max_tokens":
                raise LLMError(
                    f"{self._model} hit its {payload['max_tokens']}-token budget "
                    f"before finishing an answer. Raise FAQRAG_LLM_MAX_TOKENS or "
                    f"FAQRAG_RERANK_MAX_TOKENS."
                )
            raise LLMError(
                f"{self._model} returned no text (stop_reason="
                f"{body.get('stop_reason')!r})"
            )
        return content

    def stream_complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        payload = {
            "model": self._model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "temperature": self._temperature if temperature is None else temperature,
            "stream": True,
        }
        try:
            with httpx.stream(
                "POST", self._url, json=payload, headers=self._headers, timeout=self._timeout
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        break
                    body = json.loads(raw)
                    if body.get("type") != "content_block_delta":
                        continue
                    delta = body.get("delta", {}).get("text", "")
                    if delta:
                        yield strip_reasoning(delta, trim=False)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Anthropic stream to {self._model} failed: {describe_http_error(exc)}"
            ) from exc


_PROVIDERS: dict[str, type[LLMClient]] = {
    "ollama": OllamaLLM,
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
}


def build_llm(settings: Settings, model: str | None = None) -> LLMClient | None:
    """Instantiate the configured LLM client.

    Returns ``None`` for the ``extractive`` provider, which answers directly
    from retrieved FAQ text without a model.
    """
    if settings.llm_provider == "extractive":
        return None
    try:
        provider_cls = _PROVIDERS[settings.llm_provider]
    except KeyError:
        raise LLMError(
            f"unknown llm provider {settings.llm_provider!r}; "
            f"expected one of {sorted(_PROVIDERS)} or 'extractive'"
        ) from None
    return provider_cls(settings, model)
