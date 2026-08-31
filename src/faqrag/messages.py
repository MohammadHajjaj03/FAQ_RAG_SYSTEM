"""Asynchronous message API for integrating an external system.

Two endpoints form the contract:

* ``POST /v1/messages``       -- the other side hands us a question. Returns
  immediately with a ``message_id``; nothing is generated yet.
* ``GET  /v1/messages/{id}``  -- the other side collects the answer.

They are split because answering takes roughly ten seconds (retrieval, then
reranking, then generation). A synchronous endpoint would hold the caller's
socket open for that whole time, which browsers, gateways, and messaging
platforms all tend to cut short. Accepting the message instantly and letting the
caller collect the result decouples their timeout budget from ours.

Work runs on a small bounded thread pool rather than an unbounded set of
background tasks. The pipeline call is synchronous and CPU-idle-but-slow (it
waits on the model), so it must stay off the event loop; and because the model
serialises requests anyway, admitting more concurrent jobs than it can serve
would only convert latency into memory.

State is in-memory and TTL-bounded, which is the right scope here: a message is
a short-lived hand-off, not a record. Restarting the API drops in-flight
messages. Persisting them means swapping :class:`MessageStore` for a Redis or
database implementation -- the rest of this module does not care which.
"""

from __future__ import annotations

import asyncio
import json
import re
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Literal

from fastapi import (
    APIRouter,
    Body,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .generate import StreamDelta, StreamReset
from .models import QueryResponse, SourceCitation

logger = logging.getLogger(__name__)

MessageStatus = Literal["queued", "processing", "done", "failed"]

#: Longest a caller may block on GET waiting for an answer, in seconds.
MAX_WAIT_SECONDS = 60.0

#: Sent once when an SSE stream opens. Proxies buffer a response until enough
#: bytes have accumulated, which silences an otherwise-correct event stream when
#: it is served through a tunnel or CDN. Lines beginning with ":" are SSE
#: comments and are discarded by every client, so this only costs bandwidth.
SSE_PROXY_PADDING = ":" + " " * 2048 + "\n\n"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Message:
    """One question in flight, and its answer once ready."""

    message_id: str
    text: str
    status: MessageStatus = "queued"
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    response: QueryResponse | None = None
    error: str | None = None
    # Set when the message reaches a terminal state, so GET can block on it
    # instead of the caller hot-polling.
    _finished: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def terminal(self) -> bool:
        """Whether the message has finished, successfully or not."""
        return self.status in ("done", "failed")


class EventBroadcaster:
    """Fans events out to every connected Server-Sent Events listener.

    Answers are produced on worker *threads*, but SSE responses are served from
    the asyncio event loop. Bridging the two safely is the whole job here:
    :meth:`publish` is called from a worker thread and uses
    ``call_soon_threadsafe`` to hand each event to the loop, which is the only
    thread allowed to touch an ``asyncio.Queue``.

    Each subscriber gets its own bounded queue. A listener that stops reading
    (a laptop that slept, a tab that froze) therefore fills only its own queue
    and is dropped, rather than blocking the worker that is publishing.
    """

    #: Events buffered per listener before it is considered dead.
    QUEUE_SIZE = 100

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the event loop that worker threads will publish into."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        """Register a listener and return its queue."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.QUEUE_SIZE)
        with self._lock:
            self._subscribers.add(queue)
        logger.info("SSE listener connected (%d total)", len(self._subscribers))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Drop a listener."""
        with self._lock:
            self._subscribers.discard(queue)
        logger.info("SSE listener disconnected (%d remain)", len(self._subscribers))

    @property
    def listener_count(self) -> int:
        """How many listeners are currently connected."""
        with self._lock:
            return len(self._subscribers)

    def publish(self, event: str, data: dict[str, Any]) -> None:
        """Broadcast an event. Safe to call from any thread."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with self._lock:
            queues = list(self._subscribers)
        if not queues:
            return

        payload = {"event": event, "data": data}
        for queue in queues:
            # Hop to the event loop thread before touching the queue.
            loop.call_soon_threadsafe(self._offer, queue, payload)

    def _offer(self, queue: asyncio.Queue, payload: dict[str, Any]) -> None:
        """Enqueue for one listener, dropping it if it has stopped reading."""
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("SSE listener is not keeping up; dropping it")
            self.unsubscribe(queue)


class MessageStore:
    """Thread-safe, TTL-bounded store of in-flight and completed messages.

    Eviction is opportunistic (on write) rather than on a timer: there is no
    background thread to leak, and at this volume a sweep is trivially cheap.
    """

    def __init__(self, ttl_seconds: float = 3600.0, max_messages: int = 1000) -> None:
        self._ttl = ttl_seconds
        self._max = max_messages
        self._messages: dict[str, Message] = {}
        self._lock = threading.Lock()

    def _evict_locked(self) -> None:
        """Drop expired messages, then the oldest if still over capacity."""
        cutoff = time.time() - self._ttl
        expired = [
            mid for mid, msg in self._messages.items()
            if msg.terminal and msg.updated_at.timestamp() < cutoff
        ]
        for mid in expired:
            del self._messages[mid]

        if len(self._messages) > self._max:
            # Oldest-first, but never evict work that is still running.
            evictable = sorted(
                (m for m in self._messages.values() if m.terminal),
                key=lambda m: m.created_at,
            )
            for msg in evictable[: len(self._messages) - self._max]:
                self._messages.pop(msg.message_id, None)

    def create(self, text: str) -> Message:
        """Register a new queued message."""
        message = Message(message_id=f"msg_{uuid.uuid4().hex[:16]}", text=text)
        with self._lock:
            # Insert first, then sweep, so the cap holds *after* every create.
            # Sweeping first leaves the store at max+1. The new message is
            # queued, and only terminal messages are evictable, so it is safe.
            self._messages[message.message_id] = message
            self._evict_locked()
        return message

    def get(self, message_id: str) -> Message | None:
        """Look up a message, or ``None`` if unknown or evicted."""
        with self._lock:
            return self._messages.get(message_id)

    def mark_processing(self, message_id: str) -> None:
        """Move a message from queued to processing."""
        with self._lock:
            message = self._messages.get(message_id)
            if message and message.status == "queued":
                message.status = "processing"
                message.updated_at = _now()

    def complete(self, message_id: str, response: QueryResponse) -> None:
        """Attach the generated answer and release anyone waiting on it."""
        with self._lock:
            message = self._messages.get(message_id)
            if message is None:
                return
            message.status = "done"
            message.response = response
            message.updated_at = _now()
        message._finished.set()

    def fail(self, message_id: str, error: str) -> None:
        """Record a processing failure and release waiters."""
        with self._lock:
            message = self._messages.get(message_id)
            if message is None:
                return
            message.status = "failed"
            message.error = error
            message.updated_at = _now()
        message._finished.set()

    def latest(self) -> Message | None:
        """Return the most recently submitted message, or ``None`` if empty."""
        with self._lock:
            messages = list(self._messages.values())
        return max(messages, key=lambda m: m.created_at) if messages else None

    def list_recent(self, limit: int = 50) -> list[Message]:
        """Return the most recent messages, oldest first.

        Exists so a client that reconnects can backfill what it missed: the live
        socket does not replay events, so without this a dropped connection
        would leave a permanent gap in the transcript.
        """
        with self._lock:
            messages = list(self._messages.values())
        messages.sort(key=lambda m: m.created_at)
        return messages[-limit:]

    def stats(self) -> dict[str, int]:
        """Counts by status, for /health."""
        with self._lock:
            messages = list(self._messages.values())
        counts = {"total": len(messages)}
        for status in ("queued", "processing", "done", "failed"):
            counts[status] = sum(1 for m in messages if m.status == status)
        return counts


class MessageService:
    """Accepts messages and answers them on a bounded worker pool."""

    def __init__(
        self,
        store: MessageStore,
        pipeline_getter: Callable[[], object],
        max_workers: int = 2,
        broadcaster: "EventBroadcaster | None" = None,
    ) -> None:
        self._store = store
        self._get_pipeline = pipeline_getter
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="faqrag-msg"
        )
        self.broadcaster = broadcaster or EventBroadcaster()

    @property
    def store(self) -> MessageStore:
        """The backing message store."""
        return self._store

    @property
    def pipeline(self) -> Any:
        """The shared RAG pipeline, loaded on first use.

        Exposed for the streaming endpoint, which drives generation itself
        instead of handing the question to the worker pool.
        """
        return self._get_pipeline()

    def submit(self, text: str, top_k: int | None = None) -> Message:
        """Queue ``text`` for answering and return immediately."""
        message = self._store.create(text)
        # Announce the question before answering it, so a listening UI shows the
        # incoming message the instant it arrives rather than ten seconds later.
        self.broadcaster.publish(
            "message.received",
            {
                "message_id": message.message_id,
                "text": message.text,
                "created_at": message.created_at.isoformat(),
            },
        )
        self._pool.submit(self._process, message.message_id, text, top_k)
        return message

    def _process(self, message_id: str, text: str, top_k: int | None) -> None:
        """Answer one message on a worker thread."""
        self._store.mark_processing(message_id)
        try:
            pipeline = self._get_pipeline()
            response = pipeline.answer(text, top_k)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - must never kill the worker
            logger.exception("message %s failed", message_id)
            self._store.fail(message_id, str(exc))
            self.broadcaster.publish(
                "message.failed", {"message_id": message_id, "error": str(exc)}
            )
            return
        self._store.complete(message_id, response)
        message = self._store.get(message_id)
        if message is not None:
            self.broadcaster.publish("message.answered", to_result(message).model_dump(mode="json"))
        logger.info(
            "message %s answered in %sms (confident=%s)",
            message_id,
            response.latency_ms,
            response.confident,
        )

    def shutdown(self) -> None:
        """Stop accepting work and let running jobs finish."""
        self._pool.shutdown(wait=False)


# --------------------------------------------------------------------------
# Wire models
# --------------------------------------------------------------------------


class InboundMessage(BaseModel):
    """Body of ``POST /v1/messages`` -- text arriving from the other side."""

    text: str = Field(
        ..., min_length=1, max_length=2000, description="The user's question, Arabic or English."
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="Chunks to retrieve.")


class AcceptedMessage(BaseModel):
    """Response to ``POST /v1/messages`` -- an acknowledgement, not an answer."""

    message_id: str
    status: MessageStatus
    created_at: datetime
    poll_url: str = Field(description="GET this to collect the answer.")


class MessageResult(BaseModel):
    """Response to ``GET /v1/messages/{id}``.

    Answer fields are ``None`` until ``status`` is ``done``, so the caller
    should branch on ``status`` rather than on the presence of ``answer``.
    """

    message_id: str
    status: MessageStatus
    text: str
    created_at: datetime
    updated_at: datetime

    answer: str | None = None
    cited_faq_ids: list[str] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)
    confident: bool | None = None
    confidence: float | None = None
    language: str | None = None
    latency_ms: float | None = None
    error: str | None = None


def to_result(message: Message) -> MessageResult:
    """Render a stored message as its wire representation."""
    result = MessageResult(
        message_id=message.message_id,
        status=message.status,
        text=message.text,
        created_at=message.created_at,
        updated_at=message.updated_at,
        error=message.error,
    )
    if message.response is not None:
        response = message.response
        result.answer = response.answer
        result.cited_faq_ids = list(response.cited_faq_ids)
        result.sources = list(response.sources)
        result.confident = response.confident
        result.confidence = response.confidence
        result.language = response.language
        result.latency_ms = response.latency_ms
    return result


#: Example bodies shown as a dropdown in Swagger's "Try it out".
INBOUND_EXAMPLES = {
    "arabic_dialect": {
        "summary": "Arabic (Saudi dialect)",
        "description": "A colloquial question. The answer comes back in dialect too.",
        "value": {"text": "وش طرق الدفع عندكم؟"},
    },
    "english": {
        "summary": "English",
        "value": {"text": "What is Mwfaq Business?"},
    },
    "out_of_scope": {
        "summary": "Out of scope (expect confident=false)",
        "description": "Pricing is not in the knowledge base, so the service declines.",
        "value": {"text": "بكم الفحص الطبي؟"},
    },
}


#: Markdown the model sometimes emits. Harmless on screen, wrong out loud --
#: a synthesiser reads "**" and "-" as noise or as an unnatural pause.
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|`)")
_MD_BULLET = re.compile(r"^[ 	]*(?:[-*•]|\d+[.)])[ 	]+", re.MULTILINE)
_MD_HEADING = re.compile(r"^[ 	]*#{1,6}[ 	]*", re.MULTILINE)


def to_speech_text(answer: str) -> str:
    """Flatten an answer into one block of prose suitable for text-to-speech.

    Markdown markers are stripped, and each line is terminated with sentence
    punctuation before the lines are joined. Without that terminator a speech
    synthesiser runs a bulleted list together as a single breathless sentence;
    with it, each item gets its natural pause.
    """
    text = _MD_HEADING.sub("", _MD_BULLET.sub("", _MD_EMPHASIS.sub("", answer)))

    lines = []
    for line in (raw.strip() for raw in text.splitlines()):
        if not line:
            continue
        # Arabic and Latin sentence enders both count as already terminated.
        if line[-1] not in ".!?؟،:":
            line += "."
        lines.append(line)

    return re.sub(r"[ 	]{2,}", " ", " ".join(lines)).strip()


def _text_response(message: Message) -> PlainTextResponse:
    """Render a message as plain text.

    The status still travels, in the headers rather than the body, so a text
    client is not left guessing: ``X-Message-Status`` always, and
    ``X-Answer-Confident`` plus ``X-Cited-FAQ-Ids`` once an answer exists. A
    caller that ignores them still gets a sensible sentence, because the
    low-confidence body is itself a plainly worded "I don't have that".
    """
    headers = {"X-Message-Status": message.status}
    if not message.terminal:
        headers["Retry-After"] = "2"
        body = ""
    elif message.status == "failed":
        body = message.error or "failed"
    else:
        result = message.response
        # Speech-ready: this representation exists to be spoken or relayed, so
        # markdown that reads fine on screen is stripped here.
        body = to_speech_text(result.answer) if result else ""
        if result is not None:
            headers["X-Answer-Confident"] = str(result.confident).lower()
            headers["X-Cited-FAQ-Ids"] = ",".join(result.cited_faq_ids)

    return PlainTextResponse(
        content=body,
        media_type="text/plain; charset=utf-8",
        headers=headers,
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame.

    A frame is ``event:`` and ``data:`` lines terminated by a blank line. The
    JSON is emitted with ``ensure_ascii=False`` so Arabic travels as UTF-8
    rather than as escape sequences.
    """
    body = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n"


def build_messages_router(service: MessageService) -> APIRouter:
    """Build the ``/v1/messages`` router backed by ``service``."""
    router = APIRouter(prefix="/v1", tags=["Integration (async)"])

    @router.post(
        "/messages/stream",
        response_model=None,
        summary="Ask a question and receive the answer as it is written",
        response_description=(
            "A text/event-stream of delta events carrying answer text, closed by "
            "a single done event with citations and confidence."
        ),
        responses={
            200: {
                "content": {"text/event-stream": {}},
                "description": "An open SSE stream of answer chunks.",
            },
            422: {"description": "text was empty or longer than 2000 characters."},
        },
    )
    def stream_message(
        payload: InboundMessage = Body(..., openapi_examples=INBOUND_EXAMPLES),
    ) -> StreamingResponse:
        """Answer a question, pushing text as the model writes it.

        Same answer as `POST /v1/messages`, but the wait is filled with partial
        text instead of silence. Retrieval and reranking still run first, so the
        first `delta` lands after roughly a second, not instantly.

        Event types:

        | `event` | payload | meaning |
        |---|---|---|
        | `stream.open` | `message_id` | accepted; generation starting |
        | `delta` | `text` | append this to what you are showing |
        | `reset` | `reason` | **discard everything shown so far** |
        | `done` | the full message result | settled answer, citations, confidence |
        | `error` | `error` | generation failed; nothing further follows |

        The `done` payload is identical to what `POST /v1/messages` returns, and
        is authoritative: concatenated deltas are the same text arriving early.
        Branch on `confident` there exactly as with the blocking endpoint.

        `reset` is rare but must be handled -- it fires when the model reveals
        only partway through that the context did not support an answer. Ignoring
        it leaves a retracted answer on screen.

        ```js
        const res = await fetch("/v1/messages/stream", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({text: "وش طرق الدفع عندكم؟"}),
        });
        const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
        // then split on the blank line between frames and dispatch on `event:`
        ```

        The stream is recorded in the message store like any other message, so it
        still appears in `GET /v1/history` and is broadcast to `/v1/events` and
        `/v1/ws` listeners.
        """
        message = service.store.create(payload.text)
        service.broadcaster.publish(
            "message.received",
            {
                "message_id": message.message_id,
                "text": message.text,
                "created_at": message.created_at.isoformat(),
            },
        )

        def stream() -> Iterator[str]:
            # Same tunnel-buffering defence as GET /v1/events: without this the
            # stream is correct on localhost and silent through a proxy.
            yield SSE_PROXY_PADDING
            yield _sse("stream.open", {"message_id": message.message_id})
            service.store.mark_processing(message.message_id)
            try:
                for event in service.pipeline.answer_stream(payload.text, payload.top_k):
                    if isinstance(event, StreamDelta):
                        yield _sse("delta", {"text": event.text})
                    elif isinstance(event, StreamReset):
                        yield _sse("reset", {"reason": event.reason})
                    else:
                        # The terminal QueryResponse: store it so the message
                        # behaves like every other, then report it.
                        service.store.complete(message.message_id, event)
                        stored = service.store.get(message.message_id)
                        result = (
                            to_result(stored) if stored is not None else None
                        )
                        data = (
                            result.model_dump(mode="json")
                            if result is not None
                            else event.model_dump(mode="json")
                        )
                        yield _sse("done", data)
                        if result is not None:
                            service.broadcaster.publish("message.answered", data)
                        logger.info(
                            "message %s streamed in %sms (confident=%s)",
                            message.message_id,
                            event.latency_ms,
                            event.confident,
                        )
            except Exception as exc:  # noqa: BLE001 - the stream must report, not crash
                logger.exception("streaming message %s failed", message.message_id)
                service.store.fail(message.message_id, str(exc))
                service.broadcaster.publish(
                    "message.failed",
                    {"message_id": message.message_id, "error": str(exc)},
                )
                yield _sse(
                    "error", {"message_id": message.message_id, "error": str(exc)}
                )

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post(
        "/messages",
        response_model=None,
        status_code=200,
        summary="Ask a question and get the answer back",
        response_description="The answer, in this same response.",
        responses={
            200: {"description": "The answer.", "model": MessageResult},
            202: {
                "description": (
                    "The answer was not ready within `wait` seconds. Carries a "
                    "message_id to collect it with instead."
                ),
                "model": AcceptedMessage,
            },
            422: {"description": "text was empty or longer than 2000 characters."},
        },
    )
    def receive_message(
        response: Response,
        payload: InboundMessage = Body(..., openapi_examples=INBOUND_EXAMPLES),
        wait: float = Query(
            default=MAX_WAIT_SECONDS,
            ge=0.0,
            le=MAX_WAIT_SECONDS,
            description=(
                "Seconds to hold the request open waiting for the answer. "
                "Defaults to the maximum, so the answer comes back in this call. "
                "Set wait=0 to return immediately with a message_id instead."
            ),
        ),
        format: Literal["json", "text"] = Query(
            default="json",
            description="'text' returns just the answer as speech-ready text/plain.",
        ),
    ):
        """Ask a question and get the answer back in the same response.

        **One call. No message_id, no polling.** Send the text, wait, read the
        answer. Add `&format=text` to get speech-ready plain text instead of
        JSON, which is the form to hand straight to a text-to-speech engine.

        Generation takes roughly ten seconds, so **set your client timeout above
        60 seconds.** The request is held open for that long by default.

        If the answer is somehow not ready in time you get `202` and a
        `message_id` rather than an error, and can collect it from
        `GET /v1/messages/{message_id}`. A slow answer degrades; it never fails.

        Pass `wait=0` to opt out of waiting and always get the `message_id`
        form. That is what you want when something between you and this service
        (a browser, an API gateway) would cut a ten-second request short.
        """

        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="text must not be empty")

        message = service.submit(text, payload.top_k)

        if wait > 0:
            message._finished.wait(timeout=min(wait, MAX_WAIT_SECONDS))
            if message.terminal:
                if format == "text":
                    return _text_response(message)
                response.status_code = 200
                return to_result(message)

        # Either wait was not requested, or it elapsed first: hand back the id.
        response.status_code = 202
        return AcceptedMessage(
            message_id=message.message_id,
            status=message.status,
            created_at=message.created_at,
            poll_url=f"/v1/messages/{message.message_id}",
        )

    @router.get(
        "/messages/{message_id}",
        response_model=MessageResult,
        response_model_exclude_none=False,
        summary="2. Collect the answer",
        response_description="The message, with answer fields populated once status is done.",
        responses={
            404: {"description": "Unknown message_id, or it expired (default TTL 1 hour)."},
            422: {"description": "wait was above the 60 second maximum."},
        },
    )
    def get_message(
        message_id: str = Path(..., description="The id returned by POST /v1/messages."),
        response: Response = None,  # type: ignore[assignment]
        format: Literal["json", "text"] = Query(
            default="json",
            description=(
                "'json' returns the full result. 'text' returns only the answer "
                "as text/plain -- simplest to consume, but it drops the "
                "confident flag and the citations."
            ),
        ),
        wait: float = Query(
            default=0.0,
            ge=0.0,
            le=MAX_WAIT_SECONDS,
            description=(
                "Seconds to block waiting for the answer before returning. "
                "0 returns the current status immediately. Long-polling this "
                "way is cheaper than a tight poll loop."
            ),
        ),
    ) -> MessageResult:
        """Collect the answer for a previously accepted message.

        **Use `wait`.** Passing `wait=30` blocks server-side until the answer is
        ready (or 30s pass) and returns it in one call — far cheaper than
        polling in a tight loop. While still pending, the response carries a
        `Retry-After: 2` header.

        **Branch on `status`, not on whether `answer` is set.** Every answer
        field is `null` until `status` is `"done"`:

        | status | meaning | what to do |
        |---|---|---|
        | `queued` | accepted, not started | call again with `wait` |
        | `processing` | being answered | call again with `wait` |
        | `done` | ready | render `answer` and `sources` |
        | `failed` | something broke | read `error`; safe to resubmit |

        Then check **`confident`**: when `false`, the FAQ did not cover the
        question and `cited_faq_ids` is empty.
        """
        message = service.store.get(message_id)
        if message is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown message_id {message_id!r} (it may have expired)",
            )

        if wait > 0 and not message.terminal:
            message._finished.wait(timeout=min(wait, MAX_WAIT_SECONDS))

        if format == "text":
            return _text_response(message)

        if not message.terminal:
            # Tell a polling client how long to back off for.
            response.headers["Retry-After"] = "2"
        return to_result(message)

    @router.get(
        "/events",
        summary="Live event stream (Server-Sent Events)",
        response_description=(
            "A text/event-stream of message.received, message.answered, and "
            "message.failed events."
        ),
        responses={
            200: {
                "content": {"text/event-stream": {}},
                "description": "An open SSE stream. Events arrive as they happen.",
            }
        },
    )
    async def events(request: Request) -> StreamingResponse:
        """Subscribe to messages as they arrive, with no polling.

        Every message pushed to `POST /v1/messages` is broadcast here the moment
        it lands, and again the moment it is answered. This is what lets a chat
        UI display an inbound question immediately instead of discovering it on
        the next poll.

        Event types:

        | `event` | when | payload |
        |---|---|---|
        | `message.received` | a message arrives | `message_id`, `text`, `created_at` |
        | `message.answered` | its answer is ready | the full message result |
        | `message.failed` | answering failed | `message_id`, `error` |
        | `ping` | every 15s | keeps proxies from closing an idle stream |

        Consume it from a browser with `EventSource`:

        ```js
        const es = new EventSource("/v1/events");
        es.addEventListener("message.received", e => showQuestion(JSON.parse(e.data)));
        es.addEventListener("message.answered", e => showAnswer(JSON.parse(e.data)));
        ```

        `EventSource` reconnects on its own. Events are **not** replayed on
        reconnect, so treat this as a live feed and use
        `GET /v1/sessions/{id}/messages` to backfill anything missed.
        """
        queue = service.broadcaster.subscribe()

        async def stream() -> AsyncIterator[str]:
            try:
                # Reverse proxies (Cloudflare tunnels included) hold a response
                # back until enough bytes accumulate to be worth forwarding.
                # Without this padding the stream is correct on localhost and
                # completely silent through a tunnel. An SSE comment is ignored
                # by every client, so it costs only the bytes.
                yield SSE_PROXY_PADDING
                # Announce readiness so the client can distinguish "connected"
                # from "connected but nothing has happened yet".
                yield _sse("stream.open", {"listeners": service.broadcaster.listener_count})
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        # Idle streams get closed by proxies; a comment-only
                        # heartbeat keeps the connection alive cheaply.
                        yield ": ping\n\n"
                        continue
                    yield _sse(payload["event"], payload["data"])
            finally:
                service.broadcaster.unsubscribe(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Stops nginx and similar from buffering the stream into silence.
                "X-Accel-Buffering": "no",
            },
        )

    @router.websocket("/ws")
    async def live_socket(websocket: WebSocket) -> None:
        """Live event stream over a WebSocket.

        Carries exactly the same events as ``GET /v1/events``, as JSON frames of
        the form ``{"event": ..., "data": {...}}``.

        This exists because Server-Sent Events do not survive every proxy.
        Cloudflare quick tunnels, in particular, buffer an SSE response
        indefinitely -- the origin emits events correctly and the browser
        receives nothing. WebSockets are negotiated as a connection upgrade
        rather than a long response body, so proxies forward them without
        buffering. Prefer this endpoint whenever the service is reached through
        a tunnel or CDN.
        """
        await websocket.accept()
        queue = service.broadcaster.subscribe()
        try:
            await websocket.send_json(
                {
                    "event": "stream.open",
                    "data": {"listeners": service.broadcaster.listener_count},
                }
            )
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    # A ping keeps idle intermediaries from dropping the socket.
                    await websocket.send_json({"event": "ping", "data": {}})
                    continue
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001 - a dead socket must not raise upward
            logger.debug("websocket closed unexpectedly", exc_info=True)
        finally:
            service.broadcaster.unsubscribe(queue)

    @router.get(
        "/latest",
        response_model=MessageResult,
        summary="3. The answer to the most recent question",
        response_description="The single most recent message.",
        responses={404: {"description": "No messages have been received yet."}},
    )
    def get_latest(
        response: Response,
        format: Literal["json", "text"] = Query(
            default="json",
            description="'text' returns just the answer as text/plain.",
        ),
        wait: float = Query(
            default=0.0,
            ge=0.0,
            le=MAX_WAIT_SECONDS,
            description="Seconds to block waiting for the answer before returning.",
        ),
    ):
        """Return the answer to the **latest** question, without a message_id.

        This is the simplest possible flow: `POST /v1/messages`, then
        `GET /v1/latest?wait=60`. Nothing to track between the two calls.

        The trade-off is that "latest" is inherently racy. If two questions are
        submitted close together, this returns whichever arrived last -- which
        may not be the one you meant. That is fine for a single sequential
        conversation and wrong for concurrent ones; use
        `GET /v1/messages/{message_id}` whenever more than one question can be
        in flight at a time.
        """
        message = service.store.latest()
        if message is None:
            raise HTTPException(status_code=404, detail="no messages received yet")

        if wait > 0 and not message.terminal:
            message._finished.wait(timeout=min(wait, MAX_WAIT_SECONDS))

        if format == "text":
            return _text_response(message)

        if not message.terminal:
            response.headers["Retry-After"] = "2"
        return to_result(message)

    @router.get(
        "/history",
        response_model=list[MessageResult],
        summary="4. Recent messages (backfill after a reconnect)",
        response_description="The most recent messages, oldest first.",
    )
    def list_history(
        limit: int = Query(default=50, ge=1, le=200, description="How many to return."),
    ) -> list[MessageResult]:
        """List the most recent messages, oldest first.

        This is a transcript, not an answer -- if you want the reply to the
        question you just sent, use `GET /v1/latest` instead.

        Its purpose is reconnection: the live socket does not replay events, so
        a client that dropped its connection would otherwise carry a permanent
        gap. Call this once on (re)connect, then rely on the socket.
        """
        return [to_result(m) for m in service.store.list_recent(limit)]

    return router
