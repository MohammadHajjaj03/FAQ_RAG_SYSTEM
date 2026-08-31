# Mwfaq FAQ RAG System

Bilingual (Arabic + English) retrieval-augmented generation over the Mwfaq FAQ knowledge base —
26 FAQs, 52 chunks. Hybrid retrieval (dense + BM25 with reciprocal rank fusion), language-aware
ranking, LLM reranking, and strictly grounded generation with citations.

Embeddings run on the OpenAI API; answering and reranking run on Claude Haiku via the
Anthropic API. Both providers are swappable by config — an all-local Ollama setup is still
one `.env` edit away.

---

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Usage](#usage) — [Chat UI](#chat-ui) · [CLI](#cli) · [HTTP API](#http-api) · [Evaluation](#evaluation)
- [Re-indexing after the FAQ changes](#re-indexing-after-the-faq-changes)
- [Configuration](#configuration)
- [Debugging a bad answer](#debugging-a-bad-answer)
- [Design decisions](#design-decisions)
- [Project layout](#project-layout)

---

## Quick start

**Prerequisites:** Python 3.10+, an OpenAI API key (embeddings), and an Anthropic API key
(generation). Nothing runs locally except the index itself.

```bash
# 1. Environment
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Unix
pip install -e ".[dev]"

# 2. Config — then put your two keys in .env:
copy .env.example .env            # cp on Unix
#   FAQRAG_OPENAI_API_KEY=sk-...
#   FAQRAG_ANTHROPIC_API_KEY=sk-ant-...

# 3. Build the index (one embedding pass over 52 chunks; a few cents at most)
python -m faqrag.index

# 4. Ask something
python -m faqrag.cli "What is Mwfaq Academy?"
python -m faqrag.cli "كيف أحجز فحصي الطبي عبر موفق؟"
```

> **Windows note:** the CLI, indexer, and eval reconfigure stdout to UTF-8 so Arabic prints
> correctly on a cp1252 console. If you call the library directly from your own script, call
> `faqrag.logging_utils.ensure_utf8_stdout()` first.

---

## How it works

```
                    query ("كيف أدفع عبر Tabby؟")
                              │
                    ┌─────────▼─────────┐
                    │ language detection │  Arabic char ratio ≥ 0.2 → "ar"
                    └─────────┬─────────┘
              ┌───────────────┴───────────────┐
   ┌──────────▼──────────┐        ┌───────────▼───────────┐
   │ dense: vector cosine │        │ lexical: BM25         │
   │ over question+answer │        │ over q + a + keywords │
   │ (both languages)     │        │ (Arabic-normalised)   │
   └──────────┬──────────┘        └───────────┬───────────┘
              └───────────────┬───────────────┘
                    ┌─────────▼─────────┐
                    │  RRF fusion       │  rank-based, needs no score calibration
                    └─────────┬─────────┘
                    ┌─────────▼─────────┐
                    │ language boost    │  ×1.15 same-language — a preference,
                    └─────────┬─────────┘  not a filter
                    ┌─────────▼─────────┐
                    │ LLM rerank (0-10) │  optional; FAQRAG_RERANK_ENABLED
                    └─────────┬─────────┘
                    ┌─────────▼─────────┐
                    │ relevance ≥ 0.45? │  NO → decline, no LLM call
                    └─────────┬─────────┘
                       YES    │
                    ┌─────────▼─────────┐
                    │ grounded answer   │  same language, cites FAQ ids
                    └───────────────────┘
```

### Ranking vs. confidence

These are separate signals, and conflating them is the single easiest way to get this wrong:

| | `score` (ranking) | `relevance` (confidence) |
|---|---|---|
| Purpose | order candidates within one query | decide whether to answer at all |
| Source | fused RRF, normalised so best = 1.0 | 0.6 × rerank/10 + 0.4 × cosine |
| Comparable across queries? | **No** | **Yes** |

The best candidate of a totally out-of-scope query still has `score` 1.0 — it is the best of a bad
set. Only `relevance` is compared against `FAQRAG_MIN_RELEVANCE_SCORE`. On the eval set,
out-of-scope queries peak at 0.20 relevance while in-scope queries bottom out at 0.47, so the 0.45
threshold sits in a wide empty gap rather than being tuned to a knife edge.

---

## Usage

### Chat UI

An optional browser interface, served by the API itself:

```bash
python -m faqrag.api      # then open http://127.0.0.1:8000/chat
```

Bilingual chat with automatic RTL/LTR switching, expandable citations, a
confidence bar, and — importantly — a distinct amber treatment for
low-confidence replies, so an "I don't know" can never be mistaken for a
sourced answer. Settings (⚙) expose a top-k slider and two debugging views:
*show all retrieved chunks* (dimming those that weren't cited) and *show
retrieval scores* (per-candidate vector/BM25/rerank table).

**It is a fully detachable add-on.** Nothing in the retrieval or generation
pipeline imports it:

```bash
FAQRAG_ENABLE_CHAT_UI=false          # turn off, keep the files
rm -rf web/ src/faqrag/web.py        # remove permanently
```

Either way the API keeps serving `/health`, `/query`, and `/retrieve`
unchanged, the CLI is unaffected, and all tests still pass. See
[web/README.md](web/README.md).

### CLI

```bash
python -m faqrag.cli "What payment methods does Mwfaq accept?"
python -m faqrag.cli "ما هي أكاديمية موفق؟" --sources     # show cited FAQ entries
python -m faqrag.cli "What is Mwfaq Business?" --json      # full structured response
python -m faqrag.cli "How do I get started?" --retrieve-only   # scores only, no LLM
python -m faqrag.cli "Tabby?" --no-rerank --verbose        # faster, with score logging
```

| Flag | Effect |
|---|---|
| `--sources` | Print the cited FAQ entries beneath the answer |
| `--json` | Emit the full `QueryResponse` as JSON |
| `--retrieve-only` | Show retrieved chunks and every score; makes no LLM call |
| `--no-rerank` | Skip reranking for this query (~2s instead of ~11s) |
| `-k / --top-k` | Override the number of chunks retrieved |
| `--verbose` | Log per-candidate retrieval scores to stderr |

### HTTP API

```bash
python -m faqrag.api                                  # http://127.0.0.1:8000
uvicorn faqrag.api:app --reload                       # dev autoreload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

**`GET /health`** — liveness plus the active index and model configuration.

```json
{ "status": "ok", "indexed_chunks": 52, "embedding_model": "openai:text-embedding-3-large",
  "llm_model": "anthropic:claude-haiku-4-5", "vector_store": "numpy", "rerank_enabled": true }
```

**`POST /query`** — `{"question": "...", "top_k": 5}`

```json
{
  "answer": "Mwfaq Academy is a smart learning system offering integrated qualification...",
  "cited_faq_ids": ["012", "014"],
  "sources":   [{ "faq_id": "012", "category": "Mwfaq Academy", "lang": "en",
                  "question": "...", "answer": "...", "score": 0.897 }],
  "retrieved": [{ "faq_id": "012", "...": "..." }],
  "confidence": 0.897, "confident": true, "language": "en",
  "query": "What is Mwfaq Academy?", "reranked": true,
  "cross_lingual_fallback": false, "latency_ms": 12420.3
}
```

`sources` holds only what the model cited (render these as citations). `retrieved` holds
everything retrieval surfaced, cited or not — always populated, including on a refusal, so a
wrong decline is diagnosable.

When nothing clears the threshold, `confident` is `false`, `cited_faq_ids` is empty, and `answer`
is an "I don't know" message in the user's language. **A frontend should branch on `confident`
rather than rendering the answer unconditionally.**

**`POST /retrieve`** — same body, retrieval only. Returns every candidate with its vector score,
BM25 score, rerank score, and fused rank, for diagnosing ranking without paying for generation.

**`GET /chat`** — the optional chat UI (`/` redirects here). Absent when the UI is removed or
disabled; every other endpoint is unaffected.

#### Integration endpoints (for another system)

`/query` blocks for the ~10s a full answer takes, which browsers and gateways often cut short.
For an external integration use the async pair instead:

| | Endpoint | Returns |
|---|---|---|
| **Ask** | `POST /v1/messages` | the answer, in the same response (~10s) |
| **Ask, streamed** | `POST /v1/messages/stream` | answer text as the model writes it (SSE) |
| Ask, text only | `POST /v1/messages?format=text` | bare answer as `text/plain`, TTS-ready |
| **Live push** | `WS /v1/ws` | every message pushed on arrival (~90 ms) |
| Don't wait | `POST /v1/messages?wait=0` | `202` + `message_id` to collect later |
| Collect later | `GET /v1/messages/{id}?wait=60` | the answer once ready |
| History | `GET /v1/history?limit=50` | recent transcript (reconnect backfill) |

```bash
curl -X POST localhost:8000/v1/messages      -H 'Content-Type: application/json' -d '{"text":"وش طرق الدفع عندكم؟"}'
# → {"answer": "أبشر، عندنا عدة طرق دفع ميسرة...", "cited_faq_ids": ["011"], "confident": true}

curl -X POST "localhost:8000/v1/messages?format=text"      -H 'Content-Type: application/json' -d '{"text":"وش طرق الدفع عندكم؟"}'
# → أبشر، عندنا عدة طرق دفع ميسرة تقدر تستخدمها: تابي، تمارا، مدى...
```

The request is held open for the ~10s a generation takes, so **set your client
timeout above 60 seconds**.

Use `WS /v1/ws` for anything that must react instantly — SSE (`GET /v1/events`) is also
available but is buffered to death by Cloudflare-style tunnels.

#### Streaming the answer

`POST /v1/messages/stream` returns the same answer as `POST /v1/messages`, but fills the wait
with partial text instead of silence:

```bash
curl -N -X POST localhost:8000/v1/messages/stream      -H 'Content-Type: application/json' -d '{"text":"وش طرق الدفع عندكم؟"}'
# event: stream.open  {"message_id": "msg_..."}
# event: delta        {"text": "عندنا عدة طرق دفع ميسرة: "}
# event: delta        {"text": "تابي، تمارا، مدى..."}
# event: done         {"answer": "...", "cited_faq_ids": ["011"], "confident": true}
```

| `event` | payload | meaning |
|---|---|---|
| `stream.open` | `message_id` | accepted; generation starting |
| `delta` | `text` | append to what you are showing |
| `reset` | `reason` | **discard everything shown so far** |
| `done` | the full message result | settled answer, citations, confidence |
| `error` | `error` | generation failed; nothing further follows |

Retrieval and reranking still run before the first token, so `delta` events begin at roughly
3s and then arrive every 150-400ms. `done` carries the identical payload to the blocking
endpoint and is **authoritative** — concatenated deltas are the same text arriving early, so
branch on `confident` there exactly as before.

Two details a client must handle:

- **`reset` means retract.** It fires when the model reveals only partway through that the
  context did not support an answer, or when generation fails after text was already sent.
  Streamed text cannot be recalled, so the server says so explicitly. Ignoring it leaves a
  retracted answer on screen.
- **The `SOURCES: 011` line never appears in a `delta`.** It is withheld as the model writes
  it, so citations arrive only in `done`.

Streamed messages are recorded like any other, so they still appear in `GET /v1/history` and
are broadcast to `/v1/events` and `/v1/ws` listeners.

Answer fields are `null` until `status` is `done`, so clients must branch on `status` — and on
`confident`, exactly as the chat UI does. Full contract: **[docs/INTEGRATION.md](docs/INTEGRATION.md)**.

### Evaluation

```bash
python -m faqrag.eval                          # core suite: retrieval + generation + judge
python -m faqrag.eval --suite saudi            # 100 questions in Saudi dialect
python -m faqrag.eval --suite all              # both
python -m faqrag.eval --retrieval-only         # no LLM calls, ~2s/query
python -m faqrag.eval --no-judge               # generate, but skip the judge
python -m faqrag.eval --suite saudi --json saudi.json
python scripts/analyze_eval.py saudi.json      # per-FAQ / per-category breakdown
```

Two suites ship:

| Suite | Cases | What it measures |
|---|---|---|
| `core` | 20 | MSA Arabic + English, the baseline behaviours |
| `saudi` | 100 | Saudi colloquial dialect against an MSA corpus — the dialect gap |

20 cases, balanced Arabic/English, spanning clean questions, low-overlap paraphrases, ambiguous
questions, out-of-scope questions, and one case where the FAQ explicitly leaves a detail
unspecified.

Results below were measured on the previous stack (`bge-m3` + `deepseek-v4-flash:cloud`,
rerank on, k=5) and have not yet been re-run against OpenAI embeddings + Claude Haiku:

| Metric | Result | |
|---|---|---|
| `retrieval_hit_rate@5` | **1.00** | 17/17 in-scope cases |
| — clean / paraphrase / ambiguous | 1.00 / 1.00 / 1.00 | |
| `out_of_scope_refusal_rate` | **1.00** | 4/4 correctly declined |
| `groundedness_rate` | **1.00** | 16/16 answers fully traceable to cited chunks |
| `citation_accuracy` | **1.00** | cited FAQ actually answers the question |
| `fabrication_free_rate` | **1.00** | no invented phone/email values |
| `language_match_rate` | **1.00** | answer language always matches the query |
| `in_scope_answer_rate` | 0.94 | see note below |
| `median_latency_ms` | ~10,600 | ~2,100 with `--no-rerank` |

`in_scope_answer_rate` is 16/17 because "Can I split my payment into instalments with Tabby?"
is declined. That is correct: the corpus lists Tabby as accepted but states nothing about
instalment terms, so the honest answer is that it doesn't know. Retrieval still finds FAQ 011.

Metrics measure distinct stages and should not be collapsed: `hit_rate` scores **retrieval**
(against `retrieved`), `citation_accuracy` scores **generation** (against `cited_faq_ids`). A
model that retrieves correctly but cites the wrong FAQ fails only the second — which is exactly
how that bug was caught during development.

---

## Re-indexing after the FAQ changes

The index is a build artifact. Rebuild it whenever the FAQ content, the chunking logic, or the
embedding model changes.

**If you edited the JSON directly:**

```bash
python -m faqrag.index
```

**If you edited the markdown source** (`data/mwfaq_faq_rag.md`), regenerate the JSON first:

```bash
python scripts/md_to_json.py data/mwfaq_faq_rag.md data/mwfaq_faq_rag.json
python -m faqrag.index
python -m pytest tests/test_ingest.py -q     # asserts every FAQ still has both languages
```

`python -m faqrag.index` always rebuilds from scratch, so removed FAQs disappear from the index
rather than lingering. It writes `data/index/manifest.json` recording the source file, embedding
model, dimension, and chunk counts — check it when you are unsure what the running index contains.

Restart the API afterwards; it loads the index once at startup.

---

## Configuration

All settings live in `.env` (see `.env.example`) or as `FAQRAG_`-prefixed environment variables.
No values are hardcoded in logic modules — everything is defined in [`config.py`](src/faqrag/config.py).

| Variable | Default | Notes |
|---|---|---|
| `FAQRAG_EMBEDDING_PROVIDER` | `openai` | or `ollama` for a local daemon |
| `FAQRAG_EMBEDDING_MODEL` | `text-embedding-3-large` | multilingual; **must** handle Arabic |
| `FAQRAG_EMBEDDING_BATCH_SIZE` | `64` | texts per embedding request |
| `FAQRAG_OPENAI_API_KEY` | — | required by the `openai` embedding provider |
| `FAQRAG_VECTOR_STORE` | `numpy` | or `chroma` |
| `FAQRAG_TOP_K` | `5` | chunks passed to the generator |
| `FAQRAG_CANDIDATE_K` | `20` | candidates pulled from *each* retriever before fusion |
| `FAQRAG_FUSION_METHOD` | `rrf` | or `weighted` (uses `VECTOR_WEIGHT`/`LEXICAL_WEIGHT`) |
| `FAQRAG_LANGUAGE_BOOST` | `0.15` | multiplicative bonus for same-language chunks |
| `FAQRAG_MIN_RELEVANCE_SCORE` | `0.45` | below this the system declines to answer |
| `FAQRAG_RERANK_ENABLED` | `true` | **latency toggle**: off saves ~8s/query |
| `FAQRAG_RERANK_MAX_TOKENS` | `4096` | must leave room for reasoning models to think |
| `FAQRAG_LLM_PROVIDER` | `anthropic` | or `ollama`, `openai`, `extractive` |
| `FAQRAG_LLM_MODEL` | `claude-haiku-4-5` | used for answering **and** reranking |
| `FAQRAG_ANTHROPIC_API_KEY` | — | required by the `anthropic` provider |
| `FAQRAG_LLM_MAX_TOKENS` | `2048` | budget per generation |
| `FAQRAG_LOG_RETRIEVAL_TRACES` | `true` | append per-query traces to `logs/retrieval.jsonl` |
| `FAQRAG_ENABLE_CHAT_UI` | `true` | serve the browser chat UI at `/chat` |
| `FAQRAG_ANSWER_STYLE` | `saudi` | answer voice: `saudi` dialect or formal `msa` |
| `FAQRAG_MESSAGE_WORKERS` | `2` | concurrent workers for `/v1/messages` |
| `FAQRAG_MESSAGE_TTL_SECONDS` | `3600` | how long an answer stays collectable |
| `FAQRAG_MESSAGE_MAX_STORED` | `1000` | retained message cap |

### Swapping components

Each of these is a config change, not a code change:

```bash
# Vector store → Chroma
FAQRAG_VECTOR_STORE=chroma   # then: pip install chromadb && python -m faqrag.index

# Embeddings → cheaper OpenAI model (1536-dim instead of 3072)
FAQRAG_EMBEDDING_MODEL=text-embedding-3-small   # then re-index

# Embeddings → back to a local Ollama daemon
FAQRAG_EMBEDDING_PROVIDER=ollama
FAQRAG_EMBEDDING_MODEL=bge-m3
FAQRAG_OLLAMA_BASE_URL=http://localhost:11434   # then re-index

# Generation → a stronger Claude model
FAQRAG_LLM_MODEL=claude-sonnet-5

# Generation → local Ollama
FAQRAG_LLM_PROVIDER=ollama
FAQRAG_LLM_MODEL=deepseek-v4-flash:cloud

# No LLM at all: return the top FAQ answer verbatim (zero hallucination risk)
FAQRAG_LLM_PROVIDER=extractive
```

To add a backend, implement the relevant interface and register it in the factory:
[`VectorStore`](src/faqrag/stores/base.py) · [`EmbeddingProvider`](src/faqrag/embeddings.py) ·
[`LLMClient`](src/faqrag/llm.py) · [`Reranker`](src/faqrag/rerank.py).

---

## Debugging a bad answer

Retrieval is not a black box. Every query appends one JSON line to `logs/retrieval.jsonl` with
the detected language, each candidate's vector/BM25/rerank/fused scores, whether the language
boost applied, and which chunks reached the generator.

```bash
# What did the last query actually retrieve?
tail -1 logs/retrieval.jsonl | python -m json.tool

# Was this a retrieval fault or a generation fault?
python -m faqrag.cli "the failing question" --retrieve-only

# Every score, live
python -m faqrag.cli "the failing question" --verbose
```

| Symptom | Likely cause |
|---|---|
| Right FAQ retrieved, wrong FAQ cited | generation fault — check `prompts.py` |
| Right FAQ retrieved but declined | `MIN_RELEVANCE_SCORE` too high, or reranker scored it low |
| Wrong FAQ ranked first | check `vector_score` vs `lexical_score` in the trace |
| Answer in the wrong language | check `query_lang` in the trace |
| Slow (>10s) | the rerank call; set `FAQRAG_RERANK_ENABLED=false` |

---

## Design decisions

**One chunk per `(faq_id, lang)`, never merged.** Mixing Arabic and English in one chunk degrades
multilingual embedding quality and pollutes BM25 term statistics. It would also make
language-aware retrieval impossible. `tests/test_ingest.py` asserts neither chunk contains the
other language's text.

**Question *and* answer are embedded together.** FAQ answers carry most of the retrievable
signal, and real user phrasings often match answer content rather than the canonical question —
"Can I pay with Tabby?" matches an answer listing payment methods, not any question. Keywords are
appended because they encode product names and synonyms neither field always spells out.

**Hybrid retrieval, not pure vector.** Dense search blurs rare exact terms. Querying the office
address "U Commercial Center" ranks the correct FAQ (024) **first** on BM25 but only **third** on
cosine; searching "Tamara" or "93%" returns exactly one BM25 document and a tail of near-ties on
cosine. Conversely BM25 fails on paraphrases with no shared words — "I run a company. How do I
automate my staff's medical checks?" shares almost no terms with the FAQs that answer it. RRF is
the default fusion because it consumes only *ranks*, so it needs no calibration between bounded
cosine similarity and unbounded BM25 scores — which matters when score distributions shift as the
corpus grows.

**Arabic-aware lexical processing.** BM25 is fed through a normaliser that strips tashkeel and
tatweel, folds أ/إ/آ → ا, ة → ه, ى → ي, converts Arabic-Indic digits, and strips the definite
article ال (plus و/ف/ب/ك/ل proclitics), so "الفحوصات" and "فحوصات" share a term. Full stemming is
deliberately avoided: aggressive Arabic stemmers conflate distinct FAQ topics at this corpus size.

**Language preference, not a language filter.** Same-language chunks get a ×1.15 boost, but the
other language is never excluded — so a strongly relevant cross-lingual chunk can still win, and
`cross_lingual_fallback` flags when the query's own language had nothing strong. The measured
cross-lingual margin on this corpus is modest (0.49 same-question AR/EN vs 0.37 unrelated under
bge-m3), which is precisely why same-language retrieval is preferred and BM25 is there to back it
up. The margin is a property of the embedding model, so re-measure it after a model swap.

**Refusal happens before generation.** If nothing clears the threshold, the LLM is never called.
A model shown a weak context and asked not to use it will often use it anyway; not showing it is
a stronger guarantee than instructing against it.

**Citations are validated against the supplied context.** A cited FAQ id not present in the
context is dropped, so the model cannot cite content it never saw.

**Every failure degrades rather than breaks.** A reranker failure falls back to the fused order;
an LLM failure falls back to returning the retrieved FAQ answer verbatim — degraded, still
grounded, never invented.

### Model choices

| Component | Choice | Why |
|---|---|---|
| Embeddings | `text-embedding-3-large` via OpenAI | Strongest Arabic coverage in the OpenAI family, 3072-dim, no local daemon or PyTorch dependency |
| Generation | `claude-haiku-4-5` via Anthropic | Strong Arabic generation and instruction-following at the cheapest Claude tier ($1/$5 per MTok), fast enough to also carry the rerank call |
| Reranker | LLM-based (0–10 scoring) | A cross-encoder (`bge-reranker-v2-m3`) would need PyTorch, which this deployment avoids |

Reranking costs ~8s per query but is worth it: it lifted `hit_rate@5` from 0.88 to 1.00 and
out-of-scope refusal from 0.50 to 1.00. Turn it off with `FAQRAG_RERANK_ENABLED=false` when
latency matters more than precision.

### Scaling beyond ~1,000 chunks

The current defaults suit ~50 chunks. What changes as the corpus grows:

- **Vector store** — the NumPy backend scans exhaustively (fine to ~10k chunks). Beyond that,
  switch to `chroma`, or add a Qdrant backend behind `VectorStore`.
- **BM25** — rebuilt in memory at startup. Past ~10k chunks, persist it or move to a
  proper inverted index.
- **Reranker** — the single-call design sends all candidates in one prompt. Past ~20 candidates,
  batch it or move to a cross-encoder.
- **Thresholds** — `MIN_RELEVANCE_SCORE` is calibrated against this corpus. Re-run
  `python -m faqrag.eval` after any significant content change and re-check the gap between
  in-scope and out-of-scope confidence.

---

## Project layout

```
├── data/
│   ├── mwfaq_faq_rag.md          # source knowledge base (markdown)
│   ├── mwfaq_faq_rag.json        # generated: {"faqs": [...]}
│   └── index/                    # generated: vectors + chunks + manifest
├── scripts/
│   ├── md_to_json.py             # markdown → JSON converter
│   └── analyze_eval.py           # per-FAQ / per-category eval breakdown
├── src/faqrag/
│   ├── config.py                 # all settings; nothing hardcoded elsewhere
│   ├── models.py                 # Chunk, ScoredChunk, QueryResponse, ...
│   ├── lang.py                   # language detection + Arabic normalisation
│   ├── ingest.py                 # JSON → chunks
│   ├── embeddings.py             # EmbeddingProvider: OpenAI, Ollama
│   ├── stores/                   # VectorStore: base, numpy, chroma
│   ├── bm25.py                   # Okapi BM25 with Arabic tokenisation
│   ├── fusion.py                 # RRF, weighted fusion, language boost
│   ├── rerank.py                 # Reranker: LLM-based, no-op
│   ├── retriever.py              # hybrid retrieval orchestration
│   ├── llm.py                    # LLMClient: Anthropic, OpenAI, Ollama
│   ├── http_utils.py             # shared provider-error rendering
│   ├── prompts.py                # grounding + rerank prompts
│   ├── generate.py               # grounded answering, citation parsing
│   ├── pipeline.py               # end-to-end, shared by CLI/API/eval
│   ├── index.py                  # python -m faqrag.index
│   ├── cli.py                    # python -m faqrag.cli
│   ├── api.py                    # python -m faqrag.api
│   ├── messages.py               # async /v1/messages integration API
│   ├── eval.py                   # python -m faqrag.eval
│   ├── eval_data.py              # the 20 core evaluation cases
│   ├── eval_saudi.py             # 100 Saudi-dialect evaluation cases
│   └── web.py                    # optional: mounts the chat UI (deletable)
├── web/                          # optional: the chat UI (deletable)
│   ├── index.html                #   the entire UI - no build step, no CDN
│   └── README.md                 #   what it does, and how to remove it
├── docs/
│   └── INTEGRATION.md            # the /v1/messages contract for other systems
├── tests/                        # 172 tests, no network required
└── logs/retrieval.jsonl          # per-query score traces
```

### Tests

```bash
python -m pytest -q            # 172 tests, ~0.6s, fully offline
```

Coverage is concentrated where bugs are silent — they produce a worse answer, not an error:
chunking and the language pairing invariant, Arabic tokenisation, fusion arithmetic, score
normalisation, citation parsing, and reranker fallback behaviour. Several tests are explicit
regression guards for bugs found during development (RRF score compression, language-boost
saturation, and the candidate-numbering bug that made the model cite position 1 as FAQ "001").

---

## Out of scope for this pass

No auth, no multi-tenancy, and no deployment config. The chat UI is a local debugging and demo
surface, not a production frontend: it has no auth, no rate limiting, and no conversation
persistence, and the server binds to `127.0.0.1` by default.
