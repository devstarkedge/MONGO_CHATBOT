# Mongo RAG Pro: MongoDB, relationships, embeddings, hybrid search, LLM

Ask questions in plain language about **any** MongoDB database. The system converts documents of any schema to
natural language, detects and resolves the ObjectId relationships between collections, stores embeddings in MongoDB,
searches them (semantic + keyword), follows links between records, and lets an LLM (gpt-oss-20b, OpenAI, ...) answer
from the retrieved records only.

## 0. Run fully local with Ollama (gemma3:4b)

Nothing leaves your machine: MongoDB, the embedding model and the language model are all local.

```powershell
# 1. Install Ollama from https://ollama.com (Windows/macOS/Linux), then download the model once:
ollama pull gemma3:4b
ollama run gemma3:4b "say hi"          # quick check that it works; type /bye to exit

# 2. In this project
copy .env.example .env                  # the default .env.example IS the local profile (LLM_PROVIDER=ollama)
#    set MONGO_URI and MONGO_DATABASE in .env, nothing else is required (no API key)
python manage.py ingest
python manage.py runserver
```

The top-right pill tells you if Ollama is not running ("Ollama is not running") or the model is missing ("Run: ollama pull gemma3:4b").
The Index tab shows the model and whether Ollama is reachable. Use `.env.cloud.example` to go back to Groq/OpenAI.

**How the local profile differs.** The code talks to Ollama's native API (`/api/chat`), because the OpenAI-compatible endpoint cannot set the
context window and Ollama's default window would silently cut off long prompts. Each request sets `num_ctx` (`OLLAMA_NUM_CTX`, default 8192) and
keeps the model loaded (`OLLAMA_KEEP_ALIVE`). The query planner uses Ollama's JSON mode, so even a 4B model returns valid JSON (the plan is still
validated strictly). A shorter system prompt is used, and the defaults are smaller than for hosted models:
`TOP_K=4`, `MAX_CONTEXT_CHARS=6000`, `LIST_TOP_K=20`, `STRUCT_MAX_RECORDS=40`, `LLM_MAX_TOKENS=600`, `LLM_TIMEOUT=180`. The model is preloaded when the server starts.

**What to expect from a 4B model on a laptop**
- It needs a few GB of free RAM plus roughly 3 to 4 GB of disk for the model. Speed depends heavily on your hardware: a GPU is much faster than CPU only.
- The first answer after starting is slower (the model loads). Later answers are faster while the model stays in memory.
- Small models are weaker at following long instructions and at planning filters. Safeguards: plans are validated, wrong plans fall back to similarity search or
  a plain complete list, exact totals come from MongoDB (never from the model's counting), and if Ollama fails you still get the plain result.
- If it is too slow: set `STRUCT_PLANNER=off` (skips the planning call), lower `TOP_K` / `MAX_CONTEXT_CHARS`, use the Fast button, or raise `LLM_TIMEOUT`.
  With `STRUCT_PLANNER=off`, "list all departments" still returns the complete list, but conditions ("only medium priority") are applied by the model reading the records.
- Larger Ollama models (for example an 8B+) follow instructions better: change `LLM_MODEL`, run `ollama pull <model>`, restart.

Not verified here: this was tested against a simulated Ollama API (request format, errors, the full pipeline), not a live Ollama install.

## 0b. Use Google Gemini (gemini-3.5-flash-lite)

```powershell
copy .env.gemini.example .env       # then set MONGO_URI, MONGO_DATABASE and GEMINI_API_KEY (key from https://aistudio.google.com/apikey)
python manage.py check_llm          # tests the model with real calls and says exactly what is wrong, if anything
python manage.py ingest
python manage.py runserver
```

`.env` needs only these three lines for the model: `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...`, `LLM_MODEL=gemini-3.5-flash-lite`.
(`GOOGLE_API_KEY` also works as the key name.)

**What the `gemini` provider does differently** (from Google's documentation for Gemini 3.x):
- It uses Google's OpenAI-compatible endpoint, `https://generativelanguage.googleapis.com/v1beta/openai/`, with an AI Studio key.
  A key or URL from another provider (for example the Groq URL left in `.env`) gives "model not found" or "key rejected".
- `temperature` is **never sent**: Google has deprecated it for Gemini 3.x (ignored today, an HTTP 400 in future models).
  An old `LLM_TEMPERATURE=0` in your `.env` is harmless.
- The output limit is sent as `max_tokens`. Thinking tokens count toward it; if the limit is hit the app now says so
  ("ran out of output tokens") instead of showing an empty answer as "not found in the database".
- Leave `LLM_REASONING_EFFORT` empty: gemini-3.5-flash-lite defaults to the fastest "minimal" thinking.
- Requests always end with a user turn (Gemini 3.5 rejects a trailing model turn).

**Cost and privacy.** Google lists gemini-3.5-flash-lite at $0.30 per 1M input tokens and $2.50 per 1M output tokens (check the pricing page,
including any free-tier limits for your account). Only the retrieved records for each question are sent to Google, never the whole database,
but those records do leave your machine. For sensitive data use the local Ollama profile (section 0).

**If it does not work, run `python manage.py check_llm`:**

| `check_llm` says | Cause | Fix |
|---|---|---|
| key rejected by `api.groq.com` (or another host) | wrong `LLM_BASE_URL` or key for another provider | use `LLM_PROVIDER=gemini` and remove `LLM_BASE_URL` / `LLM_API_KEY` of other providers |
| key rejected by `generativelanguage.googleapis.com` | key invalid, expired or restricted | create a new key in AI Studio, restart the server |
| model not found at ... | wrong model id | `LLM_MODEL=gemini-3.5-flash-lite` (exact id, all lowercase) |
| rate-limited | free-tier or per-minute limit | wait, or enable billing for higher limits |
| took longer than 60s | network or slow response | retry; raise `LLM_TIMEOUT` |
| ran out of output tokens | thinking used the budget | raise `LLM_MAX_TOKENS` / `LIST_MAX_TOKENS`, clear `LLM_REASONING_EFFORT` |
| WARNING 2/2 (not valid JSON) | the planner step answered in prose | list/count questions fall back to similarity search; try again or report it |

## 1. Architecture

```
INGEST  (python manage.py ingest   or   POST /api/ingest/)
 MongoDB collections (discovered automatically)
   -> read in BATCHES
   -> relationship resolver: which collection does each ObjectId field point to? load their names in bulk
   -> universal converter: any document -> readable text (+ metadata, relationships)
   -> chunking (fits the embedding window)
   -> skip unchanged chunks (content hash)           <- incremental
   -> embeddings (local sentence-transformers or OpenAI), normalised
   -> `rag_embeddings` collection: {embedding, text, metadata, relationships}   <- source of truth
   -> FAISS index rebuilt from it (index.faiss + index_meta.json)

ASK  (POST /api/ask/)
 question -> [cache?] -> hybrid search: FAISS vectors + MongoDB $text keywords -> Reciprocal Rank Fusion
          -> multi-hop expansion (follow relationships forward and backward)
          -> context (character budget) -> LLM (OpenAI-compatible API) -> answer + sources + timings
```

## 2. Quick start

```bash
python -m venv .venv && .venv\Scripts\activate         # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt                         # dev/tests: pip install -r requirements-dev.txt
copy .env.example .env                                  # then edit .env (MONGO_URI, MONGO_DATABASE, LLM_API_KEY)
python manage.py seed_sample_data                       # optional: cross-referenced example data
python manage.py ingest                                 # build embeddings + index
python manage.py runserver                              # http://127.0.0.1:8000
```

For the local Ollama profile see section 0. For hosted models, start from `.env.cloud.example`: the LLM defaults to Groq with `openai/gpt-oss-20b`. For OpenAI set `LLM_BASE_URL=https://api.openai.com/v1`,
`LLM_MODEL=gpt-4o-mini`, `LLM_API_KEY=sk-...`. `GROQ_API_KEY` from the previous project still works as a fallback.
Async serving: `pip install uvicorn` then `uvicorn config.asgi:application`.

## 2b. The interface

Three tabs, all using the same API as your own clients:

- **Chat**: centered glass composer. *Fast* (3 records, no links), *In-depth* (6 records, 1 hop) and *Holistic* (10 records, 2 hops) set `top_k` and `hops` for `/api/ask/`. *Sources* and *Timing* toggles show or hide the source chips and per-stage timings. Answers render Markdown (bold, lists, tables, code) safely; the copy button copies the plain answer.
- **Search**: `/api/search/` with scores (match, vector, keyword) and relationships, plus an optional "Follow relationships" mode.
- **Index**: health cards (index, models, cache, MongoDB) and a button that starts `/api/ingest/` and shows the live job log. Enter the token only if `INGEST_TOKEN` is set.

Files: `templates/rag/index.html`, `static/rag/app.css`, `static/rag/app.js` (no external fonts or libraries).

## 3. API

```bash
# Ask (full RAG)
curl -X POST localhost:8000/api/ask/ -H "Content-Type: application/json" \
     -d '{"question": "Which workspace does Aisha Khan belong to?"}'

# Search only (no LLM): optional top_k, collections, hybrid, expand, hops
curl -X POST localhost:8000/api/search/ -H "Content-Type: application/json" \
     -d '{"query": "completed projects", "top_k": 5, "collections": ["projects"], "expand": true}'

# Ingest (background job; token required unless DJANGO_DEBUG=True)
curl -X POST localhost:8000/api/ingest/ -H "Authorization: Bearer $INGEST_TOKEN" -d '{"full": false}'
curl localhost:8000/api/ingest/ -H "Authorization: Bearer $INGEST_TOKEN"      # job status + log
curl localhost:8000/api/health/                                               # monitoring
```

`/api/ask/` returns `{question, answer, sources[{collection, document_id, name, score, related, via}], cached, timings_ms}`.
Errors are always `{"error": "..."}` with a meaningful status code (400 input, 401 token, 409 job running, 502 LLM, 503 MongoDB/index).

## 4. What each requirement maps to

| Requirement | Where |
|---|---|
| Mongo URI, dynamic collections, batches | `services/mongo_service.py` (`list_collections`, `iter_batches`) |
| Any schema -> natural language (ObjectId, dates, arrays, nested) | `services/document_converter.py` |
| Metadata: `_id`, ObjectId references, collection, field names | `document_converter.to_records` -> `metadata` |
| Automatic relationship detection `{field, ref_id, ref_collection}` | `services/relationships.py` |
| Embeddings, normalised, batched | `services/embedding_service.py` |
| Separate embeddings collection `{embedding, text, metadata, relationships}` | `services/embedding_store.py` |
| Semantic search (cosine) with FAISS | `services/vector_index.py` |
| Hybrid search (keyword + vector) | `services/keyword_search.py`, `retrieval.rrf_fuse` |
| Multi-hop resolution | `retrieval.expand` |
| LLM answering (gpt-oss-20b / OpenAI) | `services/llm_service.py`, `services/rag_service.py` |
| `/ingest`, `/search`, `/ask` | `views.py`, `urls.py` |
| Caching, async, logging, monitoring | `cache_service.py`, async views, `middleware.py`, `/api/health/` |

## 5. The stored record

```json
{
  "_id": "memberships:65a...:0",
  "embedding": [0.012, -0.044, "..."],
  "text": "Collection: memberships\nDocument ID: 65a...\nThis is a record in the 'memberships' collection that references users (via user), workspaces (via workspace).\nUser: Aisha Khan (ID: b00...)\nWorkspace: Acme Corp (ID: a00...)\nRole: employee\nStatus: active",
  "metadata": {"collection": "memberships", "primary_id": "65a...", "chunk_index": 0, "chunk_count": 1,
               "field_names": ["user", "workspace", "role", "status"], "reference_ids": ["b00...", "a00..."],
               "display_name": null, "content_hash": "...", "model": "local:sentence-transformers/all-MiniLM-L6-v2",
               "indexed_at": "2026-..."},
  "relationships": [{"field": "user", "ref_id": "b00...", "ref_collection": "users", "ref_name": "Aisha Khan"},
                    {"field": "workspace", "ref_id": "a00...", "ref_collection": "workspaces", "ref_name": "Acme Corp"}]
}
```

Text, metadata, relationships and the vector are one document, so they cannot drift apart. The `_id` is deterministic
(`collection:document_id:chunk`), so re-ingesting updates in place. FAISS row *i* maps to `ids[i]`, the `_id` of this document.

## 6. How the clever parts work

**Relationship detection (no schema needed).** For each field that holds ObjectIds (or 24-hex strings), a sample of its
values is checked: (1) the field name is matched to a collection (`userId`, `user_id`, `user` -> `users`; `categoryId` -> `categories`),
verified by checking those ids really exist there; (2) if the name gives nothing (`createdBy`, `assignedTo`, `owner`), the sample ids are
looked up in the other collections and the best match wins. Results are cached per `(collection, field)`. Hex strings are only treated as
references when they resolve, so hashes and tokens are left alone. Dangling references are kept as plain `ID: ...` text.

**Enrichment.** Names of referenced documents (`name`, `title`, `firstName`+`lastName`, `email`, ...) are loaded with one `$in` query per batch
and per target collection, and written into the text ("Workspace: Acme Corp"), so questions can use names instead of ids. If a referenced
name changes, the documents pointing to it get new text, a new hash, and are re-embedded on the next ingest.

**Hybrid search.** Vector results and MongoDB `$text` results (exact words, names, ids) are merged with Reciprocal Rank Fusion
(`KEYWORD_WEIGHT`, `RRF_K`). If the text index is unavailable it silently falls back to vector search.

**Multi-hop.** After retrieval, `expand` follows each hit's relationships forward (task -> project -> workspace, up to `MAX_HOPS`) and
backward (a retrieved user -> tasks assigned to them), capped by `MAX_RELATED`. Related records go to the LLM as `[R1]`, `[R2]`, marked with the link
they came from, and are listed in `sources` with `related: true`.

**Performance.** Batched reads and bulk writes; embeddings only for new/changed chunks; model loaded once; FAISS in memory; query-embedding
cache and answer cache (invalidated automatically when a new index is written, even by another process); prompt limited by `MAX_CONTEXT_CHARS`;
blocking work runs in worker threads behind async views; ingestion runs as a background job.

## 7. Configuration

See `.env.example`. Most useful: `TOP_K`, `HYBRID_SEARCH`, `MAX_HOPS`, `MAX_RELATED`, `INCLUDE_REVERSE`, `BATCH_SIZE`, `CHUNK_SIZE`,
`EXCLUDE_COLLECTIONS`, `EXCLUDE_FIELDS`, `CACHE_TTL_SECONDS` (0 = no answer cache), `EMBEDDING_PROVIDER` (`local` or `openai`).
Fields named like `password`, `token`, `secret`, `hash`, `salt`, `api_key` are never embedded.

## 8. Testing

```bash
pip install -r requirements-dev.txt
python manage.py test        # 90 tests, no MongoDB / internet / API key needed
```

Tests use an in-memory MongoDB (mongomock), a deterministic fake embedder and a fake LLM. They cover conversion, relationship detection (by name, by
probing, string ids), incremental ingestion (unchanged / changed / deleted / renamed references), search, fusion, multi-hop, caching, the context budget,
the API and auth. **Not covered by the automated tests:** MongoDB `$text` keyword search (mongomock lacks it, so the fallback path is tested), the real
embedding model, and real LLM calls. Verify those once on your own setup: run `seed_sample_data`, `ingest`, then ask the questions below.

## 9. Try these (sample data)

| Question | What it exercises |
|---|---|
| Which projects have been finished? | semantic match: finished = completed |
| Which workspace does Aisha Khan belong to? | relationship: membership -> user + workspace (names resolved) |
| What tasks are assigned to Rohan Mehta? | reverse hop: user -> tasks |
| Which workspace is the Fix login bug task in? | 2 hops: task -> project -> workspace (set `"hops": 2`) |
| What is Aisha's password? | never embedded: "I couldn't find that information in the database." |

## 9a. Exact answers for list and count questions

Similarity search returns the *nearest* records, so asking for "all 20 departments" can give 6, 8, 10 or 13 depending on the wording.
For list/count questions (and questions that use a collection name in plural, like "departments" or "tasks") the system now queries MongoDB directly:

1. A catalog of collections and fields (sampled, cached 10 minutes) is shown to the LLM, which fills a tiny JSON plan:
   `{"collection": "tasks", "operation": "list", "filters": [{"field": "priority", "op": "eq", "value": "medium"}]}`.
2. `structured_query.validate_plan` checks it strictly: known collection, known fields, whitelisted operators (`eq ne in contains gt gte lt lte`).
   Any doubt rejects the whole plan. A condition is never silently dropped, because that would make an "exact" count wrong.
   The LLM never writes MongoDB syntax; the code translates the plan.
3. `count_documents` gives the exact total; `find` gives up to `STRUCT_MAX_RECORDS` (100) records. Filters on reference fields accept names
   ("assigned to Rohan Mehta" is resolved to ids in `users` first).
4. The LLM phrases the answer from the exact result. The prompt says whether the list is complete, and the exact total is always stated.

Fallbacks: no valid plan -> if the question names a small collection (up to `STRUCT_SCAN_MAX`, 150) the whole collection is read and the LLM is told conditions were NOT applied;
zero matches -> similarity search (the plan may have misread the question); several collections or calculations -> similarity search.
Exact answers are not cached (they are always live). They cost one extra small LLM call (the planner). Turn the feature off with `STRUCTURED_QUERIES=false`.
The chat's timing line shows e.g. `exact query: 20 in departments`.

## 9a-2. Speed and "language model unavailable"

The chat's timing line shows where the time goes. Exact list/count questions report `plan` (catalog + planner LLM call), `db` (MongoDB) and `llm` (final answer);
other questions report `search`, `expand`, `llm`. Whichever number is large is the culprit.

- **Catalog.** Probing which collection each id field points to is the slowest part. It now happens once, during `python manage.py ingest`, and is saved to
  `vector_store/catalog.json`; the server loads it from disk (and refreshes it in the background after `CATALOG_MAX_AGE_HOURS`). Run `ingest` once after upgrading.
- **Plans are cached** per question (1 hour), so repeating a list question skips the planner call.
- **Only needed fields** go to the LLM ("names of all departments" sends just the names). `STRUCT_MAX_RECORDS` defaults to 60.
- **Reasoning models are slow.** gpt-oss "thinks" before answering. Set `LLM_REASONING_EFFORT=low` (dropped automatically if the provider rejects it),
  and optionally `PLANNER_MODEL=<a small fast model your provider offers>` for the planning step.
- **If the language model fails** during a list/count question, you still get the complete, exact list as a plain result (marked "plain result" in the timing line).

Error messages are specific. The server log always has the exact line: `LLM call failed: <ErrorName> (HTTP <status>): ...`.

| Message says | Meaning | Fix |
|---|---|---|
| took longer than Ns | timeout | `LLM_REASONING_EFFORT=low`, raise `LLM_TIMEOUT`, lower `LIST_MAX_TOKENS` |
| rate-limited | too many requests/tokens per minute | wait; fewer calls; higher plan |
| request was too large | token limit of your plan | lower `STRUCT_MAX_RECORDS`, `LIST_MAX_TOKENS`, `MAX_CONTEXT_CHARS` |
| temporary problem (HTTP 5xx) | provider outage/capacity | retry shortly |
| could not connect | network or wrong `LLM_BASE_URL` | check internet / URL |
| API key is invalid / model not found / access denied | configuration | fix `.env`, restart |

## 9b. "Why only a few results?" (similarity search)

Similarity search returns the *closest* records, not every match. How many is set by `top_k`: Fast = 3, In-depth = 6, Holistic = 10, API up to 50.
List-style questions ("list all...", "which projects...", "how many...", "show me...") automatically retrieve at least `LIST_TOP_K` (default 30) chunks,
with a larger prompt budget (`LIST_CONTEXT_CHARS`) and answer length (`LIST_MAX_TOKENS`). The reply's timing line shows how many records were used.
This is still bounded: for complete lists or exact counts over big data, a structured MongoDB query is needed (see limits).

## 10. Limits and next steps

- FAISS `IndexFlatIP` is exact and fine up to a few hundred thousand chunks. Beyond that use MongoDB Atlas Vector Search (the `embedding` field is already a
  plain number array; index it with `{"fields":[{"type":"vector","path":"embedding","numDimensions":384,"similarity":"cosine"},
  {"type":"filter","path":"metadata.collection"}]}`) and replace `vector_index.search` with a `$vectorSearch` aggregation. Not implemented here.
- Changing the embedding model requires `ingest --full` (detected and reported automatically).
- Exact queries cover lists and counts over ONE collection with simple conditions. Sums, averages, grouping and multi-collection joins still go through similarity search.
- PDFs and binary fields are ignored; only text stored in fields is indexed.
- The chat page and `/api/ask/` have no login. Add authentication before exposing them. `/api/ingest/` already requires `INGEST_TOKEN` outside DEBUG.
#   t e s t _ b o t 
 
 