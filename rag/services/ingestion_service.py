"""
Ingestion pipeline:  MongoDB -> text -> relationships -> embeddings -> embeddings collection -> FAISS

  for each collection:
    for each BATCH of documents:
       detect relationships + bulk-load referenced names      (relationships.py)
       convert every document to chunk records                (document_converter.py)
       skip chunks whose content hash is unchanged            (incremental)
       embed only the new/changed chunks in one batch         (embedding_service.py)
       bulk upsert them into the embeddings collection        (embedding_store.py)
    delete embeddings of documents that no longer exist
  rebuild the FAISS index from the embeddings collection

Running it again is cheap: only new/changed documents are embedded, and deleted ones are removed.
A background job runner lets POST /api/ingest/ start it without blocking the web server.
"""
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

from django.conf import settings

from . import (document_converter, embedding_service, embedding_store, mongo_service,
               relationships, structured_query, vector_index)
from .exceptions import (ChatbotError, EmptyDatabaseError, IngestionBusyError, IngestionError)

logger = logging.getLogger(__name__)


def run_ingestion(collections=None, full=False, max_docs=0, batch_size=None, log=print) -> dict:
    started = time.perf_counter()
    batch_size = batch_size or settings.BATCH_SIZE
    available = mongo_service.list_collections()
    if not available:
        raise EmptyDatabaseError("The database has no collections to index.")
    if collections:
        unknown = [c for c in collections if c not in available]
        if unknown:
            raise IngestionError(f"Unknown collection(s): {', '.join(unknown)}. Available: {', '.join(available)}")
        targets = list(collections)
    else:
        targets = available

    embedding_store.ensure_indexes()
    old_models = [m for m in embedding_store.distinct_models() if m != settings.EMBEDDING_MODEL_ID]
    if old_models:
        if set(targets) != set(available):
            raise IngestionError(f"Existing embeddings were made with {old_models}. Re-embed ALL collections "
                                 f"(run without --collections) after changing the embedding model.")
        log(f"Embedding model changed ({', '.join(old_models)} -> {settings.EMBEDDING_MODEL_ID}): re-embedding everything.")
        full = True

    resolver = relationships.RelationshipResolver(mongo_service.get_db(), available)
    log(f"Collections to index: {', '.join(targets)}  (batch size {batch_size}, "
        f"{'full re-embed' if full else 'incremental'})")
    summary = {"mode": "full" if full else "incremental", "collections": {}}

    for name in targets:
        stats = {"documents": 0, "chunks": 0, "embedded": 0, "unchanged": 0, "removed": 0}
        existing = embedding_store.existing_hashes(name)
        seen = set()
        for batch in mongo_service.iter_batches(name, batch_size, max_docs):
            resolver.prefetch(name, batch)
            pending = []
            for doc in batch:
                records = document_converter.to_records(name, doc, resolver)
                if records:
                    stats["documents"] += 1
                for record in records:
                    stats["chunks"] += 1
                    seen.add(record["_id"])
                    if not full and existing.get(record["_id"]) == record["metadata"]["content_hash"]:
                        stats["unchanged"] += 1
                    else:
                        pending.append(record)
            if pending:
                vectors = embedding_service.embed_texts([r["text"] for r in pending])
                embedding_store.upsert_many(pending, vectors)
                stats["embedded"] += len(pending)
            log(f"  {name}: {stats['documents']} documents, {stats['embedded']} embedded so far")
        if not max_docs:   # only delete stale embeddings after a complete read of the collection
            stale = [i for i in existing if i not in seen]
            if stale:
                embedding_store.delete_ids(stale)
                stats["removed"] = len(stale)
        summary["collections"][name] = stats

    if embedding_store.count() == 0:
        raise EmptyDatabaseError("No documents with indexable content were found.")
    log("Rebuilding the FAISS index ...")
    index = vector_index.rebuild_index()
    summary["total_vectors"] = len(index.ids)
    try:
        log("Saving the query catalog ...")
        structured_query.build_catalog(force=True, resolver=resolver)
    except Exception as exc:       # list/count questions rebuild it on demand
        logger.warning("Could not save the catalog: %s", exc)
    summary["seconds"] = round(time.perf_counter() - started, 1)
    return summary


# ------------------------------------------------------- background job runner
_job_lock = threading.Lock()
_current = None


def start_job(collections=None, full=False, max_docs=0) -> dict:
    global _current
    with _job_lock:
        if _current and _current["state"] == "running":
            raise IngestionBusyError()
        job = {"id": uuid.uuid4().hex[:8], "state": "running", "started_at": datetime.now(timezone.utc).isoformat(),
               "finished_at": None, "log": [], "summary": None, "error": None}
        _current = job

    def log(message):
        job["log"] = (job["log"] + [message])[-50:]
        logger.info("[ingest %s] %s", job["id"], message)

    def worker():
        try:
            job["summary"] = run_ingestion(collections, full, max_docs, log=log)
            job["state"] = "finished"
        except ChatbotError as exc:
            job["state"], job["error"] = "failed", exc.message
        except Exception:
            logger.exception("Ingestion job crashed")
            job["state"], job["error"] = "failed", "Unexpected error. See the server log."
        job["finished_at"] = datetime.now(timezone.utc).isoformat()

    threading.Thread(target=worker, daemon=True, name=f"ingest-{job['id']}").start()
    return dict(job)


def job_status() -> dict:
    return dict(_current) if _current else {"state": "idle"}
