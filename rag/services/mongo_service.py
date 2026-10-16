"""
MongoDB access: one shared client, dynamic collection discovery, batched reads.
Credentials come from settings (.env) and never appear in logs or error messages.
"""
import logging
import threading

from bson import ObjectId
from django.conf import settings
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from .exceptions import ConfigError, MongoServiceError

logger = logging.getLogger(__name__)
_client = None
_lock = threading.Lock()


def _require(name: str) -> str:
    value = (getattr(settings, name, "") or "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}. Add it to your .env file.")
    return value


def _get_client() -> MongoClient:
    global _client
    if _client is not None:
        return _client
    uri = _require("MONGO_URI")
    with _lock:
        if _client is None:
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=5000, appname="mongo-rag-pro")
                client.admin.command("ping")
            except PyMongoError as exc:
                logger.error("MongoDB connection failed: %s", type(exc).__name__)
                raise MongoServiceError() from exc
            _client = client
    return _client


def get_db():
    """The source database holding your data."""
    return _get_client()[_require("MONGO_DATABASE")]


def get_embeddings_collection():
    """The collection where embeddings + text + metadata + relationships are stored."""
    db_name = (settings.EMBEDDINGS_DATABASE or "").strip() or _require("MONGO_DATABASE")
    return _get_client()[db_name][settings.EMBEDDINGS_COLLECTION]


def ping() -> bool:
    try:
        _get_client().admin.command("ping")
        return True
    except (PyMongoError, ConfigError, MongoServiceError):
        return False


def list_collections() -> list:
    """Discover collections automatically (skips system.*, our embeddings collection, exclusions)."""
    try:
        names = get_db().list_collection_names()
    except PyMongoError as exc:
        raise MongoServiceError() from exc
    skip = set(settings.EXCLUDE_COLLECTIONS)
    if not (settings.EMBEDDINGS_DATABASE or "").strip():
        skip.add(settings.EMBEDDINGS_COLLECTION)
    return sorted(n for n in names if not n.startswith("system.") and n not in skip)


def iter_batches(collection: str, batch_size: int, limit: int = 0):
    """Yield lists of documents (batch_size at a time) so memory stays flat on big collections."""
    try:
        cursor = get_db()[collection].find({}).batch_size(batch_size)
        if limit and limit > 0:
            cursor = cursor.limit(limit)
        batch = []
        for doc in cursor:
            batch.append(doc)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    except PyMongoError as exc:
        raise MongoServiceError(f"Failed while reading collection '{collection}'.") from exc


def get_document(collection: str, document_id: str):
    """Fetch an original document using the id saved in the embedding metadata."""
    candidates = [document_id]
    if len(document_id) == 24:
        try:
            candidates.append(ObjectId(document_id))
        except Exception:
            pass
    if document_id.lstrip("-").isdigit():
        candidates.append(int(document_id))
    try:
        return get_db()[collection].find_one({"_id": {"$in": candidates}})
    except PyMongoError as exc:
        raise MongoServiceError() from exc
