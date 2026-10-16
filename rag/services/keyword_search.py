"""
Keyword half of hybrid search: MongoDB full-text ($text) search over the embedded text.
Finds exact words, names and IDs that pure semantic search can miss. If the text index is
unavailable the function returns [] and search simply falls back to vectors only.
"""
import logging

from . import embedding_store

logger = logging.getLogger(__name__)


def search(query: str, limit: int, collections=None) -> list:
    """-> [(embedding_id, text_score)] best first."""
    try:
        flt = {"$text": {"$search": query}}
        if collections:
            flt["metadata.collection"] = {"$in": list(collections)}
        cursor = (embedding_store.collection()
                  .find(flt, {"score": {"$meta": "textScore"}})
                  .sort([("score", {"$meta": "textScore"})])
                  .limit(limit))
        return [(d["_id"], float(d.get("score", 0.0))) for d in cursor]
    except Exception as exc:
        logger.warning("Keyword search unavailable (%s); using vector search only.", type(exc).__name__)
        return []
