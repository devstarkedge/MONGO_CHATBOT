"""
FAISS search index built from the embeddings collection.

Files in vector_store/:  index.faiss (vectors)  +  index_meta.json (row -> embedding _id)
Row i in FAISS <-> ids[i] <-> the MongoDB embedding document with that _id. Nothing else to keep in sync.
If the files are missing but embeddings exist in MongoDB, the index is rebuilt automatically.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from django.conf import settings

from . import embedding_store
from .exceptions import VectorStoreError

logger = logging.getLogger(__name__)
INDEX_FILE, META_FILE = "index.faiss", "index_meta.json"
MISSING = "The search index has not been built yet. Run: python manage.py ingest"


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise VectorStoreError("FAISS is not installed. Run: pip install faiss-cpu") from exc
    return faiss


class VectorIndex:
    def __init__(self):
        self.index = None
        self.ids, self.collections = [], []
        self.model, self.dimension, self.built_at = "", 0, ""

    # -------------------------------------------------------------- build/save/load
    def build_from_store(self):
        faiss = _faiss()
        ids, cols, blocks = [], [], []
        for batch_ids, batch_cols, matrix in embedding_store.iter_vectors():
            if blocks and matrix.shape[1] != blocks[0].shape[1]:
                raise VectorStoreError("Stored embeddings have different sizes. Run a full ingest: "
                                       "python manage.py ingest --full")
            ids.extend(batch_ids)
            cols.extend(batch_cols)
            blocks.append(matrix)
        if not ids:
            raise VectorStoreError(MISSING)
        matrix = np.ascontiguousarray(np.vstack(blocks), dtype="float32")
        faiss.normalize_L2(matrix)
        self.dimension = int(matrix.shape[1])
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(matrix)
        self.ids, self.collections = ids, cols
        models = embedding_store.distinct_models()
        self.model = models[0] if len(models) == 1 else settings.EMBEDDING_MODEL_ID
        self.built_at = datetime.now(timezone.utc).isoformat()

    def save(self, directory):
        faiss = _faiss()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tmp_index, tmp_meta = directory / (INDEX_FILE + ".tmp"), directory / (META_FILE + ".tmp")
        faiss.write_index(self.index, str(tmp_index))
        tmp_index.replace(directory / INDEX_FILE)
        tmp_meta.write_text(json.dumps({
            "ids": self.ids, "collections": self.collections, "model": self.model,
            "dimension": self.dimension, "built_at": self.built_at,
        }), encoding="utf-8")
        tmp_meta.replace(directory / META_FILE)   # written last: it is the "version" stamp

    def load(self, directory):
        faiss = _faiss()
        directory = Path(directory)
        try:
            self.index = faiss.read_index(str(directory / INDEX_FILE))
            meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
            self.ids, self.collections = meta["ids"], meta["collections"]
            self.model, self.dimension = meta.get("model", ""), int(meta["dimension"])
            self.built_at = meta.get("built_at", "")
        except Exception as exc:
            logger.exception("Failed to load vector index")
            raise VectorStoreError("The search index looks corrupted. Rebuild it with: "
                                   "python manage.py ingest") from exc
        if self.index.ntotal != len(self.ids) or len(self.collections) != len(self.ids):
            raise VectorStoreError("The search index and its id list do not match. Rebuild it with: "
                                   "python manage.py ingest")

    # --------------------------------------------------------------------- search
    def search(self, query_vector, k: int, collections=None) -> list:
        """-> [(embedding_id, cosine_score)] best first, optionally limited to some source collections."""
        if self.index is None or not self.ids:
            return []
        query = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise VectorStoreError("Embedding size mismatch. Run a full ingest: python manage.py ingest --full")
        allowed = set(collections) if collections else None
        total = len(self.ids)
        fetch = min(total, k * 5 if allowed else k)
        while True:
            scores, rows = self.index.search(query, fetch)
            out = []
            for score, row in zip(scores[0], rows[0]):
                if row < 0 or (allowed and self.collections[row] not in allowed):
                    continue
                out.append((self.ids[row], float(score)))
                if len(out) == k:
                    break
            if len(out) >= k or fetch >= total:
                return out
            fetch = min(total, fetch * 4)


# ------------------------------------------------------- shared instance (Django)
_index = None
_stamp = None
_lock = threading.RLock()


def _stamp_now():
    try:
        return (Path(settings.VECTOR_STORE_DIR) / META_FILE).stat().st_mtime_ns
    except FileNotFoundError:
        return None


def current_version() -> str:
    """Changes whenever a new index is written (used to invalidate the answer cache)."""
    return str(_stamp_now())


def rebuild_index() -> VectorIndex:
    """Rebuild FAISS from the embeddings collection, save it and use it immediately."""
    global _index, _stamp
    with _lock:
        fresh = VectorIndex()
        fresh.build_from_store()
        fresh.save(settings.VECTOR_STORE_DIR)
        _index, _stamp = fresh, _stamp_now()
        logger.info("FAISS index rebuilt: %d vectors, dim %d", len(fresh.ids), fresh.dimension)
        return fresh


def get_index() -> VectorIndex:
    """Return the loaded index. Reloads when another process wrote a newer one; rebuilds if missing."""
    global _index, _stamp
    with _lock:
        stamp = _stamp_now()
        if _index is not None and stamp is not None and stamp == _stamp:
            return _index
        if stamp is not None:
            fresh = VectorIndex()
            try:
                fresh.load(settings.VECTOR_STORE_DIR)
                _index, _stamp = fresh, stamp
                logger.info("FAISS index loaded: %d vectors", len(fresh.ids))
                return fresh
            except VectorStoreError:
                logger.warning("Index files unusable, rebuilding from MongoDB")
        if embedding_store.count() == 0:
            raise VectorStoreError(MISSING)
        return rebuild_index()


def reset_for_tests():
    global _index, _stamp
    _index, _stamp = None, None
