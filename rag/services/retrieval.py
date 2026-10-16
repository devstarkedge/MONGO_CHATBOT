"""
Retrieval: hybrid search (vector + keyword, fused with Reciprocal Rank Fusion)
and multi-hop expansion (follow relationships forward and backward).
"""
import logging
from dataclasses import dataclass, field

from django.conf import settings

from . import embedding_service, embedding_store, keyword_search, vector_index
from .exceptions import VectorStoreError

logger = logging.getLogger(__name__)


@dataclass
class Hit:
    id: str
    text: str
    collection: str
    primary_id: str
    chunk_index: int = 0
    display_name: str = None
    relationships: list = field(default_factory=list)
    score: float = 0.0
    vector_score: float = None
    keyword_score: float = None
    via: dict = None            # set for related records: {from_doc, field, direction}

    @property
    def doc_key(self):
        return self.collection, self.primary_id

    def to_dict(self) -> dict:
        data = {"id": self.id, "collection": self.collection, "document_id": self.primary_id,
                "name": self.display_name, "text": self.text, "score": round(self.score, 4),
                "vector_score": None if self.vector_score is None else round(self.vector_score, 4),
                "keyword_score": None if self.keyword_score is None else round(self.keyword_score, 4),
                "chunk_index": self.chunk_index, "relationships": self.relationships}
        if self.via:
            data["via"] = {"from": list(self.via["from_doc"]), "field": self.via["field"],
                           "direction": self.via["direction"]}
        return data


def _to_hit(doc: dict, **extra) -> Hit:
    meta = doc.get("metadata", {})
    return Hit(id=doc["_id"], text=doc.get("text", ""), collection=meta.get("collection", ""),
               primary_id=meta.get("primary_id", ""), chunk_index=meta.get("chunk_index", 0),
               display_name=meta.get("display_name"), relationships=doc.get("relationships", []), **extra)


def rrf_fuse(vector_hits: list, keyword_hits: list, k: int, keyword_weight: float, rrf_k: int) -> list:
    """Reciprocal Rank Fusion of two ranked lists of (id, score) -> [(id, fused_score)]."""
    fused = {}
    for rank, (doc_id, _score) in enumerate(vector_hits, 1):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (doc_id, _score) in enumerate(keyword_hits, 1):
        fused[doc_id] = fused.get(doc_id, 0.0) + keyword_weight / (rrf_k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]


def search(query: str, top_k: int = None, collections=None, hybrid: bool = None) -> list:
    """Return the top_k most relevant embedded chunks as Hit objects."""
    top_k = top_k or settings.TOP_K
    hybrid = settings.HYBRID_SEARCH if hybrid is None else hybrid
    index = vector_index.get_index()
    if index.model and index.model != settings.EMBEDDING_MODEL_ID:
        raise VectorStoreError(f"The index was built with '{index.model}' but the app is configured for "
                               f"'{settings.EMBEDDING_MODEL_ID}'. Run: python manage.py ingest --full")

    query_vector = embedding_service.embed_query(query)
    wide = top_k * 3
    vector_hits = [(i, s) for i, s in index.search(query_vector, wide, collections) if s >= settings.MIN_SCORE]
    keyword_hits = keyword_search.search(query, wide, collections) if hybrid else []
    if not vector_hits and not keyword_hits:
        return []

    vector_scores, keyword_scores = dict(vector_hits), dict(keyword_hits)
    fused = rrf_fuse(vector_hits, keyword_hits, top_k, settings.KEYWORD_WEIGHT, settings.RRF_K)
    docs = embedding_store.get_many([doc_id for doc_id, _ in fused])
    hits = []
    for doc_id, fused_score in fused:
        doc = docs.get(doc_id)
        if doc is None:      # removed since the index was built
            continue
        hits.append(_to_hit(doc, score=fused_score, vector_score=vector_scores.get(doc_id),
                            keyword_score=keyword_scores.get(doc_id)))
    return hits


def expand(hits: list, hops: int = None, max_related: int = None) -> list:
    """
    Multi-hop resolution. Starting from the hits:
      forward  follow each record's relationships to the records it points to
      reverse  find records that point AT a hit (e.g. tasks assigned to a retrieved user)
    Up to `hops` levels (reverse only at the first level), capped at `max_related` records.
    """
    hops = settings.MAX_HOPS if hops is None else hops
    max_related = settings.MAX_RELATED if max_related is None else max_related
    if hops <= 0 or max_related <= 0 or not hits:
        return []

    seen_docs = {h.doc_key for h in hits}
    related, frontier = [], hits
    for level in range(hops):
        wanted = {}                       # embedding id -> via
        for hit in frontier:
            for rel in hit.relationships:
                coll, ref_id = rel.get("ref_collection"), rel.get("ref_id")
                if coll and ref_id and (coll, ref_id) not in seen_docs:
                    wanted.setdefault(f"{coll}:{ref_id}:0", {
                        "from_doc": hit.doc_key, "field": rel["field"], "direction": "forward"})
        new_hits = []
        for doc_id, doc in embedding_store.get_many(list(wanted)[:max_related - len(related)]).items():
            hit = _to_hit(doc, via=wanted[doc_id])
            if hit.doc_key not in seen_docs:
                seen_docs.add(hit.doc_key)
                new_hits.append(hit)
        related.extend(new_hits)

        if level == 0 and settings.INCLUDE_REVERSE:
            for hit in hits:
                if len(related) >= max_related:
                    break
                for doc in embedding_store.find_referencing(hit.primary_id, limit=settings.REVERSE_PER_RECORD):
                    found = _to_hit(doc, via={"from_doc": hit.doc_key, "field": "(references this record)",
                                              "direction": "reverse"})
                    if found.doc_key not in seen_docs and len(related) < max_related:
                        seen_docs.add(found.doc_key)
                        new_hits.append(found)
                        related.append(found)
        frontier = new_hits
        if len(related) >= max_related or not frontier:
            break
    return related[:max_related]
