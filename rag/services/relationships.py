"""
Automatic relationship detection + ObjectId enrichment.

For every field that holds ObjectIds the resolver works out WHICH collection they point to:
  1. Name guess:  user / userId / user_id -> collection "users"; workspaceId -> "workspaces"
  2. Verification: the sample ids must really exist in that collection
  3. Probing: if the name gives no match (createdBy, assignedTo, owner...), the sample ids are
     looked up in the other collections and the best match wins
Results are cached per (collection, field). Names of referenced documents are loaded in
bulk ($in queries) per batch and cached, so enrichment costs a handful of queries, not one per id.
"""
import logging
import re
from collections import OrderedDict, defaultdict

from bson import ObjectId

from .document_converter import (EXTRA_NAME_FIELDS, HEX24, NAME_FIELDS, iter_ref_candidates,
                                 pick_display_name, strip_id_suffix)

logger = logging.getLogger(__name__)

MAX_PROBE_COLLECTIONS = 100
NAME_PROJECTION = {name: 1 for name in NAME_FIELDS + EXTRA_NAME_FIELDS}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def guess_collections(field_name: str, collections: list) -> list:
    """Collections whose (singular) name matches the field name, e.g. workspaceId -> workspaces."""
    key = _singular(_norm(strip_id_suffix(field_name)))
    if not key:
        return []
    return [c for c in collections if _singular(_norm(c)) == key]


def _id_variants(ids) -> list:
    variants = []
    for sid in ids:
        variants.append(sid)
        if HEX24.match(sid):
            variants.append(ObjectId(sid))
    return variants


class RelationshipResolver:
    MAX_ATTEMPTS = 3

    def __init__(self, db, collections, max_names: int = 100_000):
        self.db = db
        self.collections = list(collections)
        self._targets = {}            # (collection, field_path) -> ref collection | None
        self._attempts = {}
        self._names = OrderedDict()   # (ref_collection, id) -> display name | None
        self._max_names = max_names

    # ------------------------------------------------------------ public API
    def seed_target(self, collection: str, path: str, target: str):
        """Tell the resolver a field's target (e.g. from a saved catalog) so no probing is needed."""
        self._targets[(collection, path)] = target

    def prefetch(self, collection: str, docs: list, infer: bool = True):
        """Detect relationships for a batch and bulk-load the names of everything it references.
        With infer=False only already-known fields are used (no probing queries)."""
        ids_by_field = defaultdict(set)
        for doc in docs:
            for path, sid in iter_ref_candidates(doc):
                ids_by_field[path].add(sid)
        need = defaultdict(set)
        for path, ids in ids_by_field.items():
            target = (self._target_for(collection, path, sorted(ids)[:20]) if infer
                      else self._targets.get((collection, path)))
            if target:
                need[target].update(i for i in ids if (target, i) not in self._names)
        for target, ids in need.items():
            self._load_names(target, sorted(ids))

    def target_of(self, collection: str, path: str):
        """Collection that `collection.path` points to (after prefetch), or None."""
        return self._targets.get((collection, path))

    def target_of(self, collection: str, path: str):
        """Collection a field points to (or None)."""
        return self._targets.get((collection, path))

    def reset_names(self):
        """Forget cached display names (targets are kept) so the next lookups are fresh."""
        self._names.clear()

    def lookup(self, collection: str, path: str, sid: str):
        """-> (ref_collection, name_or_None) or None when the field is not a known reference."""
        target = self._targets.get((collection, path))
        if not target:
            return None
        return target, self._names.get((target, sid))

    # --------------------------------------------------------------- internals
    def _target_for(self, collection, path, samples):
        key = (collection, path)
        if key in self._targets:
            return self._targets[key]
        target = self._infer(path, samples)
        if target is not None:
            self._targets[key] = target
            logger.info("Relationship detected: %s.%s -> %s", collection, path, target)
            return target
        self._attempts[key] = self._attempts.get(key, 0) + 1
        if self._attempts[key] >= self.MAX_ATTEMPTS:
            self._targets[key] = None   # stop retrying: not a reference (or dangling ids)
        return None

    def _infer(self, path, samples):
        guessed = guess_collections(path.split(".")[-1], self.collections)
        for candidate in guessed:
            if self._hits(candidate, samples):
                return candidate
        best, best_hits = None, 0
        others = [c for c in self.collections if c not in guessed][:MAX_PROBE_COLLECTIONS]
        for candidate in others:
            hits = self._hits(candidate, samples)
            if hits > best_hits:
                best, best_hits = candidate, hits
        return best

    def _hits(self, collection, samples) -> int:
        try:
            found = self.db[collection].find({"_id": {"$in": _id_variants(samples)}}, {"_id": 1})
            return len(list(found))
        except Exception as exc:
            logger.warning("Probe of %s failed: %s", collection, type(exc).__name__)
            return 0

    def _remember(self, key, name):
        self._names[key] = name
        self._names.move_to_end(key)
        while len(self._names) > self._max_names:
            self._names.popitem(last=False)

    def _load_names(self, target, ids):
        for start in range(0, len(ids), 1000):
            chunk = ids[start:start + 1000]
            found = set()
            try:
                for doc in self.db[target].find({"_id": {"$in": _id_variants(chunk)}}, NAME_PROJECTION):
                    sid = str(doc["_id"])
                    found.add(sid)
                    self._remember((target, sid), pick_display_name(doc))
            except Exception as exc:
                logger.warning("Loading names from %s failed: %s", target, type(exc).__name__)
            for sid in chunk:
                if sid not in found:
                    self._remember((target, sid), None)   # dangling reference
