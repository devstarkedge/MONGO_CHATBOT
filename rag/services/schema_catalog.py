"""
Schema catalog: a compact, always-correct description of what is in MongoDB, built by sampling.
The query planner (structured_query.py) reads it so the LLM only ever refers to REAL collections,
fields and values (e.g. it learns that "finished" means status = "completed").

For each collection:  document count, field paths, field types, which fields are references to other
collections, and the actual values of low-cardinality fields. Sensitive fields are never included.
The catalog is cached (CATALOG_TTL_SECONDS) and rebuilt automatically.
"""
import logging
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

from bson import ObjectId
from bson.decimal128 import Decimal128
from django.conf import settings

from . import mongo_service
from .document_converter import HEX24, is_sensitive
from .relationships import RelationshipResolver

logger = logging.getLogger(__name__)

MAX_COLLECTIONS, MAX_FIELDS, MAX_DEPTH = 60, 30, 2
MAX_VALUES, MAX_EXAMPLES, LOW_CARDINALITY = 12, 3, 25


@dataclass
class FieldInfo:
    path: str
    type: str = "string"            # string | number | boolean | date | reference | array | object
    ref: str = None                 # collection a reference points to
    element: str = None             # element type for arrays
    values: list = field(default_factory=list)      # all values when the field has few distinct values
    examples: list = field(default_factory=list)    # a few sample values otherwise


@dataclass
class CollectionProfile:
    name: str
    count: int
    fields: "OrderedDict" = field(default_factory=OrderedDict)   # path -> FieldInfo


@dataclass
class Catalog:
    profiles: dict
    resolver: RelationshipResolver
    built_at: float

    def text(self, only=None) -> str:
        return render_catalog([p for n, p in self.profiles.items() if not only or n in only])

    def restricted(self, only):
        return {n: p for n, p in self.profiles.items() if n in set(only)}


def _kind(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, ObjectId):
        return "reference"
    if isinstance(value, (int, float, Decimal, Decimal128)):
        return "number"
    if isinstance(value, (datetime, date)):
        return "date"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return None


class _Acc:
    def __init__(self):
        self.kinds, self.elements, self.values, self.ids = Counter(), Counter(), Counter(), []


def _collect(doc, prefix, depth, acc):
    for key, value in doc.items():
        key = str(key)
        if key in ("_id", "__v") or is_sensitive(key):
            continue
        path = f"{prefix}.{key}" if prefix else key
        kind = _kind(value)
        if kind is None:
            continue
        slot = acc.setdefault(path, _Acc())
        slot.kinds[kind] += 1
        if kind == "object":
            if depth < MAX_DEPTH:
                _collect(value, path, depth + 1, acc)
        elif kind == "array":
            for item in list(value)[:20]:
                ek = _kind(item)
                if ek in (None, "object", "array"):
                    continue
                slot.elements[ek] += 1
                if ek == "reference":
                    slot.ids.append(str(item))
                elif ek in ("string", "boolean"):
                    slot.values[str(item)[:40]] += 1
        elif kind == "reference":
            slot.ids.append(str(value))
        elif kind in ("string", "boolean"):
            text = ("Yes" if value else "No") if kind == "boolean" else " ".join(value.split())[:40]
            if text:
                slot.values[text] += 1
            if kind == "string" and HEX24.match(value.strip()):
                slot.ids.append(value.strip())


def _sample(db, name):
    size = settings.CATALOG_SAMPLE_SIZE
    try:
        return list(db[name].aggregate([{"$sample": {"size": size}}]))
    except Exception:
        return list(db[name].find({}).limit(size))


def _profile(db, name, resolver) -> CollectionProfile:
    docs = _sample(db, name)
    try:
        count = db[name].estimated_document_count()
    except Exception:
        count = len(docs)
    acc = OrderedDict()
    for doc in docs:
        _collect(doc, "", 0, acc)
    resolver.prefetch(name, docs)
    profile = CollectionProfile(name=name, count=count)
    for path, slot in list(acc.items())[:MAX_FIELDS * 2]:
        kind = slot.kinds.most_common(1)[0][0]
        info = FieldInfo(path=path, type=kind)
        target = resolver.target_of(name, path)
        if target:
            info.type, info.ref = "reference", target
        elif kind == "array":
            info.element = slot.elements.most_common(1)[0][0] if slot.elements else None
        if info.type in ("string", "boolean", "array") and slot.values:
            if len(slot.values) <= LOW_CARDINALITY and sum(slot.values.values()) >= min(5, len(docs)):
                info.values = [v for v, _ in slot.values.most_common(MAX_VALUES)]
            else:
                info.examples = [v for v, _ in slot.values.most_common(MAX_EXAMPLES)]
        profile.fields[path] = info
    return profile


def render_catalog(profiles) -> str:
    lines = []
    for p in profiles:
        lines.append(f"- {p.name} ({p.count} documents)")
        for info in list(p.fields.values())[:MAX_FIELDS]:
            if info.type == "reference":
                desc = f"reference to {info.ref}" if info.ref else "id"
            elif info.type == "array":
                desc = f"array of {info.element or 'values'}"
            else:
                desc = info.type
            extra = ""
            if info.values:
                extra = " values: " + ", ".join(f'"{v}"' for v in info.values)
            elif info.examples:
                extra = " e.g. " + ", ".join(f'"{v}"' for v in info.examples)
            lines.append(f"    {info.path} ({desc}){extra}")
    return "\n".join(lines)


_catalog = None
_lock = threading.Lock()


def get_catalog() -> Catalog:
    global _catalog
    with _lock:
        if _catalog is not None and time.monotonic() - _catalog.built_at < settings.CATALOG_TTL_SECONDS:
            return _catalog
        started = time.perf_counter()
        db = mongo_service.get_db()
        names = mongo_service.list_collections()
        resolver = RelationshipResolver(db, names)
        profiles = {}
        for name in names[:MAX_COLLECTIONS]:
            try:
                profiles[name] = _profile(db, name, resolver)
            except Exception as exc:
                logger.warning("Could not profile %s: %s", name, type(exc).__name__)
        _catalog = Catalog(profiles=profiles, resolver=resolver, built_at=time.monotonic())
        logger.info("Schema catalog built: %d collections in %.1fs", len(profiles), time.perf_counter() - started)
        return _catalog


def reset():
    global _catalog
    _catalog = None
