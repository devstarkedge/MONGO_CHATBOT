"""
Universal MongoDB document -> natural-language converter. Works with ANY schema.

    ObjectId       -> "User: Aisha Khan (ID: 65a...)"   (name filled in when the reference resolves)
    unresolved id  -> "Workspace ID: 65a..."
    datetime       -> "August 12, 2026 06:38 UTC"
    list           -> "Tags: a, b, c"
    nested object  -> recursive: "Settings Notifications Email: Yes"
    sensitive      -> skipped (password, token, secret, ...), also empty values, binary data

The converter is pure: relationship lookups are delegated to a `resolver` object
(see relationships.py). Without a resolver, ObjectIds are simply labelled.
"""
import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal

from bson import ObjectId
from bson.decimal128 import Decimal128
from django.conf import settings

from . import chunking

HEX24 = re.compile(r"^[0-9a-fA-F]{24}$")
ID_SUFFIX = re.compile(r"(?:_ids?|Ids?|_refs?|Refs?)$")
LABEL_ID_SUFFIX = re.compile(r"\s+Ids?$", re.IGNORECASE)
SENSITIVE_PARTS = ("password", "passwd", "secret", "token", "apikey", "api_key",
                   "private_key", "salt", "hash", "otp")
NAME_FIELDS = ("name", "fullName", "full_name", "displayName", "display_name", "title",
               "label", "projectName", "project", "subject", "username", "email")
EXTRA_NAME_FIELDS = ("firstName", "first_name", "lastName", "last_name")
MAX_LIST_ITEMS = 20
MAX_DEPTH = 4
MAX_FIELD_NAMES = 200


# ------------------------------------------------------------------ helpers
def humanize_key(key) -> str:
    """'workspaceId' / 'workspace_id' -> 'Workspace Id'."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(key))
    return re.sub(r"[_\-\s]+", " ", text).strip().title()


def is_sensitive(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return lowered in settings.EXCLUDE_FIELDS or any(p in lowered for p in SENSITIVE_PARTS)


def strip_id_suffix(field_name: str) -> str:
    return ID_SUFFIX.sub("", field_name)

def pick_display_name(doc: dict):
    """Best human-readable name of a document (or None)."""
    for key in NAME_FIELDS:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:100]
    first = doc.get("firstName") or doc.get("first_name")
    last = doc.get("lastName") or doc.get("last_name")
    parts = [p.strip() for p in (first, last) if isinstance(p, str) and p.strip()]
    return " ".join(parts)[:100] if parts else None


def iter_ref_candidates(doc: dict):
    """Yield (field_path, id_string) for every ObjectId (or 24-hex string) in a document."""
    def walk(value, path, depth):
        if isinstance(value, dict):
            if depth >= MAX_DEPTH:
                return
            for key, child in value.items():
                key = str(key)
                if (depth == 0 and key in ("_id", "__v")) or is_sensitive(key):
                    continue
                yield from walk(child, f"{path}.{key}" if path else key, depth + 1)
        elif isinstance(value, (list, tuple, set)):
            for item in list(value)[:MAX_LIST_ITEMS]:
                yield from walk(item, path, depth + 1)
        elif isinstance(value, ObjectId):
            yield path, str(value)
        elif isinstance(value, str) and HEX24.match(value.strip()):
            yield path, value.strip()

    yield from walk(doc, "", 0)


def _format_datetime(value: datetime) -> str:
    text = f"{value:%B} {value.day}, {value.year}"
    if (value.hour, value.minute, value.second) != (0, 0, 0):
        tz = value.tzinfo.tzname(value) if value.tzinfo else "UTC"
        text += f" {value:%H:%M} {tz or 'UTC'}"
    return text


# ---------------------------------------------------------------- converter
@dataclass
class Converted:
    primary_id: str
    lines: list = field(default_factory=list)
    field_names: list = field(default_factory=list)
    relationships: list = field(default_factory=list)   # [{field, ref_id, ref_collection, ref_name}]
    reference_ids: list = field(default_factory=list)   # every ObjectId found
    display_name: str = None


class _Walker:
    def __init__(self, collection, resolver):
        self.collection, self.resolver = collection, resolver
        self.lines, self.fields, self.rels, self.ref_ids = [], [], [], []
        self._rel_seen, self._field_seen = set(), set()

    def walk(self, value, path, label, depth):
        if isinstance(value, dict):
            if depth >= MAX_DEPTH:
                return
            for key, child in value.items():
                key = str(key)
                if (depth == 0 and key in ("_id", "__v")) or is_sensitive(key):
                    continue
                child_path = f"{path}.{key}" if path else key
                if child_path not in self._field_seen and len(self.fields) < MAX_FIELD_NAMES:
                    self._field_seen.add(child_path)
                    self.fields.append(child_path)
                self.walk(child, child_path, f"{label} {humanize_key(key)}".strip(), depth + 1)
        elif isinstance(value, (list, tuple, set)):
            items = list(value)[:MAX_LIST_ITEMS]
            if all(not isinstance(i, (dict, list, tuple, set)) for i in items):
                texts, refs = [], []
                for item in items:
                    result = self._scalar(item, path)
                    if result is None:
                        continue
                    (refs if result[0] == "ref" else texts).append(result[1])
                if texts:
                    self.lines.append(f"{label}: {', '.join(texts)}")
                if refs:
                    self.lines.append(self._ref_line(label, refs))
            else:
                for number, item in enumerate(items, 1):
                    self.walk(item, path, f"{label} {number}", depth + 1)
        else:
            result = self._scalar(value, path)
            if result is None:
                return
            if result[0] == "ref":
                self.lines.append(self._ref_line(label, [result[1]]))
            else:
                self.lines.append(f"{label}: {result[1]}")

    # -- scalar handling: returns ("text", str) | ("ref", (id, name)) | None
    def _scalar(self, value, path):
        if value is None or isinstance(value, (bytes, bytearray)):
            return None
        if isinstance(value, bool):
            return "text", "Yes" if value else "No"
        if isinstance(value, ObjectId):
            return "ref", self._reference(str(value), path, certain=True)
        if isinstance(value, datetime):
            return "text", _format_datetime(value)
        if isinstance(value, date):
            return "text", f"{value:%B} {value.day}, {value.year}"
        if isinstance(value, (int, float, Decimal, Decimal128)):
            return "text", str(value)
        text = " ".join(str(value).split())
        if not text:
            return None
        if HEX24.match(text) and self.resolver is not None:
            ref = self._reference(text, path, certain=False)
            if ref:
                return "ref", ref
        if len(text) > settings.MAX_FIELD_CHARS:
            text = text[:settings.MAX_FIELD_CHARS].rstrip() + "…"
        return "text", text

    def _reference(self, sid, path, certain):
        found = self.resolver.lookup(self.collection, path, sid) if self.resolver is not None else None
        if found is None and not certain:
            return None
        if sid not in self.ref_ids:
            self.ref_ids.append(sid)
        name = None
        if found is not None:
            ref_collection, name = found
            if (path, sid) not in self._rel_seen:
                self._rel_seen.add((path, sid))
                self.rels.append({"field": path, "ref_id": sid,
                                  "ref_collection": ref_collection, "ref_name": name})
        return sid, name

    @staticmethod
    def _ref_line(label, refs):
        base = LABEL_ID_SUFFIX.sub("", label).strip() or "Reference"
        if len(refs) == 1:
            sid, name = refs[0]
            return f"{base}: {name} (ID: {sid})" if name else f"{base} ID: {sid}"
        if any(name for _, name in refs):
            return f"{base}: " + ", ".join(f"{n} (ID: {i})" if n else f"ID {i}" for i, n in refs)
        return f"{base} IDs: " + ", ".join(i for i, _ in refs)


def _summary(collection, rels) -> str:
    domain_type = "project" if collection == "boards" else "task" if collection == "cards" else "record"
    text = f"This is a {domain_type} in the '{collection}' collection"
    pairs = sorted({(r["ref_collection"], humanize_key(strip_id_suffix(r["field"].split(".")[-1])).lower())
                    for r in rels})
    if pairs:
        text += " that references " + ", ".join(f"{coll} (via {name})" for coll, name in pairs)
    return text + "."


def convert_document(collection: str, doc: dict, resolver=None) -> Converted:
    walker = _Walker(collection, resolver)
    walker.walk(doc, "", "", 0)
    primary = str(doc.get("_id", ""))
    lines = list(walker.lines)
    if lines:  # only add the framing lines when there is real content
        lines = [f"Document ID: {primary}", _summary(collection, walker.rels)] + lines
    return Converted(primary_id=primary, lines=lines, field_names=walker.fields,
                     relationships=walker.rels, reference_ids=walker.ref_ids,
                     display_name=pick_display_name(doc))


def to_records(collection: str, doc: dict, resolver=None) -> list:
    """
    Convert one document into embedding-ready records (one per chunk):
        {_id, text, metadata{...}, relationships[...]}
    Short documents give exactly one record. Returns [] if there is nothing to index.
    """
    conv = convert_document(collection, doc, resolver)
    if not conv.primary_id or not conv.lines:
        return []

    kept, total = [], 0           # enforce the whole-document cap on whole lines
    for line in conv.lines:
        if total + len(line) > settings.MAX_DOC_CHARS and kept:
            break
        kept.append(line[:settings.MAX_DOC_CHARS])
        total += len(line) + 1

    bodies = chunking.chunk_lines(kept, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    identity = f"{collection} {conv.primary_id}" + (f" ({conv.display_name})" if conv.display_name else "")
    records = []
    for index, body in enumerate(bodies):
        header = f"Collection: {collection}" + (f"\nRecord: {identity}" if index else "")
        text = f"{header}\n{body}"
        records.append({
            "_id": f"{collection}:{conv.primary_id}:{index}",
            "text": text,
            "metadata": {
                "collection": collection,
                "primary_id": conv.primary_id,
                "chunk_index": index,
                "chunk_count": len(bodies),
                "field_names": conv.field_names,
                "reference_ids": conv.reference_ids,
                "display_name": conv.display_name,
                "content_hash": hashlib.sha1(text.encode("utf-8")).hexdigest(),
            },
            "relationships": conv.relationships,
        })
    return records
