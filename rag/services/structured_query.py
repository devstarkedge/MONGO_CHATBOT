"""
Exact answers for LIST and COUNT questions.

Similarity search returns the nearest few records, so it can never reliably list or count "all" of something.
For such questions this module queries MongoDB directly:

  1. a CATALOG of collections and fields is loaded (built once at ingest time, saved to vector_store/catalog.json)
  2. the LLM turns the question into a small JSON PLAN: {collection, operation, filters, show}   (plans are cached)
  3. the plan is VALIDATED strictly (known collection and fields, whitelisted operators). Any doubt rejects the
     whole plan: a condition is never silently dropped, because that would make the "exact" count wrong
  4. count_documents() gives the exact total, find() gives the records (up to STRUCT_MAX_RECORDS)
  5. the LLM phrases the answer. If it is unavailable, the caller still shows the plain result

The LLM never writes MongoDB syntax. Filters on reference fields accept NAMES ("assigned to Rohan Mehta"):
the name is resolved to ids in the referenced collection first.

Speed notes: the slow parts (probing which collection each id field points to, sampling field values, reading
the field list) happen once, at ingest, and are saved. A question then costs one planner call (cached for
repeated questions), one count, one find and the answer call. Time is reported as plan / db / llm.
"""
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from bson import ObjectId
from django.conf import settings

from . import cache_service, document_converter, embedding_store, llm_service, mongo_service, relationships
from .document_converter import EXTRA_NAME_FIELDS, HEX24, NAME_FIELDS, humanize_key, is_sensitive, strip_id_suffix
from .exceptions import ChatbotError
from .relationships import _norm, _singular

logger = logging.getLogger(__name__)

CATALOG_FILE = "catalog.json"
OPS = {"eq", "ne", "in", "contains", "gt", "gte", "lt", "lte"}
MAX_FILTERS = 5
SAMPLE_DOCS = 25
MAX_FIELDS = 40
MAX_SHOW = 8
MAX_RECORD_CHARS = 400

_memory = cache_service.TTLCache(maxsize=4, ttl=3600)      # "catalog" -> (file_stamp, data); "names"
_plans = cache_service.TTLCache(maxsize=500, ttl=3600)                # normalised question -> {"plan": ...}
_name_ids = cache_service.TTLCache(maxsize=256, ttl=300)              # (collection, name) -> ids
_resolver = None
_lock = threading.Lock()
_refreshing = False


@dataclass
class ExactResult:
    collection: str
    total: int
    records: list                      # [{document_id, name, text}]
    conditions: str
    complete: bool
    source: str                        # "planner" | "name-match"
    operation: str = "list"
    context: str = ""
    notes: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def reset_for_tests():
    global _resolver
    _memory.clear()
    _plans.clear()
    _name_ids.clear()
    _resolver = None


# ----------------------------------------------------------- collections
def known_collections() -> list:
    names = _memory.get("names")
    if names is None:
        names = mongo_service.list_collections()
        _memory.set("names", names)
    return names


def _exec_resolver(catalog: dict):
    """Resolver for question time: targets come from the catalog, so no probing queries are needed."""
    global _resolver
    names = known_collections()
    with _lock:
        if _resolver is None or _resolver.collections != names:
            _resolver = relationships.RelationshipResolver(mongo_service.get_db(), names)
        for cname, info in catalog.items():
            for path, meta in info["fields"].items():
                if meta.get("ref"):
                    _resolver.seed_target(cname, path, meta["ref"])
        _resolver.reset_names()
        return _resolver


# --------------------------------------------------------------- catalog
def _scalars(doc, path="", depth=0):
    """Yield (field_path, scalar_value) for the leaves of a document (sensitive keys skipped)."""
    if isinstance(doc, dict):
        if depth >= 3:
            return
        for key, value in doc.items():
            key = str(key)
            if (depth == 0 and key in ("_id", "__v")) or is_sensitive(key):
                continue
            yield from _scalars(value, f"{path}.{key}" if path else key, depth + 1)
    elif isinstance(doc, (list, tuple)):
        for item in doc[:5]:
            yield from _scalars(item, path, depth + 1)
    elif doc is not None and not isinstance(doc, (bytes, bytearray)):
        yield path, doc


def _build(resolver=None) -> dict:
    """Sample every collection. This is the slow part (it probes id fields), so it runs at ingest, not per question."""
    db = mongo_service.get_db()
    names = mongo_service.list_collections()
    _memory.set("names", names)
    resolver = resolver or relationships.RelationshipResolver(db, names)
    catalog = {}
    for name in names:
        try:
            total = db[name].estimated_document_count()
            samples = list(db[name].find({}).limit(SAMPLE_DOCS))
        except Exception as exc:
            logger.warning("Catalog: skipping %s (%s)", name, type(exc).__name__)
            continue
        if not samples:
            continue
        resolver.prefetch(name, samples)
        fields = {}
        for doc in samples:
            for path, value in _scalars(doc):
                is_id = isinstance(value, ObjectId) or (isinstance(value, str) and HEX24.match(value))
                target = resolver.target_of(name, path) if is_id else None
                if is_id and not target:
                    continue                      # an id we cannot translate to names: not filterable
                if path not in fields and len(fields) >= MAX_FIELDS:
                    continue
                meta = fields.setdefault(path, {"values": [], "types": set(), "ref": None})
                if target:
                    meta["ref"] = target
                    meta["types"].add("ref")
                    continue
                kind = "bool" if isinstance(value, bool) else "num" if isinstance(value, (int, float)) else "str"
                meta["types"].add(kind)
                text = " ".join(str(value).split())[:30]
                if kind == "str" and text and text not in meta["values"] and len(meta["values"]) < 6:
                    meta["values"].append(text)
        valid = set(fields)
        try:   # field names seen on ALL ingested documents, not just the sample
            valid |= set(embedding_store.collection().distinct("metadata.field_names", {"metadata.collection": name}))
        except Exception:
            pass
        catalog[name] = {"count": total, "fields": fields, "valid": sorted(valid)}
    return catalog


def _catalog_path() -> Path:
    return Path(settings.VECTOR_STORE_DIR) / CATALOG_FILE


def _stamp(path: Path):
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _read(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))["catalog"]
        for info in data.values():
            for meta in info["fields"].values():
                meta["types"] = set(meta["types"])
        return data
    except Exception as exc:
        logger.warning("Saved catalog unreadable (%s); rebuilding.", type(exc).__name__)
        return None


def _write(path: Path, catalog: dict):
    plain = {n: {**info, "fields": {p: {**m, "types": sorted(m["types"])} for p, m in info["fields"].items()}}
             for n, info in catalog.items()}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps({"catalog": plain}), encoding="utf-8")
        tmp.replace(path)
        return _stamp(path)
    except OSError as exc:
        logger.warning("Could not save the catalog (%s); keeping it in memory only.", type(exc).__name__)
        return None


def _refresh_in_background():
    global _refreshing
    if _refreshing:
        return
    _refreshing = True

    def work():
        global _refreshing
        try:
            build_catalog(force=True)
        except Exception:
            logger.exception("Background catalog refresh failed")
        finally:
            _refreshing = False

    threading.Thread(target=work, daemon=True, name="catalog-refresh").start()


def build_catalog(force: bool = False, resolver=None) -> dict:
    """
    Return the catalog: memory -> saved file -> build (slow, once). `ingest` calls this with force=True.
    A file older than CATALOG_MAX_AGE_HOURS is still used right away and refreshed in the background.
    """
    path = _catalog_path()
    stamp = _stamp(path)
    mem = _memory.get("catalog")
    if not force and mem is not None and mem[0] == stamp:
        return mem[1]
    if not force and stamp is not None:
        data = _read(path)
        if data is not None:
            _memory.set("catalog", (stamp, data))
            _plans.clear()
            if time.time() - path.stat().st_mtime > settings.CATALOG_MAX_AGE_HOURS * 3600:
                _refresh_in_background()
            return data
    started = time.perf_counter()
    catalog = _build(resolver)
    new_stamp = _write(path, catalog)
    _memory.set("catalog", (new_stamp, catalog))
    _plans.clear()
    logger.info("Catalog built for %d collections in %.1fs", len(catalog), time.perf_counter() - started)
    return catalog


def catalog_text(catalog: dict, limit: int = None) -> str:
    """Compact description of collections and fields for the planner. Sample values are dropped first when
    the text would exceed `limit` (small local models have small contexts)."""
    limit = limit or settings.CATALOG_CHARS

    def render(with_values: bool) -> str:
        lines = []
        for name, info in catalog.items():
            parts = []
            for path, meta in info["fields"].items():
                if meta["ref"]:
                    parts.append(f"{path} (reference to {meta['ref']}; filter it by that record's NAME)")
                elif meta["values"] and with_values:
                    parts.append(f"{path} [{', '.join(meta['values'])}]")
                else:
                    parts.append(path)
            lines.append(f"- {name} ({info['count']} records): " + "; ".join(parts))
        return "\n".join(lines)

    text = render(True)
    return (text if len(text) <= limit else render(False))[:limit]


# ------------------------------------------------------------------ plans
def _to_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def validate_plan(plan, catalog: dict):
    """Return a clean plan or None. Strict on purpose: reject rather than guess."""
    if not isinstance(plan, dict):
        return None
    collection = plan.get("collection")
    if not isinstance(collection, str) or collection not in catalog:
        return None
    operation = plan.get("operation", "list")
    operation = operation if operation in ("list", "count") else "list"
    info = catalog[collection]
    valid = set(info.get("valid") or info["fields"])
    clean = []
    for item in (plan.get("filters") or [])[:MAX_FILTERS + 1]:
        if len(clean) >= MAX_FILTERS or not isinstance(item, dict):
            return None
        fname, op, value = item.get("field"), item.get("op", "eq"), item.get("value")
        if fname not in valid or op not in OPS or value is None or isinstance(value, dict):
            return None
        if op == "in" and not (isinstance(value, list) and value and all(isinstance(v, (str, int, float)) for v in value)):
            return None
        if op != "in" and isinstance(value, list):
            return None
        if op in ("gt", "gte", "lt", "lte") and _to_number(value) is None:
            return None
        clean.append({"field": fname, "op": op, "value": value})
    show = [f for f in (plan.get("show") or []) if isinstance(f, str) and f in valid][:MAX_SHOW]   # display only
    return {"collection": collection, "operation": operation, "filters": clean, "show": show}


def _regex_exact(text):
    return re.compile(f"^{re.escape(str(text).strip())}$", re.IGNORECASE)


def _ids_by_name(target: str, value: str) -> list:
    """Ids of records in `target` whose name matches every word of `value` (handles first/last name fields)."""
    key = (target, " ".join(value.lower().split()))
    cached = _name_ids.get(key)
    if cached is not None:
        return cached
    tokens = re.findall(r"\w+", value)[:4]
    if not tokens:
        return []
    fields = NAME_FIELDS + EXTRA_NAME_FIELDS
    cond = {"$and": [{"$or": [{f: re.compile(re.escape(t), re.IGNORECASE)} for f in fields]} for t in tokens]}
    ids = []
    for doc in mongo_service.get_db()[target].find(cond, {"_id": 1}).limit(50):
        ids.extend([doc["_id"], str(doc["_id"])])
    _name_ids.set(key, ids)
    return ids


def _condition(info: dict, fname: str, op: str, value, notes: list):
    """-> (mongo_condition, readable_text)"""
    meta = info["fields"].get(fname, {})
    target = meta.get("ref")
    names = value if isinstance(value, list) else [value]

    if target and any(isinstance(v, str) and not HEX24.match(v) for v in names) and op in ("eq", "ne", "contains", "in"):
        ids = []
        for v in names:
            found = _ids_by_name(target, str(v))
            if not found:
                notes.append(f"No record in '{target}' matches the name '{v}'.")
            ids.extend(found)
        text = f"{fname} = {', '.join(map(str, names))} (resolved in '{target}')"
        if op == "ne":
            return {fname: {"$nin": ids}}, text.replace("=", "≠")
        return {fname: {"$in": ids}}, text            # empty ids -> matches nothing

    numeric = meta.get("types") == {"num"}
    if op in ("gt", "gte", "lt", "lte"):
        return {fname: {f"${op}": _to_number(value)}}, f"{fname} {dict(gt='>', gte='>=', lt='<', lte='<=')[op]} {value}"
    if op == "in":
        conds = [_condition(info, fname, "eq", v, notes)[0] for v in value]
        return {"$or": conds}, f"{fname} in {value}"

    if isinstance(value, bool):
        return ({fname: value} if op == "eq" else {fname: {"$ne": value}}), f"{fname} {'=' if op == 'eq' else '≠'} {value}"
    if numeric and _to_number(value) is not None:
        number = _to_number(value)
        return ({fname: number} if op == "eq" else {fname: {"$ne": number}}), f"{fname} {'=' if op == 'eq' else '≠'} {number}"
    if op == "contains":
        return {fname: re.compile(re.escape(str(value).strip()), re.IGNORECASE)}, f"{fname} contains '{value}'"
    pattern = _regex_exact(value)
    if op == "eq":
        return {fname: pattern}, f"{fname} = {value}"
    return {fname: {"$not": pattern}}, f"{fname} ≠ {value}"


def match_collection(question: str, names: list):
    """No-LLM fallback: the one collection whose name appears in the question (singular or plural)."""
    words = {_singular(w) for w in re.findall(r"[a-z0-9]+", question.lower())}
    matches = [n for n in names if _singular(_norm(n)) in words]
    return matches[0] if len(matches) == 1 else None


def mentions_collection(question: str) -> bool:
    """True when the question uses a collection name in its plural form ("departments", "tasks")."""
    try:
        names = known_collections()
    except ChatbotError:
        return False
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    return any(_norm(n) in words and _norm(n).endswith("s") for n in names)


# -------------------------------------------------------------- execution
def _count(collection, flt) -> int:
    try:
        return collection.count_documents(flt, maxTimeMS=settings.STRUCT_TIMEOUT_MS)
    except (TypeError, NotImplementedError):      # drivers/mocks without maxTimeMS support
        return collection.count_documents(flt)


def _labels_for(show: list) -> list:
    labels = []
    for path in show:
        parts = path.split(".")
        parts[-1] = strip_id_suffix(parts[-1]) or parts[-1]
        labels.append(" ".join(humanize_key(p) for p in parts))
    return labels


def _select_lines(lines: list, labels: list) -> list:
    kept = [l for l in lines if any(l.startswith(lab + ":") or l.startswith(lab + " ID") for lab in labels)]
    return kept or lines


def execute(plan: dict, catalog: dict, source: str):
    started = time.perf_counter()
    name, info = plan["collection"], catalog[plan["collection"]]
    notes, conds, texts = [], [], []
    for f in plan["filters"]:
        cond, text = _condition(info, f["field"], f["op"], f["value"], notes)
        conds.append(cond)
        texts.append(text)
    flt = {"$and": conds} if conds else {}
    collection = mongo_service.get_db()[name]
    total = _count(collection, flt)
    if total == 0 and conds:
        return None                       # probably a mis-read question: let similarity search try instead
    wanted = 10 if plan["operation"] == "count" else settings.STRUCT_MAX_RECORDS
    cursor = collection.find(flt).limit(wanted)
    try:
        cursor = cursor.max_time_ms(settings.STRUCT_TIMEOUT_MS)
    except Exception:
        pass
    docs = list(cursor)
    resolver = _exec_resolver(catalog)
    resolver.prefetch(name, docs, infer=False)      # names only: targets are already known
    labels = _labels_for(plan.get("show") or [])
    records = []
    for doc in docs:
        conv = document_converter.convert_document(name, doc, resolver)
        body = conv.lines[2:] if len(conv.lines) > 2 else conv.lines       # drop "Document ID" + summary lines
        if labels:
            body = _select_lines(body, labels)
        records.append({"document_id": conv.primary_id, "name": conv.display_name,
                        "text": "; ".join(body)[:MAX_RECORD_CHARS]})
    return ExactResult(collection=name, total=total, records=records, conditions="; ".join(texts) or "none",
                       complete=total <= len(records), source=source, operation=plan["operation"], notes=notes,
                       timings={"db": round((time.perf_counter() - started) * 1000)})


def _context(result: ExactResult) -> str:
    lines = [f"EXACT QUERY RESULT from the '{result.collection}' collection (read live from the database, not a similarity search).",
             f"Conditions applied: {result.conditions}"]
    if result.source == "name-match":
        lines.append("Conditions were NOT applied by the system. These are ALL records of the collection: "
                     "read them and apply the question's conditions yourself.")
        lines.append(f"Exact number of records in the collection: {result.total}")
    else:
        lines.append(f"Exact number of matching records in the database: {result.total}")
    if result.complete:
        lines.append(f"All {len(result.records)} records are listed below: the list is complete.")
    elif result.operation == "count":
        lines.append(f"Only {len(result.records)} examples are listed below. The total above is exact.")
    else:
        lines.append(f"Only the first {len(result.records)} of {result.total} are listed below: the list is NOT complete. "
                     "The total above is exact.")
    lines.extend(f"Note: {n}" for n in result.notes)
    lines.append("")
    lines.extend(f"[{i}] {r['text']}" for i, r in enumerate(result.records, 1))
    return "\n".join(lines)


def try_structured(question: str):
    """-> ExactResult, or None when similarity search should handle the question."""
    if not settings.STRUCTURED_QUERIES:
        return None
    try:
        began = time.perf_counter()
        catalog = build_catalog()
        if not catalog:
            return None
        key = " ".join(question.lower().split())
        cached = _plans.get(key)
        plan, source, planner_ok = None, "planner", True
        if cached is not None:
            plan = cached["plan"]
        elif settings.STRUCT_PLANNER == "off":
            plan = None                     # no planning call: rely on the no-LLM collection-name fallback below
        else:
            try:
                plan = validate_plan(llm_service.plan_query(question, catalog_text(catalog)), catalog)
            except ChatbotError as exc:
                planner_ok = False
                logger.warning("Query planner unavailable: %s", exc.message)
            if planner_ok:
                _plans.set(key, {"plan": plan})
        plan_ms = round((time.perf_counter() - began) * 1000)
        if plan is None:
            name = match_collection(question, list(catalog))
            if name and catalog[name]["count"] <= settings.STRUCT_SCAN_MAX:
                plan, source = {"collection": name, "operation": "list", "filters": [], "show": []}, "name-match"
        if plan is None:
            return None
        result = execute(plan, catalog, source)
        if result is None:
            return None
        result.context = _context(result)
        result.timings = {"plan": plan_ms, **result.timings}
        logger.info("Exact query on %s via %s: %d matching (%s) timings=%s", result.collection, source,
                    result.total, result.conditions, result.timings)
        return result
    except ChatbotError as exc:
        logger.warning("Exact query skipped: %s", exc.message)
        return None
    except Exception:
        logger.exception("Exact query failed; falling back to similarity search")
        return None
