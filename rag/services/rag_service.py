"""
The full RAG pipeline for one question:

  question -> hybrid search (vector + keyword) -> multi-hop expansion -> context -> LLM -> answer

Only a few relevant records are sent to the LLM, never the whole database.
Answers are cached (keyed by question + index version), so repeated questions are free.
"""
import copy
import logging
import re
import time

from django.conf import settings

from . import (cache_service, embedding_service, embedding_store, llm_service, retrieval,
               chat_memory, structured_query, vector_index)
from .exceptions import ChatbotError, InvalidRequestError, LLMError
from .llm_service import NOT_FOUND_ANSWER

logger = logging.getLogger(__name__)

MIN_QUESTION_CHARS, MAX_QUESTION_CHARS = 2, 500
_answer_cache = cache_service.TTLCache(settings.CACHE_MAX_ENTRIES, settings.CACHE_TTL_SECONDS)


# Questions that expect SEVERAL answers. They automatically retrieve more records (LIST_TOP_K).
LIST_PATTERN = re.compile(
    r"\b(?:all|every|each|list|enumerate|how many|show me|give me|give|tell me|what are|who are|"
    r"can you|last|latest|recent|when did|status of|what was|what is|which\s+(?!is\b|are\b)\w{3,}s?|"
    r"completed|complete|done|finished|total)\b", re.IGNORECASE)



def is_list_question(question: str) -> bool:
    return bool(LIST_PATTERN.search(question))


GREETING_WORDS = {
    "hello", "hlo", "hi", "hey", "hiya", "greetings", "namaste", "hola",
    "good morning", "good afternoon", "good evening", "good day",
    "hey there", "hello there", "hi there"
}

HOW_ARE_YOU_WORDS = {
    "how are you", "how are you doing", "how is it going", "hows it going",
    "how do you do", "what is up", "whats up", "wassup"
}

THANKS_WORDS = {
    "thanks", "thank you", "thank you so much", "thanks a lot", "thx",
    "many thanks", "appreciate it"
}

GOODBYE_WORDS = {
    "bye", "goodbye", "see you", "see ya", "take care", "cya", "good night", "have a good day"
}

CAPABILITIES_WORDS = {
    "who are you", "what are you", "what can you do", "help", "help me",
    "what do you do", "what is this", "what is this app"
}


def check_conversational(question: str) -> str | None:
    cleaned = " ".join(re.sub(r"[^a-z0-9\s]", "", question.lower()).split())
    if not cleaned:
        return None
    if cleaned in GREETING_WORDS:
        return "Hello! How can I help you today? You can ask me about employees, workspaces, projects, tasks, departments, or other data in the database."
    if cleaned in HOW_ARE_YOU_WORDS:
        return "I'm doing well, thank you! How can I assist you with your projects, employees, or database records today?"
    if cleaned in THANKS_WORDS:
        return "You're welcome! Let me know if you need any other information from the database."
    if cleaned in GOODBYE_WORDS:
        return "Goodbye! Have a great day ahead. Feel free to come back whenever you need database insights."
    if cleaned in CAPABILITIES_WORDS:
        return ("I am your MongoDB database assistant. You can ask me to list employees, "
                "check assigned tasks, view departments, inspect projects, or count records.")
    return None


def strip_greeting(text: str) -> str:
    cleaned = re.sub(
        r"^(?:hello|hlo|hi|hey|good\s+(?:morning|afternoon|evening)|greetings|namaste)\b[\s,!:\.\-]*",
        "", text, flags=re.IGNORECASE
    ).strip()
    return cleaned or text


def validate_question(question) -> str:
    if not isinstance(question, str):
        raise InvalidRequestError("The question must be text.")
    question = " ".join(question.split())
    if len(question) < MIN_QUESTION_CHARS:
        raise InvalidRequestError("Please type a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise InvalidRequestError(f"Please keep your question under {MAX_QUESTION_CHARS} characters.")
    return question


def _group(hits: list) -> list:
    """Merge chunks of the same document; best-scoring document first."""
    groups = {}
    for hit in hits:
        groups.setdefault(hit.doc_key, []).append(hit)
    ordered = sorted(groups.values(), key=lambda g: max(h.score for h in g), reverse=True)
    return [sorted(g, key=lambda h: h.chunk_index) for g in ordered]


def build_context(main_groups: list, related_groups: list, budget: int = None):
    """Build the prompt context under a character budget. Returns (context, included_main, included_related)."""
    budget, used = budget or settings.MAX_CONTEXT_CHARS, 0
    blocks, main_done, related_done = [], [], []
    label_of = {}

    def add(label, group, note=""):
        nonlocal used
        text = "\n...\n".join(h.text for h in group)
        block = f"[{label}]{note}\n{text}"
        if blocks and used + len(block) > budget:
            return False
        blocks.append(block)
        used += len(block) + 2
        return True

    for number, group in enumerate(main_groups, 1):
        if add(str(number), group):
            main_done.append(group)
            label_of[group[0].doc_key] = number
    for number, group in enumerate(related_groups, 1):
        via = group[0].via or {}
        origin = label_of.get(via.get("from_doc"))
        note = f" (linked from record {origin} via '{via.get('field')}')" if origin else ""
        if add(f"R{number}", group, note):
            related_done.append(group)
    return "\n\n".join(blocks), main_done, related_done


def _source(group, related=False) -> dict:
    head = group[0]
    source = {"collection": head.collection, "document_id": head.primary_id, "name": head.display_name,
              "score": round(max(h.score for h in group), 4), "related": related}
    if related and head.via:
        source["via"] = {"from": list(head.via["from_doc"]), "field": head.via["field"],
                         "direction": head.via["direction"]}
    return source


SOURCE_CAP = 25


def _plain_answer(exact) -> str:
    """The exact result without the LLM, used when the language model is unavailable."""
    head = (f"The language model is unavailable right now, so this is the plain result read directly from the "
            f"database: {exact.total} matching record(s) in '{exact.collection}' (conditions: {exact.conditions}).")
    if exact.operation == "count":
        return head
    items = "\n".join(f"- {r['name'] or r['text'][:160]}" for r in exact.records)
    more = "" if exact.complete else f"\n\nShowing the first {len(exact.records)} of {exact.total}."
    return f"{head}\n\n{items}{more}"


def _retrieval_query(question: str, history: list) -> str:
    """Add a small amount of recent context so embeddings can resolve follow-up references."""
    if not history:
        return question
    recent = history[-2:]
    context = " ".join(f"{turn['question']} {turn['answer']}" for turn in recent)
    return f"Recent conversation: {context}\nCurrent question: {question}"[-2000:]


def _remember(session_id, question, result):
    if session_id:
        chat_memory.add(session_id, question, result["answer"])


def _exact_answer(question: str, exact, history=None) -> dict:
    """Answer from an exact database query (complete list / exact count)."""
    timings = dict(exact.timings)
    mark = time.perf_counter()
    degraded = False
    try:
        answer = llm_service.generate(question, exact.context, max_tokens=settings.LIST_MAX_TOKENS,
                                      history=history)
    except LLMError as exc:          # the data is already in hand: still give the user the complete list
        logger.warning("LLM unavailable for an exact result (%s); returning the plain result.", exc.message)
        answer, degraded = _plain_answer(exact), True
    timings["llm"] = round((time.perf_counter() - mark) * 1000)
    not_found = not degraded and answer.lower().startswith(NOT_FOUND_ANSWER.lower().rstrip("."))
    sources = [] if not_found else [
        {"collection": exact.collection, "document_id": r["document_id"], "name": r["name"],
         "score": None, "related": False} for r in exact.records[:SOURCE_CAP]]
    result = {"question": question, "answer": answer, "sources": sources, "cached": False, "mode": "structured",
            "collection": exact.collection, "total_matching": exact.total, "records_used": len(exact.records),
            "conditions": exact.conditions, "list_mode": True, "degraded": degraded, "timings_ms": timings}
    return result


def ask(question, top_k=None, collections=None, hops=None, session_id=None) -> dict:
    question = validate_question(question)
    if session_id is not None and (not isinstance(session_id, str) or not re.fullmatch(r"[A-Fa-f0-9-]{1,64}", session_id)):
        raise InvalidRequestError("'session_id' must be a valid session identifier.")
    conversational = check_conversational(question)
    if conversational is not None:
        result = {
            "question": question,
            "answer": conversational,
            "sources": [],
            "cached": False,
            "mode": "conversational",
            "timings_ms": {"total": 0}
        }
        _remember(session_id, question, result)
        return result

    history = chat_memory.get(session_id) if session_id else []
    query_text = strip_greeting(question)
    top_k = top_k or settings.TOP_K
    list_mode = bool(settings.LIST_TOP_K) and is_list_question(query_text)
    if list_mode:
        top_k = max(top_k, settings.LIST_TOP_K)     # never fewer than the user's mode, more for list questions
    if settings.STRUCTURED_QUERIES and not collections and (list_mode or structured_query.mentions_collection(query_text)):
        exact = structured_query.try_structured(query_text)    # exact totals straight from MongoDB
        if exact is not None:
            result = _exact_answer(question, exact, history)
            _remember(session_id, question, result)
            return result
    history_key = tuple((turn["question"], turn["answer"]) for turn in history)
    cache_key = (session_id, question.lower(), top_k, tuple(sorted(collections or [])), hops,
                 history_key, vector_index.current_version())
    cached = _answer_cache.get(cache_key)
    if cached is not None:
        result = copy.deepcopy(cached)
        result["cached"] = True
        _remember(session_id, question, result)
        return result

    timings, started = {}, time.perf_counter()

    def mark(stage):
        nonlocal started
        now = time.perf_counter()
        timings[stage] = round((now - started) * 1000)
        started = now

    hits = retrieval.search(_retrieval_query(question, history), top_k=top_k, collections=collections)
    mark("search")
    if not hits:
        fallback_msg = (
            f"{NOT_FOUND_ANSWER} You can ask me about team members, active projects, tasks, "
            "departments, logged hours, sales leads, or company announcements!"
        )
        result = {"question": question, "answer": fallback_msg, "sources": [], "cached": False,
                  "timings_ms": timings}
        _remember(session_id, question, result)
        return result

    related = retrieval.expand(hits, hops=hops)
    mark("expand")
    budget = max(settings.MAX_CONTEXT_CHARS, settings.LIST_CONTEXT_CHARS) if list_mode else settings.MAX_CONTEXT_CHARS
    context, main_done, related_done = build_context(_group(hits), _group(related), budget)
    answer = llm_service.generate(question, context, max_tokens=settings.LIST_MAX_TOKENS if list_mode else None,
                                  history=history)
    mark("llm")

    is_pure_refusal = (
        len(answer.strip()) < 90 and
        answer.lower().startswith(NOT_FOUND_ANSWER.lower().rstrip("."))
    )
    sources = [] if is_pure_refusal else (
        [_source(g) for g in main_done] + [_source(g, related=True) for g in related_done])
    result = {"question": question, "answer": answer, "sources": sources, "cached": False,
              "records_used": len(main_done), "list_mode": list_mode, "timings_ms": timings}
    _answer_cache.set(cache_key, copy.deepcopy(result))
    _remember(session_id, question, result)
    logger.info("ask: %d hits, %d related, timings=%s", len(hits), len(related), timings)
    return result


def search(query, top_k=None, collections=None, hybrid=None, expand=False, hops=None) -> dict:
    """Semantic (and keyword) search without the LLM."""
    query = validate_question(query)
    started = time.perf_counter()
    hits = retrieval.search(query, top_k=top_k, collections=collections, hybrid=hybrid)
    result = {"query": query, "results": [h.to_dict() for h in hits]}
    if expand:
        result["related"] = [h.to_dict() for h in retrieval.expand(hits, hops=hops)]
    result["timings_ms"] = {"total": round((time.perf_counter() - started) * 1000)}
    return result


def health() -> dict:
    from . import mongo_service

    info = {"mongo": mongo_service.ping(), "embedding_model": settings.EMBEDDING_MODEL_ID,
            "llm_model": settings.LLM_MODEL, "llm": llm_service.status(), "hybrid_search": settings.HYBRID_SEARCH,
            "cache": _answer_cache.stats(), "index": {"ready": False}, "embeddings": 0}
    if info["mongo"]:
        try:
            info["embeddings"] = embedding_store.count()
            index = vector_index.get_index()
            info["index"] = {"ready": True, "vectors": len(index.ids), "built_at": index.built_at}
        except ChatbotError as exc:
            info["index"] = {"ready": False, "message": exc.message}
    info["status"] = "ok" if info["mongo"] and info["index"]["ready"] else "degraded"
    return info


def warm_up():
    try:
        structured_query.build_catalog()      # first list question is then fast
    except Exception as exc:
        logger.info("Catalog warm-up skipped: %s", exc)
    vector_index.get_index()
    if settings.EMBEDDING_PROVIDER == "local":
        embedding_service.get_model()
    try:
        llm_service.warm_up()            # loads a local Ollama model into memory (no-op for hosted APIs)
    except ChatbotError as exc:
        logger.info("LLM warm-up skipped: %s", exc.message)
