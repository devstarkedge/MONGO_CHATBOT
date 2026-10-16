"""
HTTP layer. Thin on purpose: parse + validate input, call a service, return JSON.

  GET  /                 chat page
  POST /api/ask/         full RAG pipeline (search + LLM answer)
  POST /api/search/      semantic/hybrid search only
  POST /api/ingest/      start ingestion (background job)   GET /api/ingest/  job status
  GET  /api/health/      monitoring

ask/search are async views: the blocking work runs in a worker thread, so one slow
LLM call never blocks other requests under an ASGI server (uvicorn).
"""
import asyncio
import hmac
import json
import logging

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .services import ingestion_service, rag_service, chat_memory
from .services.exceptions import AuthError, ChatbotError, InvalidRequestError

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- helpers
def _payload(request) -> dict:
    try:
        data = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        raise InvalidRequestError("The request body must be valid JSON.")
    if not isinstance(data, dict):
        raise InvalidRequestError("The request body must be a JSON object.")
    return data


def _int(data, key, lo, hi):
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise InvalidRequestError(f"'{key}' must be an integer between {lo} and {hi}.")
    return value


def _collections(data):
    value = data.get("collections")
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise InvalidRequestError("'collections' must be a list of collection names.")
    return value or None


def _error(exc: Exception) -> JsonResponse:
    if isinstance(exc, ChatbotError):
        return JsonResponse({"error": exc.message}, status=exc.status_code)
    logger.exception("Unexpected error")
    return JsonResponse({"error": "Something went wrong on the server. Please try again."}, status=500)


async def _run(func, *args, **kwargs) -> JsonResponse:
    try:
        return JsonResponse(await asyncio.to_thread(func, *args, **kwargs))
    except Exception as exc:
        return _error(exc)


def _check_ingest_auth(request):
    token = settings.INGEST_TOKEN
    if not token:
        if settings.DEBUG:
            return
        raise AuthError("Ingestion over HTTP is disabled. Set INGEST_TOKEN in .env.")
    supplied = request.headers.get("Authorization", "")
    if not hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
        raise AuthError()


# ------------------------------------------------------------------- views
@require_GET
def index(request):
    return render(request, "rag/index.html")


@csrf_exempt
@require_POST
async def ask(request):
    try:
        data = _payload(request)
        args = dict(top_k=_int(data, "top_k", 1, 50), collections=_collections(data), hops=_int(data, "hops", 0, 3),
                    session_id=data.get("session_id"))
    except Exception as exc:
        return _error(exc)
    return await _run(rag_service.ask, data.get("question", ""), **args)


@csrf_exempt
@require_POST
async def search(request):
    try:
        data = _payload(request)
        hybrid = data.get("hybrid")
        if hybrid is not None and not isinstance(hybrid, bool):
            raise InvalidRequestError("'hybrid' must be true or false.")
        args = dict(top_k=_int(data, "top_k", 1, 50), collections=_collections(data), hybrid=hybrid,
                    expand=bool(data.get("expand", False)), hops=_int(data, "hops", 0, 3))
    except Exception as exc:
        return _error(exc)
    return await _run(rag_service.search, data.get("query", ""), **args)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def ingest(request):
    try:
        _check_ingest_auth(request)
        if request.method == "GET":
            return JsonResponse(ingestion_service.job_status())
        data = _payload(request)
        job = ingestion_service.start_job(
            collections=_collections(data), full=bool(data.get("full", False)),
            max_docs=_int(data, "max_docs", 0, 10_000_000) or 0)
        return JsonResponse(job, status=202)
    except Exception as exc:
        return _error(exc)


@require_GET
def health(request):
    try:
        return JsonResponse(rag_service.health())
    except Exception as exc:
        return _error(exc)


@require_GET
def chats_list(request):
    try:
        return JsonResponse({"sessions": chat_memory.list_sessions()})
    except Exception as exc:
        return _error(exc)


@require_GET
def chat_detail(request, session_id):
    try:
        turns = chat_memory.get(session_id)
        return JsonResponse({"session_id": session_id, "turns": turns})
    except Exception as exc:
        return _error(exc)


@csrf_exempt
@require_http_methods(["DELETE", "POST"])
def chat_delete(request, session_id):
    try:
        chat_memory.clear(session_id)
        return JsonResponse({"success": True})
    except Exception as exc:
        return _error(exc)

    

# -------------------------------------------------------------------------------------------





# rag/views.py
import json
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect

from .models import ChatSession  # adjust to your actual model


@require_POST
@csrf_protect
def rename_chat(request, session_id):
    """Rename a chat session. Called from the sidebar's ✎ button."""
    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)

    title = (body.get("title") or "").strip()
    if not title:
        return JsonResponse({"ok": False, "error": "Title cannot be empty"}, status=400)

    title = title[:120]  # hard cap, matches the model field

    updated = ChatSession.objects.filter(session_id=session_id).update(title=title)
    if not updated:
        return JsonResponse({"ok": False, "error": "Chat not found"}, status=404)

    return JsonResponse({"ok": True, "title": title})
