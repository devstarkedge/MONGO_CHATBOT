"""Logs one line per request (method, path, status, duration). Works for sync and async views."""
import logging
from asyncio import iscoroutinefunction
from time import perf_counter

from django.utils.decorators import sync_and_async_middleware

logger = logging.getLogger("rag.http")


def _log(request, response, started):
    if not request.path.startswith("/static/"):
        logger.info("%s %s -> %s in %.0f ms", request.method, request.path,
                    response.status_code, (perf_counter() - started) * 1000)


@sync_and_async_middleware
def request_log_middleware(get_response):
    if iscoroutinefunction(get_response):
        async def middleware(request):
            started = perf_counter()
            response = await get_response(request)
            _log(request, response, started)
            return response
    else:
        def middleware(request):
            started = perf_counter()
            response = get_response(request)
            _log(request, response, started)
            return response
    return middleware
