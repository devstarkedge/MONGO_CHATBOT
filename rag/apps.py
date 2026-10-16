"""Warms up the vector index + embedding model in the background when runserver starts."""
import logging
import os
import sys
import threading

from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger(__name__)


def _warm_up():
    from .services import rag_service

    try:
        rag_service.warm_up()
        logger.info("Warm-up finished.")
    except Exception as exc:
        logger.info("Warm-up skipped: %s", exc)


class RagConfig(AppConfig):
    name = "rag"

    def ready(self):
        if not settings.WARMUP_ON_START or "runserver" not in sys.argv:
            return
        if "--noreload" not in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return
        threading.Thread(target=_warm_up, daemon=True).start()
