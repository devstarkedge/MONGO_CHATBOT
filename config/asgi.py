"""ASGI entry point. Run with:  uvicorn config.asgi:application  (async views run natively)."""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
application = get_asgi_application()
