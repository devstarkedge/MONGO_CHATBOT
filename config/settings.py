"""
All configuration lives here and comes from environment variables (.env).
Nothing sensitive is hardcoded or sent to the browser.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _list(name, default=""):
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


# LLM backend: "openai" (any OpenAI-compatible API: Groq, OpenAI, ...), "gemini" (Google Gemini) or "ollama" (local model).
# The local profile uses smaller prompts and longer timeouts by default: a 4B model on a laptop is slower and has
# a smaller effective context than a hosted model.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()   # openai | ollama | gemini
_LOCAL = LLM_PROVIDER == "ollama"
_GEMINI = LLM_PROVIDER == "gemini"      # Google Gemini through its OpenAI-compatible endpoint

# ------------------------------------------------------------------ Django
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = _bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = _list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,.onrender.com")
_render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
if _render_host and _render_host not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(_render_host)

CSRF_TRUSTED_ORIGINS = _list(
    "DJANGO_CSRF_TRUSTED_ORIGINS",
    "http://localhost,http://127.0.0.1,https://*.onrender.com",
)
if _render_host:
    CSRF_TRUSTED_ORIGINS.append(f"https://{_render_host}")
INSTALLED_APPS = ["django.contrib.staticfiles", "rag.apps.RagConfig"]
# No cookies/sessions/auth are used: the JSON API is stateless, so no CSRF middleware is needed.
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "rag.middleware.request_log_middleware",
]
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [BASE_DIR / "templates"],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
}]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"std": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "std"}},
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
    "loggers": {"httpx": {"level": "WARNING"}, "httpcore": {"level": "WARNING"},
                "urllib3": {"level": "WARNING"}, "huggingface_hub": {"level": "WARNING"}},
}

# ----------------------------------------------------------------- MongoDB
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "")
# Embeddings are stored in their own collection (optionally in a different database).
EMBEDDINGS_COLLECTION = os.getenv("EMBEDDINGS_COLLECTION", "rag_embeddings")
EMBEDDINGS_DATABASE = os.getenv("EMBEDDINGS_DATABASE", "")
EXCLUDE_COLLECTIONS = _list("EXCLUDE_COLLECTIONS")
EXCLUDE_FIELDS = [f.lower() for f in _list("EXCLUDE_FIELDS")]
BATCH_SIZE = max(10, _int("BATCH_SIZE", 500))
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")   # required for POST /api/ingest/ outside DEBUG

# --------------------------------------------------------- Text conversion
MAX_FIELD_CHARS = _int("MAX_FIELD_CHARS", 20000)
MAX_DOC_CHARS = _int("MAX_DOC_CHARS", 50000)
CHUNK_SIZE = max(200, _int("CHUNK_SIZE", 700))
CHUNK_OVERLAP = max(0, min(_int("CHUNK_OVERLAP", 100), CHUNK_SIZE // 2))

# -------------------------------------------------------------- Embeddings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()  # local | openai
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2" if EMBEDDING_PROVIDER == "local" else "text-embedding-3-small",
)
EMBEDDING_MODEL_ID = f"{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")      # only for provider=openai
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
VECTOR_STORE_DIR = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "vector_store")

# ------------------------------------------------------------------ Search
TOP_K = max(1, _int("TOP_K", 4 if _LOCAL else 5))
MIN_SCORE = _float("MIN_SCORE", 0.15)              # min cosine similarity for vector hits
HYBRID_SEARCH = _bool("HYBRID_SEARCH", True)       # vector + MongoDB $text keyword search
KEYWORD_WEIGHT = _float("KEYWORD_WEIGHT", 0.5)
RRF_K = _int("RRF_K", 60)
MAX_HOPS = max(0, _int("MAX_HOPS", 1))             # relationship hops followed in /ask
MAX_RELATED = max(0, _int("MAX_RELATED", 6))
INCLUDE_REVERSE = _bool("INCLUDE_REVERSE", True)   # also find records that point AT a hit
REVERSE_PER_RECORD = max(1, _int("REVERSE_PER_RECORD", 3))
MAX_CONTEXT_CHARS = _int("MAX_CONTEXT_CHARS", 6000 if _LOCAL else 12000)
# List-style questions ("list all...", "which projects...", "how many...") automatically retrieve more.
# Set LIST_TOP_K=0 to turn this off.
LIST_TOP_K = max(0, _int("LIST_TOP_K", 20 if _LOCAL else 30))
LIST_CONTEXT_CHARS = _int("LIST_CONTEXT_CHARS", 9000 if _LOCAL else 30000)
LIST_MAX_TOKENS = _int("LIST_MAX_TOKENS", 900 if _LOCAL else 2048)

# ------------------------------------------- exact answers for list/count questions
STRUCTURED_QUERIES = _bool("STRUCTURED_QUERIES", True)   # list/count/total questions are answered from MongoDB directly
STRUCTURED_MAX_ROWS = max(1, _int("STRUCTURED_MAX_ROWS", 500))
STRUCTURED_PLANNER_TOKENS = _int("STRUCTURED_PLANNER_TOKENS", 1200)
CATALOG_TTL_SECONDS = _int("CATALOG_TTL_SECONDS", 600)   # how long the sampled schema is reused
CATALOG_SAMPLE_SIZE = max(20, _int("CATALOG_SAMPLE_SIZE", 200))
QUERY_TIMEOUT_MS = _int("QUERY_TIMEOUT_MS", 15000)

# ------------------------------------------------ Exact (structured) queries
# List / count questions ("list all departments", "how many tasks are medium priority") are answered by
# querying MongoDB directly, so totals are exact. Set STRUCTURED_QUERIES=false to use similarity search only.

STRUCT_MAX_RECORDS = max(1, _int("STRUCT_MAX_RECORDS", 40 if _LOCAL else 60))   # records handed to the LLM (total is always exact)
STRUCT_SCAN_MAX = _int("STRUCT_SCAN_MAX", 150)                 # no-LLM fallback: read a whole collection up to this size
STRUCT_TIMEOUT_MS = _int("STRUCT_TIMEOUT_MS", 10000)
# "on": the LLM plans list/count questions (filters). "off": no planning call (faster on a slow local model):
# a question that names one small collection still returns its complete list, but conditions are applied by the LLM.
STRUCT_PLANNER = os.getenv("STRUCT_PLANNER", "on").strip().lower()
CATALOG_CHARS = _int("CATALOG_CHARS", 3500 if _LOCAL else 8000)    # size of the catalog shown to the planner
CATALOG_MAX_AGE_HOURS = _int("CATALOG_MAX_AGE_HOURS", 24)       # saved catalog is refreshed in the background after this

# --------------------------------------------------------------------- LLM
# Any OpenAI-compatible API. Defaults: Groq + gpt-oss-20b.
# OpenAI: LLM_BASE_URL=https://api.openai.com/v1  LLM_MODEL=gpt-4o-mini
LLM_API_KEY = (os.getenv("LLM_API_KEY") or
               (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "") if _GEMINI else os.getenv("GROQ_API_KEY", "")))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/" if _GEMINI
                         else "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b" if _LOCAL else "gemini-3.5-flash-lite" if _GEMINI else "openai/gpt-oss-20b")
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE", "" if _GEMINI else "0")   # never sent to Gemini (deprecated there)   # empty = provider default
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.0) if not _GEMINI else None
STRUCT_PLANNER_ENABLED = _bool("STRUCT_PLANNER", True)   # not "on"/"off"
# Name of the "maximum output tokens" request field. Gemini's OpenAI-compatible endpoint documents max_tokens.
LLM_MAX_TOKENS_PARAM = os.getenv("LLM_MAX_TOKENS_PARAM", "max_tokens" if _GEMINI else "max_completion_tokens")
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 600 if _LOCAL else 1500)
LLM_TIMEOUT = _int("LLM_TIMEOUT", 180 if _LOCAL else 60)
# --- Ollama (LLM_PROVIDER=ollama): talks to the native API, which can set the context window per request
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
OLLAMA_NUM_CTX = _int("OLLAMA_NUM_CTX", 8192)          # tokens; Ollama's own default is smaller and silently cuts prompts
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")   # keep the model in memory between questions
# Speed: reasoning models (gpt-oss) "think" before answering. LLM_REASONING_EFFORT=low makes them much faster.
# Leave empty if your provider does not support it (it is dropped automatically if the provider rejects it).
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip()
# Optional smaller/faster model just for the query-planning step of list/count questions.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "").strip()

# ------------------------------------------------------------------- Cache
CACHE_TTL_SECONDS = _int("CACHE_TTL_SECONDS", 600)   # 0 disables the answer cache
CACHE_MAX_ENTRIES = _int("CACHE_MAX_ENTRIES", 500)
WARMUP_ON_START = _bool("WARMUP_ON_START", True)
