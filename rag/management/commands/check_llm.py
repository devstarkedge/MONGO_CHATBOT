"""
python manage.py check_llm

Tests the language-model configuration with the same kind of calls the app makes, and says exactly what is wrong
(wrong key, wrong model name, wrong base URL, timeout, token limit, Ollama not running ...). Your API key is
never printed. Run it whenever the chat reports that the language model is unavailable.
"""
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from rag.services import llm_service
from rag.services.exceptions import ChatbotError


class Command(BaseCommand):
    help = "Check the LLM configuration with a real test call."

    def handle(self, *args, **options):
        s = settings
        out = self.stdout.write
        if s.LLM_PROVIDER == "ollama":
            where = s.OLLAMA_HOST
        else:
            where = s.LLM_BASE_URL
        key = "not needed (local)" if s.LLM_PROVIDER == "ollama" else (
            f"set ({len(s.LLM_API_KEY)} characters)" if s.LLM_API_KEY else "MISSING")
        sends_temperature = bool((s.LLM_TEMPERATURE or "").strip()) and s.LLM_PROVIDER != "gemini"
        out(f"Provider        : {s.LLM_PROVIDER}")
        out(f"Model           : {s.LLM_MODEL}")
        out(f"Endpoint        : {where}")
        out(f"API key         : {key}")
        out(f"Temperature     : {'sent (' + s.LLM_TEMPERATURE + ')' if sends_temperature else 'not sent'}")
        out(f"Output limit    : field '{s.LLM_MAX_TOKENS_PARAM}' = {s.LLM_MAX_TOKENS}" if s.LLM_PROVIDER != "ollama"
            else f"Output limit    : num_predict = {s.LLM_MAX_TOKENS}, num_ctx = {s.OLLAMA_NUM_CTX}")
        out(f"Reasoning effort: {s.LLM_REASONING_EFFORT or 'provider default'}")
        out(f"Timeout         : {s.LLM_TIMEOUT}s")
        out("")

        failures = 0
        failures += self._step("1/2 simple answer", lambda: llm_service._chat(
            [{"role": "user", "content": "Reply with the single word: ok"}], 200),
            lambda text: f"reply: {text[:60]!r}")
        failures += self._step("2/2 query-plan JSON (used for list/count questions)", lambda: llm_service.plan_query(
            "list all departments", "- departments (3 records): name [Engineering, Finance]"),
            lambda plan: (f"plan: {plan}" if isinstance(plan, dict) else None))
        out("")
        if failures:
            raise CommandError("The language model is not working yet. Fix the message above, restart the server, "
                               "and run this command again.")
        out(self.style.SUCCESS("The language model is configured correctly."))

    def _step(self, label: str, call, describe) -> int:
        started = time.perf_counter()
        try:
            result = call()
        except ChatbotError as exc:
            self.stdout.write(self.style.ERROR(f"  FAILED  {label}\n          {exc.message}"))
            return 1
        except Exception as exc:                     # never show a traceback for a configuration check
            self.stdout.write(self.style.ERROR(f"  FAILED  {label}\n          {type(exc).__name__}: {str(exc)[:200]}"))
            return 1
        detail = describe(result)
        seconds = time.perf_counter() - started
        if detail is None:
            self.stdout.write(self.style.WARNING(
                f"  WARNING {label} ({seconds:.1f}s)\n          the model answered, but not with valid JSON; "
                "list/count questions will fall back to similarity search"))
            return 0
        self.stdout.write(self.style.SUCCESS(f"  OK      {label} ({seconds:.1f}s)\n          {detail}"))
        return 0
