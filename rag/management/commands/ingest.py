"""
python manage.py ingest                       incremental: embeds only new/changed documents, drops deleted ones
python manage.py ingest --full                re-embed everything
python manage.py ingest --collections users projects
python manage.py ingest --max-docs 500        quick experiment (never deletes anything)
"""
from django.core.management.base import BaseCommand, CommandError

from rag.services import ingestion_service
from rag.services.exceptions import ChatbotError


class Command(BaseCommand):
    help = "Read MongoDB, convert documents to text, detect relationships, embed and index them."

    def add_arguments(self, parser):
        parser.add_argument("--full", action="store_true", help="Re-embed every document.")
        parser.add_argument("--collections", nargs="+", metavar="NAME", help="Only these collections.")
        parser.add_argument("--max-docs", type=int, default=0, help="Max documents per collection (0 = all).")
        parser.add_argument("--batch-size", type=int, default=0, help="Documents per batch (default: BATCH_SIZE).")

    def handle(self, *args, **options):
        try:
            summary = ingestion_service.run_ingestion(
                collections=options["collections"], full=options["full"], max_docs=options["max_docs"],
                batch_size=options["batch_size"] or None, log=self.stdout.write)
        except ChatbotError as exc:
            raise CommandError(exc.message)
        self.stdout.write("")
        for name, s in summary["collections"].items():
            self.stdout.write(f"  {name}: {s['documents']} docs, {s['chunks']} chunks, {s['embedded']} embedded, "
                              f"{s['unchanged']} unchanged, {s['removed']} removed")
        self.stdout.write(self.style.SUCCESS(
            f"Done ({summary['mode']}) in {summary['seconds']}s. {summary['total_vectors']} vectors in the index."))
