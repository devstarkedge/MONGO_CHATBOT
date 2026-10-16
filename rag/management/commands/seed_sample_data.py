"""
python manage.py seed_sample_data
Creates workspaces, users, memberships, projects and tasks that REFERENCE each other with ObjectIds
(different field-name styles on purpose) so you can see relationship detection working.
Collections that already contain data are left alone.
"""
from bson import ObjectId
from django.core.management.base import BaseCommand, CommandError
from pymongo.errors import PyMongoError

from rag.services import mongo_service
from rag.services.exceptions import ChatbotError

WS_ACME, WS_BETA = ObjectId("a00000000000000000000001"), ObjectId("a00000000000000000000002")
U_AISHA, U_ROHAN, U_MEERA = (ObjectId(f"b0000000000000000000000{i}") for i in (1, 2, 3))
P_H4T, P_SHOP, P_BLOG = (ObjectId(f"c0000000000000000000000{i}") for i in (1, 2, 3))

SAMPLE = {
    "workspaces": [
        {"_id": WS_ACME, "name": "Acme Corp", "plan": "pro", "industry": "travel"},
        {"_id": WS_BETA, "name": "Beta Labs", "plan": "free", "industry": "retail"},
    ],
    "users": [
        {"_id": U_AISHA, "name": "Aisha Khan", "role": "UI designer", "email": "aisha@example.com",
         "password": "$2b$10$notarealhash"},
        {"_id": U_ROHAN, "name": "Rohan Mehta", "role": "backend developer", "email": "rohan@example.com"},
        {"_id": U_MEERA, "name": "Meera Iyer", "role": "project manager", "email": "meera@example.com"},
    ],
    "memberships": [
        {"user": U_AISHA, "workspace": WS_ACME, "role": "employee", "status": "active"},
        {"user": U_ROHAN, "workspace": WS_BETA, "role": "employee", "status": "active"},
        {"user": U_MEERA, "workspace": WS_ACME, "role": "admin", "status": "inactive"},
    ],
    "projects": [
        {"_id": P_H4T, "project": "Health4Travel", "status": "completed", "priority": "medium",
         "source": "Upwork", "workspaceId": WS_ACME, "createdBy": U_MEERA, "tags": ["travel", "health"]},
        {"_id": P_SHOP, "project": "ShopEase", "status": "in progress", "priority": "high",
         "source": "Fiverr", "workspaceId": WS_BETA, "createdBy": U_ROHAN, "tags": ["retail"]},
        {"_id": P_BLOG, "project": "BlogRevamp", "status": "on hold", "priority": "low",
         "source": "Direct client", "workspaceId": WS_ACME, "createdBy": U_MEERA},
    ],
    "tasks": [
        {"title": "Design landing page", "projectId": P_H4T, "assignedTo": U_AISHA,
         "status": "done", "priority": "medium"},
        {"title": "Set up payment gateway", "projectId": P_SHOP, "assignedTo": U_ROHAN,
         "status": "in progress", "priority": "high"},
        {"title": "Fix login bug", "projectId": P_SHOP, "assignedTo": U_ROHAN,
         "status": "todo", "priority": "medium"},
        {"title": "Write release notes", "projectId": P_BLOG, "assignedTo": U_AISHA,
         "status": "todo", "priority": "low"},
    ],
}


class Command(BaseCommand):
    help = "Insert example, cross-referenced documents (only into empty collections)."

    def handle(self, *args, **options):
        try:
            db = mongo_service.get_db()
            for name, docs in SAMPLE.items():
                if db[name].count_documents({}) > 0:
                    self.stdout.write(f"  {name}: already has data, skipped")
                    continue
                db[name].insert_many([dict(d) for d in docs])
                self.stdout.write(f"  {name}: inserted {len(docs)} document(s)")
        except ChatbotError as exc:
            raise CommandError(exc.message)
        except PyMongoError as exc:
            raise CommandError(f"MongoDB error: {type(exc).__name__}")
        self.stdout.write(self.style.SUCCESS("Sample data ready. Now run: python manage.py ingest"))
