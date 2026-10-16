# No Django models: data lives in MongoDB (see rag/services/embedding_store.py).
# rag/models.py
from django.db import models

class ChatSession(models.Model):
    session_id = models.CharField(max_length=64, unique=True, db_index=True)
    title = models.CharField(max_length=120, default="New chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.title} ({self.session_id[:8]})"