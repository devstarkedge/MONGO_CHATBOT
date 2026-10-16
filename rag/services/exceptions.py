"""Custom errors. Each has a safe user-facing message and an HTTP status code."""


class ChatbotError(Exception):
    status_code = 500
    default_message = "Something went wrong."

    def __init__(self, message=None):
        self.message = message or self.default_message
        super().__init__(self.message)


class ConfigError(ChatbotError):
    default_message = "The server is missing required configuration."


class MongoServiceError(ChatbotError):
    status_code = 503
    default_message = "Could not reach MongoDB. Check MONGO_URI and that the server is running."


class EmptyDatabaseError(ChatbotError):
    status_code = 404
    default_message = "The database has no documents to index."


class VectorStoreError(ChatbotError):
    status_code = 503
    default_message = "The search index is unavailable. Run: python manage.py ingest"


class EmbeddingError(ChatbotError):
    default_message = "Could not generate embeddings."


class IngestionError(ChatbotError):
    status_code = 400
    default_message = "Ingestion failed."


class IngestionBusyError(ChatbotError):
    status_code = 409
    default_message = "An ingestion job is already running."


class AuthError(ChatbotError):
    status_code = 401
    default_message = "Missing or invalid token."


class LLMError(ChatbotError):
    status_code = 502
    default_message = "The language model is unavailable right now. Please try again shortly."


class InvalidRequestError(ChatbotError):
    status_code = 400
    default_message = "Invalid request."
