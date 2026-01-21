"""Custom exceptions"""


class BaseAppException(Exception):
    """Base exception for the application"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(BaseAppException):
    """Configuration related errors"""
    pass


class OllamaError(BaseAppException):
    """Ollama LLM related errors"""
    pass


class RAGError(BaseAppException):
    """RAG pipeline related errors"""
    pass


class CacheError(BaseAppException):
    """Cache related errors"""
    pass


class GuardrailViolation(BaseAppException):
    """Guardrail violation detected"""
    pass


class EmbeddingError(BaseAppException):
    """Embedding generation errors"""
    pass


class ValidationError(BaseAppException):
    """Input validation errors"""
    pass
