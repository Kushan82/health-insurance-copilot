"""Application configuration management"""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = Field(default="Health Insurance Copilot")
    app_version: str = Field(default="0.1.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    
    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    
    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:3b")
    ollama_temperature: float = Field(default=0.7)
    ollama_max_tokens: int = Field(default=2048)
    ollama_timeout: int = Field(default=120)
    
    # ChromaDB
    chroma_persist_directory: str = Field(default="./data/vector_store")
    chromadb_collection_name: str = Field(default="insurance_policies")
    
    # Embeddings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)
    
    # LangSmith
    langchain_tracing_v2: bool = Field(default=False)
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com")
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="health-insurance-copilot")
    
    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: str = Field(default="")
    cache_ttl: int = Field(default=3600)
    
    # RAG
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    top_k_retrieval: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)
    
    # Cache
    enable_query_cache: bool = Field(default=True)
    enable_semantic_cache: bool = Field(default=True)
    enable_response_cache: bool = Field(default=True)
    enable_retrieval_cache: bool = Field(default=True)
    semantic_cache_threshold: float = Field(default=0.95)
    
    # Guardrails
    enable_guardrails: bool = Field(default=True)
    toxicity_threshold: float = Field(default=0.7)
    enable_pii_detection: bool = Field(default=True)
    enable_bias_detection: bool = Field(default=True)
    
    # Data Paths
    data_dir: Path = Field(default=Path("./data"))
    raw_data_dir: Path = Field(default=Path("./data/raw"))
    processed_data_dir: Path = Field(default=Path("./data/processed"))
    policies_file: Path = Field(default=Path("./data/processed/policies.json"))
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501", "http://localhost:8000"]
    )
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == "production"
    
    @property
    def langsmith_enabled(self) -> bool:
        """Check if LangSmith tracing is enabled"""
        return self.langchain_tracing_v2 and bool(self.langchain_api_key)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
