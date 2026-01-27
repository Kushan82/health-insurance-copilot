"""Application configuration management with optimized RAG settings"""

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
    
    # ==========================================
    # 🚀 OPTIMIZED OLLAMA SETTINGS
    # ==========================================
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:3b")
    
    # ✅ OPTIMIZED: Lower temperature for factual consistency
    ollama_temperature: float = Field(default=0.1)
    
    # ✅ OPTIMIZED: Reduced max tokens for faster responses
    ollama_max_tokens: int = Field(default=512)
    
    # ✅ OPTIMIZED: Reduced timeout for faster failure detection
    ollama_timeout: int = Field(default=60)
    
    # ✅ NEW: Response caching
    ollama_enable_cache: bool = Field(default=True)
    ollama_cache_ttl: int = Field(default=3600)  # 1 hour
    
    # ChromaDB
    chroma_persist_directory: str = Field(default="./data/vector_store")
    chromadb_collection_name: str = Field(default="insurance_policies")
    
    # Embeddings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)
    
    # ==========================================
    # 🚀 OPTIMIZED RAG SETTINGS
    # ==========================================
    
    # Chunking - Smaller chunks for better precision
    chunk_size: int = Field(default=512)  # Reduced from 500
    chunk_overlap: int = Field(default=128)  # Increased from 50 for better context
    min_chunk_size: int = Field(default=100)
    
    # Retrieval - Multi-stage approach
    top_k_retrieval: int = Field(default=5)  # Final top K
    initial_retrieval_k: int = Field(default=20)  # For hybrid retrieval (4x top_k)
    
    # ✅ OPTIMIZED: Higher similarity threshold for quality
    similarity_threshold: float = Field(default=0.5)  # Increased from 0.3
    
    # ✅ NEW: Retrieval thresholds
    initial_similarity_threshold: float = Field(default=0.25)  # For initial retrieval
    final_similarity_threshold: float = Field(default=0.5)  # For final results
    
    # ✅ NEW: Hybrid retrieval weights
    semantic_weight: float = Field(default=0.40)
    keyword_weight: float = Field(default=0.25)
    phrase_weight: float = Field(default=0.15)
    importance_weight: float = Field(default=0.10)
    query_type_weight: float = Field(default=0.10)
    
    # ✅ NEW: Confidence calculation weights
    avg_similarity_weight: float = Field(default=0.40)
    top_similarity_weight: float = Field(default=0.30)
    consistency_weight: float = Field(default=0.20)
    answer_quality_weight: float = Field(default=0.10)
    
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
    
    # Cache - Enhanced caching strategy
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
    
    # ==========================================
    # COMPUTED PROPERTIES
    # ==========================================
    
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
    
    @property
    def retrieval_multiplier(self) -> int:
        """Calculate how many times top_k to retrieve initially"""
        return self.initial_retrieval_k // self.top_k_retrieval
    
    def get_retrieval_config(self) -> dict:
        """Get all retrieval-related configuration"""
        return {
            "top_k": self.top_k_retrieval,
            "initial_k": self.initial_retrieval_k,
            "similarity_threshold": self.similarity_threshold,
            "initial_threshold": self.initial_similarity_threshold,
            "final_threshold": self.final_similarity_threshold,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
    
    def get_llm_config(self) -> dict:
        """Get all LLM-related configuration"""
        return {
            "model": self.ollama_model,
            "temperature": self.ollama_temperature,
            "max_tokens": self.ollama_max_tokens,
            "timeout": self.ollama_timeout,
            "base_url": self.ollama_base_url,
            "enable_cache": self.ollama_enable_cache,
            "cache_ttl": self.ollama_cache_ttl,
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
