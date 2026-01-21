"""FastAPI application entry point"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import get_settings
from src.monitoring.logger import setup_logging

settings = get_settings()
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("="*60)
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info("="*60)
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Ollama Model: {settings.ollama_model}")
    logger.info(f"Ollama URL: {settings.ollama_base_url}")
    logger.info(f"Debug Mode: {settings.debug}")
    
    if settings.langsmith_enabled:
        logger.info(f"✅ LangSmith Project: {settings.langchain_project}")
    else:
        logger.warning("⚠️  LangSmith tracing disabled")
    
    logger.info("="*60)
    
    yield
    
    # Shutdown
    logger.info("="*60)
    logger.info("👋 Shutting down application")
    logger.info("="*60)


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-grade AI health insurance advisory chatbot with RAG, caching, and guardrails",
    lifespan=lifespan,
    debug=settings.debug,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "langsmith_enabled": settings.langsmith_enabled,
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "description": "AI-powered health insurance advisory chatbot",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "openapi": "/openapi.json"
        }
    }


# Will add routers later as we build features
# from src.api.routes import chat, policies, admin
# app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
# app.include_router(policies.router, prefix="/api/v1/policies", tags=["Policies"])
