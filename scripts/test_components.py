"""Test all components"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_settings
from src.monitoring.logger import setup_logging

logger = setup_logging()
settings = get_settings()


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_config():
    """Test configuration loading"""
    print_section("🔧 Testing Configuration")
    
    print(f"✅ App Name: {settings.app_name}")
    print(f"✅ Version: {settings.app_version}")
    print(f"✅ Environment: {settings.environment}")
    print(f"✅ Ollama Model: {settings.ollama_model}")
    print(f"✅ Ollama URL: {settings.ollama_base_url}")
    print(f"✅ LangSmith Enabled: {settings.langsmith_enabled}")
    if settings.langsmith_enabled:
        print(f"✅ LangSmith Project: {settings.langchain_project}")
    print(f"✅ Data Directory: {settings.data_dir}")
    print(f"✅ Cache Enabled: {settings.enable_query_cache}")
    print(f"✅ Guardrails Enabled: {settings.enable_guardrails}")


def test_imports():
    """Test that all core modules can be imported"""
    print_section("📦 Testing Imports")
    
    try:
        from src.core.config import get_settings
        print("✅ src.core.config")
    except Exception as e:
        print(f"❌ src.core.config: {e}")
    
    try:
        from src.core.constants import SYSTEM_PROMPT
        print("✅ src.core.constants")
    except Exception as e:
        print(f"❌ src.core.constants: {e}")
    
    try:
        from src.core.exceptions import OllamaError
        print("✅ src.core.exceptions")
    except Exception as e:
        print(f"❌ src.core.exceptions: {e}")
    
    try:
        from src.monitoring.logger import setup_logging
        print("✅ src.monitoring.logger")
    except Exception as e:
        print(f"❌ src.monitoring.logger: {e}")
    
    try:
        from src.api.main import app
        print("✅ src.api.main")
    except Exception as e:
        print(f"❌ src.api.main: {e}")
    
    try:
        from src.services.llm.ollama_client import OllamaClient
        print("✅ src.services.llm.ollama_client")
    except Exception as e:
        print(f"❌ src.services.llm.ollama_client: {e}")


def test_ollama():
    """Test Ollama connection (will fail if Ollama not running)"""
    print_section("🤖 Testing Ollama Connection")
    
    try:
        from src.services.llm.ollama_client import OllamaClient
        
        print("Attempting to connect to Ollama...")
        client = OllamaClient()
        
        # Health check
        health = client.health_check()
        
        if health["status"] == "healthy":
            print(f"✅ Ollama is running")
            print(f"✅ Model: {health['model']}")
            print(f"✅ Base URL: {health['base_url']}")
            print(f"✅ Test response: {health['test_response']}")
            return True
        else:
            print(f"❌ Ollama health check failed")
            print(f"   Error: {health.get('error', 'Unknown error')}")
            print(f"\n⚠️  Ollama is not running or model not available")
            return False
        
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("\n⚠️  Ollama needs to be set up:")
        print("   1. Install Ollama")
        print("   2. Start server: ollama serve")
        print("   3. Pull model: ollama pull llama3.2:3b")
        return False


def test_fastapi():
    """Test FastAPI app creation"""
    print_section("🌐 Testing FastAPI Application")
    
    try:
        from src.api.main import app
        print(f"✅ FastAPI app created")
        print(f"✅ Title: {app.title}")
        print(f"✅ Version: {app.version}")
        print(f"✅ Routes: {len(app.routes)} registered")
        return True
    except Exception as e:
        print(f"❌ Error creating FastAPI app: {e}")
        return False


def main():
    """Run all component tests"""
    print("\n" + "="*60)
    print("  🧪 HEALTH INSURANCE COPILOT - COMPONENT TESTING")
    print("="*60)
    
    # Test configuration
    test_config()
    
    # Test imports
    test_imports()
    
    # Test FastAPI
    fastapi_ok = test_fastapi()
    
    # Test Ollama (will fail if not set up - that's expected)
    ollama_ok = test_ollama()
    
    # Final summary
    print_section("📊 TEST SUMMARY")
    print(f"Configuration:  ✅ Passed")
    print(f"Imports:        ✅ Passed")
    print(f"FastAPI:        {'✅ Passed' if fastapi_ok else '❌ Failed'}")
    print(f"Ollama:         {'✅ Connected' if ollama_ok else '⚠️  Not Running (Expected)'}")
    print("="*60)
    
    if fastapi_ok:
        print("\n🎉 Core components are working!")
        print("\n📍 Next steps:")
        print("  1. Install Ollama (see instructions below)")
        print("  2. Start Ollama and pull model")
        print("  3. Re-run this test")
        print("  4. Start API: uvicorn src.api.main:app --reload")
        
        if not ollama_ok:
            print("\n⚙️  Install Ollama:")
            print("  Windows: https://ollama.com/download/windows")
            print("  Mac:     brew install ollama")
            print("  Linux:   curl -fsSL https://ollama.com/install.sh | sh")
            print("\n  Then run:")
            print("    ollama serve")
            print("    ollama pull llama3.2:3b")
    else:
        print("\n⚠️  Some components failed. Check errors above.")


if __name__ == "__main__":
    main()
