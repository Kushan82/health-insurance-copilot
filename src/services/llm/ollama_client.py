"""Ollama LLM client with LangSmith tracing"""
from typing import Dict, List, Optional
from langchain_ollama import OllamaLLM
from langsmith import traceable
from src.core.config import get_settings
from src.core.exceptions import OllamaError
from src.monitoring.logger import setup_logging

settings = get_settings()
logger = setup_logging()


class OllamaClient:    
    def __init__(self):
        self.settings = settings
        self.llm = self._initialize_llm()
        
        if settings.langsmith_enabled:
            logger.info("✅ LangSmith tracing enabled")
        else:
            logger.warning("⚠️  LangSmith tracing disabled")
    
    def _initialize_llm(self) -> OllamaLLM:
        try:
            return OllamaLLM(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=settings.ollama_temperature,
                num_predict=settings.ollama_max_tokens,
            )
        except Exception as e:
            raise OllamaError(f"Failed to initialize Ollama: {e}")
    
    @traceable(name="ollama_generate", metadata={"service": "llm"})
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate response from Ollama
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            
        Returns:
            Generated response
        """
        try:
            # Construct full prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = prompt
            
            # Override temperature if provided
            if temperature is not None:
                self.llm.temperature = temperature
            
            logger.info(f"Generating response for prompt (length: {len(prompt)})")
            
            response = self.llm.invoke(full_prompt)
            
            logger.info(f"Response generated (length: {len(response)})")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise OllamaError(f"Generation failed: {e}")
    
    @traceable(name="ollama_batch_generate", metadata={"service": "llm"})
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            
        Returns:
            List of generated responses
        """
        responses = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing batch {i+1}/{len(prompts)}")
            response = self.generate(prompt, system_prompt)
            responses.append(response)
        
        return responses
    
    def health_check(self) -> Dict[str, any]:
        """
        Check if Ollama is running and model is available
        
        Returns:
            Health check status
        """
        try:
            # Try a simple generation
            test_response = self.llm.invoke("Hello")
            
            return {
                "status": "healthy",
                "model": settings.ollama_model,
                "base_url": settings.ollama_base_url,
                "test_response": test_response[:50] + "..." if len(test_response) > 50 else test_response
            }
            
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": settings.ollama_model,
                "base_url": settings.ollama_base_url
            }
