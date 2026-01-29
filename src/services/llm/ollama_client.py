"""Optimized Ollama LLM client with caching and performance improvements"""

from typing import Dict, List, Optional
import time
import hashlib
from langchain_ollama import OllamaLLM
from langsmith import traceable
from src.core.config import get_settings
from src.core.exceptions import OllamaError
from src.monitoring.logger import setup_logging

settings = get_settings()
logger = setup_logging()


class OllamaClient:
    """
    Optimized Ollama client with:
    - Response caching for repeated queries
    - Configurable token limits
    - Performance monitoring
    - Timeout handling
    """
    
    def __init__(self):
        self.settings = settings
        self.llm = self._initialize_llm()
        
        # ✅ NEW: Response cache
        self._response_cache: Dict[str, Dict] = {}
        self._cache_enabled = settings.ollama_enable_cache
        self._cache_ttl = settings.ollama_cache_ttl
        
        # ✅ NEW: Performance metrics
        self._call_count = 0
        self._cache_hits = 0
        self._total_latency = 0.0
        
        if settings.langsmith_enabled:
            logger.info("✅ LangSmith tracing enabled")
        else:
            logger.warning("⚠️  LangSmith tracing disabled")
        
        logger.info(
            f"✅ Ollama client initialized - "
            f"Model: {settings.ollama_model}, "
            f"Temp: {settings.ollama_temperature}, "
            f"MaxTokens: {settings.ollama_max_tokens}, "
            f"Cache: {'enabled' if self._cache_enabled else 'disabled'}"
        )
    
    def _initialize_llm(self) -> OllamaLLM:
        """Initialize Ollama LLM with optimized settings"""
        try:
            return OllamaLLM(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=settings.ollama_temperature,
                num_predict=settings.ollama_max_tokens,
                timeout=settings.ollama_timeout,
            )
        except Exception as e:
            raise OllamaError(f"Failed to initialize Ollama: {e}")
    
    def _generate_cache_key(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Generate cache key from prompt parameters"""
        # Use first 500 chars of prompt + system prompt + temperature
        content = f"{prompt}_{system_prompt}_{temperature}" 
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available and not expired"""
        if not self._cache_enabled or cache_key not in self._response_cache:
            return None
        
        cached = self._response_cache[cache_key]
        age = time.time() - cached['timestamp']
        
        if age > self._cache_ttl:
            # Cache expired
            del self._response_cache[cache_key]
            return None
        
        self._cache_hits += 1
        logger.info(f"✅ Cache hit! (age: {age:.1f}s, hits: {self._cache_hits}/{self._call_count})")
        return cached['response']
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache a response"""
        if self._cache_enabled:
            self._response_cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
            # Limit cache size to 100 entries
            if len(self._response_cache) > 100:
                # Remove oldest entry
                oldest_key = min(self._response_cache.keys(), 
                               key=lambda k: self._response_cache[k]['timestamp'])
                del self._response_cache[oldest_key]
    
    @traceable(name="ollama_generate", metadata={"service": "llm"})
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Generate response from Ollama with caching and optimization
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override (default: 0.1 for factual)
            max_tokens: Optional max tokens override (default: 512)
            use_cache: Whether to use caching (default: True)
        
        Returns:
            Generated response
        """
        self._call_count += 1
        start_time = time.time()
        
        # Use optimized defaults if not provided
        temperature = temperature if temperature is not None else settings.ollama_temperature
        max_tokens = max_tokens if max_tokens is not None else settings.ollama_max_tokens
        system_prompt = system_prompt or ""
        
        # Check cache
        if use_cache and self._cache_enabled:
            cache_key = self._generate_cache_key(prompt, system_prompt, temperature)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        try:
            # Construct full prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = prompt
            
            # Override settings if needed
            original_temp = self.llm.temperature
            original_tokens = self.llm.num_predict
            
            self.llm.temperature = temperature
            self.llm.num_predict = max_tokens
            
            logger.info(
                f"Generating response (prompt: {len(prompt)} chars, "
                f"temp: {temperature}, max_tokens: {max_tokens})"
            )
            
            # Generate response
            response = self.llm.invoke(full_prompt)
            
            # Restore original settings
            self.llm.temperature = original_temp
            self.llm.num_predict = original_tokens
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000
            self._total_latency += latency
            
            logger.info(
                f"Response generated in {latency:.0f}ms "
                f"({len(response)} chars, "
                f"avg: {self._total_latency/self._call_count:.0f}ms)"
            )
            
            # Cache the response
            if use_cache and self._cache_enabled:
                self._cache_response(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise OllamaError(f"Generation failed: {e}")
    
    @traceable(name="ollama_batch_generate", metadata={"service": "llm"})
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        Generate responses for multiple prompts with caching
        
        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            temperature: Optional temperature override
        
        Returns:
            List of generated responses
        """
        responses = []
        total_start = time.time()
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing batch {i+1}/{len(prompts)}")
            response = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature
            )
            responses.append(response)
        
        total_time = (time.time() - total_start) * 1000
        logger.info(
            f"✅ Batch complete: {len(prompts)} prompts in {total_time:.0f}ms "
            f"(avg: {total_time/len(prompts):.0f}ms/prompt)"
        )
        
        return responses
    
    def health_check(self) -> Dict[str, any]:
        """
        Check if Ollama is running and model is available
        
        Returns:
            Health check status with performance metrics
        """
        try:
            # Try a simple generation
            start_time = time.time()
            test_response = self.llm.invoke("Hello")
            latency = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "model": settings.ollama_model,
                "base_url": settings.ollama_base_url,
                "test_response": test_response[:50] + "..." if len(test_response) > 50 else test_response,
                "test_latency_ms": round(latency, 2),
                "performance": self.get_performance_stats()
            }
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": settings.ollama_model,
                "base_url": settings.ollama_base_url
            }
    
    def get_performance_stats(self) -> Dict[str, any]:
        """Get performance statistics"""
        if self._call_count == 0:
            return {
                "total_calls": 0,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "avg_latency_ms": 0.0
            }
        
        return {
            "total_calls": self._call_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(self._cache_hits / self._call_count * 100, 2),
            "avg_latency_ms": round(self._total_latency / self._call_count, 2),
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._response_cache),
            "cache_ttl_seconds": self._cache_ttl
        }
    
    def clear_cache(self):
        """Clear the response cache"""
        self._response_cache.clear()
        logger.info("✅ Response cache cleared")
    
    def set_cache_enabled(self, enabled: bool):
        """Enable or disable caching"""
        self._cache_enabled = enabled
        logger.info(f"✅ Cache {'enabled' if enabled else 'disabled'}")
