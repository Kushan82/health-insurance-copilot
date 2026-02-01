"""
Semantic Cache - Cache query results based on semantic similarity
Speeds up similar queries by 10x+ by avoiding re-retrieval and LLM calls
"""
from typing import Optional, Dict, List, Tuple
import hashlib
import time
from dataclasses import dataclass
from loguru import logger
import numpy as np


@dataclass
class CacheEntry:
    """Cached query-response pair"""
    query: str
    query_embedding: np.ndarray
    answer: str
    sources: List[Dict]
    confidence: float
    timestamp: float
    hit_count: int = 0


class SemanticCache:
    """
    Semantic-aware cache that matches similar queries
    
    Unlike exact-match caching (e.g., Redis), this cache:
    - Matches semantically similar queries ("What is PED?" ≈ "Explain pre-existing disease")
    - Uses embedding similarity threshold
    - Implements TTL (time-to-live) and LRU (least recently used) eviction
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_cache_size: int = 1000,
        ttl_seconds: int = 3600  # 1 hour default
    ):
        """
        Args:
            similarity_threshold: Min similarity to consider cache hit (0.90-0.95 recommended)
            max_cache_size: Max number of cached entries
            ttl_seconds: Cache entry lifetime in seconds
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        logger.info(
            f"Semantic cache initialized (threshold: {similarity_threshold}, "
            f"max_size: {max_cache_size}, ttl: {ttl_seconds}s)"
        )
    
    def get(
        self,
        query: str,
        query_embedding: np.ndarray
    ) -> Optional[Tuple[str, List[Dict], float]]:
        """
        Try to get cached result for query
        
        Args:
            query: User query
            query_embedding: Query embedding vector
        
        Returns:
            (answer, sources, confidence) if cache hit, None otherwise
        """
        # Clean expired entries first
        self._evict_expired()
        
        # Find most similar cached query
        best_match = None
        best_similarity = 0.0
        
        for cache_id, entry in self.cache.items():
            # Compute cosine similarity
            similarity = self._cosine_similarity(query_embedding, entry.query_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        # Check if similarity exceeds threshold
        if best_match and best_similarity >= self.similarity_threshold:
            # Cache hit!
            best_match.hit_count += 1
            self.stats['hits'] += 1
            
            logger.info(
                f"✅ Cache HIT! (similarity: {best_similarity:.3f}) "
                f"Original query: '{best_match.query[:50]}...'"
            )
            
            return (best_match.answer, best_match.sources, best_match.confidence)
        
        # Cache miss
        self.stats['misses'] += 1
        logger.debug(f"Cache MISS (best similarity: {best_similarity:.3f})")
        
        return None
    
    def set(
        self,
        query: str,
        query_embedding: np.ndarray,
        answer: str,
        sources: List[Dict],
        confidence: float
    ):
        """
        Store query-response in cache
        
        Args:
            query: User query
            query_embedding: Query embedding
            answer: Generated answer
            sources: Retrieved sources
            confidence: Answer confidence
        """
        # Check cache size limit (LRU eviction)
        if len(self.cache) >= self.max_cache_size:
            self._evict_lru()
        
        # Create cache key (hash of query)
        cache_id = self._generate_cache_id(query)
        
        # Store entry
        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            answer=answer,
            sources=sources,
            confidence=confidence,
            timestamp=time.time(),
            hit_count=0
        )
        
        self.cache[cache_id] = entry
        logger.debug(f"Cached query: '{query[:50]}...'")
    
    def _generate_cache_id(self, query: str) -> str:
        """Generate unique cache ID from query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _evict_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if (current_time - entry.timestamp) > self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
            self.stats['evictions'] += 1
        
        if expired_keys:
            logger.debug(f"Evicted {len(expired_keys)} expired cache entries")
    
    def _evict_lru(self):
        """Evict least recently used (lowest hit count) entry"""
        if not self.cache:
            return
        
        # Find entry with lowest hit count and oldest timestamp
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (self.cache[k].hit_count, self.cache[k].timestamp)
        )
        
        del self.cache[lru_key]
        self.stats['evictions'] += 1
        logger.debug("Evicted LRU cache entry")
    
    def clear(self):
        """Clear all cache entries"""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cache cleared ({count} entries removed)")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_cache_size,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%",
            'evictions': self.stats['evictions'],
            'ttl_seconds': self.ttl_seconds
        }
