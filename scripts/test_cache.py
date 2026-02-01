"""Test semantic cache functionality"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag.rag_chain import RAGChain
import time


def test_semantic_cache():
    print("\n" + "="*60)
    print("  🚀 TESTING SEMANTIC CACHE")
    print("="*60)
    
    # Initialize RAG chain with cache
    rag = RAGChain(enable_cache=True, enable_conversation=True)
    
    # Query 1: First time (cache miss)
    query1 = "What is the waiting period for pre-existing diseases in Optima Restore?"
    print(f"\n📝 Query 1 (First time): {query1}")
    
    start = time.time()
    response1 = rag.generate_answer(query1, use_enhanced_retrieval=True)
    time1 = time.time() - start
    
    print(f"⏱️  Time: {time1:.2f}s")
    print(f"📊 Confidence: {response1.confidence:.2%}")
    print(f"💾 Cached: {response1.cached}")
    
    # Query 2: Similar query (should hit cache)
    query2 = "Tell me about PED waiting time in Optima Restore policy"
    print(f"\n📝 Query 2 (Similar): {query2}")
    
    start = time.time()
    response2 = rag.generate_answer(query2, use_enhanced_retrieval=True)
    time2 = time.time() - start
    
    print(f"⏱️  Time: {time2:.2f}s")
    print(f"📊 Confidence: {response2.confidence:.2%}")
    print(f"💾 Cached: {response2.cached}")
    
    # Show speedup
    if response2.cached:
        speedup = time1 / time2
        print(f"\n🚀 SPEEDUP: {speedup:.1f}x faster!")
    
    # Show cache stats
    print(f"\n📊 Cache Statistics:")
    cache_stats = rag.cache.get_stats()
    for key, value in cache_stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    test_semantic_cache()
    
    print("\n" + "="*60)
    print("  ✅ CACHE TEST COMPLETE!")
    print("="*60)
