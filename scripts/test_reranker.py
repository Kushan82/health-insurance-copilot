"""
Test Cross-Encoder Re-ranking
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag.reranker import CrossEncoderReranker
from loguru import logger


def test_reranker():
    print("\n" + "="*60)
    print("  🎯 TESTING CROSS-ENCODER RE-RANKER")
    print("="*60)
    
    # Initialize reranker
    reranker = CrossEncoderReranker()
    
    if not reranker.model:
        print("\n❌ Cross-encoder model not loaded")
        print("Install: pip install sentence-transformers")
        return
    
    # Test query
    query = "What is the waiting period for pre-existing diseases?"
    
    # Mock chunks (simulating retrieval results)
    chunks = [
        {
            'text': 'The waiting period for specific diseases like cataract is 24 months.',
            'similarity': 0.65,
            'chunk_id': 'chunk_1'
        },
        {
            'text': 'Pre-existing diseases have a waiting period of 36 months in most policies.',
            'similarity': 0.60,
            'chunk_id': 'chunk_2'
        },
        {
            'text': 'Initial waiting period is 30 days for all illnesses except accidents.',
            'similarity': 0.58,
            'chunk_id': 'chunk_3'
        },
        {
            'text': 'Room rent is capped at 1% of sum insured per day.',
            'similarity': 0.40,
            'chunk_id': 'chunk_4'
        },
    ]
    
    print(f"\n📝 Query: {query}")
    print(f"\n📊 Original Ranking (by embedding similarity):")
    for i, chunk in enumerate(chunks, 1):
        print(f"   {i}. Similarity: {chunk['similarity']:.3f}")
        print(f"      Text: {chunk['text'][:80]}...")
    
    # Re-rank
    print(f"\n🔄 Re-ranking with cross-encoder...")
    reranked = reranker.rerank(query, chunks.copy())
    
    print(f"\n📊 After Re-ranking:")
    for i, chunk in enumerate(reranked, 1):
        original = chunk.get('original_similarity', 0)
        print(f"   {i}. Similarity: {chunk['similarity']:.3f} (was {original:.3f})")
        print(f"      Text: {chunk['text'][:80]}...")
    
    # Show improvement
    print(f"\n📈 Improvement Analysis:")
    print(f"   Original top chunk: \"{chunks[0]['text'][:60]}...\"")
    print(f"   Re-ranked top chunk: \"{reranked[0]['text'][:60]}...\"")
    
    if chunks[0]['chunk_id'] != reranked[0]['chunk_id']:
        print(f"   ✅ Ranking changed! Re-ranker found more relevant chunk.")
    else:
        print(f"   ✓ Ranking confirmed. Top chunk was already most relevant.")


if __name__ == "__main__":
    test_reranker()
    
    print("\n" + "="*60)
    print("  ✅ RE-RANKER TEST COMPLETE!")
    print("="*60)
