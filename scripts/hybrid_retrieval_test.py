"""Test improved hybrid retrieval"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag.embedding_service import EmbeddingService
from src.services.rag.vector_store import VectorStore
from src.services.rag.retriever import Retriever

def main():
    print("\n" + "="*60)
    print("  🔬 Testing IMPROVED Hybrid Retrieval")
    print("="*60)
    
    embedding_service = EmbeddingService()
    vector_store = VectorStore(reset=False)
    retriever = Retriever(embedding_service, vector_store)
    
    test_queries = [
        "What is the waiting period for pre-existing diseases in HDFC Optima Restore?",
        "Compare maternity coverage across policies",
        "What are AYUSH treatment benefits?"
    ]
    
    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        print(f"{'─'*60}")
        
        # Basic retrieval (baseline)
        basic = retriever.retrieve(query=query, top_k=5)
        basic_avg = sum(c.similarity for c in basic) / len(basic) if basic else 0
        
        # Improved hybrid retrieval
        hybrid = retriever.retrieve_hybrid(query=query, top_k=5)
        hybrid_avg = sum(c.similarity for c in hybrid) / len(hybrid) if hybrid else 0
        
        print(f"\n📊 Basic Retrieval:")
        print(f"   Chunks: {len(basic)}")
        print(f"   Avg similarity: {basic_avg:.3f}")
        if basic:
            print(f"   Top result: {basic[0].similarity:.3f} - {basic[0].source}")
        
        print(f"\n🚀 IMPROVED Hybrid Retrieval:")
        print(f"   Chunks: {len(hybrid)}")
        print(f"   Avg similarity: {hybrid_avg:.3f}")
        if hybrid:
            print(f"   Top result: {hybrid[0].similarity:.3f} - {hybrid[0].source}")
            
            # Show all results
            print(f"\n   📚 All {len(hybrid)} Results:")
            for i, chunk in enumerate(hybrid, 1):
                print(f"      {i}. {chunk.similarity:.3f} | {chunk.source}")
        
        # Performance comparison
        if basic_avg > 0 and hybrid_avg > 0:
            diff = hybrid_avg - basic_avg
            if abs(diff) < 0.01:
                status = "≈"
                result = "EQUAL"
            elif diff > 0:
                status = "↑"
                improvement = (diff / basic_avg) * 100
                result = f"+{improvement:.1f}%"
            else:
                status = "↓"
                decrease = (abs(diff) / basic_avg) * 100
                result = f"-{decrease:.1f}%"
            
            print(f"\n   {status} Hybrid vs Basic: {result}")
    
    print("\n" + "="*60)
    print("  ✅ Improved Hybrid Test Complete")
    print("="*60)

if __name__ == "__main__":
    main()
