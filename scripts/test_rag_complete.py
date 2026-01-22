"""Test RAG pipeline with REAL ingested policy data"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag.rag_chain import RAGChain
from src.services.rag.retriever import Retriever
from src.services.rag.embedding_service import EmbeddingService
from src.services.rag.vector_store import VectorStore


def check_ingested_data():
    """Verify that data has been ingested"""
    print("\n" + "="*60)
    print("  📊 CHECKING INGESTED DATA")
    print("="*60)
    
    # Initialize vector store (DO NOT reset!)
    vector_store = VectorStore(reset=False)
    
    # Get collection stats
    collection = vector_store.collection
    doc_count = collection.count()
    
    print(f"📦 Total chunks in vector store: {doc_count}")
    
    if doc_count == 0:
        print("\n❌ ERROR: No data found in vector store!")
        print("\n📍 Please run ingestion first:")
        print("   python scripts/ingest_policies.py")
        return False
    
    # Sample some documents
    sample = collection.get(limit=5, include=["metadatas", "documents"])
    
    print(f"\n📁 Sample sources:")
    unique_sources = set()
    for metadata in sample["metadatas"]:
        if "filename" in metadata:
            unique_sources.add(metadata["filename"])
    
    for source in sorted(unique_sources):
        print(f"   • {source}")
    
    return True


def test_rag_with_real_data():
    """Test RAG chain with real ingested policy data"""
    print("\n" + "="*60)
    print("  🤖 TESTING RAG WITH REAL POLICY DATA")
    print("="*60)
    
    # Initialize components (NO RESET - use existing data!)
    embedding_service = EmbeddingService()
    vector_store = VectorStore(reset=False)
    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store
    )
    
    # Initialize RAG chain
    rag_chain = RAGChain(
        embedding_service=embedding_service,
        vector_store=vector_store,
        retriever=retriever
    )
    
    # Test queries based on REAL policy content
    test_queries = [
        # Query 1: Pre-existing disease waiting periods
        {
            "query": "What is the waiting period for pre-existing diseases in HDFC Optima Restore?",
            "expected_answer": "36 months (3 years)",
            "policy": "HDFC Optima Restore"
        },
        
        # Query 2: Care Supreme maternity coverage
        {
            "query": "Does Care Supreme cover maternity expenses?",
            "expected_answer": "Maternity is generally excluded (check policy wording)",
            "policy": "Care Supreme"
        },
        
        # Query 3: Room rent limits
        {
            "query": "What are the room rent limits in Medicare Plus?",
            "expected_answer": "Varies by plan - check specific sub-limits",
            "policy": "Medicare Plus"
        },
        
        # Query 4: Restoration benefit
        {
            "query": "Explain the restore benefit in Optima Restore",
            "expected_answer": "100% instant addition of sum insured after partial/complete utilization",
            "policy": "HDFC Optima Restore"
        },
        
        # Query 5: AYUSH treatment coverage
        {
            "query": "Which policies cover AYUSH treatments?",
            "expected_answer": "Multiple policies including Care Supreme, Optima Restore",
            "policy": "Multiple"
        },
        
        # Query 6: Day care procedures
        {
            "query": "Are day care procedures covered in Care Ultimate?",
            "expected_answer": "Yes, procedures taking less than 24 hours",
            "policy": "Care Ultimate"
        },
        
        # Query 7: Policy comparison
        {
            "query": "Compare coverage amounts between Care Supreme and Optima Secure",
            "expected_answer": "Care Supreme: ₹5L-₹1Cr, Optima Secure: varies",
            "policy": "Multiple"
        },
        
        # Query 8: Exclusions
        {
            "query": "What are the standard exclusions in health insurance policies?",
            "expected_answer": "Cosmetic surgery, self-inflicted injuries, pre-existing diseases during waiting period",
            "policy": "General"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected = test_case["expected_answer"]
        policy = test_case["policy"]
        
        print(f"\n{'─'*60}")
        print(f"Query {i}: {query}")
        print(f"Expected Policy: {policy}")
        print(f"Expected Answer: {expected}")
        print(f"{'─'*60}")
        
        try:
            # Generate answer
            response = rag_chain.generate_answer(query, top_k=5)
            
            print(f"\n📝 RAG Answer:")
            print(response.answer)
            
            print(f"\n📊 Metadata:")
            print(f"   Confidence: {response.confidence:.2%}")
            print(f"   Retrieved chunks: {response.retrieved_chunks_count}")
            print(f"   Generation time: {response.generation_time_ms:.0f}ms")
            
            if response.sources:
                print(f"\n📚 Sources ({len(response.sources)}):")
                for j, source in enumerate(response.sources[:3], 1):  # Show top 3
                    print(f"   {j}. {source.filename}")
                    print(f"      Similarity: {source.similarity:.2%}")
                    print(f"      Preview: {source.preview[:150]}...")
            else:
                print("\n⚠️  No sources retrieved!")
            
            # Store result
            results.append({
                "query": query,
                "answer": response.answer,
                "confidence": response.confidence,
                "sources": len(response.sources) if response.sources else 0,
                "time_ms": response.generation_time_ms
            })
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                "query": query,
                "answer": f"ERROR: {e}",
                "confidence": 0,
                "sources": 0,
                "time_ms": 0
            })
    
    # Summary
    print("\n" + "="*60)
    print("  📊 TEST SUMMARY")
    print("="*60)
    
    total_queries = len(results)
    successful_queries = sum(1 for r in results if r["confidence"] > 0)
    avg_confidence = sum(r["confidence"] for r in results) / total_queries if total_queries > 0 else 0
    avg_time = sum(r["time_ms"] for r in results) / total_queries if total_queries > 0 else 0
    
    print(f"\n✅ Successful queries: {successful_queries}/{total_queries}")
    print(f"📊 Average confidence: {avg_confidence:.2%}")
    print(f"⚡ Average response time: {avg_time:.0f}ms")
    print(f"📚 Average sources per query: {sum(r['sources'] for r in results) / total_queries:.1f}")
    
    # Show chain info
    print("\n" + "="*60)
    print("  ℹ️  RAG CHAIN CONFIGURATION")
    print("="*60)
    info = rag_chain.get_chain_info()
    print(f"LLM Model: {info['llm_model']}")
    print(f"Embedding Model: {info['embedding_model']}")
    print(f"Vector Store: {info['vector_store']}")
    print(f"Top-K Retrieval: {info['top_k']}")
    print(f"Similarity Threshold: {info['similarity_threshold']}")
    
    return results


def main():
    """Run RAG test with real data"""
    print("\n" + "="*60)
    print("  🧪 RAG PIPELINE TEST - REAL DATA")
    print("="*60)
    
    # Check if data is ingested
    if not check_ingested_data():
        return
    
    # Run tests
    test_rag_with_real_data()
    
    print("\n" + "="*60)
    print("  ✅ RAG TEST COMPLETE!")
    print("="*60)
    
    print("\n📍 Next Steps:")
    print("   1. Review answers for accuracy")
    print("   2. Compare with actual policy documents")
    print("   3. Tune retrieval parameters if needed")
    print("   4. Test with more complex queries")
    print("   5. Build Chat API (Phase 3)")


if __name__ == "__main__":
    main()
