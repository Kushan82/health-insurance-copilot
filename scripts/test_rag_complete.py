"""Complete RAG pipeline test with real policy data"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_settings
from src.services.rag.embedding_service import EmbeddingService
from src.services.rag.vector_store import VectorStore
from src.services.rag.retriever import Retriever
from src.services.rag.rag_chain import RAGChain
from src.services.llm.ollama_client import OllamaClient
from src.core.exceptions import RAGError

settings = get_settings()

def check_ingested_data():
    """Check what data is in vector store"""
    print("\n" + "="*60)
    print("  📊 CHECKING INGESTED DATA")
    print("="*60)
    
    vector_store = VectorStore(reset=False)
    stats = vector_store.get_stats()
    
    # ✅ FIXED: Use 'total_chunks' instead of 'count'
    print(f"📦 Total chunks in vector store: {stats.get('total_chunks', 0)}")
    print(f"📁 Unique sources: {stats.get('unique_sources', 0)}")
    
    # Get sample data
    if stats.get('total_chunks', 0) > 0:
        print(f"\n📚 Sample sources:")
        for source in stats.get('sources', [])[:5]:
            print(f"   • {source}")

def test_rag_with_real_data():
    """Test RAG with real insurance policy queries"""
    print("\n" + "="*60)
    print("  🤖 TESTING RAG WITH REAL POLICY DATA")
    print("="*60)
    
    # Initialize components
    embedding_service = EmbeddingService()
    vector_store = VectorStore(reset=False)
    retriever = Retriever(embedding_service, vector_store)
    llm_client = OllamaClient()
    
    # ✅ NO MONKEY PATCHING - RAG chain handles hybrid automatically
    
    # Initialize RAG chain
    rag_chain = RAGChain(
        llm_client=llm_client,
        retriever=retriever,
        embedding_service=embedding_service,
        vector_store=vector_store
    )
    
    # Test queries
    test_queries = [
        {
            "query": "What is the waiting period for pre-existing diseases in HDFC Optima Restore?",
            "expected_policy": "HDFC Optima Restore",
            "expected_answer": "36 months (3 years)"
        },
        {
            "query": "Does Care Supreme cover maternity expenses?",
            "expected_policy": "Care Supreme",
            "expected_answer": "Maternity is generally excluded (check policy wording)"
        },
        {
            "query": "What are the room rent limits in Medicare Plus?",
            "expected_policy": "Medicare Plus",
            "expected_answer": "Varies by plan - check specific sub-limits"
        },
        {
            "query": "Explain the restore benefit in Optima Restore",
            "expected_policy": "HDFC Optima Restore",
            "expected_answer": "100% instant addition of sum insured after partial/complete utilization"
        },
        {
            "query": "Which policies cover AYUSH treatments?",
            "expected_policy": "Multiple",
            "expected_answer": "Multiple policies including Care Supreme, Optima Restore"
        },
        {
            "query": "Are day care procedures covered in Care Ultimate?",
            "expected_policy": "Care Ultimate",
            "expected_answer": "Yes, procedures taking less than 24 hours"
        },
        {
            "query": "Compare coverage amounts between Care Supreme and Optima Secure",
            "expected_policy": "Multiple",
            "expected_answer": "Care Supreme: ₹5L-₹1Cr, Optima Secure: varies"
        },
        {
            "query": "What are the standard exclusions in health insurance policies?",
            "expected_policy": "General",
            "expected_answer": "Cosmetic surgery, self-inflicted injuries, pre-existing diseases during waiting period"
        }
    ]
    
    results = []
    total_time = 0
    
    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        expected_policy = test["expected_policy"]
        expected_answer = test["expected_answer"]
        
        print(f"\n{'─'*60}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print(f"Expected Policy: {expected_policy}")
        print(f"Expected Answer: {expected_answer}")
        print(f"{'─'*60}")
        
        # Analyze query first
        from src.services.rag.query_processor import QueryProcessor
        processor = QueryProcessor()
        query_data = processor.process_query(query)
        
        print(f"\n🔍 Query Analysis:")
        print(f"   Type: {query_data['query_type']}")
        print(f"   Detected Policy: {query_data['detected_policy']}")
        if query_data['expanded_terms']:
            print(f"   Expanded Terms: {', '.join(query_data['expanded_terms'][:3])}")
        
        try:
            # Generate answer (hybrid is used automatically inside rag_chain)
            response = rag_chain.generate_answer(query, top_k=5)
            total_time += response.generation_time_ms
            
            print(f"\n✅ Answer Generated:")
            print(f"   Confidence: {response.confidence:.2%}")
            print(f"   Retrieved: {response.retrieved_chunks_count} chunks")
            print(f"   Time: {response.generation_time_ms:.0f}ms ({response.generation_time_ms/1000:.1f}s)")
            print(f"\n📝 Answer:\n{response.answer}\n")
            
            if response.sources:
                print(f"📚 Sources ({len(response.sources)}):")
                for j, source in enumerate(response.sources[:3], 1):
                    page_info = f", Page {source.page}" if source.page else ""
                    print(f"   {j}. {source.filename}{page_info} (similarity: {source.similarity:.2f})")
            
            results.append({
                "query": query,
                "success": True,
                "confidence": response.confidence,
                "time_ms": response.generation_time_ms,
                "sources": len(response.sources),
                "answer": response.answer
            })
            
        except RAGError as e:
            print(f"\n❌ Error: {e}")
            results.append({
                "query": query,
                "success": False,
                "confidence": 0.0,
                "time_ms": 0,
                "sources": 0,
                "answer": None
            })
        except Exception as e:
            print(f"\n❌ Unexpected Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "query": query,
                "success": False,
                "confidence": 0.0,
                "time_ms": 0,
                "sources": 0,
                "answer": None
            })
    
    # Print summary
    print("\n" + "="*60)
    print("  📊 TEST SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r["success"])
    avg_confidence = sum(r["confidence"] for r in results if r["success"]) / max(successful, 1)
    avg_time = sum(r["time_ms"] for r in results if r["success"]) / max(successful, 1)
    avg_sources = sum(r["sources"] for r in results if r["success"]) / max(successful, 1)
    
    print(f"\n✅ Success Rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"📊 Average Confidence: {avg_confidence:.2%}")
    print(f"⚡ Average Response Time: {avg_time:.0f}ms ({avg_time/1000:.1f}s)")
    print(f"🕐 Total Time: {total_time/1000:.1f}s")
    print(f"📚 Average Sources: {avg_sources:.1f}")
    
    # Show confidence breakdown
    if successful > 0:
        print(f"\n📈 Confidence Breakdown:")
        confidences = [r["confidence"] for r in results if r["success"]]
        high_conf = sum(1 for c in confidences if c >= 0.70)
        med_conf = sum(1 for c in confidences if 0.60 <= c < 0.70)
        low_conf = sum(1 for c in confidences if c < 0.60)
        
        print(f"   High (≥70%): {high_conf}")
        print(f"   Medium (60-70%): {med_conf}")
        print(f"   Low (<60%): {low_conf}")
    
    # Print config info
    print("\n" + "="*60)
    print("  ℹ️  RAG CHAIN CONFIGURATION")
    print("="*60)
    chain_info = rag_chain.get_chain_info()
    print(f"LLM Model: {chain_info['llm_model']}")
    print(f"Embedding Model: {chain_info['embedding_model']}")
    print(f"Top-K Retrieval: {chain_info['top_k']}")
    print(f"Similarity Threshold: {chain_info['similarity_threshold']}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🧪 RAG PIPELINE COMPLETE TEST")
    print("="*60)
    
    try:
        # Step 1: Check ingested data
        check_ingested_data()
        
        # Step 2: Test RAG with queries
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
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
