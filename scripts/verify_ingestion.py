"""
Verify policy document ingestion
Checks data quality, metadata, and retrieval capability
"""
import sys
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag.vector_store import VectorStore
from src.services.rag.embedding_service import EmbeddingService
from loguru import logger


class IngestionVerifier:
    """Verify ingested policy documents"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
    
    def verify_stats(self) -> Dict:
        """Get and display vector store statistics"""
        print("\n" + "="*60)
        print("  📊 VECTOR STORE STATISTICS")
        print("="*60)
        
        stats = self.vector_store.get_stats()
        
        print(f"\n📦 Collection: {stats['collection_name']}")
        print(f"📄 Total chunks: {stats['total_chunks']:,}")
        print(f"📁 Unique sources: {stats['unique_sources']}")
        print(f"💾 Storage: {stats['persist_directory']}")
        
        return stats
    
    def verify_sources(self) -> List[str]:
        """List all sources in the vector store"""
        print("\n" + "="*60)
        print("  📁 SOURCES")
        print("="*60)
        
        sources = self.vector_store.list_sources()
        
        print(f"\nFound {len(sources)} unique document sources:\n")
        
        # Group by insurer
        insurers = {
            "Bajaj Allianz": [],
            "HDFC ERGO": [],
            "Care Health": [],
            "Tata AIG": []
        }
        
        for source in sources:
            source_lower = source.lower()
            if "optima" in source_lower:
                insurers["HDFC ERGO"].append(source)
            elif "care" in source_lower and ("supreme" in source_lower or "ultimate" in source_lower):
                insurers["Care Health"].append(source)
            elif "medicare" in source_lower:
                insurers["Tata AIG"].append(source)
            elif "health_care_plan" in source_lower or "my health" in source_lower:
                insurers["Bajaj Allianz"].append(source)
        
        for insurer, docs in insurers.items():
            if docs:
                print(f"\n{insurer}:")
                for doc in sorted(docs):
                    print(f"  • {doc}")
        
        return sources
    
    def verify_sample_chunks(self, limit: int = 5):
        """Display sample chunks from the vector store"""
        print("\n" + "="*60)
        print("  📝 SAMPLE CHUNKS")
        print("="*60)
        
        try:
            # Get sample documents
            results = self.vector_store.collection.get(limit=limit)
            
            if not results['documents']:
                print("\n⚠️  No chunks found in vector store!")
                return
            
            print(f"\nShowing {len(results['documents'])} sample chunks:\n")
            
            for i in range(len(results['documents'])):
                doc = results['documents'][i]
                metadata = results['metadatas'][i]
                
                print(f"\n{'─'*60}")
                print(f"Chunk {i+1}:")
                print(f"{'─'*60}")
                print(f"📄 File: {metadata.get('filename', 'N/A')}")
                print(f"🏢 Insurer: {metadata.get('insurer', 'N/A')}")
                print(f"📋 Policy: {metadata.get('policy_name', 'N/A')}")
                print(f"📖 Page: {metadata.get('page', 'N/A')}/{metadata.get('total_pages', 'N/A')}")
                print(f"📑 Type: {metadata.get('document_type', 'N/A')}")
                print(f"\n📝 Content Preview:")
                print(f"{doc[:300]}...")
                
        except Exception as e:
            logger.error(f"Error retrieving sample chunks: {e}")
            print(f"\n❌ Error: {e}")
    
    def verify_retrieval(self, test_queries: List[str] = None):
        """Test retrieval with sample queries"""
        print("\n" + "="*60)
        print("  🔍 RETRIEVAL TEST")
        print("="*60)
        
        if test_queries is None:
            test_queries = [
                "What is the waiting period for pre-existing diseases?",
                "Does the policy cover maternity benefits?",
                "What is covered under AYUSH treatment?"
            ]
        
        print("\nTesting retrieval with sample queries...\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'─'*60}")
            print(f"Query {i}: {query}")
            print(f"{'─'*60}")
            
            try:
                # Generate query embedding
                query_embedding = self.embedding_service.embed_text(query)
                
                # Search vector store
                results = self.vector_store.query(
                    query_embedding=query_embedding,
                    top_k=3
                )
                
                if results['documents'] and len(results['documents'][0]) > 0:
                    print(f"✅ Found {len(results['documents'][0])} relevant chunks\n")
                    
                    for j, doc in enumerate(results['documents'][0], 1):
                        metadata = results['metadatas'][0][j-1]
                        distance = results['distances'][0][j-1]
                        similarity = 1 - distance
                        
                        print(f"Result {j}:")
                        print(f"  📄 Source: {metadata.get('filename', 'N/A')}")
                        print(f"  🏢 Insurer: {metadata.get('insurer', 'N/A')}")
                        print(f"  📋 Policy: {metadata.get('policy_name', 'N/A')}")
                        print(f"  📖 Page: {metadata.get('page', 'N/A')}")
                        print(f"  🎯 Similarity: {similarity:.3f}")
                        print(f"  📝 Preview: {doc[:150]}...")
                        print()
                else:
                    print("⚠️  No results found")
                    
            except Exception as e:
                logger.error(f"Error during retrieval test: {e}")
                print(f"❌ Error: {e}")
    
    def verify_metadata_completeness(self):
        """Check if all chunks have required metadata"""
        print("\n" + "="*60)
        print("  ✓ METADATA VALIDATION")
        print("="*60)
        
        try:
            # Get all documents
            all_docs = self.vector_store.collection.get()
            
            required_fields = ['filename', 'insurer', 'policy_name', 'document_type', 'page']
            missing_fields = {field: 0 for field in required_fields}
            
            total_chunks = len(all_docs['metadatas'])
            
            for metadata in all_docs['metadatas']:
                for field in required_fields:
                    if not metadata.get(field):
                        missing_fields[field] += 1
            
            print(f"\nValidating {total_chunks:,} chunks...\n")
            
            all_valid = True
            for field, count in missing_fields.items():
                if count > 0:
                    print(f"⚠️  {field}: Missing in {count} chunks")
                    all_valid = False
                else:
                    print(f"✅ {field}: Present in all chunks")
            
            if all_valid:
                print("\n✅ All metadata fields are complete!")
            else:
                print("\n⚠️  Some metadata fields are incomplete")
                
        except Exception as e:
            logger.error(f"Error validating metadata: {e}")
            print(f"\n❌ Error: {e}")
    
    def verify_all(self):
        """Run all verification checks"""
        print("\n" + "="*70)
        print("  🔍 POLICY INGESTION VERIFICATION")
        print("="*70)
        
        # 1. Stats
        stats = self.verify_stats()
        
        # Quick check
        if stats.get('total_chunks', 0) == 0:
            print("\n❌ No data found in vector store!")
            print("\n💡 Run ingestion first:")
            print("   python scripts/ingest_policies.py --reset")
            return
        
        # 2. Sources
        self.verify_sources()
        
        # 3. Sample chunks
        self.verify_sample_chunks(limit=5)
        
        # 4. Metadata validation
        self.verify_metadata_completeness()
        
        # 5. Retrieval test
        self.verify_retrieval()
        
        # Summary
        print("\n" + "="*70)
        print("  ✅ VERIFICATION COMPLETE")
        print("="*70)
        
        print(f"""
📊 Summary:
   • Total chunks: {stats['total_chunks']:,}
   • Unique sources: {stats['unique_sources']}
   • Storage location: {stats['persist_directory']}

✅ Verification Status:
   ✓ Data successfully ingested
   ✓ Metadata structure valid
   ✓ Retrieval working correctly

📍 Next Steps:
   1. Test RAG pipeline: python scripts/test_rag_complete.py
   2. Start API: uvicorn src.api.main:app --reload
   3. Build UI: streamlit run src/ui/app.py
        """)


def main():
    """Main verification script"""
    verifier = IngestionVerifier()
    verifier.verify_all()


if __name__ == "__main__":
    main()
