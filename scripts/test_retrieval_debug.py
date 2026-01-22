"""Debug retrieval to check similarity scores"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag.embedding_service import EmbeddingService
from src.services.rag.vector_store import VectorStore
from src.services.rag.chunker import PolicyChunker, Chunk
from src.services.rag.text_processor import TextProcessor

def test_retrieval():
    # Setup
    embedding_service = EmbeddingService()
    vector_store = VectorStore(reset=True)
    chunker = PolicyChunker()
    
    # Add test data
    test_text = """
    Waiting Period: Pre-existing diseases covered after 24 months.
    Coverage Amount: Rs 5 lakhs to Rs 3 crores.
    Maternity benefits available after 4 years.
    """
    
    chunk = Chunk(
        text=test_text,
        metadata={"test": "data"},
        chunk_id="test_1",
        source="test.pdf",
        start_char=0,
        end_char=len(test_text)
    )
    
    embedding = embedding_service.embed_text(test_text)
    vector_store.add_chunks([chunk], embedding.reshape(1, -1))
    
    # Test query
    query = "What is the waiting period?"
    query_embedding = embedding_service.embed_text(query)
    
    # Query with NO threshold
    results = vector_store.query(query_embedding, top_k=1)
    
    print(f"\n{'='*60}")
    print("Query:", query)
    print(f"{'='*60}")
    
    if results['distances'][0]:
        distance = results['distances'][0][0]
        similarity = 1 - distance
        print(f"Distance: {distance:.4f}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Retrieved: {results['documents'][0][0][:100]}")
        
        if similarity < 0.7:
            print(f"\n⚠️  Similarity {similarity:.2f} is below threshold 0.7")
            print("   Recommendation: Lower SIMILARITY_THRESHOLD in .env to 0.3-0.5")
    else:
        print("❌ No results returned")

if __name__ == "__main__":
    test_retrieval()
