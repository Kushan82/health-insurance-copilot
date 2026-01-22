"""Test RAG components: embeddings, vector store, retrieval"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag.embedding_service import EmbeddingService
from src.services.rag.vector_store import VectorStore
from src.services.rag.retriever import Retriever
from src.services.rag.chunker import Chunk


def test_embedding_service():
    """Test embedding generation"""
    print("\n" + "="*60)
    print("  🧮 Testing Embedding Service")
    print("="*60)
    
    embedding_service = EmbeddingService()
    
    # Test single embedding
    text = "Health insurance covers medical expenses"
    embedding = embedding_service.embed_text(text)
    
    print(f"✅ Single embedding generated")
    print(f"   Dimension: {len(embedding)}")
    print(f"   Shape: {embedding.shape}")
    print(f"   Sample values: {embedding[:5]}")
    
    # Test batch embedding
    texts = [
        "Health insurance policy",
        "Life insurance coverage",
        "Car insurance premium"
    ]
    embeddings = embedding_service.embed_batch(texts, show_progress=False)
    
    print(f"✅ Batch embeddings generated")
    print(f"   Count: {len(embeddings)}")
    print(f"   Shape: {embeddings.shape}")
    
    # Test similarity
    similarity = embedding_service.compute_similarity(embeddings[0], embeddings[1])
    print(f"✅ Similarity computed: {similarity:.4f}")
    
    return embedding_service


def test_vector_store(embedding_service):
    """Test vector store operations"""
    print("\n" + "="*60)
    print("  💾 Testing Vector Store")
    print("="*60)
    
    # Initialize with reset
    vector_store = VectorStore(reset=True)
    
    # Create sample chunks
    sample_chunks = [
        Chunk(
            text="Health insurance provides financial protection against medical expenses.",
            metadata={"filename": "sample1.pdf", "page": 1},
            chunk_id="chunk_1",
            source="sample1.pdf",
            start_char=0,
            end_char=100
        ),
        Chunk(
            text="Coverage includes hospitalization, surgery, and outpatient care.",
            metadata={"filename": "sample1.pdf", "page": 2},
            chunk_id="chunk_2",
            source="sample1.pdf",
            start_char=100,
            end_char=200
        ),
        Chunk(
            text="Premiums are calculated based on age, health conditions, and coverage amount.",
            metadata={"filename": "sample2.pdf", "page": 1},
            chunk_id="chunk_3",
            source="sample2.pdf",
            start_char=0,
            end_char=100
        )
    ]
    
    # Generate embeddings
    texts = [chunk.text for chunk in sample_chunks]
    embeddings = embedding_service.embed_batch(texts, show_progress=False)
    
    # Add to vector store
    vector_store.add_chunks(sample_chunks, embeddings)
    
    print(f"✅ Added chunks to vector store")
    
    # Get stats
    stats = vector_store.get_stats()
    print(f"✅ Vector store stats:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Unique sources: {stats['unique_sources']}")
    print(f"   Sources: {stats['sources']}")
    
    return vector_store


def test_retriever(embedding_service, vector_store):
    """Test retrieval"""
    print("\n" + "="*60)
    print("  🔍 Testing Retriever")
    print("="*60)
    
    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store
    )
    
    # Test query
    query = "What does health insurance cover?"
    results = retriever.retrieve(query, top_k=2)
    
    print(f"✅ Retrieved {len(results)} chunks for query: '{query}'")
    
    for i, chunk in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"   Similarity: {chunk.similarity:.4f}")
        print(f"   Source: {chunk.source}")
        print(f"   Text: {chunk.text[:100]}...")
    
    # Test with filters
    print(f"\n   Testing filtered retrieval...")
    filtered_results = retriever.retrieve(
        query="insurance premiums",
        top_k=2,
        filters={"filename": "sample2.pdf"}
    )
    
    print(f"✅ Filtered retrieval: {len(filtered_results)} chunks from sample2.pdf")
    
    # Get stats
    stats = retriever.get_retriever_stats()
    print(f"\n✅ Retriever stats:")
    print(f"   Embedding model: {stats['embedding_model']['model_name']}")
    print(f"   Total docs in store: {stats['vector_store']['total_chunks']}")


def main():
    """Run all RAG component tests"""
    print("\n" + "="*60)
    print("  🧪 RAG COMPONENTS TEST")
    print("="*60)
    
    # Test embedding service
    embedding_service = test_embedding_service()
    
    # Test vector store
    vector_store = test_vector_store(embedding_service)
    
    # Test retriever
    test_retriever(embedding_service, vector_store)
    
    print("\n" + "="*60)
    print("  ✅ ALL RAG COMPONENTS WORKING!")
    print("="*60)
    print("\n   Next: Build complete RAG chain and add real policy PDFs")


if __name__ == "__main__":
    main()
