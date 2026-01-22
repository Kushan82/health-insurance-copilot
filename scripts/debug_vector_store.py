"""Debug script to check what's actually in the vector store"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag.vector_store import VectorStore

# Initialize vector store
vector_store = VectorStore(reset=False)

# Get collection
collection = vector_store.collection

# Check count
count = collection.count()
print(f"\n📦 Total chunks in vector store: {count}")

# Get all data
if count > 0:
    all_data = collection.get(
        limit=min(count, 100),  # Get up to 100 samples
        include=["metadatas", "documents"]
    )
    
    print(f"\n📁 Sample sources:")
    sources = set()
    for metadata in all_data["metadatas"]:
        if "filename" in metadata:
            sources.add(metadata["filename"])
    
    for source in sorted(sources):
        print(f"   • {source}")
    
    print(f"\n📊 Total unique sources: {len(sources)}")
else:
    print("\n❌ Vector store is EMPTY!")
    print("\n🔍 Possible causes:")
    print("   1. Ingestion didn't run with --reset flag")
    print("   2. ChromaDB persistence issue")
    print("   3. Path to vector_store is different")
    
    print("\n💡 Try:")
    print("   python scripts/ingest_policies.py --reset")
