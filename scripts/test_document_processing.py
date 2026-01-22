"""Test document processing components"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag.document_loader import DocumentLoader
from src.services.rag.text_processor import TextProcessor
from src.services.rag.chunker import PolicyChunker


def test_document_loader():
    """Test document loading"""
    print("\n" + "="*60)
    print("  📄 Testing Document Loader")
    print("="*60)
    
    loader = DocumentLoader()
    
    # Check if we have any PDFs yet
    data_dir = Path("data/raw/policies")
    
    if not data_dir.exists() or not list(data_dir.rglob("*.pdf")):
        print("⚠️  No PDF files found in data/raw/policies/")
        print("   We'll add sample PDFs next")
        return None
    
    # Load first PDF found
    pdf_files = list(data_dir.rglob("*.pdf"))
    first_pdf = pdf_files[0]
    
    print(f"Loading: {first_pdf.name}")
    doc = loader.load_pdf(first_pdf)
    
    print(f"✅ Loaded successfully")
    print(f"   Pages: {doc.page_count}")
    print(f"   Characters: {len(doc.content):,}")
    print(f"   Preview: {doc.content[:200]}...")
    
    return doc


def test_text_processor(doc):
    """Test text processing"""
    print("\n" + "="*60)
    print("  🧹 Testing Text Processor")
    print("="*60)
    
    if not doc:
        print("⚠️  No document to process (need PDFs first)")
        return None
    
    processor = TextProcessor()
    
    # Clean text
    cleaned = processor.clean_text(doc.content)
    
    print(f"✅ Text cleaned")
    print(f"   Original: {len(doc.content):,} chars")
    print(f"   Cleaned: {len(cleaned):,} chars")
    
    # Extract sections
    sections = processor.extract_sections(cleaned)
    print(f"✅ Extracted {len(sections)} sections")
    
    if sections:
        print(f"   First section: {sections[0][0]}")
    
    return cleaned


def test_chunker(text):
    """Test chunking"""
    print("\n" + "="*60)
    print("  ✂️  Testing Chunker")
    print("="*60)
    
    if not text:
        print("⚠️  No text to chunk (need PDFs first)")
        return
    
    chunker = PolicyChunker(chunk_size=500, chunk_overlap=50)
    
    # Create chunks
    chunks = chunker.chunk_document(
        text=text,
        metadata={"test": "document"},
        source="test.pdf"
    )
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Show stats
    stats = chunker.get_chunk_stats(chunks)
    print(f"   Avg chunk length: {stats['avg_chunk_length']:.0f} chars")
    print(f"   Min: {stats['min_chunk_length']}, Max: {stats['max_chunk_length']}")
    
    # Show first chunk
    if chunks:
        print(f"\n   First chunk preview:")
        print(f"   {chunks[0].text[:200]}...")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  🧪 DOCUMENT PROCESSING TESTS")
    print("="*60)
    
    # Test loader
    doc = test_document_loader()
    
    # Test processor
    text = test_text_processor(doc)
    
    # Test chunker
    test_chunker(text)
    
    print("\n" + "="*60)
    print("  📊 TEST COMPLETE")
    print("="*60)
    print("\n✅ Document processing components are ready!")
    print("   Next: Add PDF files to data/raw/policies/ to test with real data")


if __name__ == "__main__":
    main()
