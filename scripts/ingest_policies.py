"""
Policy document ingestion pipeline
Processes PDFs from data/raw/policies/ into vector store
"""
import sys
from pathlib import Path
from typing import List, Dict
import time
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_settings
from src.services.rag.document_loader import DocumentLoader
from src.services.rag.text_processor import TextProcessor
from src.services.rag.chunker import PolicyChunker, Chunk
from src.services.rag.embedding_service import EmbeddingService
from src.services.rag.vector_store import VectorStore
from src.services.rag.semantic_chunker import SemanticPolicyChunker

settings = get_settings()
class PolicyIngestion:
    """Ingest policy documents into vector store"""
    
    def __init__(self, reset: bool = False):
        logger.info("Initializing ingestion pipeline...")
        
        self.loader = DocumentLoader()
        self.processor = TextProcessor()
        self.chunker = SemanticPolicyChunker(
            chunk_size=800,
            chunk_overlap=150
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(reset=reset)
        
        self.stats = {
            "files_processed": 0,
            "total_chunks": 0,
            "total_pages": 0,
            "insurers": set(),
            "policies": set()
        }
    
    def discover_policies(self, base_dir: str = "data/raw/policies") -> List[Path]:
        """Discover all PDF files in policy directories"""
        base_path = Path(base_dir)
        
        if not base_path.exists():
            logger.error(f"Policy directory not found: {base_dir}")
            return []
        
        # Find all PDFs recursively
        pdf_files = list(base_path.rglob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        return sorted(pdf_files)
    
    def extract_metadata_from_path(self, file_path: Path) -> Dict:
        """Extract metadata from file path structure"""
        # Path structure: data/raw/policies/{insurer}/{filename}.pdf
        parts = file_path.parts
        
        # Extract insurer from parent directory
        insurer = parts[-2] if len(parts) >= 2 else "unknown"
        
        # Clean insurer name mapping
        insurer_map = {
            "bajaj": "Bajaj Allianz",
            "bajaj_allianz": "Bajaj Allianz",
            "hdfc_ergo": "HDFC ERGO",
            "care_health": "Care Health Insurance",
            "tata_aig": "Tata AIG"
        }
        insurer_name = insurer_map.get(insurer.lower(), insurer.replace("_", " ").title())
        
        # Extract policy name and document type from filename
        filename = file_path.stem  # Without .pdf
        
        # Determine document type
        doc_type = "unknown"
        if "wording" in filename.lower():
            doc_type = "policy_wording"
        elif "brochure" in filename.lower():
            doc_type = "brochure"
        elif "prospectus" in filename.lower():
            doc_type = "prospectus"
        
        # Extract policy name (remove document type suffixes)
        policy_name = filename
        for suffix in ["_wording", "_brochure", "_prospectus"]:
            policy_name = policy_name.replace(suffix, "")
        
        # Clean policy name
        policy_name = policy_name.replace("_", " ").title()
        
        return {
            "filename": file_path.name,
            "insurer": insurer_name,
            "policy_name": policy_name,
            "document_type": doc_type,
            "file_path": str(file_path)
        }
    
    def process_document(self, file_path: Path) -> int:
        """Process a single PDF document"""
        logger.info(f"Processing: {file_path.name}")
        
        try:
            # Extract metadata
            base_metadata = self.extract_metadata_from_path(file_path)
            
            # Create unique source identifier (insurer + filename)
            unique_source = f"{base_metadata['insurer']}_{file_path.stem}"
            
            # Load document
            pages = self.loader.load_pdf_pages(str(file_path))
            
            if not pages:
                logger.warning(f"No pages extracted from {file_path.name}")
                return 0
            
            logger.info(f"  Loaded {len(pages)} pages")
            
            # Process each page
            all_chunks = []
            
            for page in pages:
                # Clean text
                cleaned_text = self.processor.clean_text(page["text"])
                
                if not cleaned_text or len(cleaned_text.strip()) < 50:
                    continue  # Skip empty/tiny pages
                
                # Create page metadata
                page_metadata = {
                    **base_metadata,
                    "page": page["page"],
                    "total_pages": len(pages)
                }
                
                # Chunk the page with unique source
                semantic_chunks = self.chunker.chunk_with_context(
                    text=cleaned_text,
                    metadata=page_metadata,
                    source=unique_source  
                )
                regular_chunks = [
                    Chunk(
                        text=sc.text,
                        metadata=sc.metadata,
                        chunk_id=sc.chunk_id,
                        source=sc.source,
                        start_char=0,
                        end_char=len(sc.text)
                    )
                    for sc in semantic_chunks
                ]
                all_chunks.extend(regular_chunks)
            
            if not all_chunks:
                logger.warning(f"No chunks created from {file_path.name}")
                return 0
            
            logger.info(f"  Created {len(all_chunks)} chunks")
            
            # Generate embeddings
            texts = [chunk.text for chunk in all_chunks]
            embeddings = self.embedding_service.embed_batch(
                texts,
                show_progress=False
            )
            
            # Add to vector store
            self.vector_store.add_chunks(all_chunks, embeddings)
            
            # Update stats
            self.stats["files_processed"] += 1
            self.stats["total_chunks"] += len(all_chunks)
            self.stats["total_pages"] += len(pages)
            self.stats["insurers"].add(base_metadata["insurer"])
            policy_name_normalized = base_metadata["policy_name"].strip().title()
            self.stats["policies"].add(policy_name_normalized)
            
            logger.info(f"  ✅ Successfully indexed {file_path.name}")
            
            return len(all_chunks)
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {file_path.name}: {e}")
            return 0

    
    def ingest_all(self, base_dir: str = "data/raw/policies") -> None:
        """Ingest all policy documents"""
        start_time = time.time()
        
        print("\n" + "="*60)
        print("  📚 POLICY DOCUMENT INGESTION")
        print("="*60)
        
        # Discover files
        pdf_files = self.discover_policies(base_dir)
        
        if not pdf_files:
            logger.error("No PDF files found!")
            return
        
        print(f"\nFound {len(pdf_files)} PDF files to process\n")
        
        # Process each file
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
            print("-" * 60)
            
            self.process_document(pdf_file)
        
        # Show final stats
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("  ✅ INGESTION COMPLETE!")
        print("="*60)
        print(f"\n📊 Statistics:")
        print(f"   Files processed: {self.stats['files_processed']}")
        print(f"   Total pages: {self.stats['total_pages']}")
        print(f"   Total chunks: {self.stats['total_chunks']}")
        print(f"   Unique insurers: {len(self.stats['insurers'])}")
        print(f"   Unique policies: {len(self.stats['policies'])}")
        print(f"   Time taken: {elapsed:.1f}s")
        
        print(f"\n📋 Insurers indexed:")
        for insurer in sorted(self.stats['insurers']):
            print(f"   • {insurer}")
        
        print(f"\n📋 Policies indexed:")
        for policy in sorted(self.stats['policies']):
            print(f"   • {policy}")
        
        # Show vector store stats
        print("\n" + "="*60)
        print("  🗄️  Vector Store Status")
        print("="*60)
        
        vs_stats = self.vector_store.get_stats()
        print(f"\nCollection: {vs_stats['collection_name']}")
        print(f"Total chunks: {vs_stats['total_chunks']}")
        print(f"Unique sources: {vs_stats['unique_sources']}")


def main():
    """Main ingestion script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest policy documents")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store before ingestion"
    )
    parser.add_argument(
        "--dir",
        default="data/raw/policies",
        help="Directory containing policy PDFs"
    )
    
    args = parser.parse_args()
    
    if args.reset:
        print("\n⚠️  WARNING: This will delete all existing data in vector store!")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
    
    # Run ingestion
    ingestion = PolicyIngestion(reset=args.reset)
    ingestion.ingest_all(base_dir=args.dir)
    
    print("\n✅ Ingestion pipeline complete!")
    print("\n📍 Next steps:")
    print("   1. Run: python scripts/verify_ingestion.py")
    print("   2. Test: python scripts/test_rag_complete.py")
    print("   3. Start API: uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
