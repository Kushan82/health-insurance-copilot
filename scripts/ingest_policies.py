"""
Enhanced Policy Document Ingestion Pipeline
Supports multi-level directory structure organized by company
"""

from pathlib import Path
from typing import List, Dict, Optional
import time
import sys
from loguru import logger

from src.core.config import get_settings
from src.services.rag.document_loader import DocumentLoader
from src.services.rag.chunker import PolicyChunker
from src.services.rag.embedding_service import EmbeddingService
from src.services.rag.vector_store import VectorStore

settings = get_settings()


class PolicyIngestionPipeline:
    """Enhanced pipeline for ingesting policy documents into vector store"""
    
    def __init__(
        self,
        source_dir: str = "./data/raw/policies",
        persist_dir: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.source_dir = Path(source_dir)
        self.persist_dir = persist_dir or settings.chroma_persist_directory
        
        # Initialize components
        self.loader = DocumentLoader()
        self.chunker = PolicyChunker(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = None
        
        # Statistics tracking
        self.stats = {
            'total_docs': 0,
            'total_chunks': 0,
            'failed_docs': 0,
            'total_chars': 0,
            'processing_time_ms': 0,
            'doc_details': [],
            'companies': {}
        }
        
        logger.info(f"📁 Ingestion pipeline initialized")
        logger.info(f"   Source: {self.source_dir}")
        logger.info(f"   Vector Store: {self.persist_dir}")
        logger.info(f"   Chunk Size: {self.chunker.chunk_size} (Overlap: {self.chunker.chunk_overlap})")
    
    def ingest_all(self, reset_store: bool = False) -> Dict:
        """Ingest all policy documents from source directory (supports subdirectories)"""
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("  🚀 STARTING POLICY INGESTION PIPELINE")
        logger.info("=" * 60)
        
        # Initialize vector store
        try:
            if reset_store:
                logger.warning("⚠️  Resetting vector store...")
                self.vector_store = VectorStore(persist_directory=self.persist_dir, reset=True)
                logger.info("✅ Vector store reset complete")
            else:
                self.vector_store = VectorStore(persist_directory=self.persist_dir)
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            return {'error': str(e)}
        
        # Find all PDFs recursively
        pdf_files = sorted(self.source_dir.rglob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"⚠️  No PDF files found in {self.source_dir}")
            return self.stats
        
        # Group by company
        companies = {}
        for pdf_path in pdf_files:
            company = pdf_path.parent.name
            if company not in companies:
                companies[company] = []
            companies[company].append(pdf_path)
        
        logger.info(f"📄 Found {len(pdf_files)} policy documents across {len(companies)} companies")
        for company, files in companies.items():
            logger.info(f"   • {company}: {len(files)} files")
        logger.info("")
        
        # Process each document
        for idx, pdf_path in enumerate(pdf_files, 1):
            company = pdf_path.parent.name
            logger.info(f"[{idx}/{len(pdf_files)}] Processing: {company}/{pdf_path.name}")
            
            try:
                doc_stats = self._ingest_document(pdf_path, company)
                self.stats['doc_details'].append(doc_stats)
                self.stats['total_docs'] += 1
                
                # Track per-company stats
                if company not in self.stats['companies']:
                    self.stats['companies'][company] = {'docs': 0, 'chunks': 0, 'chars': 0}
                self.stats['companies'][company]['docs'] += 1
                self.stats['companies'][company]['chunks'] += doc_stats['chunks']
                self.stats['companies'][company]['chars'] += doc_stats['chars']
                
                logger.info(
                    f"   ✅ {doc_stats['chunks']} chunks | "
                    f"{doc_stats['chars']:,} chars | "
                    f"{doc_stats['time_ms']:.0f}ms\n"
                )
            except Exception as e:
                logger.error(f"   ❌ Failed: {e}")
                self.stats['failed_docs'] += 1
        
        # Calculate statistics
        self.stats['processing_time_ms'] = (time.time() - start_time) * 1000
        self.stats['total_chunks'] = sum(d['chunks'] for d in self.stats['doc_details'])
        self.stats['total_chars'] = sum(d['chars'] for d in self.stats['doc_details'])
        
        self._log_summary()
        return self.stats
    
    def _ingest_document(self, pdf_path: Path, company: str) -> Dict:
        """Ingest a single document"""
        doc_start = time.time()
        
        # Step 1: Load document pages
        pages = self.loader.load_pdf_pages(str(pdf_path))
        
        if not pages:
            raise ValueError("No content extracted from PDF")
        
        # Step 2: Combine pages
        full_text = "\n\n".join([page["text"] for page in pages])
        char_count = len(full_text)
        
        # ✅ ENHANCED METADATA
        base_metadata = {
            'source': pdf_path.name,
            'filename': pdf_path.name,
            'company': company,  # Keep for backward compatibility
            'insurer': self._normalize_company_name(company),  # NEW
            'policy_name': self._extract_policy_name(pdf_path.name),  # NEW
            'document_type': self._classify_document(pdf_path.name),  # Already exists
            'total_pages': len(pages),
            'ingested_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Step 3: Chunk
        chunks_objs = self.chunker.chunk_document(
            text=full_text,
            metadata=base_metadata,
            source=f"{company}/{pdf_path.name}"
        )
        
        if not chunks_objs:
            raise ValueError("No chunks generated")
        
        # Step 4: Filter
        valid_chunk_objs = []
        for chunk_obj in chunks_objs:
            text = chunk_obj.text.strip()
            if len(text) < 50 or len(text.split()) < 10:
                continue
            alpha_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text)
            if alpha_ratio < 0.6:
                continue
            valid_chunk_objs.append(chunk_obj)
        
        # Step 5: Generate embeddings
        chunk_texts = [chunk.text for chunk in valid_chunk_objs]
        embeddings = self.embedding_service.embed_batch(chunk_texts)  # ✅ FIXED: Changed from embed_documents to embed_batch
        
        # Step 6: Store
        self.vector_store.add_chunks(chunks=valid_chunk_objs, embeddings=embeddings)
        
        processing_time = (time.time() - doc_start) * 1000
        
        return {
            'filename': pdf_path.name,
            'company': company,
            'chunks': len(valid_chunk_objs),
            'chars': char_count,
            'pages': len(pages),
            'time_ms': processing_time,
            'avg_chunk_size': char_count // len(valid_chunk_objs) if valid_chunk_objs else 0
        }
    def _normalize_company_name(self, company: str) -> str:
        """Convert folder name to display name"""
        mapping = {
            'bajaj': 'Bajaj Allianz',
            'care_health': 'Care Health Insurance',
            'hdfc_ergo': 'HDFC ERGO',
            'tata_aig': 'Tata AIG'
        }
        return mapping.get(company.lower(), company)

    def _extract_policy_name(self, filename: str) -> str:
        """Extract policy name from filename"""
        filename_lower = filename.lower()
        
        # HDFC ERGO
        if 'optima_restore' in filename_lower:
            return 'Optima Restore'
        elif 'optima_secure' in filename_lower:
            return 'Optima Secure'
        
        # Care Health
        elif 'care_supreme' in filename_lower:
            return 'Care Supreme'
        elif 'care_ultimate' in filename_lower:
            return 'Care Ultimate'
        
        # Tata AIG
        elif 'medicare_plus' in filename_lower:
            return 'Medicare Plus'
        elif 'medicare_premier' in filename_lower:
            return 'Medicare Premier'
        
        # Bajaj
        elif 'my_health_care_plan_1' in filename_lower:
            return 'My Health Care Plan 1'
        elif 'my_health_care_plan_6' in filename_lower:
            return 'My Health Care Plan 6'
        
        return 'Unknown Policy'
    
    def _classify_document(self, filename: str) -> str:
        """Classify document type"""
        filename_lower = filename.lower()
        if 'wording' in filename_lower:
            return 'policy_wording'
        elif 'prospectus' in filename_lower:
            return 'prospectus'
        elif 'brochure' in filename_lower:
            return 'brochure'
        return 'general'
    
    def _log_summary(self):
        """Log summary"""
        logger.info("\n" + "=" * 60)
        logger.info("  📊 INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successfully processed: {self.stats['total_docs']} documents")
        logger.info(f"❌ Failed: {self.stats['failed_docs']} documents")
        logger.info(f"📦 Total chunks: {self.stats['total_chunks']:,}")
        logger.info(f"📝 Total characters: {self.stats['total_chars']:,}")
        logger.info(f"⏱️  Processing time: {self.stats['processing_time_ms']/1000:.2f}s")
        
        if self.stats['total_docs'] > 0:
            avg_chunks = self.stats['total_chunks'] / self.stats['total_docs']
            avg_time = self.stats['processing_time_ms'] / self.stats['total_docs']
            logger.info(f"📊 Avg chunks/doc: {avg_chunks:.1f}")
            logger.info(f"⚡ Avg time/doc: {avg_time:.0f}ms")
        
        if self.stats['companies']:
            logger.info("\n🏢 Per-Company Statistics:")
            logger.info("-" * 60)
            for company, stats in sorted(self.stats['companies'].items()):
                logger.info(
                    f"   {company:<20} | Docs: {stats['docs']:>3} | "
                    f"Chunks: {stats['chunks']:>5} | Chars: {stats['chars']:>8,}"
                )
        
        logger.info("=" * 60 + "\n")
    
    def get_store_stats(self) -> Dict:
        """Get store stats"""
        try:
            temp_store = VectorStore(persist_directory=self.persist_dir) if self.vector_store is None else self.vector_store
            count = temp_store.collection.count()
            
            if count > 0:
                # ✅ FIX: Get ALL documents, not just 10
                results = temp_store.collection.get(
                    limit=count,  # Changed from min(10, count)
                    include=['metadatas']
                )
                companies = set()
                if results and 'metadatas' in results:
                    for metadata in results['metadatas']:
                        if metadata and 'company' in metadata:
                            companies.add(metadata['company'])
            else:
                companies = set()
            
            return {
                'total_chunks': count,
                'companies': sorted(list(companies)),
                'storage_location': self.persist_dir
            }
        except Exception as e:
            return {'error': str(e)}



def main():
    """Main function"""
    logger.info("\n" + "=" * 60)
    logger.info("  🏥 HEALTH INSURANCE POLICY INGESTION")
    logger.info("=" * 60 + "\n")
    
    reset_store = '--reset' in sys.argv
    
    pipeline = PolicyIngestionPipeline(
        source_dir="./data/raw/policies",
        persist_dir="./data/vector_store"
    )
    
    if not reset_store:
        logger.info("🔍 Checking existing vector store...")
        store_stats = pipeline.get_store_stats()
        if 'error' not in store_stats and store_stats['total_chunks'] > 0:
            print("\n⚠️  Vector store already contains data!")
            response = input("Reset and re-ingest? (yes/no): ").strip().lower()
            reset_store = response in ['yes', 'y']
    else:
        logger.info("🔄 Reset flag detected")
    
    print()
    stats = pipeline.ingest_all(reset_store=reset_store)
    
    if 'error' in stats:
        logger.error(f"\n❌ INGESTION FAILED: {stats['error']}")
        return
    
    if stats.get('total_chunks', 0) > 0:
        logger.info("🔍 Verifying ingestion...")
        final_stats = pipeline.get_store_stats()
        if 'error' not in final_stats:
            logger.info(f"\n✅ VERIFICATION SUCCESSFUL")
            logger.info(f"   • Total chunks: {final_stats['total_chunks']:,}")
            logger.info(f"   • Companies: {', '.join(final_stats['companies'])}\n")
    else:
        logger.error("\n❌ No chunks created")
    
    logger.info("=" * 60)
    logger.info("  ✅ INGESTION COMPLETE")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
