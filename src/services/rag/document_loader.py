from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import pymupdf  # PyMuPDF
import pdfplumber
from loguru import logger

from src.core.exceptions import RAGError


@dataclass
class Document:
    """reps loaded document"""
    content: str
    metadata: Dict[str, Any]
    source: str
    page_count: int


class DocumentLoader:
    """Load and parse PDF documents"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def load_pdf(self, pdf_path: str | Path) -> Document:
        """
        Load a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Document object with content and metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise RAGError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() not in self.supported_extensions:
            raise RAGError(f"Unsupported file type: {pdf_path.suffix}")
        
        logger.info(f"Loading PDF: {pdf_path.name}")
        
        try:
            # Extract text using PyMuPDF (faster)
            content = self._extract_text_pymupdf(pdf_path)
            
            # Get metadata
            metadata = self._extract_metadata(pdf_path)
            
            # Get page count
            with pymupdf.open(pdf_path) as doc:
                page_count = len(doc)
            
            logger.info(f"✅ Loaded {pdf_path.name} - {page_count} pages, {len(content)} characters")
            
            return Document(
                content=content,
                metadata=metadata,
                source=str(pdf_path),
                page_count=page_count
            )
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path.name}: {e}")
            raise RAGError(f"Failed to load PDF: {e}")
    
    def load_pdf_pages(self, pdf_path: str | Path) -> List[Dict[str, Any]]:
        """
        Load PDF and return list of page dicts (for ingestion pipeline)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dicts: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise RAGError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() not in self.supported_extensions:
            raise RAGError(f"Unsupported file type: {pdf_path.suffix}")
        
        logger.info(f"Loading PDF: {pdf_path.name}")
        
        try:
            pages = []
            total_chars = 0
            
            with pymupdf.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    # Extract text from page
                    text = page.get_text()
                    
                    pages.append({
                        "page": page_num,
                        "text": text
                    })
                    
                    total_chars += len(text)
            
            logger.info(
                f"✅ Loaded {pdf_path.name} - {len(pages)} pages, {total_chars} characters"
            )
            
            return pages
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path.name}: {e}")
            raise RAGError(f"Failed to load PDF: {e}")
    
    def load_multiple(self, pdf_paths: List[str | Path]) -> List[Document]:
        """
        Load multiple PDF files
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for pdf_path in pdf_paths:
            try:
                doc = self.load_pdf(pdf_path)
                documents.append(doc)
            except RAGError as e:
                logger.warning(f"Skipping {pdf_path}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(documents)}/{len(pdf_paths)} documents")
        return documents
    
    def load_directory(
        self, 
        directory: str | Path, 
        recursive: bool = True
    ) -> List[Document]:
        """
        Load all PDFs from a directory
        
        Args:
            directory: Directory path
            recursive: Search subdirectories
            
        Returns:
            List of Document objects
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise RAGError(f"Directory not found: {directory}")
        
        # Find all PDFs
        if recursive:
            pdf_files = list(directory.rglob("*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        return self.load_multiple(pdf_files)
    
    def _extract_text_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF (fast)"""
        text_parts = []
        
        with pymupdf.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                # Extract text from page
                text = page.get_text()
                
                if text.strip():
                    # Add page marker for reference
                    text_parts.append(f"\n--- Page {page_num} ---\n")
                    text_parts.append(text)
        
        return "\n".join(text_parts)
    
    def _extract_text_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber (better for tables, slower)"""
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text()
                
                if text:
                    text_parts.append(f"\n--- Page {page_num} ---\n")
                    text_parts.append(text)
        
        return "\n".join(text_parts)
    
    def _extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract PDF metadata"""
        metadata = {
            "filename": pdf_path.name,
            "file_size": pdf_path.stat().st_size,
            "file_path": str(pdf_path),
        }
        
        try:
            with pymupdf.open(pdf_path) as doc:
                # Get PDF metadata
                pdf_metadata = doc.metadata
                
                if pdf_metadata:
                    metadata.update({
                        "title": pdf_metadata.get("title", ""),
                        "author": pdf_metadata.get("author", ""),
                        "subject": pdf_metadata.get("subject", ""),
                        "creator": pdf_metadata.get("creator", ""),
                    })
                
                # Add page count
                metadata["page_count"] = len(doc)
        
        except Exception as e:
            logger.warning(f"Could not extract metadata from {pdf_path.name}: {e}")
        
        return metadata
    
    def extract_tables(self, pdf_path: str | Path) -> List[List[List[str]]]:
        """
        Extract tables from PDF (useful for coverage tables)
        
        Returns:
            List of tables, where each table is a list of rows
        """
        pdf_path = Path(pdf_path)
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
            
            logger.info(f"Extracted {len(tables)} tables from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
        
        return tables
