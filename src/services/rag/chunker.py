from typing import List, Optional
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.core.config import get_settings
from src.services.rag.text_processor import TextProcessor

settings = get_settings()


@dataclass
class Chunk:
    text: str
    metadata: dict
    chunk_id: str
    source: str
    start_char: int
    end_char: int


class PolicyChunker:
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.text_processor = TextProcessor()
        
        # Initialize LangChain text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(
            f"Chunker initialized: size={self.chunk_size}, overlap={self.chunk_overlap}"
        )
    
    def chunk_document(
        self,
        text: str,
        metadata: dict,
        source: str
    ) -> List[Chunk]:
        """
        Chunk a document using recursive character splitting
        
        Args:
            text: Document text
            metadata: Document metadata
            source: Source identifier (file path)
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for chunking: {source}")
            return []
        
        # Clean text first
        text = self.text_processor.clean_text(text)
        
        # Split into chunks
        text_chunks = self.splitter.split_text(text)
        
        # Create Chunk objects
        chunks = []
        char_position = 0
        
        # Extract page number from metadata if available
        page_num = metadata.get("page", 0)
        
        for i, chunk_text in enumerate(text_chunks):
            # Find actual position in original text
            start_char = text.find(chunk_text[:50], char_position)
            if start_char == -1:
                start_char = char_position
            
            end_char = start_char + len(chunk_text)
            
            # Generate UNIQUE chunk ID with page number
            chunk_id = f"{source}_page{page_num}_chunk{i}"
            
            # Create chunk with metadata
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks)
                },
                chunk_id=chunk_id,
                source=source,
                start_char=start_char,
                end_char=end_char
            )
            
            chunks.append(chunk)
            char_position = end_char
        
        logger.info(f"Created {len(chunks)} chunks from {source}")
        
        return chunks

    def chunk_multiple_documents(
        self,
        documents: List[tuple[str, dict, str]]
    ) -> List[Chunk]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of (text, metadata, source) tuples
            
        Returns:
            List of all chunks
        """
        all_chunks = []
        
        for text, metadata, source in documents:
            chunks = self.chunk_document(text, metadata, source)
            all_chunks.extend(chunks)
        
        logger.info(
            f"Created {len(all_chunks)} total chunks from {len(documents)} documents"
        )
        
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> dict:
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "total_characters": sum(chunk_lengths)
        }
