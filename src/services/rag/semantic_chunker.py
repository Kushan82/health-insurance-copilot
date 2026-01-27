from typing import List, Dict, Optional
import re
from dataclasses import dataclass
from loguru import logger

@dataclass
class SemanticChunk:
    text: str
    metadata: dict
    chunk_id: str
    source: str
    section_type: str  # NEW: "coverage", "exclusion", "definition", etc.
    importance_score: float  # NEW: 0-1 score

class SemanticPolicyChunker:
    """Enhanced chunker that understands policy document structure"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Policy-specific section patterns
        self.section_patterns = {
            'coverage': r'(coverage|benefits?|what is covered|scope)',
            'exclusion': r'(exclusion|not covered|limitations?)',
            'definition': r'(definition|means|refers to)',
            'waiting_period': r'(waiting period|eligibility)',
            'claim': r'(claim procedure|how to claim)',
            'premium': r'(premium|payment|cost)',
            'terms': r'(terms? and conditions?|general terms?)'
        }
    
    def identify_section_type(self, text: str) -> tuple[str, float]:
        """
        Identify section type and importance
        Returns: (section_type, importance_score)
        """
        text_lower = text.lower()
        
        # Check for section headers
        for section_type, pattern in self.section_patterns.items():
            if re.search(pattern, text_lower[:200]):  # Check first 200 chars
                # Higher importance for coverage, exclusion, waiting_period
                importance = 1.0 if section_type in ['coverage', 'exclusion', 'waiting_period'] else 0.7
                return section_type, importance
        
        return 'general', 0.5
    
    def chunk_with_context(
        self, 
        text: str, 
        metadata: dict, 
        source: str
    ) -> List[SemanticChunk]:
        """
        Chunk text while preserving semantic boundaries
        """
        chunks = []
        
        # Split by major sections first (preserving structure)
        sections = self._split_into_sections(text)
        
        for section_idx, section in enumerate(sections):
            # Identify section type
            section_type, importance = self.identify_section_type(section)
            
            # Further split if section is too large
            if len(section) > self.chunk_size:
                sub_chunks = self._split_large_section(section)
            else:
                sub_chunks = [section]
            
            # Create chunks with enhanced metadata
            for chunk_idx, chunk_text in enumerate(sub_chunks):
                chunk = SemanticChunk(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        'section_type': section_type,
                        'importance_score': importance,
                        'chunk_index': chunk_idx,
                        'section_index': section_idx
                    },
                    chunk_id=f"{source}_p{metadata.get('page', 0)}_s{section_idx}_c{chunk_idx}",
                    source=source,
                    section_type=section_type,
                    importance_score=importance
                )
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} semantic chunks from {source}")
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text at natural boundaries (headers, paragraphs)"""
        # Split on double newlines or section markers
        sections = re.split(r'\n{2,}|\n(?=[A-Z][^a-z]{3,})', text)
        return [s.strip() for s in sections if s.strip()]
    
    def _split_large_section(self, section: str) -> List[str]:
        """Split large sections with overlap"""
        chunks = []
        start = 0
        
        while start < len(section):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(section):
                # Find last period within chunk
                last_period = section.rfind('.', start, end)
                if last_period > start + self.chunk_size // 2:
                    end = last_period + 1
            
            chunks.append(section[start:end].strip())
            start = end - self.chunk_overlap
        
        return chunks
