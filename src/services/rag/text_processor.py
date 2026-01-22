import re
from typing import List
from loguru import logger


class TextProcessor:
    
    def __init__(self):
        # Common insurance document headers/footers patterns
        self.header_footer_patterns = [
            r"Page \d+ of \d+",
            r"^\d+$",  # Page numbers alone
            r"^[A-Z\s]{10,}$",  # All caps headers
            r"www\.[a-z0-9\-]+\.(com|in|org)",  # URLs
            r"©.*?\d{4}",  # Copyright notices
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Main text cleaning pipeline
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        # Apply cleaning steps
        text = self.normalize_whitespace(text)
        text = self.fix_hyphenation(text)
        text = self.remove_excessive_newlines(text)
        text = self.fix_bullet_points(text)
        text = self.remove_page_markers(text)
        
        return text.strip()
    
    def normalize_whitespace(self, text: str) -> str:
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Fix zero-width spaces and similar
        text = text.replace('\u200b', '')
        text = text.replace('\ufeff', '')
        
        return text
    
    def fix_hyphenation(self, text: str) -> str:
        # Pattern: word- \n word → word
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text
    
    def remove_excessive_newlines(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    
    def fix_bullet_points(self, text: str) -> str:
        # Replace various bullet characters with standard bullet
        bullet_chars = ['•', '◦', '▪', '▫', '●', '○', '■', '□', '–', '—']
        for char in bullet_chars:
            text = text.replace(f'{char} ', '• ')
        
        return text
    
    def remove_page_markers(self, text: str) -> str:
        text = re.sub(r'\n--- Page \d+ ---\n', '\n\n', text)
        return text
    
    def remove_headers_footers(self, text: str) -> str:
        """
        Remove common header/footer patterns
        (Use cautiously - might remove important content)
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Check if line matches any header/footer pattern
            is_header_footer = False
            
            for pattern in self.header_footer_patterns:
                if re.search(pattern, line.strip(), re.IGNORECASE):
                    is_header_footer = True
                    break
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_sections(self, text: str) -> List[tuple[str, str]]:
        """
        Extract document sections based on headings
        
        Returns:
            List of (section_name, section_content) tuples
        """
        sections = []
        
        # Pattern for section headings (all caps, numbered, etc.)
        heading_pattern = r'^(?:[0-9]+\.?\s+)?([A-Z][A-Z\s]{5,})$'
        
        lines = text.split('\n')
        current_section = "Introduction"
        current_content = []
        
        for line in lines:
            # Check if line is a heading
            match = re.match(heading_pattern, line.strip())
            
            if match and len(line.strip()) < 100:  # Reasonable heading length
                # Save previous section
                if current_content:
                    sections.append((
                        current_section,
                        '\n'.join(current_content).strip()
                    ))
                
                # Start new section
                current_section = match.group(1).strip().title()
                current_content = []
            else:
                current_content.append(line)
        
        # Add last section
        if current_content:
            sections.append((
                current_section,
                '\n'.join(current_content).strip()
            ))
        
        logger.info(f"Extracted {len(sections)} sections from document")
        
        return sections
    
    def split_into_sentences(self, text: str) -> List[str]:
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
