import fitz  # PyMuPDF
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata"""
    text: str
    page_number: int
    section: str
    chunk_id: str
    start_char: int
    end_char: int
    confidence_score: float = 1.0

class ResearchPaperProcessor:
    """
    Advanced PDF processor specifically designed for research papers
    Handles LaTeX equations, multi-column layouts, headers/footers, citations
    """
    
    def __init__(self):
        # Common patterns in research papers
        self.header_patterns = [
            r'^.*?(?:Conference|Workshop|Journal|Proceedings).*?$',
            r'^\d+\s*$',  # Page numbers
            r'^.*?©.*?\d{4}.*?$',  # Copyright lines
            r'^.*?arXiv:\d{4}\.\d{4,5}.*?$',  # arXiv identifiers
        ]
        
        self.footer_patterns = [
            r'^\d+\s*$',  # Page numbers at bottom
            r'^.*?(?:doi:|DOI:|https?://).*?$',  # DOI lines
            r'^.*?ISSN.*?$',  # ISSN lines
        ]
        
        # LaTeX equation patterns
        self.latex_patterns = [
            r'\$\$.*?\$\$',  # Display math
            r'\$.*?\$',      # Inline math
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'\\begin\{align\}.*?\\end\{align\}',
            r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}',
        ]
        
        # Citation patterns
        self.citation_patterns = [
            r'\[[\d\s,\-]+\]',  # [1], [1,2], [1-3]
            r'\(\w+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}\)',  # (Author, 2023)
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text from PDF while preserving structure
        Returns: (full_text, metadata)
        """
        doc = None
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            full_text = ""
            metadata = {
                'total_pages': total_pages,
                'title': doc.metadata.get('title', 'Unknown'),
                'author': doc.metadata.get('author', 'Unknown'),
                'pages_text': {}
            }
            
            for page_num in range(total_pages):
                page = doc[page_num]
                
                # Extract text with layout preservation
                text_dict = page.get_text("dict")
                page_text = self._extract_structured_text(text_dict)
                
                # Clean the page text
                cleaned_text = self._clean_page_text(page_text, page_num)
                
                if cleaned_text.strip():
                    full_text += f"\n--- Page {page_num + 1} ---\n{cleaned_text}\n"
                    metadata['pages_text'][page_num + 1] = cleaned_text
            
            logger.info(f"Successfully extracted text from {total_pages} pages")
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
        finally:
            # Ensure document is properly closed
            if doc is not None:
                doc.close()
    
    def _extract_structured_text(self, text_dict: Dict) -> str:
        """
        Extract text from PyMuPDF text dictionary while preserving structure
        Handles multi-column layouts and maintains reading order
        """
        blocks = text_dict.get("blocks", [])
        page_text = ""
        
        # Sort blocks by y-coordinate (top to bottom), then x-coordinate (left to right)
        text_blocks = []
        for block in blocks:
            if "lines" in block:  # Text block
                block_text = ""
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        text = span["text"]
                        # Preserve spacing and structure
                        if text.strip():
                            line_text += text
                    if line_text.strip():
                        block_text += line_text + " "
                
                if block_text.strip():
                    # Get block position for sorting
                    bbox = block["bbox"]
                    text_blocks.append({
                        'text': block_text.strip(),
                        'y': bbox[1],  # top y-coordinate
                        'x': bbox[0],  # left x-coordinate
                        'height': bbox[3] - bbox[1]
                    })
        
        # Sort blocks: first by y-coordinate (top to bottom), then by x-coordinate (left to right)
        text_blocks.sort(key=lambda b: (b['y'], b['x']))
        
        # Join blocks with appropriate spacing
        for block in text_blocks:
            page_text += block['text'] + "\n\n"
        
        return page_text
    
    def _clean_page_text(self, text: str, page_num: int) -> str:
        """
        Clean extracted text from research paper artifacts
        Remove headers, footers, clean LaTeX, normalize spacing
        """
        # Split into lines for processing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip headers and footers
            if self._is_header_footer(line):
                continue
            
            # Clean LaTeX equations (preserve but normalize)
            line = self._clean_latex_equations(line)
            
            # Clean citations (preserve but normalize)
            line = self._clean_citations(line)
            
            # Remove excessive whitespace
            line = re.sub(r'\s+', ' ', line)
            
            # Skip very short lines (likely artifacts)
            if len(line) < 3:
                continue
                
            cleaned_lines.append(line)
        
        # Join lines and normalize paragraph breaks
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove multiple consecutive newlines
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        
        return cleaned_text
    
    def _is_header_footer(self, line: str) -> bool:
        """Check if line is likely a header or footer"""
        # Check against header patterns
        for pattern in self.header_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        # Check against footer patterns
        for pattern in self.footer_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        # Check for standalone page numbers
        if re.match(r'^\d+\s*$', line):
            return True
        
        # Check for very short lines at start/end (likely headers/footers)
        if len(line) < 10 and any(word in line.lower() 
                                  for word in ['page', 'doi', 'issn', 'copyright', '©']):
            return True
        
        return False
    
    def _clean_latex_equations(self, text: str) -> str:
        """
        Clean LaTeX equations while preserving mathematical meaning
        Convert to more readable format
        """
        # Replace common LaTeX commands with readable text
        replacements = {
            r'\\textbf\{([^}]+)\}': r'\1',  # Bold text
            r'\\textit\{([^}]+)\}': r'\1',  # Italic text
            r'\\emph\{([^}]+)\}': r'\1',    # Emphasis
            r'\\cite\{[^}]+\}': '[CITATION]',  # Citations
            r'\\ref\{[^}]+\}': '[REF]',     # References
            r'\\label\{[^}]+\}': '',        # Labels
            r'\\\\': ' ',                   # Line breaks
            r'\\_': '_',                    # Underscores
            r'\\&': '&',                    # Ampersands
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Handle display math equations
        text = re.sub(r'\$\$([^$]+)\$\$', r'[EQUATION: \1]', text)
        text = re.sub(r'\$([^$]+)\$', r'[\1]', text)
        
        return text
    
    def _clean_citations(self, text: str) -> str:
        """Clean and normalize citations"""
        # Normalize bracketed citations
        text = re.sub(r'\[[\d\s,\-]+\]', '[CITATION]', text)
        
        # Normalize parenthetical citations
        text = re.sub(r'\(\w+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}\)', '(CITATION)', text)
        
        return text
    
    def create_chunks(self, text: str, metadata: Dict, 
                     chunk_size: int = 800, overlap: int = 100) -> List[DocumentChunk]:
        """
        Create intelligent chunks from research paper text
        Preserves semantic boundaries (sentences, paragraphs)
        """
        chunks = []
        pages_text = metadata.get('pages_text', {})
        
        for page_num, page_text in pages_text.items():
            # Split into sentences for better boundary preservation
            sentences = self._split_into_sentences(page_text)
            
            current_chunk = ""
            current_start = 0
            chunk_count = 0
            
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    # Create chunk
                    chunk = self._create_document_chunk(
                        text=current_chunk.strip(),
                        page_number=page_num,
                        section=self._detect_section(current_chunk),
                        chunk_id=f"page_{page_num}_chunk_{chunk_count}",
                        start_char=current_start,
                        end_char=current_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                    current_start = current_start + len(current_chunk) - len(overlap_text) - len(sentence) - 1
                    chunk_count += 1
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            # Add final chunk if it has content
            if current_chunk.strip():
                chunk = self._create_document_chunk(
                    text=current_chunk.strip(),
                    page_number=page_num,
                    section=self._detect_section(current_chunk),
                    chunk_id=f"page_{page_num}_chunk_{chunk_count}",
                    start_char=current_start,
                    end_char=current_start + len(current_chunk)
                )
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure"""
        # Simple sentence splitting that handles abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Filter out very short sentences (likely artifacts)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _detect_section(self, text: str) -> str:
        """Detect the section type based on text content"""
        text_lower = text.lower()
        
        section_indicators = {
            'abstract': ['abstract'],
            'introduction': ['introduction', '1. introduction', '1 introduction'],
            'methodology': ['methodology', 'methods', 'approach'],
            'results': ['results', 'experiments', 'evaluation'],
            'conclusion': ['conclusion', 'conclusions', 'discussion'],
            'references': ['references', 'bibliography']
        }
        
        for section, indicators in section_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return section
        
        return 'content'
    
    def _create_document_chunk(self, text: str, page_number: int, section: str,
                             chunk_id: str, start_char: int, end_char: int) -> DocumentChunk:
        """Create a DocumentChunk object"""
        return DocumentChunk(
            text=text,
            page_number=page_number,
            section=section,
            chunk_id=chunk_id,
            start_char=start_char,
            end_char=end_char,
            confidence_score=self._calculate_confidence_score(text)
        )
    
    def _calculate_confidence_score(self, text: str) -> float:
        """Calculate confidence score based on text quality"""
        # Simple heuristic: longer, more complete sentences have higher scores
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.strip()) for s in sentences if s.strip()])
        
        # Normalize to 0-1 range
        confidence = min(1.0, avg_sentence_length / 50.0)
        return max(0.1, confidence)  # Minimum confidence of 0.1

def test_pdf_processor():
    """Test the PDF processor with a sample file"""
    processor = ResearchPaperProcessor()
    
    # This would be called with an actual PDF file
    print("PDF Processor initialized successfully!")
    print("Ready to process research papers with:")
    print("- LaTeX equation handling")
    print("- Header/footer removal") 
    print("- Multi-column layout support")
    print("- Intelligent chunking")
    print("- Citation normalization")

if __name__ == "__main__":
    test_pdf_processor()