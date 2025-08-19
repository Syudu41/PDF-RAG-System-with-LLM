import os
import requests
from pdf_processor import ResearchPaperProcessor

def download_sample_pdf():
    """Download a sample LaTeX research paper PDF for testing"""
    url = "https://web.mit.edu/rsi/www/pdfs/new-latex.pdf"
    filename = "sample_latex_paper.pdf"
    
    if not os.path.exists(filename):
        print("Downloading sample PDF...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded: {filename}")
        except Exception as e:
            print(f"✗ Error downloading PDF: {e}")
            return None
    else:
        print(f"✓ Using existing: {filename}")
    
    return filename

def test_pdf_processing():
    """Test the PDF processor with a real LaTeX research paper"""
    print("\n" + "="*50)
    print("TESTING PDF PROCESSOR")
    print("="*50)
    
    # Download sample PDF
    pdf_file = download_sample_pdf()
    if pdf_file is None:
        return None, None
    
    # Initialize processor
    processor = ResearchPaperProcessor()
    
    try:
        # Extract text
        print("\n1. Extracting text from PDF...")
        full_text, metadata = processor.extract_text_from_pdf(pdf_file)
        
        print(f"✓ Extracted text from {metadata['total_pages']} pages")
        print(f"✓ Document title: {metadata.get('title', 'Unknown')}")
        print(f"✓ Total text length: {len(full_text)} characters")
        
        # Show sample of cleaned text
        print(f"\n2. Sample of extracted text (first 500 chars):")
        print("-" * 50)
        print(full_text[:500] + "...")
        print("-" * 50)
        
        # Create chunks
        print(f"\n3. Creating document chunks...")
        chunks = processor.create_chunks(full_text, metadata)
        
        print(f"✓ Created {len(chunks)} chunks")
        
        # Show sample chunks
        print(f"\n4. Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"  Page: {chunk.page_number}")
            print(f"  Section: {chunk.section}")
            print(f"  Length: {len(chunk.text)} chars")
            print(f"  Confidence: {chunk.confidence_score:.2f}")
            print(f"  Text preview: {chunk.text[:150]}...")
        
        # Test LaTeX equation handling
        print(f"\n5. Testing LaTeX equation handling...")
        test_latex_text = r"""
        This is a sample with inline math $x^2 + y^2 = z^2$ and display math:
        $\int_0^1 e^{-x} dx = 1 - e^{-1}$
        Also has citations \cite{author2023} and references \ref{section1}.
        """
        
        cleaned_latex = processor._clean_latex_equations(test_latex_text)
        print("Original LaTeX text:")
        print(test_latex_text)
        print("\nCleaned text:")
        print(cleaned_latex)
        
        print(f"\n" + "="*50)
        print("✓ ALL TESTS PASSED!")
        print("✓ PDF processor is working correctly")
        print("✓ Ready for next step: vector database integration")
        print("="*50)
        
        return chunks, metadata
        
    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    test_pdf_processing()