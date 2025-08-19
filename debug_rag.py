import os
from dotenv import load_dotenv

load_dotenv()

# Use environment variables
hf_token = os.getenv('HF_TOKEN')
gemini_key = os.getenv('GEMINI_API_KEY')

#!/usr/bin/env python3
"""
Simple debug test for RAG system
"""
import os
from dotenv import load_dotenv

def test_basic_rag():
    """Simple test with verbose output"""
    print("Starting RAG debug test...")
    
    try:
        # Import components
        print("1. Importing components...")
        from vector_store import VectorStore
        from rag_engine import RAGEngine
        from test_pdf_processor import test_pdf_processing
        print("   ✓ Imports successful")
        
        # Set up vector store
        print("2. Setting up vector store...")
        vs = VectorStore()
        print("   ✓ Vector store initialized")
        
        # Process PDF
        print("3. Processing PDF...")
        chunks, metadata = test_pdf_processing()
        if chunks is None:
            print("   ✗ PDF processing failed")
            return False
        print(f"   ✓ Got {len(chunks)} chunks")
        
        # Add to vector store
        print("4. Adding chunks to vector store...")
        vs.reset_database()
        success = vs.add_document_chunks(chunks, "test_doc")
        if not success:
            print("   ✗ Failed to add chunks")
            return False
        print("   ✓ Chunks added successfully")
        
        # Initialize RAG with token
        print("5. Initializing RAG engine...")
        hf_token = os.getenv('HF_TOKEN')
        rag = RAGEngine(vs, hf_token=hf_token)
        print("   ✓ RAG engine initialized")
        
        # Test simple query with debugging
        print("6. Testing simple query...")
        query = "What is this paper about?"
        print(f"   Query: {query}")
        
        # First, let's see what chunks we get
        print("   6a. Getting raw chunks...")
        raw_chunks = vs.search_similar_chunks(query, n_results=3)
        for i, chunk in enumerate(raw_chunks):
            print(f"      Chunk {i+1}: similarity={chunk['similarity_score']:.3f}, page={chunk['metadata']['page_number']}")
        
        response = rag.query(query, top_k=3, min_similarity=0.0)  # Accept any similarity
        
        print(f"7. Results:")
        print(f"   Response time: {response.response_time:.2f}s")
        print(f"   Confidence: {response.confidence_score:.3f}")
        print(f"   Sources found: {len(response.sources)}")
        print(f"   Answer length: {len(response.answer)} chars")
        
        print(f"\n8. Answer:")
        print("-" * 50)
        print(response.answer)
        print("-" * 50)
        
        if response.sources:
            print(f"\n9. Sources:")
            for i, source in enumerate(response.sources[:2]):
                print(f"   Source {i+1}: Page {source['page_number']}, {source['section']}")
                print(f"   Similarity: {source['similarity_score']:.3f}")
                print(f"   Preview: {source['text_preview'][:100]}...")
                print()
        
        print("✓ RAG test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_rag()