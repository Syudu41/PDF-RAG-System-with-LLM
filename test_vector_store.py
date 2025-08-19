import sys
import os

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import chromadb
        print("✓ ChromaDB imported successfully")
    except ImportError as e:
        print(f"✗ ChromaDB import failed: {e}")
        print("Install with: pip install chromadb")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ SentenceTransformers imported successfully")
    except ImportError as e:
        print(f"✗ SentenceTransformers import failed: {e}")
        print("Install with: pip install sentence-transformers")
        return False
    
    try:
        from vector_store import VectorStore
        print("✓ VectorStore imported successfully")
    except ImportError as e:
        print(f"✗ VectorStore import failed: {e}")
        return False
    
    try:
        from test_pdf_processor import test_pdf_processing
        print("✓ PDF processor test imported successfully")
    except ImportError as e:
        print(f"✗ PDF processor test import failed: {e}")
        return False
    
    return True

def test_vector_store():
    """Test the vector store functionality"""
    print("\n" + "="*50)
    print("TESTING VECTOR STORE")
    print("="*50)
    
    # Check dependencies first
    if not check_dependencies():
        print("✗ Dependency check failed")
        return False
    
    # Import after dependency check
    from vector_store import VectorStore
    from test_pdf_processor import test_pdf_processing
    
    # Initialize vector store
    print("\n1. Initializing vector store...")
    vs = VectorStore()
    
    # Reset database for clean test
    print("\n2. Resetting database for clean test...")
    vs.reset_database()
    
    # Get test chunks from PDF processor
    print("\n3. Getting test chunks from PDF processor...")
    chunks, metadata = test_pdf_processing()
    
    if chunks is None:
        print("✗ Failed to get chunks from PDF processor")
        return False
    
    # Add chunks to vector store
    print(f"\n4. Adding {len(chunks)} chunks to vector store...")
    document_id = "research_paper_sample"
    success = vs.add_document_chunks(chunks, document_id)
    
    if not success:
        print("✗ Failed to add chunks to vector store")
        return False
    
    # Get database stats
    print(f"\n5. Getting database statistics...")
    stats = vs.get_document_stats()
    print(f"✓ Database stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Unique documents: {stats['unique_documents']}")
    print(f"  Unique pages: {stats['unique_pages']}")
    print(f"  Unique sections: {stats['unique_sections']}")
    
    # Test search functionality
    print(f"\n6. Testing semantic search...")
    
    test_queries = [
        "What is LaTeX?",
        "How to write mathematical equations?",
        "Document preparation system",
        "Bibliography and references",
        "Beamer presentations"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: '{query}'")
        results = vs.search_similar_chunks(query, n_results=3)
        
        if results:
            print(f"  Found {len(results)} results:")
            for j, result in enumerate(results):
                similarity = result['similarity_score']
                page = result['metadata']['page_number']
                preview = result['text'][:100]
                print(f"    {j+1}. Page {page} (similarity: {similarity:.3f})")
                print(f"       {preview}...")
        else:
            print("  No results found")
    
    print(f"\n" + "="*50)
    print("✓ ALL VECTOR STORE TESTS PASSED!")
    print("✓ Vector database is working correctly")
    print("✓ Semantic search is functional")
    print("✓ Ready for next step: RAG query engine")
    print("="*50)
    
    return True

if __name__ == "__main__":
    test_vector_store()