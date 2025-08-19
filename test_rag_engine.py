import sys
import os

def check_dependencies():
    """Check if all RAG dependencies are available"""
    try:
        from rag_engine import RAGEngine
        print("✓ RAG Engine imported successfully")
    except ImportError as e:
        print(f"✗ RAG Engine import failed: {e}")
        return False
    
    try:
        from vector_store import VectorStore
        print("✓ Vector Store imported successfully")
    except ImportError as e:
        print(f"✗ Vector Store import failed: {e}")
        return False
    
    try:
        from test_pdf_processor import test_pdf_processing
        print("✓ PDF processor test imported successfully")
    except ImportError as e:
        print(f"✗ PDF processor test import failed: {e}")
        return False
    
    return True

def setup_rag_system():
    """Set up the complete RAG system with test data"""
    from vector_store import VectorStore
    from rag_engine import RAGEngine
    from test_pdf_processor import test_pdf_processing
    
    print("\n" + "="*60)
    print("SETTING UP COMPLETE RAG SYSTEM")
    print("="*60)
    
    # Initialize vector store
    print("\n1. Initializing vector store...")
    vs = VectorStore()
    
    # Reset for clean test
    print("2. Resetting database...")
    vs.reset_database()
    
    # Process PDF and add to vector store
    print("3. Processing research paper...")
    chunks, metadata = test_pdf_processing()
    
    if chunks is None:
        print("✗ Failed to process PDF")
        return None, None
    
    print(f"4. Adding {len(chunks)} chunks to vector database...")
    document_id = "research_paper_sample"
    success = vs.add_document_chunks(chunks, document_id)
    
    if not success:
        print("✗ Failed to add chunks to vector store")
        return None, None
    
    # Initialize RAG engine with environment variables
    print("5. Initializing RAG engine with Gemini API...")
    rag = RAGEngine(vs)  # Will read from environment variables
    
    print("✓ RAG system setup complete!")
    return rag, vs

def test_rag_queries():
    """Test the RAG system with various queries"""
    print("\n" + "="*60)
    print("TESTING RAG QUERY ENGINE")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("✗ Dependency check failed")
        return False
    
    # Set up RAG system
    rag, vs = setup_rag_system()
    if rag is None:
        return False
    
    # Test queries - designed for the research paper content
    test_queries = [
        "What is the main research question of this paper?",
        "What mathematical equations are presented in this research?",
        "What are the key findings or results?", 
        "How is the methodology described?",
        "What references are cited in this work?",
        "What are the implications of this research?",
        "What data or experiments are discussed?",
        "What is the conclusion of this paper?"
    ]
    
    print(f"\n6. Testing RAG queries...")
    print("="*60)
    
    successful_queries = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 QUERY {i}: {query}")
        print("-" * 50)
        
        try:
            # Execute RAG query with lower similarity threshold
            response = rag.query(query, top_k=3, min_similarity=0.0)
            
            print(f"⏱️  Response time: {response.response_time:.2f}s")
            sys.stdout.flush()  # Force output
            print(f"🎯 Confidence: {response.confidence_score:.2f}")
            sys.stdout.flush()
            print(f"🤖 Model used: {response.model_used}")
            sys.stdout.flush()
            
            print(f"\n💬 ANSWER:")
            sys.stdout.flush()
            print(response.answer)
            sys.stdout.flush()
            
            if response.sources:
                print(f"\n📚 SOURCES:")
                for j, source in enumerate(response.sources[:2], 1):  # Show top 2 sources
                    page = source['page_number']
                    section = source['section']
                    similarity = source['similarity_score']
                    preview = source['text_preview']
                    
                    print(f"  {j}. Page {page} ({section}) - Similarity: {similarity:.3f}")
                    print(f"     {preview}")
            
            if response.confidence_score > 0.3:
                successful_queries += 1
                print("✅ Query successful")
            else:
                print("⚠️  Low confidence response")
            
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Summary
    print("="*60)
    print("RAG TESTING SUMMARY")
    print("="*60)
    print(f"Total queries tested: {len(test_queries)}")
    print(f"Successful responses: {successful_queries}")
    print(f"Success rate: {(successful_queries/len(test_queries)*100):.1f}%")
    
    # System stats
    stats = rag.get_stats()
    print(f"\nSystem Configuration:")
    print(f"  Model: {stats['current_model']}")
    print(f"  Documents in DB: {stats['vector_database']['unique_documents']}")
    print(f"  Total chunks: {stats['vector_database']['total_chunks']}")
    print(f"  Pages processed: {stats['vector_database']['unique_pages']}")
    
    if successful_queries >= len(test_queries) * 0.6:  # 60% success rate
        print("\n🎉 RAG SYSTEM WORKING SUCCESSFULLY!")
        print("✅ Ready for Streamlit interface development")
        return True
    else:
        print("\n⚠️  RAG system needs improvement")
        print("❌ Check model availability and context quality")
        return False

def test_single_query():
    """Test a single query interactively"""
    print("\n" + "="*60)
    print("INTERACTIVE RAG QUERY TEST")
    print("="*60)
    
    # Set up system with token
    print("Setting up RAG system with HuggingFace token...")
    rag, vs = setup_rag_system()
    if rag is None:
        return
    
    print("\n📝 Enter your question about the research paper:")
    print("(Type 'quit' to exit)")
    
    while True:
        query = input("\n🔍 Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        try:
            response = rag.query(query)
            
            print(f"\n💬 Answer:")
            print(response.answer)
            
            print(f"\n📊 Metadata:")
            print(f"  Confidence: {response.confidence_score:.2f}")
            print(f"  Response time: {response.response_time:.2f}s")
            print(f"  Sources: {len(response.sources)}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Run automatic test
    success = test_rag_queries()
    
    # Optionally run interactive test
    if success:
        user_input = input("\n🤔 Want to try interactive queries? (y/n): ").strip().lower()
        if user_input in ['y', 'yes']:
            test_single_query()