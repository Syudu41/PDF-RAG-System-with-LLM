import streamlit as st
import os
import tempfile
import requests
from vector_store import VectorStore
from pdf_processor import ResearchPaperProcessor
from rag_engine import RAGEngine

# PDF Viewer import
try:
    import streamlit_pdf_viewer
    PDF_VIEWER_AVAILABLE = True
except ImportError:
    PDF_VIEWER_AVAILABLE = False

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Use environment variables
hf_token = os.getenv('HF_TOKEN')
gemini_key = os.getenv('GEMINI_API_KEY')

# Page configuration
st.set_page_config(
    page_title="PDF RAG System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'current_document' not in st.session_state:
    st.session_state.current_document = None
if 'current_pdf_content' not in st.session_state:
    st.session_state.current_pdf_content = None

def initialize_system():
    """Initialize the RAG system"""
    if st.session_state.vector_store is None:
        with st.spinner("Initializing system..."):
            st.session_state.vector_store = VectorStore()
            st.session_state.rag_engine = RAGEngine(st.session_state.vector_store)
        return True
    return False

def download_sample_pdf(doc_choice):
    """Download sample PDF and return the file content - using working URLs"""
    sample_papers = {
        "doc1": {
            "url": "https://pascalmichaillat.org/a.pdf",
            "name": "Economic Research Paper",
            "filename": "economics_research.pdf",
            "description": "Economic research - simple formatting"
        },
        "doc2": {
            "url": "https://jmlr.org/papers/volume3/blei03a/blei03a.pdf",
            "name": "Latent Dirichlet Allocation",
            "filename": "lda_paper.pdf", 
            "description": "Machine Learning paper - medium complexity"
        },
        "doc3": {
            "url": "https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf",
            "name": "Attention Mechanism Paper",
            "filename": "attention_paper.pdf",
            "description": "Advanced ML paper - complex mathematics"
        }
    }
    
    paper_info = sample_papers[doc_choice]
    
    try:
        with st.spinner(f"Downloading {paper_info['name']}..."):
            response = requests.get(paper_info['url'], timeout=60)
            response.raise_for_status()
            
            # Verify it's actually a PDF
            if response.headers.get('content-type', '').startswith('application/pdf') or len(response.content) > 1000:
                return response.content, paper_info['filename'], paper_info['name']
            else:
                st.error(f"Downloaded file doesn't appear to be a valid PDF")
                return None, None, None
                
    except Exception as e:
        st.error(f"Failed to download {paper_info['name']}: {e}")
        return None, None, None

def process_pdf_content(pdf_content, filename):
    """Process PDF content and clear history for new document"""
    try:
        # Clear previous query history when loading new document
        st.session_state.query_history = []
        
        # Reset database for new document
        st.session_state.vector_store.reset_database()
        
        # Save content to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_content)
            tmp_file_path = tmp_file.name
        
        # Process PDF
        processor = ResearchPaperProcessor()
        
        with st.spinner("📄 Extracting text from PDF..."):
            full_text, metadata = processor.extract_text_from_pdf(tmp_file_path)
        
        with st.spinner("✂️ Creating document chunks..."):
            chunks = processor.create_chunks(full_text, metadata)
        
        with st.spinner("🧠 Generating embeddings..."):
            document_id = f"current_doc"
            success = st.session_state.vector_store.add_document_chunks(chunks, document_id)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        if success:
            # Store document info and PDF content
            st.session_state.current_document = {
                'id': document_id,
                'name': filename,
                'pages': metadata['total_pages'],
                'chunks': len(chunks),
                'title': metadata.get('title', 'Unknown'),
                'text_sample': full_text[:500] + "..." if len(full_text) > 500 else full_text
            }
            
            # Store PDF content for viewer
            st.session_state.current_pdf_content = pdf_content
            
            st.success(f"✅ Successfully processed {filename}")
            st.info(f"📊 Extracted {len(chunks)} chunks from {metadata['total_pages']} pages")
            
            return True
        else:
            st.error("❌ Failed to add document to vector store")
            return False
            
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return False

def main():
    # Title
    st.title("🔬 PDF RAG System")
    st.markdown("**Upload research papers and ask questions about their content**")
    
    # Initialize system
    if initialize_system():
        st.success("✅ System initialized!")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Sample document section
        st.subheader("📥 Try Sample Documents")
        
        sample_choice = st.selectbox(
            "Choose complexity level:",
            ["doc1", "doc2", "doc3"],
            format_func=lambda x: {
                "doc1": "🟢 Economics (Simple)",
                "doc2": "🟡 ML Topic Model (Medium)", 
                "doc3": "🔴 Attention Mechanism (Complex)"
            }[x]
        )
        
        sample_descriptions = {
            "doc1": "Economic research with basic equations and standard formatting",
            "doc2": "Machine learning paper with mathematical notation and algorithms",
            "doc3": "Advanced ML paper with complex mathematical formulations"
        }
        
        st.caption(sample_descriptions[sample_choice])
        
        if st.button("📄 Download & Process Sample", type="secondary", use_container_width=True):
            pdf_content, filename, paper_name = download_sample_pdf(sample_choice)
            if pdf_content:
                if process_pdf_content(pdf_content, filename):
                    st.rerun()
        
        st.divider()
        
        # File upload section
        st.subheader("📤 Upload Your PDF")
        uploaded_file = st.file_uploader(
            "Choose PDF file",
            type=['pdf'],
            help="Upload a research paper or academic document"
        )
        
        if uploaded_file is not None:
            # Show file info
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {len(uploaded_file.getvalue())/1024/1024:.2f} MB")
            st.write(f"**Type:** {uploaded_file.type}")
            
            if st.button("🚀 Process Uploaded PDF", type="primary", use_container_width=True):
                pdf_content = uploaded_file.getvalue()
                
                # Validate PDF content
                if len(pdf_content) < 1000:
                    st.error("❌ File seems too small to be a valid PDF")
                elif not uploaded_file.type == 'application/pdf':
                    st.error("❌ Please upload a PDF file")
                else:
                    st.info(f"Processing {uploaded_file.name}...")
                    if process_pdf_content(pdf_content, uploaded_file.name):
                        st.rerun()
                    else:
                        st.error("❌ Failed to process the uploaded PDF. Check the error messages above.")
        
        st.divider()
        
        # Query settings
        st.subheader("⚙️ Query Settings")
        top_k = st.slider("Sources to retrieve", 1, 20, 8)
        min_similarity = st.slider("Similarity threshold", 0.0, 1.0, 0.15, 0.05)  # Back to 0.15 default
        
        st.divider()
        
        # Database management
        st.subheader("🗄️ Database")
        
        if st.button("🔄 Reset Database", help="Clear all documents", use_container_width=True):
            if st.session_state.vector_store:
                st.session_state.vector_store.reset_database()
                st.session_state.current_document = None
                st.session_state.query_history = []
                st.session_state.current_pdf_content = None
                st.success("✅ Database reset!")
                st.rerun()
        
        # Show current document
        if st.session_state.current_document:
            st.subheader("📚 Current Document")
            doc = st.session_state.current_document
            st.write(f"**Name:** {doc['name']}")
            st.write(f"**Pages:** {doc['pages']}")
            st.write(f"**Chunks:** {doc['chunks']}")
        
        # System stats
        if st.session_state.rag_engine:
            stats = st.session_state.rag_engine.get_stats()
            st.subheader("⚙️ System Status")
            st.write(f"**Model:** {stats['current_model']}")
            st.write(f"**Documents:** {stats['vector_database']['unique_documents']}")
            st.write(f"**Total Chunks:** {stats['vector_database']['total_chunks']}")
            
            # Debug: Check if chunks are actually there
            if stats['vector_database']['total_chunks'] == 0:
                st.error("⚠️ No chunks in database!")
            else:
                st.success(f"✅ Database has {stats['vector_database']['total_chunks']} chunks")
    
    # Main content area
    if st.session_state.current_document:
        # Document info
        doc = st.session_state.current_document
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Document", doc['name'][:20] + "...")
        with col2:
            st.metric("📄 Pages", doc['pages'])
        
        # Document preview
        with st.expander("📖 Document Preview"):
            st.write(f"**Title:** {doc['title']}")
            
            # PDF Viewer
            if PDF_VIEWER_AVAILABLE and st.session_state.current_pdf_content:
                st.write("**PDF Viewer:**")
                try:
                    streamlit_pdf_viewer.pdf_viewer(st.session_state.current_pdf_content, width=700, height=400)
                except Exception as e:
                    st.error(f"PDF viewer error: {e}")
                    # Fallback to text preview
                    if 'text_sample' in doc:
                        st.write("**Sample Content:**")
                        st.text_area("First 500 characters:", doc['text_sample'], height=150, disabled=True)
            else:
                # Text preview fallback
                if 'text_sample' in doc:
                    st.write("**Sample Content:**")
                    st.text_area("First 500 characters:", doc['text_sample'], height=150, disabled=True)
                else:
                    st.write("**Sample Content:** Not available for this document")
                
                # PDF download option
                if st.session_state.current_pdf_content:
                    st.download_button(
                        label="📄 Download PDF",
                        data=st.session_state.current_pdf_content,
                        file_name=doc['name'],
                        mime='application/pdf'
                    )
        
        st.divider()
        
        # Chat Interface with History
        st.subheader("💬 Chat Interface")
        
        # Document caption at top of chat
        st.caption(f"📄 Chatting with: {doc['title'][:60]}..." if len(doc['title']) > 60 else f"📄 Chatting with: {doc['title']}")
        
        # Chat container for scrollable history
        chat_container = st.container()
        
        with chat_container:
            # Display all chat history
            if st.session_state.query_history:
                for i, chat in enumerate(st.session_state.query_history):
                    # User message
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 15px; margin: 10px 0; border-left: 4px solid #1f77b4;'>
                        <strong style='color: #000000;'>You:</strong> <span style='color: #000000;'>{chat['query']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # AI response
                    answer_text = chat.get('answer', 'No response generated')
                    st.markdown(f"""
                    <div style='background-color: #e8f4fd; padding: 15px; border-radius: 15px; margin: 10px 0; border-left: 4px solid #28a745;'>
                        <strong style='color: #000000;'>AI Assistant:</strong><br>
                        <span style='color: #000000;'>{answer_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Response metadata
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        model_name = chat.get('model', 'unknown')
                        st.caption(f"🤖 {model_name.split('/')[-1] if model_name else 'unknown'}")
                    with col2:
                        confidence = chat.get('confidence', 0.0)
                        st.caption(f"🎯 Confidence: {confidence:.2f}")
                    with col3:
                        sources_count = chat.get('sources', 0)
                        st.caption(f"📄 {sources_count} sources")
                    with col4:
                        st.caption(f"💭 Query #{i+1}")
                    
                    # Sources in expandable section
                    source_details = chat.get('source_details', [])
                    if source_details and len(source_details) > 0:
                        with st.expander(f"📚 View {len(source_details)} Sources", expanded=False):
                            for j, source in enumerate(source_details, 1):
                                st.markdown(f"""
                                **📄 Source {j}:** Page {source.get('page_number', 'N/A')} ({source.get('section', 'N/A')}) 
                                - Similarity: {source.get('similarity_score', 0.0):.3f}
                                """)
                                st.text_area(
                                    f"Source {j} Content", 
                                    source.get('text_preview', 'No content available'), 
                                    height=100, 
                                    disabled=True,
                                    key=f"source_{i}_{j}"
                                )
                    
                    st.divider()
            else:
                st.info("💡 Start a conversation by asking a question about the document!")
        
        # Query input at bottom (always visible)
        st.markdown("### 🔍 Ask a Question")
        
        # Query input with button
        col_input, col_button = st.columns([4, 1])
        
        with col_input:
            query = st.text_input(
                "Type your question here:",
                placeholder="What is this paper about?",
                label_visibility="collapsed",
                key="query_input"
            )
        
        with col_button:
            ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
        
        # Process query
        if ask_button and query:
            # First check if we have any documents
            if not st.session_state.current_document:
                st.error("❌ No document loaded. Please upload or process a document first.")
                return
                
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    # Debug: Check database content
                    stats = st.session_state.rag_engine.get_stats()
                    if stats['vector_database']['total_chunks'] == 0:
                        st.error("❌ No chunks in database. Please reprocess your document.")
                        return
                    
                    st.info(f"🔍 Searching {stats['vector_database']['total_chunks']} chunks with similarity ≥ {min_similarity}")
                    
                    response = st.session_state.rag_engine.query(
                        query, 
                        top_k=top_k, 
                        min_similarity=min_similarity
                    )
                    
                    # If no sources found, try with lower threshold
                    if len(response.sources) == 0 and min_similarity > 0.1:
                        st.warning(f"⚠️ No sources found with similarity ≥ {min_similarity}. Trying with lower threshold...")
                        response = st.session_state.rag_engine.query(
                            query, 
                            top_k=top_k, 
                            min_similarity=0.1
                        )
                        st.info(f"🔄 Retried with similarity ≥ 0.1, found {len(response.sources)} sources")
                    
                    # Add to history only if response is successful
                    if response and response.answer:
                        st.session_state.query_history.append({
                            'query': query,
                            'answer': response.answer,
                            'confidence': response.confidence_score,
                            'model': response.model_used,
                            'sources': len(response.sources),
                            'source_details': response.sources
                        })
                        st.success(f"✅ Generated response with {len(response.sources)} sources")
                    else:
                        st.error("❌ Failed to generate response. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
            
            # Clear input and refresh
            st.rerun()
    
    else:
        # Welcome screen
        st.info("👆 Please select and process a document from the sidebar to get started")
        
        # Instructions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Quick Start")
            st.markdown("""
            1. **Choose a sample document** from the sidebar
            2. Click **"📄 Download & Process Sample"**
            3. Wait for processing to complete
            4. **Ask questions** about the paper content
            """)
        
        with col2:
            st.subheader("💡 Example Questions")
            examples = [
                "What is the main research question?",
                "What methodology was used?",
                "What are the key findings?",
                "What mathematical equations are presented?",
                "What are the conclusions?",
                "What references are cited?"
            ]
            for example in examples:
                st.write(f"• {example}")

if __name__ == "__main__":
    main()