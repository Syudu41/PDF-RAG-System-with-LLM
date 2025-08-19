import logging
from typing import List, Dict, Optional, Tuple
import requests
import json
import time
import os
from dataclasses import dataclass
from vector_store import VectorStore
from pdf_processor import ResearchPaperProcessor

# Use environment variables
hf_token = os.getenv('HF_TOKEN')
gemini_key = os.getenv('GEMINI_API_KEY')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Generative AI not available. Install with: pip install google-generativeai")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Response from RAG query with sources and metadata"""
    answer: str
    sources: List[Dict]
    query: str
    response_time: float
    confidence_score: float
    model_used: str

class RAGEngine:
    """
    RAG (Retrieval Augmented Generation) Engine
    Combines vector search with text generation for research paper Q&A
    Uses Gemini API as primary, HuggingFace as fallback, templates as final fallback
    """
    
    def __init__(self, vector_store: VectorStore, gemini_api_key: Optional[str] = None, hf_token: Optional[str] = None):
        """
        Initialize RAG engine
        
        Args:
            vector_store: Initialized VectorStore instance
            gemini_api_key: Gemini API key (optional, will try env variable)
            hf_token: HuggingFace API token (optional, will try env variable)
        """
        self.vector_store = vector_store
        
        # Get API keys from parameters or environment
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        
        # Initialize Gemini
        self.gemini_model = None
        if GEMINI_AVAILABLE and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                # Use the correct model name for current API
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✓ Gemini API initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
        
        # HuggingFace backup configuration
        self.hf_api_base = "https://api-inference.huggingface.co/models"
        self.hf_models = [
            "microsoft/DialoGPT-large",
            "google/flan-t5-large", 
            "microsoft/DialoGPT-medium"
        ]
        
        # Response configuration
        self.max_context_length = int(os.getenv('MAX_CONTEXT_LENGTH', 2000))
        self.max_response_length = int(os.getenv('MAX_RESPONSE_LENGTH', 500))
        self.temperature = 0.7
        
        # Determine available models
        self.available_models = []
        if self.gemini_model:
            self.available_models.append("gemini-1.5-flash")
        if self.hf_token:
            self.available_models.extend(self.hf_models)
        self.available_models.append("template-fallback")
        
        logger.info(f"✓ RAG Engine initialized with models: {self.available_models}")
    
    def query(self, question: str, top_k: int = 5, min_similarity: float = 0.15) -> RAGResponse:
        """
        Process a query using RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve  
            min_similarity: Minimum similarity threshold for chunks
            
        Returns:
            RAGResponse with answer and sources
        """
        start_time = time.time()
        model_used = "none"
        
        try:
            # Step 1: Retrieve relevant chunks
            logger.info(f"Retrieving relevant chunks for: '{question}'")
            chunks = self.vector_store.search_similar_chunks(question, n_results=top_k)
            
            # Filter by similarity threshold
            relevant_chunks = [
                chunk for chunk in chunks 
                if chunk['similarity_score'] >= min_similarity
            ]
            
            if not relevant_chunks:
                return RAGResponse(
                    answer="I couldn't find relevant information in the document to answer your question. Please try rephrasing or asking about different topics covered in the paper.",
                    sources=[],
                    query=question,
                    response_time=time.time() - start_time,
                    confidence_score=0.0,
                    model_used="none"
                )
            
            logger.info(f"Found {len(relevant_chunks)} relevant chunks")
            
            # Step 2: Prepare context
            context = self._prepare_context(relevant_chunks)
            
            # Step 3: Generate response (try multiple models)
            answer, model_used = self._generate_response_with_fallback(question, context)
            
            # Step 4: Calculate confidence
            confidence = self._calculate_confidence(relevant_chunks, answer, model_used)
            
            # Step 5: Format sources
            sources = self._format_sources(relevant_chunks)
            
            response_time = time.time() - start_time
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                query=question,
                response_time=response_time,
                confidence_score=confidence,
                model_used=model_used
            )
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return RAGResponse(
                answer=f"An error occurred while processing your question: {str(e)}",
                sources=[],
                query=question,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                model_used="error"
            )
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context from retrieved chunks"""
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk['text']
            page_num = chunk['metadata']['page_number']
            section = chunk['metadata']['section']
            
            # Add source info to chunk
            chunk_with_source = f"[Page {page_num}, {section}]: {chunk_text}"
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_with_source) > self.max_context_length:
                break
            
            context_parts.append(chunk_with_source)
            current_length += len(chunk_with_source)
        
        return "\n\n".join(context_parts)
    
    def _generate_response_with_fallback(self, question: str, context: str) -> Tuple[str, str]:
        """Generate response with model fallback chain"""
        
        # Try Gemini first
        if self.gemini_model:
            try:
                response = self._generate_with_gemini(question, context)
                if response:
                    return response, "gemini-1.5-flash"
            except Exception as e:
                logger.warning(f"Gemini failed: {e}")
        
        # Try HuggingFace models
        if self.hf_token:
            for model in self.hf_models:
                try:
                    response = self._generate_with_hf(model, question, context)
                    if response:
                        return response, model
                except Exception as e:
                    logger.warning(f"HF model {model} failed: {e}")
        
        # Final fallback: Enhanced templates
        response = self._create_enhanced_template_response(question, context)
        return response, "template-enhanced"
    
    def _generate_with_gemini(self, question: str, context: str) -> Optional[str]:
        """Generate response using Gemini API"""
        prompt = self._create_gemini_prompt(question, context)
        
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_response_length,
                    temperature=self.temperature,
                )
            )
            
            if response.text:
                return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None
    
    def _create_gemini_prompt(self, question: str, context: str) -> str:
        """Create optimized prompt for Gemini"""
        return f"""You are an AI assistant helping users understand research papers. Based on the provided excerpts from a research paper, answer the user's question clearly and accurately.

Context from research paper:
{context}

User question: {question}

Instructions:
- Answer based only on the provided context
- Be clear, concise, and informative
- Use natural language, not raw technical formatting
- If you mention specific information, reference the page number
- If the context doesn't fully answer the question, mention what information is available
- Write in a helpful, academic tone

Answer:"""
    
    def _generate_with_hf(self, model: str, question: str, context: str) -> Optional[str]:
        """Generate response using HuggingFace API"""
        url = f"{self.hf_api_base}/{model}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.hf_token}"
        }
        
        prompt = f"""Based on the research paper excerpts below, answer this question: {question}

Context: {context}

Answer:"""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_response_length,
                "temperature": self.temperature,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').strip()
                elif isinstance(result, dict) and 'generated_text' in result:
                    return result['generated_text'].strip()
            else:
                logger.warning(f"HF API returned status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling HF API: {e}")
            return None
    
    def _create_enhanced_template_response(self, question: str, context: str) -> str:
        """Create enhanced template-based response that looks more natural"""
        context_snippets = context.split('\n\n')
        
        if not context_snippets:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Analyze question type for better responses
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what is', 'what does', 'define', 'meaning']):
            return self._create_definition_style_response(context_snippets, question)
        elif any(word in question_lower for word in ['how', 'method', 'process', 'approach']):
            return self._create_process_style_response(context_snippets, question)
        elif any(word in question_lower for word in ['why', 'reason', 'purpose', 'benefit']):
            return self._create_explanation_style_response(context_snippets, question)
        elif any(word in question_lower for word in ['result', 'finding', 'conclusion', 'outcome']):
            return self._create_results_style_response(context_snippets, question)
        else:
            return self._create_general_style_response(context_snippets, question)
    
    def _create_definition_style_response(self, snippets: List[str], question: str) -> str:
        """Create definition-style response"""
        main_content = self._extract_clean_content(snippets[0]) if snippets else ""
        page_ref = self._extract_page_reference(snippets[0]) if snippets else ""
        
        if not main_content:
            return "I found some information but couldn't extract a clear definition from the available content."
        
        response = f"Based on the research paper, {main_content.lower()}"
        if page_ref:
            response += f" (as mentioned on {page_ref})"
        
        if len(snippets) > 1:
            additional = self._extract_clean_content(snippets[1])
            if additional and len(additional) > 20:
                response += f". Additionally, {additional.lower()}"
        
        return response + "."
    
    def _create_process_style_response(self, snippets: List[str], question: str) -> str:
        """Create process/methodology style response"""
        processes = []
        for snippet in snippets[:3]:
            content = self._extract_clean_content(snippet)
            page_ref = self._extract_page_reference(snippet)
            if content and len(content) > 20:
                processes.append(f"• {content} ({page_ref})" if page_ref else f"• {content}")
        
        if not processes:
            return "I found some relevant information but couldn't extract clear methodological details."
        
        intro = "According to the research paper, the approach involves:\n\n"
        return intro + "\n".join(processes)
    
    def _create_explanation_style_response(self, snippets: List[str], question: str) -> str:
        """Create explanation style response"""
        main_content = self._extract_clean_content(snippets[0]) if snippets else ""
        page_ref = self._extract_page_reference(snippets[0]) if snippets else ""
        
        if not main_content:
            return "I found some information but couldn't extract a clear explanation."
        
        response = f"The research paper explains that {main_content.lower()}"
        if page_ref:
            response += f" ({page_ref})"
        
        return response + "."
    
    def _create_results_style_response(self, snippets: List[str], question: str) -> str:
        """Create results/findings style response"""
        findings = []
        for snippet in snippets[:2]:
            content = self._extract_clean_content(snippet)
            page_ref = self._extract_page_reference(snippet)
            if content:
                findings.append(f"{content} ({page_ref})" if page_ref else content)
        
        if not findings:
            return "I found some relevant sections but couldn't extract clear findings."
        
        if len(findings) == 1:
            return f"According to the research, {findings[0].lower()}."
        else:
            return f"The research presents several findings: {findings[0].lower()}, and {findings[1].lower()}."
    
    def _create_general_style_response(self, snippets: List[str], question: str) -> str:
        """Create general style response"""
        main_content = self._extract_clean_content(snippets[0]) if snippets else ""
        page_ref = self._extract_page_reference(snippets[0]) if snippets else ""
        
        if not main_content:
            return "I found some potentially relevant information, but couldn't extract a clear answer to your question."
        
        response = f"Based on the available information, {main_content.lower()}"
        if page_ref:
            response += f" (from {page_ref})"
        
        return response + "."
    
    def _extract_clean_content(self, snippet: str) -> str:
        """Extract clean content from snippet, removing formatting artifacts"""
        if not snippet or '[Page' not in snippet:
            return snippet[:200] if snippet else ""
        
        # Extract content after page reference
        parts = snippet.split(']: ', 1)
        if len(parts) != 2:
            return snippet[:200]
        
        content = parts[1].strip()
        
        # Clean up common artifacts
        content = content.replace('LATEX', 'LaTeX')
        content = content.replace('[CITATION]', '')
        content = content.replace('[REF]', 'reference')
        content = content.replace('\\', '')
        content = content.replace('{', '').replace('}', '')
        content = content.replace('begin', '').replace('end', '')
        content = content.replace('vspace', '').replace('hspace', '')
        
        # Remove excessive whitespace and special characters
        content = ' '.join(content.split())
        
        # Remove incomplete sentences and formatting commands
        sentences = content.split('.')
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip if too short or contains too many formatting artifacts
            if len(sentence) > 10 and not any(cmd in sentence.lower() for cmd in ['documentclass', 'usepackage', 'section{', 'caption{']):
                clean_sentences.append(sentence)
        
        if clean_sentences:
            result = '. '.join(clean_sentences[:2])  # Take first 2 sentences
            if len(result) > 150:
                result = result[:150] + "..."
            return result
        
        # Fallback: just clean the raw content
        if len(content) > 150:
            content = content[:150] + "..."
        
        return content
    
    def _extract_page_reference(self, snippet: str) -> str:
        """Extract page reference from snippet"""
        if '[Page' in snippet:
            try:
                page_part = snippet.split(']: ')[0]
                return page_part.replace('[', '').replace(']', '')
            except:
                return ""
        return ""
    
    def _calculate_confidence(self, chunks: List[Dict], answer: str, model_used: str) -> float:
        """Calculate confidence score based on chunk similarity, answer quality, and model used"""
        if not chunks:
            return 0.0
        
        # Base confidence from similarity scores
        avg_similarity = sum(chunk['similarity_score'] for chunk in chunks[:3]) / min(3, len(chunks))
        
        # Model quality multiplier
        model_multipliers = {
            "gemini-1.5-flash": 1.0,
            "gemini-pro": 1.0,
            "microsoft/DialoGPT-large": 0.9,
            "google/flan-t5-large": 0.85,
            "template-enhanced": 0.7,
            "template-fallback": 0.6
        }
        
        model_quality = model_multipliers.get(model_used, 0.5)
        
        # Answer quality indicators
        answer_quality = 0.5
        if len(answer) > 100:
            answer_quality += 0.2
        if any(word in answer.lower() for word in ['according to', 'based on', 'research', 'paper']):
            answer_quality += 0.1
        if 'page' in answer.lower():
            answer_quality += 0.1
        if len(chunks) >= 3:
            answer_quality += 0.1
        
        # Combine scores
        confidence = (avg_similarity * 0.4) + (model_quality * 0.4) + (min(1.0, answer_quality) * 0.2)
        return min(1.0, max(0.3, confidence))
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Format source information for response"""
        sources = []
        for chunk in chunks:
            metadata = chunk['metadata']
            source = {
                'chunk_id': metadata['chunk_id'],
                'page_number': metadata['page_number'],
                'section': metadata['section'],
                'similarity_score': chunk['similarity_score'],
                'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            }
            sources.append(source)
        return sources
    
    def get_stats(self) -> Dict:
        """Get RAG engine statistics"""
        vector_stats = self.vector_store.get_document_stats()
        
        current_model = "template-fallback"
        if self.gemini_model:
            current_model = "gemini-1.5-flash"
        elif self.hf_token:
            current_model = self.hf_models[0]
        
        return {
            'available_models': self.available_models,
            'current_model': current_model,
            'primary_model': 'gemini-1.5-flash' if self.gemini_model else 'fallback',
            'gemini_available': bool(self.gemini_model),
            'hf_available': bool(self.hf_token),
            'vector_database': vector_stats,
            'configuration': {
                'max_context_length': self.max_context_length,
                'max_response_length': self.max_response_length,
                'temperature': self.temperature
            }
        }