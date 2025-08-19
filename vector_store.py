import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Dict, Tuple
import uuid
import os
from pdf_processor import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector database using ChromaDB for semantic search
    Handles embedding generation and similarity search for document chunks
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize vector store with ChromaDB and sentence transformer"""
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        print("Loading SentenceTransformer model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("✓ Initialized SentenceTransformer model: all-MiniLM-L6-v2")
        
        # Create or get collection
        self.collection_name = "research_papers"
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"✓ Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Research paper document chunks"}
            )
            logger.info(f"✓ Created new collection: {self.collection_name}")
    
    def add_document_chunks(self, chunks: List[DocumentChunk], document_id: str) -> bool:
        """
        Add document chunks to vector database
        
        Args:
            chunks: List of DocumentChunk objects
            document_id: Unique identifier for the document
            
        Returns:
            bool: Success status
        """
        try:
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Prepare metadata
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_{chunk.chunk_id}"
                ids.append(chunk_id)
                
                metadata = {
                    "document_id": document_id,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "chunk_id": chunk.chunk_id,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "confidence_score": chunk.confidence_score,
                    "text_length": len(chunk.text)
                }
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✓ Successfully added {len(chunks)} chunks to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector database: {str(e)}")
            return False
    
    def search_similar_chunks(self, query: str, n_results: int = 5, 
                            filter_metadata: Dict = None) -> List[Dict]:
        """
        Search for similar chunks using semantic similarity
        
        Args:
            query: Search query string
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of dictionaries containing chunk info and similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                # Convert distance to similarity - distances are typically 0-2 for cosine
                # Higher distance = lower similarity
                similarity = max(0.0, 1.0 - (distance / 2.0))  # Normalize to 0-1 range
                
                result = {
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity_score": similarity,
                    "distance": distance
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar chunks for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}")
            return []
    
    def get_document_stats(self) -> Dict:
        """Get statistics about the vector database"""
        try:
            count = self.collection.count()
            
            if count == 0:
                return {
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "unique_pages": 0,
                    "unique_sections": 0,
                    "documents": [],
                    "pages": [],
                    "sections": []
                }
            
            # Get sample of metadata to analyze
            sample = self.collection.get(limit=min(100, count), include=["metadatas"])
            
            stats = {
                "total_chunks": count,
                "documents": set(),
                "pages": set(),
                "sections": set()
            }
            
            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    stats["documents"].add(metadata.get("document_id", "unknown"))
                    stats["pages"].add(metadata.get("page_number", 0))
                    stats["sections"].add(metadata.get("section", "unknown"))
            
            stats["unique_documents"] = len(stats["documents"])
            stats["unique_pages"] = len(stats["pages"])
            stats["unique_sections"] = len(stats["sections"])
            
            # Convert sets back to lists for JSON serialization
            stats["documents"] = list(stats["documents"])
            stats["pages"] = list(stats["pages"])
            stats["sections"] = list(stats["sections"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {"error": str(e)}
    
    def reset_database(self) -> bool:
        """Reset the entire vector database (use with caution!)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Research paper document chunks"}
            )
            logger.info("✓ Vector database reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            return False