from typing import List, Optional, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import numpy as np
from loguru import logger

from src.core.config import get_settings
from src.core.exceptions import RAGError
from src.services.rag.chunker import Chunk

settings = get_settings()


class VectorStore:
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        reset: bool = False
    ):
        self.collection_name = collection_name or settings.chromadb_collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        
        try:
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            if reset:
                logger.warning(f"Resetting collection: {self.collection_name}")
                try:
                    self.client.delete_collection(name=self.collection_name)
                except:
                    pass
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(f"✅ Collection '{self.collection_name}' ready")
            logger.info(f"   Current documents: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RAGError(f"ChromaDB initialization failed: {e}")
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray
    ) -> None:
        """
        Add chunks with embeddings to the vector store
        
        Args:
            chunks: List of Chunk objects
            embeddings: Array of embeddings (n_chunks, embedding_dim)
        """
        if len(chunks) != len(embeddings):
            raise RAGError(
                f"Chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        try:
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            embeddings_list = embeddings.tolist()
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_list
            )
            
            logger.info(f"✅ Added {len(chunks)} chunks to vector store")
            logger.info(f"   Total documents: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise RAGError(f"Failed to add chunks: {e}")
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store with an embedding
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"filename": "policy.pdf"})
            
        Returns:
            Dictionary with ids, documents, metadatas, distances
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise RAGError(f"Query failed: {e}")
    
    def search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (ChromaDB will handle embedding)
        
        Note: This uses ChromaDB's default embedding function.
        For production, use query() with your own embeddings.
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise RAGError(f"Search failed: {e}")
    
    def delete_by_source(self, source: str) -> int:
        try:
            # Get all documents with this source
            results = self.collection.get(
                where={"source": source}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                count = len(results['ids'])
                logger.info(f"Deleted {count} chunks from source: {source}")
                return count
            
            return 0
            
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            raise RAGError(f"Deletion failed: {e}")
    
    def delete_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise RAGError(f"Collection deletion failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            count = self.collection.count()
            
            # Get ALL documents to count unique sources
            if count > 0:
                all_docs = self.collection.get()
                sources = set()
                
                if all_docs['metadatas']:
                    # Check both 'filename' and 'source' fields
                    for m in all_docs['metadatas']:
                        source = m.get('filename') or m.get('policy_name') or 'unknown'
                        sources.add(source)
                
                return {
                    "collection_name": self.collection_name,
                    "total_chunks": count,
                    "unique_sources": len(sources),
                    "sources": sorted(list(sources)),
                    "persist_directory": self.persist_directory
                }
            else:
                return {
                    "collection_name": self.collection_name,
                    "total_chunks": 0,
                    "unique_sources": 0,
                    "sources": [],
                    "persist_directory": self.persist_directory
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

    
    def list_sources(self) -> List[str]:
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            sources = set()
            if all_docs['metadatas']:
                sources = {m.get('filename', 'unknown') for m in all_docs['metadatas']}
            
            return sorted(list(sources))
            
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            return []
