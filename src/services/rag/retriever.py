from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from loguru import logger

from src.core.config import get_settings
from src.services.rag.embedding_service import EmbeddingService
from src.services.rag.vector_store import VectorStore

settings = get_settings()


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    similarity: float
    chunk_id: str
    source: str


class Retriever:
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        
        logger.info("Retriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filters: Metadata filters (e.g., {"filename": "specific_policy.pdf"})
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of RetrievedChunk objects
        """
        top_k = top_k or settings.top_k_retrieval
        similarity_threshold = similarity_threshold or settings.similarity_threshold
        
        logger.info(f"Retrieving chunks for query: '{query[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Query vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        
        for i in range(len(results['ids'][0])):
            # Convert distance to similarity (cosine distance: similarity = 1 - distance)
            distance = results['distances'][0][i]
            similarity = 1 - distance
            
            # Apply similarity threshold
            if similarity < similarity_threshold:
                logger.debug(
                    f"Chunk filtered out: similarity {similarity:.3f} < threshold {similarity_threshold}"
                )
                continue
            
            chunk = RetrievedChunk(
                text=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                similarity=similarity,
                chunk_id=results['ids'][0][i],
                source=results['metadatas'][0][i].get('filename', 'unknown')
            )
            
            retrieved_chunks.append(chunk)
        
        logger.info(
            f"✅ Retrieved {len(retrieved_chunks)} chunks "
            f"(threshold: {similarity_threshold:.2f})"
        )
        
        return retrieved_chunks
    
    def retrieve_with_reranking(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve with two-stage process: initial retrieval + reranking
        
        Args:
            query: User query
            top_k: Final number of chunks to return
            retrieval_k: Initial number of chunks to retrieve (before reranking)
            filters: Metadata filters
            
        Returns:
            Reranked list of RetrievedChunk objects
        """
        # Stage 1: Retrieve more chunks than needed
        initial_chunks = self.retrieve(
            query=query,
            top_k=retrieval_k,
            filters=filters,
            similarity_threshold=0.0  # No threshold for initial retrieval
        )
        
        if not initial_chunks:
            return []
        
        # Stage 2: Rerank using query embedding similarity
        # (In production, could use cross-encoder model here)
        query_embedding = self.embedding_service.embed_text(query)
        
        # Recompute similarities with full precision
        for chunk in initial_chunks:
            chunk_embedding = self.embedding_service.embed_text(chunk.text)
            similarity = self.embedding_service.compute_similarity(
                query_embedding,
                chunk_embedding
            )
            chunk.similarity = similarity
        
        # Sort by similarity and take top_k
        reranked_chunks = sorted(
            initial_chunks,
            key=lambda x: x.similarity,
            reverse=True
        )[:top_k]
        
        logger.info(f"✅ Reranked to top {len(reranked_chunks)} chunks")
        
        return reranked_chunks
    
    def retrieve_by_source(
        self,
        query: str,
        source_filename: str,
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        return self.retrieve(
            query=query,
            top_k=top_k,
            filters={"filename": source_filename}
        )
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        vector_stats = self.vector_store.get_stats()
        embedding_info = self.embedding_service.model_info()
        
        return {
            "embedding_model": embedding_info,
            "vector_store": vector_stats,
            "default_top_k": settings.top_k_retrieval,
            "default_threshold": settings.similarity_threshold
        }
