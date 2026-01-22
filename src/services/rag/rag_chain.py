from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import time
from loguru import logger

from src.core.config import get_settings
from src.core.constants import SYSTEM_PROMPT
from src.core.exceptions import RAGError
from src.services.llm.ollama_client import OllamaClient
from src.services.rag.retriever import Retriever, RetrievedChunk
from src.services.rag.embedding_service import EmbeddingService
from src.services.rag.vector_store import VectorStore

settings = get_settings()


@dataclass
class Source:
    filename: str
    page: Optional[int]
    chunk_id: str
    similarity: float
    preview: str  # First 100 chars of the chunk


@dataclass
class RAGResponse:
    answer: str
    sources: List[Source]
    confidence: float
    retrieved_chunks_count: int
    generation_time_ms: float
    query: str


class RAGChain:
    
    def __init__(
        self,
        llm_client: Optional[OllamaClient] = None,
        retriever: Optional[Retriever] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.llm_client = llm_client or OllamaClient()
        
        # Initialize RAG components if not provided
        if retriever:
            self.retriever = retriever
        else:
            embedding_service = embedding_service or EmbeddingService()
            vector_store = vector_store or VectorStore()
            self.retriever = Retriever(
                embedding_service=embedding_service,
                vector_store=vector_store
            )
        
        logger.info("RAG Chain initialized")
    
    def generate_answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        use_sources: bool = True
    ) -> RAGResponse:
        """
        Generate answer using RAG pipeline
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filters: Metadata filters for retrieval
            temperature: LLM temperature override
            use_sources: Whether to include source citations
            
        Returns:
            RAGResponse with answer and sources
        """
        start_time = time.time()
        
        logger.info(f"RAG Query: '{query[:50]}...'")
        
        try:
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=filters
            )
            
            if not retrieved_chunks:
                logger.warning("No relevant chunks found for query")
                return self._create_fallback_response(query, start_time)
            
            # Step 2: Build context from chunks
            context = self._build_context(retrieved_chunks)
            
            # Step 3: Construct prompt
            prompt = self._construct_prompt(query, context)
            
            # Step 4: Generate answer with LLM
            answer = self.llm_client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=temperature
            )
            
            # Step 5: Calculate confidence based on similarity scores
            confidence = self._calculate_confidence(retrieved_chunks)
            
            # Step 6: Format sources
            sources = self._format_sources(retrieved_chunks) if use_sources else []
            
            # Calculate generation time
            generation_time = (time.time() - start_time) * 1000  # Convert to ms
            
            logger.info(
                f"✅ RAG answer generated in {generation_time:.0f}ms "
                f"(confidence: {confidence:.2f})"
            )
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                retrieved_chunks_count=len(retrieved_chunks),
                generation_time_ms=generation_time,
                query=query
            )
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise RAGError(f"RAG generation failed: {e}")
    
    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Format each chunk with metadata
            source_info = f"[Source: {chunk.source}]"
            
            # Add page number if available
            if 'page' in chunk.metadata:
                source_info = f"[Source: {chunk.source}, Page {chunk.metadata['page']}]"
            
            context_parts.append(f"{source_info}\n{chunk.text}\n")
        
        context = "\n---\n".join(context_parts)
        
        logger.debug(f"Built context from {len(chunks)} chunks ({len(context)} chars)")
        
        return context
    
    def _construct_prompt(self, query: str, context: str) -> str:
        prompt = f"""Based on the following information from health insurance policy documents, please answer the user's question.

Context from policy documents:
{context}

User Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite specific policies when relevant
- Be clear and concise
- Use simple language

Answer:"""
        
        return prompt
    
    def _calculate_confidence(self, chunks: List[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        
        # Average of top similarity scores (weighted towards top results)
        weights = [1.0 / (i + 1) for i in range(len(chunks))]
        weighted_sum = sum(chunk.similarity * weight for chunk, weight in zip(chunks, weights))
        total_weight = sum(weights)
        
        confidence = weighted_sum / total_weight
        
        return round(confidence, 3)
    
    def _format_sources(self, chunks: List[RetrievedChunk]) -> List[Source]:
        sources = []
        
        for chunk in chunks:
            source = Source(
                filename=chunk.source,
                page=chunk.metadata.get('page'),
                chunk_id=chunk.chunk_id,
                similarity=chunk.similarity,
                preview=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            )
            sources.append(source)
        
        return sources
    
    def _create_fallback_response(self, query: str, start_time: float) -> RAGResponse:
        fallback_answer = (
            "I apologize, but I couldn't find relevant information in the policy documents "
            "to answer your question. This could mean:\n"
            "1. The information isn't covered in the loaded policies\n"
            "2. The question might need to be rephrased\n"
            "3. More policy documents need to be added\n\n"
            "Please try asking in a different way or contact the insurance provider directly."
        )
        
        generation_time = (time.time() - start_time) * 1000
        
        return RAGResponse(
            answer=fallback_answer,
            sources=[],
            confidence=0.0,
            retrieved_chunks_count=0,
            generation_time_ms=generation_time,
            query=query
        )
    
    def batch_generate(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[RAGResponse]:
        responses = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing batch query {i}/{len(queries)}")
            response = self.generate_answer(query, top_k=top_k)
            responses.append(response)
        
        return responses
    
    def get_chain_info(self) -> Dict[str, Any]:
        return {
            "llm_model": settings.ollama_model,
            "embedding_model": settings.embedding_model,
            "vector_store": settings.chroma_persist_directory,
            "top_k": settings.top_k_retrieval,
            "similarity_threshold": settings.similarity_threshold,
            "retriever_stats": self.retriever.get_retriever_stats()
        }
