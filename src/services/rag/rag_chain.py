"""RAG Chain with optimized retrieval and confidence scoring"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import time
from loguru import logger

from src.core.config import get_settings
from src.core.constants import SYSTEM_PROMPT
from src.core.exceptions import RAGError
from src.services.llm.ollama_client import OllamaClient
from src.services.rag.retriever import Retriever, RetrievedChunk
from src.services.rag.query_processor import QueryProcessor
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
        self.query_processor = QueryProcessor()
        
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
        Generate answer using enhanced RAG pipeline
        """
        start_time = time.time()
        logger.info(f"RAG Query: '{query[:50]}...'")
        
        try:
            # Step 1: Hybrid retrieval
            try:
                retrieved_chunks = self.retriever.retrieve_hybrid(
                    query=query,
                    top_k=top_k or 5,
                    filters=filters
                )
                logger.debug("✅ Used hybrid retrieval")
            except Exception as e:
                logger.warning(f"Hybrid retrieval failed ({e}), using standard retrieval")
                retrieved_chunks = self.retriever.retrieve(
                    query=query,
                    top_k=top_k or 5,
                    filters=filters
                )
            
            if not retrieved_chunks:
                logger.warning("No relevant chunks found for query")
                return self._create_fallback_response(query, start_time)
            
            # Step 2: Build context from chunks
            context = self._build_context(retrieved_chunks)
            
            # Step 3: Construct enhanced prompt
            prompt = self._construct_prompt(query, context, retrieved_chunks)
            
            # Step 4: Generate answer with LLM
            answer = self.llm_client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=temperature or 0.1  # Lower temp for factual answers
            )
            
            # Step 5: Calculate confidence with answer quality
            confidence = self._calculate_confidence(retrieved_chunks, answer)
            
            # Step 6: Format sources
            sources = self._format_sources(retrieved_chunks) if use_sources else []
            
            # Calculate generation time
            generation_time = (time.time() - start_time) * 1000
            
            logger.info(
                f"✅ RAG answer generated in {generation_time:.0f}ms "
                f"(confidence: {confidence:.2f}, chunks: {len(retrieved_chunks)})"
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
        """
        Build context string from retrieved chunks
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Format each chunk with metadata
            source_info = f"[Source: {chunk.source}]"
            
            # Add page number if available
            if 'page' in chunk.metadata:
                source_info = f"[Source: {chunk.source}, Page {chunk.metadata['page']}]"
            
            # Add similarity score for transparency
            # source_info += f" [Relevance: {chunk.similarity:.2f}]"
            
            context_parts.append(f"{source_info}\n{chunk.text}\n")
        
        context = "\n---\n".join(context_parts)
        logger.debug(f"Built context from {len(chunks)} chunks ({len(context)} chars)")
        
        return context
    
    def _construct_prompt(self, query: str, context: str, chunks: List[RetrievedChunk]) -> str:
        """Enhanced prompt with better instructions"""
        
        # Add metadata about retrieval quality
        avg_similarity = sum(c.similarity for c in chunks) / len(chunks) if chunks else 0
        
        prompt = f"""Based on the following information from health insurance policy documents, please answer the user's question.

Context from policy documents:
{context}

User Question: {query}

Instructions:
- Answer based ONLY on the provided context above
- If the context contains the answer, provide it clearly and concisely
- If the context doesn't contain enough information, explicitly state what is missing
- Cite specific policy names when mentioned in the context
- Use bullet points for multiple items
- Be specific with numbers, percentages, and timeframes when available
- Keep your answer focused and avoid unnecessary elaboration

Answer:"""
        
        return prompt
    
    def _calculate_confidence(self, chunks: List[RetrievedChunk], answer: str = "") -> float:
        """
        Enhanced multi-factor confidence calculation
        
        Factors:
        1. Average similarity of retrieved chunks (35%)
        2. Top chunk quality (35%)
        3. Consistency across top chunks (15%)
        4. Answer quality indicators (15%)
        """
        if not chunks:
            return 0.0
        
        # Factor 1: Average similarity (weighted towards top results) - 35%
        weights = [1.0 / (i + 1) for i in range(len(chunks))]
        weighted_sum = sum(chunk.similarity * weight for chunk, weight in zip(chunks, weights))
        total_weight = sum(weights)
        avg_similarity = weighted_sum / total_weight
        
        # Factor 2: Top chunk quality - 35% (increased importance)
        top_similarity = chunks[0].similarity
        
        # Factor 3: Consistency across top 3 chunks - 15%
        if len(chunks) >= 3:
            top3_sims = [c.similarity for c in chunks[:3]]
            similarity_range = max(top3_sims) - min(top3_sims)
            consistency = max(0, 1.0 - similarity_range)  # Inverse of range
        else:
            consistency = 0.8  # Higher default
        
        # Factor 4: Answer quality (length and specificity) - 15%
        if answer:
            answer_length = len(answer)
            # Optimal range: 150-600 chars (adjusted for concise answers)
            if 150 <= answer_length <= 600:
                answer_quality = 1.0
            elif answer_length < 150:
                answer_quality = max(0.5, answer_length / 150)
            else:
                answer_quality = max(0.6, 1.0 - (answer_length - 600) / 1500)
        else:
            answer_quality = 0.8  # Higher default
        
        # Combined confidence with balanced weights
        confidence = (
            avg_similarity * 0.35 +
            top_similarity * 0.35 +
            consistency * 0.15 +
            answer_quality * 0.15
        )
        
        # ✅ NEW: Boost for high-quality chunks
        if top_similarity >= 0.55:
            confidence = min(confidence * 1.1, 1.0)  # 10% boost
        
        return round(confidence, 3)

    
    def _format_sources(self, chunks: List[RetrievedChunk]) -> List[Source]:
        """Format retrieved chunks into Source objects"""
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
        """Create fallback response when no chunks are found"""
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
        """Generate answers for multiple queries"""
        responses = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing batch query {i}/{len(queries)}")
            response = self.generate_answer(query, top_k=top_k)
            responses.append(response)
        
        return responses
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get RAG chain configuration info"""
        return {
            "llm_model": settings.ollama_model,
            "embedding_model": settings.embedding_model,
            "vector_store": settings.chroma_persist_directory,
            "top_k": settings.top_k_retrieval,
            "similarity_threshold": settings.similarity_threshold,
            "retriever_stats": self.retriever.get_retriever_stats()
        }
