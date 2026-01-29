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
    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        ✅ IMPROVED: Enhanced hybrid retrieval with balanced multi-signal reranking
        
        Key improvements:
        1. Higher initial threshold (0.3 instead of 0.15) for better quality
        2. Adjusted reranking weights to preserve semantic scores
        3. Smarter deduplication that preserves diversity
        """
        from src.services.rag.query_processor import QueryProcessor
        
        # Step 1: Process and expand query
        processor = QueryProcessor()
        query_data = processor.process_query(query)
        logger.info(f"Hybrid retrieval for: {query_data['query_type']} query")
        
        # Step 2: Add policy filter if detected
        if query_data['detected_policy'] and not filters:
            filters = {'policy': query_data['detected_policy']}
        
        # Step 3: Retrieve MORE chunks initially - but with HIGHER threshold
        initial_k = top_k * 3  # ✅ CHANGED: 3x instead of 4x (less noise)
        expanded_query = query_data['processed']
        
        initial_threshold = max(0.30, settings.similarity_threshold)  
        
        initial_results = self.retrieve(
            query=expanded_query,
            top_k=initial_k,
            filters=filters,
            similarity_threshold=initial_threshold  # ✅ Higher quality baseline
        )
        
        if not initial_results:
            logger.warning(f"No results found for query: {query}")
            return []
        
        logger.debug(f"Initial retrieval: {len(initial_results)} chunks (threshold: {initial_threshold:.2f})")
        
        # Step 4: Multi-signal reranking with IMPROVED weighting
        reranked = self._multi_signal_rerank_improved(
            chunks=initial_results,
            original_query=query,
            expanded_query=expanded_query,
            query_type=query_data['query_type']
        )
        
        # Step 5: Smart deduplication (less aggressive)
        deduplicated = self._deduplicate_chunks_smart(reranked)
        
        # Step 6: Return top K without harsh final threshold
        # ✅ CHANGED: No final threshold filtering - trust the reranking
        final_results = deduplicated[:top_k]
        
        avg_sim = sum(c.similarity for c in final_results) / len(final_results) if final_results else 0
        logger.info(
            f"✅ Hybrid retrieval: {len(final_results)} chunks "
            f"(avg similarity: {avg_sim:.2f})"
        )
        
        return final_results


    def _multi_signal_rerank_improved(
        self,
        chunks: List[RetrievedChunk],
        original_query: str,
        expanded_query: str,
        query_type: str
    ) -> List[RetrievedChunk]:
        """
        ✅ IMPROVED: Balanced multi-signal reranking that PRESERVES high semantic scores
        
        New weighting:
        - Semantic similarity: 60% (increased from 40%)
        - Keyword overlap: 20% (decreased from 25%)
        - Phrase matching: 10% (decreased from 15%)
        - Query type bonus: 10% (kept same)
        
        Removed: Document importance (unreliable without proper metadata)
        """
        query_lower = original_query.lower()
        query_words = self._extract_keywords(query_lower)
        
        for chunk in chunks:
            text_lower = chunk.text.lower()
            
            # Signal 1: Base semantic similarity (60% weight) ✅ INCREASED
            semantic_score = chunk.similarity
            
            # Signal 2: Keyword overlap (20% weight) ✅ DECREASED
            keyword_score = self._calculate_keyword_overlap(query_words, text_lower)
            
            # Signal 3: Exact phrase matching (10% weight) ✅ DECREASED
            phrase_score = self._calculate_phrase_match(query_lower, text_lower)
            
            # Signal 4: Query type bonus (10% weight)
            type_bonus = self._get_query_type_bonus(chunk, query_type, text_lower)
            
            # ✅ IMPROVED: New combined score that preserves semantic quality
            # Semantic score is dominant, others provide small boosts
            combined_score = (
                semantic_score * 0.60 +  # ✅ Semantic is primary
                keyword_score * 0.20 +   # ✅ Keywords add context
                phrase_score * 0.10 +    # ✅ Exact matches are bonus
                type_bonus * 0.10        # ✅ Type-specific boost
            )
            
            # ✅ BONUS: If semantic score is already high (>0.65), preserve it more
            if semantic_score >= 0.65:
                combined_score = max(combined_score, semantic_score * 0.95)
            
            chunk.similarity = combined_score
        
        # Sort by combined score
        return sorted(chunks, key=lambda x: x.similarity, reverse=True)


    def _deduplicate_chunks_smart(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        ✅ IMPROVED: Smarter deduplication that preserves diversity
        
        Changes:
        - Use hash of first 500 chars (not 300) for better uniqueness
        - Only deduplicate if chunks are from SAME source AND have high text overlap
        - Preserve chunks from different sources even if similar
        """
        if not chunks:
            return []
        
        deduplicated = []
        seen_signatures = set()
        
        for chunk in chunks:
            # Create signature from first 500 chars (not 300)
            text_normalized = chunk.text[:500].lower().strip()
            
            # Create hash to avoid memory issues with long strings
            import hashlib
            text_hash = hashlib.md5(text_normalized.encode()).hexdigest()[:16]
            
            # Include source in signature (only dedup within same source)
            source_info = f"{chunk.source}"
            signature = f"{text_hash}_{source_info}"
            
            # ✅ CHANGED: Only skip if exact match from same source
            if signature not in seen_signatures:
                deduplicated.append(chunk)
                seen_signatures.add(signature)
        
        if len(deduplicated) < len(chunks):
            logger.debug(f"Deduplicated: {len(chunks)} → {len(deduplicated)} chunks")
        
        return deduplicated




    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords (remove stop words)"""
        stop_words = {
            'the', 'is', 'in', 'and', 'or', 'a', 'an', 'of', 'to', 'for',
            'on', 'at', 'by', 'with', 'from', 'as', 'are', 'was', 'were',
            'be', 'been', 'has', 'have', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'it', 'its', 'what', 'which', 'who', 'when',
            'where', 'why', 'how'
        }
        words = text.split()
        return {w for w in words if w not in stop_words and len(w) > 2}

    def _calculate_keyword_overlap(self, query_words: set, text: str) -> float:
        """Calculate keyword overlap between query and text"""
        if not query_words:
            return 0.0
        
        text_words = self._extract_keywords(text)
        overlap = len(query_words & text_words)
        
        # Normalize by query word count
        return min(overlap / len(query_words), 1.0)

    def _calculate_phrase_match(self, query: str, text: str) -> float:
        """Calculate if query phrases appear in text"""
        # Extract 2-3 word phrases from query
        query_words = query.split()
        phrases = []
        
        # 2-word phrases
        for i in range(len(query_words) - 1):
            phrases.append(f"{query_words[i]} {query_words[i+1]}")
        
        # 3-word phrases
        for i in range(len(query_words) - 2):
            phrases.append(f"{query_words[i]} {query_words[i+1]} {query_words[i+2]}")
        
        if not phrases:
            return 0.0
        
        # Count how many phrases appear in text
        matches = sum(1 for phrase in phrases if phrase in text)
        return min(matches / len(phrases), 1.0)

    def _get_query_type_bonus(self, chunk: RetrievedChunk, query_type: str, text_lower: str) -> float:
        """Boost chunks that match query type characteristics"""
        
        if query_type == 'policy_specific':
            # Boost if chunk mentions policy features
            keywords = ['policy', 'coverage', 'benefit', 'premium', 'sum insured', 'covered', 'plan']
            matches = sum(1 for kw in keywords if kw in text_lower)
            return min(matches / 4, 1.0)  # Normalize to max 1.0
        
        elif query_type == 'comparison':
            # Boost if chunk has comparative language or mentions multiple policies
            keywords = ['compared', 'versus', 'vs', 'difference', 'between', 'than', 'while']
            matches = sum(1 for kw in keywords if kw in text_lower)
            return min(matches / 3, 1.0)
        
        elif query_type == 'exclusion':
            # Boost if chunk mentions exclusions
            keywords = ['excluded', 'not covered', 'limitation', 'restriction', 'except', 'does not cover']
            matches = sum(1 for kw in keywords if kw in text_lower)
            return min(matches / 3, 1.0)
        
        elif query_type == 'claim_related':
            # Boost if chunk mentions claims
            keywords = ['claim', 'reimbursement', 'settlement', 'procedure', 'process']
            matches = sum(1 for kw in keywords if kw in text_lower)
            return min(matches / 3, 1.0)
        
        return 0.5  # Default neutral bonus
