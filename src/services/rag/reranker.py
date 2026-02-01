"""
Cross-Encoder Re-ranking for better retrieval precision
Uses a more powerful model to re-score query-document pairs
"""
from typing import List, Dict, Optional
from loguru import logger
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed. Re-ranking disabled.")
    CROSS_ENCODER_AVAILABLE = False


class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using cross-encoder model
    
    Cross-encoders are more accurate than bi-encoders (embeddings) because:
    - They process query + document together
    - Capture fine-grained interactions
    - Better at understanding context
    
    Trade-off: Slower, so only use for final re-ranking of top candidates
    """
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'):
        """
        Initialize cross-encoder model
        
        Args:
            model_name: HuggingFace model name
                Recommended models:
                - 'cross-encoder/ms-marco-MiniLM-L-12-v2' (fast, 120MB)
                - 'cross-encoder/ms-marco-MiniLM-L-6-v2' (faster, 80MB)
                - 'cross-encoder/ms-marco-TinyBERT-L-2-v2' (fastest, 60MB)
        """
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("Cross-encoder not available. Install: pip install sentence-transformers")
            self.model = None
            return
        
        try:
            logger.info(f"Loading cross-encoder model: {model_name}")
            self.model = CrossEncoder(model_name, max_length=512)
            self.model_name = model_name
            logger.info("✅ Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            self.model = None
    
    def rerank(
        self,
        query: str,
        chunks: List,  # ✅ FIXED: Accept RetrievedChunk objects
        top_k: Optional[int] = None
    ) -> List:  # ✅ FIXED: Return RetrievedChunk objects
        """
        Re-rank chunks using cross-encoder
        
        Args:
            query: User query
            chunks: List of RetrievedChunk objects
            top_k: Return top K chunks (default: return all)
            
        Returns:
            Re-ranked RetrievedChunk objects with updated similarity scores
        """
        if not self.model or not chunks:
            logger.warning("Cross-encoder not available or no chunks to rerank")
            return chunks
        
        try:
            # ✅ FIX: Prepare query-document pairs using .text attribute
            pairs = [[query, chunk.text] for chunk in chunks]
            
            # Get cross-encoder scores
            logger.debug(f"Re-ranking {len(pairs)} chunks with cross-encoder...")
            scores = self.model.predict(pairs)
            
            # Normalize scores to 0-1 range (cross-encoder scores can be negative)
            scores = self._normalize_scores(scores)
            
            # ✅ FIX: Update chunk similarities with new scores
            for chunk, score in zip(chunks, scores):
                chunk.metadata['original_similarity'] = chunk.similarity  # Store original
                chunk.similarity = float(score)  # Update with CE score
                chunk.metadata['reranked'] = True
            
            # ✅ FIX: Sort by new similarity scores using attribute
            reranked = sorted(chunks, key=lambda x: x.similarity, reverse=True)
            
            # Return top_k if specified
            if top_k:
                reranked = reranked[:top_k]
            
            # ✅ FIX: Access similarity as attribute
            logger.info(f"✅ Re-ranked {len(chunks)} chunks (top score: {reranked[0].similarity:.3f})")
            
            return reranked
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}. Returning original chunks.")
            return chunks
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to 0-1 range using sigmoid
        
        Cross-encoder scores are logits (can be negative).
        We apply sigmoid to convert to probabilities.
        """
        # Sigmoid normalization
        normalized = 1 / (1 + np.exp(-scores))
        return normalized
    
    def batch_rerank(
        self,
        queries: List[str],
        chunks_list: List[List],
        top_k: Optional[int] = None
    ) -> List[List]:
        """
        Batch re-rank for multiple queries
        
        Args:
            queries: List of queries
            chunks_list: List of chunk lists (one per query)
            top_k: Return top K per query
            
        Returns:
            List of re-ranked chunk lists
        """
        if not self.model:
            return chunks_list
        
        reranked_all = []
        for query, chunks in zip(queries, chunks_list):
            reranked = self.rerank(query, chunks, top_k)
            reranked_all.append(reranked)
        
        return reranked_all
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_name': self.model_name if self.model else None,
            'available': self.model is not None,
            'type': 'cross-encoder'
        }


class HybridReranker:
    """
    Combines multiple re-ranking strategies
    
    Strategies:
    1. Cross-encoder scores
    2. BM25 lexical scores
    3. Embedding similarity
    4. Metadata boosting (e.g., boost chunks from specific policies)
    """
    
    def __init__(self, cross_encoder_weight: float = 0.7):
        """
        Args:
            cross_encoder_weight: Weight for cross-encoder scores (0-1)
        """
        self.cross_encoder = CrossEncoderReranker()
        self.cross_encoder_weight = cross_encoder_weight
        self.embedding_weight = 1.0 - cross_encoder_weight
    
    def rerank(
        self,
        query: str,
        chunks: List,  # ✅ FIXED: RetrievedChunk objects
        top_k: Optional[int] = None,
        metadata_boost: Optional[Dict[str, float]] = None
    ) -> List:  # ✅ FIXED: RetrievedChunk objects
        """
        Hybrid re-ranking combining multiple signals
        
        Args:
            query: User query
            chunks: Retrieved chunks (RetrievedChunk objects)
            top_k: Return top K
            metadata_boost: Dict of metadata field -> boost factor
                Example: {'policy': {'optima restore': 1.2}}
        
        Returns:
            Re-ranked chunks
        """
        if not chunks:
            return chunks
        
        # Step 1: Get cross-encoder scores
        if self.cross_encoder.model:
            reranked = self.cross_encoder.rerank(query, chunks)
            
            # ✅ FIX: Combine cross-encoder + original embedding scores
            for chunk in reranked:
                ce_score = chunk.similarity
                orig_score = chunk.metadata.get('original_similarity', 0.5)
                
                # Weighted combination
                chunk.similarity = (
                    self.cross_encoder_weight * ce_score +
                    self.embedding_weight * orig_score
                )
        else:
            reranked = chunks
        
        # ✅ FIX: Step 2 - Apply metadata boosting
        if metadata_boost:
            for chunk in reranked:
                boost = 1.0
                for field, boosts in metadata_boost.items():
                    if field in chunk.metadata:
                        value = chunk.metadata[field]
                        if value in boosts:
                            boost *= boosts[value]
                chunk.similarity *= boost
        
        # ✅ FIX: Step 3 - Re-sort and return top_k
        reranked = sorted(reranked, key=lambda x: x.similarity, reverse=True)
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked
