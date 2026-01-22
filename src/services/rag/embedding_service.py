from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.core.config import get_settings
from src.core.exceptions import EmbeddingError

settings = get_settings()


class EmbeddingService:
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self.embedding_dim = settings.embedding_dimension
        
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            actual_dim = self.model.get_sentence_embedding_dimension()
            
            if actual_dim != self.embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                    f"got {actual_dim}. Updating config."
                )
                self.embedding_dim = actual_dim
            
            logger.info(f"✅ Embedding model loaded - Dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(f"Could not load model {self.model_name}: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise EmbeddingError(f"Embedding generation failed: {e}")
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts (batched for efficiency)
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            raise EmbeddingError("Cannot embed empty list")
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")
        
        if not valid_texts:
            raise EmbeddingError("All texts are empty")
        
        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts...")
            
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise EmbeddingError(f"Batch embedding generation failed: {e}")
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return float(similarity)
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
    
    def model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_seq_length": self.model.max_seq_length,
        }
