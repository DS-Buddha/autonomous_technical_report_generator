"""
Embeddings manager using Google's text-embedding-004 model.
Generates vector embeddings for text documents.
"""

import numpy as np
from typing import List, Union
from google import genai
from google.genai import types

from src.config.settings import settings
from src.utils.logger import get_logger
from src.utils.retry import retry_with_exponential_backoff

logger = get_logger(__name__)


class EmbeddingsManager:
    """
    Manager for generating text embeddings using Google's embedding models.

    Uses text-embedding-004 model which produces 768-dimensional vectors.
    """

    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialize the embeddings manager.

        Args:
            model: Embedding model name (default from settings)
            api_key: Google API key (default from settings)
        """
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.google_api_key

        # Initialize Google Genai client
        self.client = genai.Client(api_key=self.api_key)

        logger.info(f"Embeddings manager initialized with model={self.model}")

    @retry_with_exponential_backoff(
        max_retries=3,
        initial_delay=1.0,
        exceptions=(Exception,)
    )
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Numpy array of shape [dimension]
        """
        try:
            response = self.client.models.embed_content(
                model=self.model,
                content=text
            )
            embedding = np.array(response.embedding, dtype='float32')
            logger.debug(f"Generated embedding for text ({len(text)} chars)")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def embed_texts(self, texts: List[str], batch_size: int = 10) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process in each batch

        Returns:
            Numpy array of shape [n_texts, dimension]
        """
        if not texts:
            return np.array([], dtype='float32')

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1} ({len(batch)} texts)")

            for text in batch:
                try:
                    emb = self.embed_text(text)
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Failed to embed text, using zero vector: {e}")
                    # Use zero vector as fallback
                    embeddings.append(np.zeros(settings.embedding_dimension, dtype='float32'))

        result = np.array(embeddings, dtype='float32')
        logger.info(f"Generated {len(result)} embeddings")
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.

        Note: For text-embedding-004, there's no difference between document
        and query embeddings, but this method is provided for API consistency.

        Args:
            query: Search query text

        Returns:
            Numpy array of shape [dimension]
        """
        return self.embed_text(query)

    def cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)

    def batch_cosine_similarity(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between a query and multiple embeddings.

        Args:
            query_embedding: Query embedding vector [dimension]
            embeddings: Matrix of embeddings [n_embeddings, dimension]

        Returns:
            Array of similarity scores [n_embeddings]
        """
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / np.maximum(norms, 1e-10)

        # Compute similarities
        similarities = np.dot(embeddings_norm, query_norm)
        return similarities
