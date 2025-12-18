"""
FAISS-based vector store for efficient similarity search.
Stores embeddings and associated metadata for cross-agent memory.
"""

import faiss
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import pickle

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorMemoryStore:
    """
    FAISS vector store with metadata management for agent memory.

    Uses IndexFlatL2 for exact similarity search with L2 distance.
    Persists both the FAISS index and metadata to disk.
    """

    def __init__(
        self,
        dimension: int = None,
        persist_path: str = None
    ):
        """
        Initialize the vector store.

        Args:
            dimension: Dimension of embedding vectors (default from settings)
            persist_path: Path to store index and metadata (default from settings)
        """
        self.dimension = dimension or settings.embedding_dimension
        self.persist_path = Path(persist_path or settings.faiss_index_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        # Initialize FAISS index (IndexFlatL2 for exact L2 distance search)
        self.index = faiss.IndexFlatL2(self.dimension)

        # Metadata storage (FAISS only stores vectors)
        self.metadata: List[Dict] = []

        # Load existing index if available
        self.load()

        logger.info(
            f"Vector store initialized with dimension={self.dimension}, "
            f"vectors={self.index.ntotal}"
        )

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict]
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            texts: List of text documents (for logging/debugging)
            embeddings: Numpy array of embeddings (shape: [n_docs, dimension])
            metadata: List of metadata dicts (one per document)
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Embeddings count ({embeddings.shape[0]}) must match "
                f"metadata count ({len(metadata)})"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) must match "
                f"store dimension ({self.dimension})"
            )

        # Add vectors to FAISS index
        self.index.add(embeddings.astype('float32'))

        # Store metadata
        self.metadata.extend(metadata)

        logger.info(f"Added {len(metadata)} documents to vector store")

        # Persist changes
        self.save()

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_fn: Optional[callable] = None
    ) -> List[Dict]:
        """
        Retrieve top-k most similar documents.

        Args:
            query_embedding: Query embedding vector (shape: [dimension])
            k: Number of results to return
            filter_fn: Optional function to filter results (applied to metadata)

        Returns:
            List of dicts with metadata, distance, and similarity score
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty, returning no results")
            return []

        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(k * 2, self.index.ntotal)  # Get extra for filtering
        )

        # Build results with metadata
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]

                # Apply filter if provided
                if filter_fn and not filter_fn(meta):
                    continue

                results.append({
                    'metadata': meta,
                    'distance': float(distance),
                    'score': 1 / (1 + float(distance))  # Convert distance to similarity
                })

                if len(results) >= k:
                    break

        logger.debug(f"Retrieved {len(results)} results from vector store")
        return results

    def search_by_metadata(
        self,
        filter_fn: callable,
        k: Optional[int] = None
    ) -> List[Dict]:
        """
        Search documents by metadata filter only (no vector search).

        Args:
            filter_fn: Function to filter metadata
            k: Optional limit on results

        Returns:
            List of matching metadata dicts
        """
        results = [meta for meta in self.metadata if filter_fn(meta)]

        if k:
            results = results[:k]

        logger.debug(f"Found {len(results)} results by metadata filter")
        return results

    def save(self) -> None:
        """Persist index and metadata to disk."""
        try:
            # Save FAISS index
            index_path = self.persist_path / "index.faiss"
            faiss.write_index(self.index, str(index_path))

            # Save metadata
            metadata_path = self.persist_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)

            logger.debug(f"Saved vector store to {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def load(self) -> None:
        """Load existing index and metadata from disk."""
        index_path = self.persist_path / "index.faiss"
        metadata_path = self.persist_path / "metadata.pkl"

        try:
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded {len(self.metadata)} metadata records")
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {e}")
            # Reset to empty state
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []

    def clear(self) -> None:
        """Clear all data from the vector store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.save()
        logger.info("Cleared vector store")

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'metadata_count': len(self.metadata),
            'persist_path': str(self.persist_path)
        }
