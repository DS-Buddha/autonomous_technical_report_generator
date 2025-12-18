"""
Memory manager for cross-agent context sharing.
Coordinates embeddings generation and vector store operations.
"""

from typing import List, Dict, Optional
import numpy as np

from src.memory.vector_store import VectorMemoryStore
from src.memory.embeddings import EmbeddingsManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    High-level memory manager coordinating embeddings and vector storage.

    Provides simple interfaces for agents to store and retrieve context:
    - Research findings from literature search
    - Code patterns and implementations
    - Section drafts and explanations
    """

    def __init__(self):
        """Initialize memory manager with embeddings and vector store."""
        self.embeddings = EmbeddingsManager()
        self.vector_store = VectorMemoryStore()
        logger.info("Memory manager initialized")

    def add_research_findings(self, findings: List[Dict]) -> None:
        """
        Store research findings in vector memory.

        Args:
            findings: List of dicts with keys:
                - title: Paper title
                - abstract: Paper abstract
                - key_findings: List of insights
                - ... (other metadata)
        """
        if not findings:
            logger.warning("No findings to add")
            return

        # Create text representations for embedding
        texts = []
        metadata = []

        for finding in findings:
            # Combine title and abstract for rich embedding
            text = f"{finding.get('title', '')}: {finding.get('abstract', '')}"
            texts.append(text)

            # Store complete metadata
            meta = {
                'type': 'research',
                **finding
            }
            metadata.append(meta)

        # Generate embeddings
        logger.info(f"Embedding {len(texts)} research findings...")
        embeddings = self.embeddings.embed_texts(texts)

        # Store in vector database
        self.vector_store.add_documents(texts, embeddings, metadata)
        logger.info(f"Added {len(findings)} research findings to memory")

    def add_code_patterns(self, code_blocks: List[Dict]) -> None:
        """
        Store code patterns in vector memory.

        Args:
            code_blocks: List of dicts with keys:
                - id: Unique identifier
                - code: Python code
                - description: What the code does
                - ... (other metadata)
        """
        if not code_blocks:
            logger.warning("No code blocks to add")
            return

        # Create text representations
        texts = []
        metadata = []

        for block in code_blocks:
            # Combine description and code for embedding
            text = f"{block.get('description', '')}\n```python\n{block.get('code', '')}\n```"
            texts.append(text)

            # Store complete metadata
            meta = {
                'type': 'code',
                **block
            }
            metadata.append(meta)

        # Generate embeddings
        logger.info(f"Embedding {len(texts)} code patterns...")
        embeddings = self.embeddings.embed_texts(texts)

        # Store in vector database
        self.vector_store.add_documents(texts, embeddings, metadata)
        logger.info(f"Added {len(code_blocks)} code patterns to memory")

    def add_sections(self, sections: List[Dict]) -> None:
        """
        Store report sections in vector memory.

        Args:
            sections: List of dicts with keys:
                - title: Section title
                - content: Section content
                - ... (other metadata)
        """
        if not sections:
            logger.warning("No sections to add")
            return

        texts = []
        metadata = []

        for section in sections:
            text = f"{section.get('title', '')}\n{section.get('content', '')}"
            texts.append(text)

            meta = {
                'type': 'section',
                **section
            }
            metadata.append(meta)

        logger.info(f"Embedding {len(texts)} sections...")
        embeddings = self.embeddings.embed_texts(texts)

        self.vector_store.add_documents(texts, embeddings, metadata)
        logger.info(f"Added {len(sections)} sections to memory")

    def retrieve_context(
        self,
        query: str,
        k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant context for a query.

        Args:
            query: Search query text
            k: Number of results to return
            filter_type: Optional filter by type ('research', 'code', 'section')

        Returns:
            List of dicts with metadata and similarity scores
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Define filter function if type specified
        filter_fn = None
        if filter_type:
            filter_fn = lambda meta: meta.get('type') == filter_type

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
            filter_fn=filter_fn
        )

        logger.info(
            f"Retrieved {len(results)} results for query: '{query[:50]}...'"
        )
        return results

    def retrieve_research(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve research findings relevant to query.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of research finding metadata
        """
        return self.retrieve_context(query, k=k, filter_type='research')

    def retrieve_code(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve code patterns relevant to query.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of code pattern metadata
        """
        return self.retrieve_context(query, k=k, filter_type='code')

    def get_all_by_type(self, doc_type: str) -> List[Dict]:
        """
        Get all documents of a specific type.

        Args:
            doc_type: Type filter ('research', 'code', 'section')

        Returns:
            List of matching documents
        """
        return self.vector_store.search_by_metadata(
            filter_fn=lambda meta: meta.get('type') == doc_type
        )

    def clear(self) -> None:
        """Clear all memory."""
        self.vector_store.clear()
        logger.info("Cleared all memory")

    def get_stats(self) -> Dict:
        """
        Get memory statistics.

        Returns:
            Dict with statistics about stored documents
        """
        stats = self.vector_store.get_stats()

        # Count by type
        type_counts = {}
        for meta in self.vector_store.metadata:
            doc_type = meta.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        stats['type_counts'] = type_counts
        return stats
