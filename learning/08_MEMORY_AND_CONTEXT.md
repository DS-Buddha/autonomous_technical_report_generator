# Memory and Context Management: The Knowledge Layer

## 🎯 The Memory Problem

### Why Memory Matters

**Without memory:**
```python
# Iteration 1
researcher.run("Find papers on RAG")  # Searches, finds 10 papers

# Iteration 2 (after feedback: "add more papers on hybrid RAG")
researcher.run("Find more papers on hybrid RAG")
# Problem: No memory of previous 10 papers! Might find duplicates.
```

**With memory:**
```python
# Iteration 1
researcher.run("Find papers on RAG")
memory.store(papers)  # Store findings

# Iteration 2
previous_papers = memory.query("RAG papers")  # Retrieve previous work
researcher.run("Find hybrid RAG papers", exclude=previous_papers)
# Builds on previous work, no duplicates
```

### The Context Window Problem

**LLMs have finite context:**
- GPT-4: 128K tokens (~300 pages)
- Gemini: 2M tokens (~5000 pages)

**But we might have:**
- 20 research papers (200 pages)
- 10 code examples (50 pages)
- 5 iterations of feedback (100 pages)
- **Total: 350 pages → Doesn't fit!**

**Solution: Selective retrieval**
```python
# Don't send all 20 papers to LLM
all_papers = memory.get_all()  # Too much!

# Query for relevant subset
relevant_papers = memory.query("how does attention work", k=3)  # Just 3 most relevant
# Fits in context window ✓
```

---

## 🧠 Our Memory Architecture

### Two-Layer Memory

**Layer 1: State (Short-term)**
- Current workflow data
- Flows through LangGraph
- Cleared after workflow completes
- Example: Current iteration's papers

**Layer 2: FAISS Vector Store (Long-term)**
- Persistent across workflows
- Semantic search enabled
- Infinite storage (limited by disk)
- Example: All papers ever researched

```
┌─────────────────────────────────────────┐
│           LangGraph State               │
│  (Short-term: current workflow only)    │
│  ┌─────────────────────────────────┐   │
│  │ topic: "RAG systems"            │   │
│  │ research_papers: [paper1, ...]  │   │
│  │ iteration_count: 2              │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                     │
                     ▼ (store)
┌─────────────────────────────────────────┐
│        FAISS Vector Store               │
│  (Long-term: persistent memory)         │
│  ┌─────────────────────────────────┐   │
│  │ "Attention Is All You Need"     │   │
│  │ "RAG for QA systems"            │   │
│  │ "Hybrid search approaches"      │   │
│  │ ... (1000s of documents)        │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                     │
                     ▼ (query)
        Query: "How does attention work?"
                     │
                     ▼
        Results: Top 3 most relevant docs
```

---

## 📦 FAISS Vector Store Deep-Dive

### What is FAISS?

**FAISS** = Facebook AI Similarity Search

**Purpose:** Fast nearest-neighbor search in high-dimensional vector spaces.

**Use case:** "Find documents similar to this query"

### How It Works

1. **Text → Embeddings**
   ```python
   text = "Attention mechanisms in transformers"
   embedding = model.encode(text)  # [0.23, -0.15, 0.87, ..., 0.34]
   # Converts text to 384-dimensional vector
   ```

2. **Store embeddings in FAISS**
   ```python
   index.add(embedding)  # O(1) insertion
   ```

3. **Search by similarity**
   ```python
   query = "How does attention work?"
   query_embedding = model.encode(query)
   distances, indices = index.search(query_embedding, k=5)
   # Returns 5 most similar documents
   ```

### Our Implementation

```python
# src/memory/memory_manager.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pickle
import os

class MemoryManager:
    """
    Long-term memory using FAISS vector store.

    Features:
    - Semantic search (not keyword-based)
    - Fast retrieval (milliseconds for 1M+ documents)
    - Persistent storage (saves to disk)
    - Metadata support (store paper IDs, sources, etc.)
    """

    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        storage_path: str = 'data/vector_store'
    ):
        """
        Initialize memory manager.

        Args:
            embedding_model: SentenceTransformer model name
            storage_path: Directory to save FAISS index
        """
        # Load embedding model
        self.model = SentenceTransformer(embedding_model)
        self.dimension = 384  # Model output dimension

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        # IndexFlatL2 = exhaustive search with L2 distance
        # Good for <1M documents; use IndexIVFFlat for >1M

        # Storage
        self.storage_path = storage_path
        self.documents = []  # Raw text
        self.metadata = []   # Associated metadata

        # Load existing index if available
        self._load_from_disk()

    def store(
        self,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Store content in vector memory.

        Args:
            content: Text to store
            metadata: Optional metadata dict

        Example:
            memory.store(
                content="Attention mechanisms enable...",
                metadata={'paper_id': 'arxiv:1706.03762', 'citations': 70000}
            )
        """
        # Generate embedding
        embedding = self.model.encode([content])[0]  # Shape: (384,)

        # Add to FAISS index
        self.index.add(np.array([embedding], dtype=np.float32))

        # Store document and metadata
        self.documents.append(content)
        self.metadata.append(metadata or {})

        logger.debug(f"Stored document (total: {len(self.documents)})")

    def query(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of dicts with keys:
            - content (str): Document text
            - metadata (dict): Associated metadata
            - score (float): Similarity score (L2 distance, lower = more similar)

        Example:
            results = memory.query("How does attention work?", k=3)
            for r in results:
                print(f"Score: {r['score']}, Content: {r['content'][:100]}")
        """
        if self.index.ntotal == 0:
            logger.warning("Memory is empty")
            return []

        # Generate query embedding
        query_embedding = self.model.encode([query])[0]

        # Search FAISS
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            min(k, self.index.ntotal)  # Don't request more than available
        )

        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                doc_metadata = self.metadata[idx]

                # Apply metadata filter if provided
                if filter_metadata:
                    if not all(doc_metadata.get(k) == v for k, v in filter_metadata.items()):
                        continue  # Skip if doesn't match filter

                results.append({
                    'content': self.documents[idx],
                    'metadata': doc_metadata,
                    'score': float(dist)
                })

        return results

    def save_to_disk(self) -> None:
        """Save index, documents, and metadata to disk."""
        os.makedirs(self.storage_path, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(self.storage_path, 'faiss_index.bin')
        faiss.write_index(self.index, index_path)

        # Save documents and metadata
        data_path = os.path.join(self.storage_path, 'documents.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)

        logger.info(f"Saved memory to {self.storage_path}")

    def _load_from_disk(self) -> None:
        """Load index, documents, and metadata from disk."""
        index_path = os.path.join(self.storage_path, 'faiss_index.bin')
        data_path = os.path.join(self.storage_path, 'documents.pkl')

        if os.path.exists(index_path) and os.path.exists(data_path):
            # Load FAISS index
            self.index = faiss.read_index(index_path)

            # Load documents and metadata
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']

            logger.info(f"Loaded {len(self.documents)} documents from memory")

    def clear(self) -> None:
        """Clear all memory (use with caution!)."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
        logger.warning("Memory cleared")
```

---

## 🔄 Memory Usage Patterns

### Pattern 1: Store-and-Retrieve

**Researcher stores, Coder retrieves:**

```python
# Researcher stores papers
def researcher_node(state: AgentState) -> Dict:
    papers = search_arxiv(state['topic'])

    memory = get_memory_manager()
    for paper in papers:
        memory.store(
            content=f"Title: {paper['title']}\nAbstract: {paper['abstract']}",
            metadata={
                'type': 'research_paper',
                'paper_id': paper['entry_id'],
                'source': 'arxiv',
                'topic': state['topic']
            }
        )

    memory.save_to_disk()  # Persist

    return {'research_papers': papers}

# Coder retrieves relevant papers
def coder_node(state: AgentState) -> Dict:
    memory = get_memory_manager()

    # Query for implementation details
    implementation_context = memory.query(
        "implementation details and code examples",
        k=3
    )

    # Generate code using retrieved context
    context_text = "\n\n".join([r['content'] for r in implementation_context])

    prompt = f"""
    Generate Python code based on:

    {context_text}

    Topic: {state['topic']}
    """

    code = llm.generate(prompt)

    return {'generated_code': {'example': code}}
```

### Pattern 2: Filtered Retrieval

**Retrieve only specific types of content:**

```python
# Query with metadata filter
results = memory.query(
    "attention mechanisms",
    k=10,
    filter_metadata={'source': 'arxiv', 'topic': 'transformers'}
)
# Returns only arxiv papers about transformers
```

### Pattern 3: Iterative Enrichment

**Each iteration adds more context:**

```python
# Iteration 1: Basic research
researcher.run("transformers")
memory.store(papers, metadata={'iteration': 1})

# Iteration 2: Feedback-driven research
feedback = "Add more on hybrid attention"
additional_papers = researcher.run(feedback)
memory.store(additional_papers, metadata={'iteration': 2, 'refinement': True})

# Iteration 3: Query accumulated knowledge
all_context = memory.query("hybrid attention", k=20)
# Gets relevant papers from both iterations
```

### Pattern 4: Deduplication

**Avoid storing duplicates:**

```python
def store_if_new(content: str, metadata: Dict) -> bool:
    """Store only if not already in memory."""
    # Check if similar content exists
    existing = memory.query(content, k=1)

    if existing and existing[0]['score'] < 0.1:  # Very similar (low distance)
        logger.info("Duplicate detected, skipping")
        return False

    # Not a duplicate, store it
    memory.store(content, metadata)
    return True
```

---

## 📏 Context Window Management

### Problem: Can't Fit Everything

**Example scenario:**
```python
# We have:
research_papers = 20 papers * 2 pages each = 40 pages
code_examples = 5 examples * 3 pages each = 15 pages
feedback = 10 pages
# Total: 65 pages

# But GPT-4 context window = ~300 pages
# Gemini context window = ~5000 pages

# Seems fine? But consider:
# - System prompt: 5 pages
# - Generated output: 20 pages
# - Safety margin: 50 pages
# Usable: 225 pages

# Now multiply by 3 iterations:
# 65 pages * 3 = 195 pages
# Getting close to limit!
```

### Solution 1: Selective Retrieval

**Don't send everything, query for relevance:**

```python
def generate_code_with_context(state: AgentState) -> str:
    """Generate code using only relevant research."""
    memory = get_memory_manager()

    # Instead of: "Here are all 20 papers"
    # Do: "What papers are relevant for this specific task?"

    task = state.get('current_subtask', state['topic'])

    # Query for top 3 most relevant papers
    relevant_context = memory.query(task, k=3)

    # Much smaller context (3 papers instead of 20)
    context = "\n\n".join([r['content'] for r in relevant_context])

    prompt = f"""
    Generate code for: {task}

    Relevant research:
    {context}
    """

    return llm.generate(prompt)
```

### Solution 2: Summarization

**Compress information before sending to LLM:**

```python
def summarize_papers(papers: List[Dict]) -> str:
    """Summarize multiple papers into compact overview."""
    # Extract key points from each paper
    summaries = []
    for paper in papers:
        summary = f"- {paper['title']}: {paper['abstract'][:200]}..."
        summaries.append(summary)

    # Return compact summary (not full papers)
    return "\n".join(summaries)

# Usage
papers = state['research_papers']
compact_context = summarize_papers(papers)  # Much smaller!

code = llm.generate(f"Based on:\n{compact_context}\nGenerate code...")
```

### Solution 3: Hierarchical Context

**Different agents get different levels of detail:**

```python
# Planner: High-level overview only
def planner_node(state: AgentState) -> Dict:
    papers = state['research_papers']
    overview = [f"{p['title']} ({p['year']})" for p in papers]
    # Just titles, not full abstracts

    plan = llm.generate(f"Create plan based on:\n{overview}")
    return {'plan': plan}

# Coder: Detailed implementation context
def coder_node(state: AgentState) -> Dict:
    memory = get_memory_manager()
    detailed_context = memory.query("implementation", k=5)
    # Full abstracts + methodologies

    code = llm.generate(f"Implement based on:\n{detailed_context}")
    return {'generated_code': code}
```

### Solution 4: Sliding Window

**Keep only recent messages:**

```python
def get_recent_messages(state: AgentState, window_size: int = 10) -> List[Dict]:
    """Get only recent messages for context."""
    messages = state.get('messages', [])
    return messages[-window_size:]  # Last 10 messages only

# Usage in agent
def agent_node(state: AgentState) -> Dict:
    recent_messages = get_recent_messages(state)

    response = llm.generate(messages=recent_messages)
    # Not overwhelmed by full conversation history

    return {'messages': [{'role': 'assistant', 'content': response}]}
```

---

## 🎯 Context Retrieval Strategies

### Strategy 1: Query Decomposition

**Problem:** Single query might miss relevant content.

**Solution:** Break query into sub-queries.

```python
def comprehensive_retrieve(query: str, k: int = 10) -> List[Dict]:
    """Retrieve using multiple query angles."""
    memory = get_memory_manager()

    # Decompose query
    sub_queries = [
        f"What is {query}",                    # Definition
        f"How does {query} work",              # Mechanism
        f"Implementation of {query}",          # Code
        f"Applications of {query}",            # Use cases
        f"Comparison {query} vs alternatives"  # Comparison
    ]

    all_results = []
    for sq in sub_queries:
        results = memory.query(sq, k=2)  # 2 per sub-query
        all_results.extend(results)

    # Deduplicate and rank
    unique_results = deduplicate_by_content(all_results)
    return unique_results[:k]
```

### Strategy 2: Hybrid Search

**Combine semantic search with keyword matching:**

```python
def hybrid_search(query: str, keywords: List[str], k: int = 5) -> List[Dict]:
    """Combine FAISS semantic search with keyword filtering."""
    memory = get_memory_manager()

    # Semantic search (top 20)
    semantic_results = memory.query(query, k=20)

    # Keyword filter
    filtered_results = []
    for result in semantic_results:
        content = result['content'].lower()
        # Check if any keyword present
        if any(kw.lower() in content for kw in keywords):
            filtered_results.append(result)

    # Return top k after filtering
    return filtered_results[:k]

# Usage
results = hybrid_search(
    query="transformer architectures",
    keywords=["attention", "encoder", "decoder"],
    k=5
)
```

### Strategy 3: Re-ranking

**Re-rank results using different criteria:**

```python
def retrieve_with_reranking(query: str, k: int = 5) -> List[Dict]:
    """Retrieve and re-rank by multiple factors."""
    memory = get_memory_manager()

    # Initial retrieval (get more than needed)
    candidates = memory.query(query, k=k * 3)

    # Re-rank by composite score
    def composite_score(result):
        # Factor 1: Semantic similarity (from FAISS)
        similarity = 1.0 / (1.0 + result['score'])  # Lower distance = higher score

        # Factor 2: Citation count (from metadata)
        citations = result['metadata'].get('citations', 0)
        citation_score = min(citations / 10000.0, 1.0)  # Normalize

        # Factor 3: Recency (from metadata)
        year = result['metadata'].get('year', 2000)
        recency_score = (year - 2000) / 24.0  # Normalize to 0-1

        # Weighted combination
        return 0.5 * similarity + 0.3 * citation_score + 0.2 * recency_score

    # Re-rank
    ranked = sorted(candidates, key=composite_score, reverse=True)

    return ranked[:k]
```

---

## 🧪 Testing Memory

### Test Memory Storage and Retrieval

```python
# tests/test_memory.py

def test_memory_store_and_query():
    """Test that memory stores and retrieves correctly."""
    from src.memory.memory_manager import MemoryManager

    memory = MemoryManager()

    # Store test documents
    memory.store(
        "Attention mechanisms enable models to focus on relevant parts",
        metadata={'topic': 'attention'}
    )
    memory.store(
        "Convolutional layers are good for spatial features",
        metadata={'topic': 'cnn'}
    )

    # Query
    results = memory.query("How does attention work?", k=2)

    # Assert: Attention doc is most relevant
    assert len(results) > 0
    assert 'attention' in results[0]['content'].lower()
    assert results[0]['score'] < results[1]['score']  # First is more similar


def test_memory_persistence():
    """Test that memory persists to disk."""
    from src.memory.memory_manager import MemoryManager
    import tempfile
    import shutil

    # Create temporary storage
    temp_dir = tempfile.mkdtemp()

    try:
        # Create and populate memory
        memory1 = MemoryManager(storage_path=temp_dir)
        memory1.store("Test document")
        memory1.save_to_disk()

        # Create new instance (simulates restart)
        memory2 = MemoryManager(storage_path=temp_dir)

        # Assert: Document was loaded
        assert len(memory2.documents) == 1
        assert memory2.documents[0] == "Test document"

    finally:
        shutil.rmtree(temp_dir)
```

---

## 💡 Best Practices

### 1. Store Structured Content

```python
# Bad: Unstructured
memory.store("paper about transformers with 70k citations")

# Good: Structured
memory.store(
    content="Title: Attention Is All You Need\nAbstract: We propose...",
    metadata={
        'type': 'paper',
        'title': 'Attention Is All You Need',
        'citations': 70000,
        'year': 2017
    }
)
```

### 2. Use Meaningful Queries

```python
# Bad: Vague
results = memory.query("stuff", k=5)

# Good: Specific
results = memory.query(
    "How to implement multi-head attention in PyTorch",
    k=5
)
```

### 3. Save Periodically

```python
# After storing batch of documents
for paper in papers:
    memory.store(paper['abstract'], metadata={'paper_id': paper['id']})

memory.save_to_disk()  # Don't forget!
```

### 4. Deduplicate Before Storing

```python
def store_unique(content: str, metadata: Dict):
    """Only store if not already present."""
    existing = memory.query(content, k=1)

    if not existing or existing[0]['score'] > 0.5:  # Not too similar
        memory.store(content, metadata)
    else:
        logger.info("Skipping duplicate")
```

---

## 🚀 Key Takeaways

1. **Two-layer memory**
   - State = short-term (current workflow)
   - FAISS = long-term (persistent)

2. **Semantic search > keyword search**
   - "How does attention work?" finds relevant papers
   - Even if they don't contain exact words

3. **Context is limited**
   - Can't send everything to LLM
   - Query for relevance
   - Summarize when possible

4. **Retrieval strategy matters**
   - Query decomposition
   - Hybrid search
   - Re-ranking

5. **Memory enables iteration**
   - Build on previous work
   - No duplicate effort
   - Accumulate knowledge

---

## 🚀 Next Steps

**Next:** `09_ASYNC_EXECUTION.md` → Production-grade asynchronous code execution

**Exercise:** Modify the memory manager to support:
- Full-text search (keyword matching)
- Date range filtering
- Citation count filtering

**Advanced Exercise:** Implement a "smart retrieval" function that:
1. Analyzes the query
2. Decomposes into sub-queries
3. Retrieves from multiple angles
4. Re-ranks by relevance + citations + recency
5. Returns diverse set of results (not all from same paper)

---

**Key Insight:** Memory is what transforms a collection of independent agents into a cohesive system. Without memory, agents can't learn from experience, can't build on previous work, and can't handle tasks larger than their context window. Memory is the knowledge layer that makes true multi-agent intelligence possible.
