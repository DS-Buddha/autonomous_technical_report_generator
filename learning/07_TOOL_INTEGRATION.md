# Tool Integration: Connecting Agents to the Outside World

## 🎯 Why Tools Matter

### The Problem: LLMs Can't Do Everything

**LLMs are great at:**
- Text generation
- Code generation
- Reasoning and analysis
- Pattern recognition

**LLMs cannot:**
- Search the web
- Query databases
- Execute code
- Access real-time data
- Call APIs

**Solution: Tools**

Tools extend LLM capabilities by letting them invoke external functions.

```python
# Without tools (limited)
response = llm.generate("What papers exist on RAG systems?")
# Result: Hallucinated papers, outdated info

# With tools (powerful)
papers = search_arxiv("RAG systems")  # Real API call!
response = llm.generate(f"Summarize these papers: {papers}")
# Result: Accurate, up-to-date information
```

---

## 🔧 Our Tool Architecture

### Tool Categories

We have **three categories** of tools:

**1. Research Tools** (`src/tools/research_tools.py`)
- Search academic papers (arXiv, Semantic Scholar)
- Extract citations and metadata
- Download and parse PDFs

**2. Code Tools** (`src/tools/code_tools.py`)
- Execute Python code
- Validate syntax
- Capture stdout/stderr
- Handle timeouts

**3. Memory Tools** (`src/memory/memory_manager.py`)
- Store embeddings in FAISS
- Semantic search
- Context retrieval

---

## 📚 Research Tools Deep-Dive

### Tool 1: arXiv Search

**Purpose:** Search academic papers on arXiv.

**Code:**
```python
# src/tools/research_tools.py

import arxiv
from typing import List, Dict

@retry_with_exponential_backoff(max_retries=3)
def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
) -> List[Dict]:
    """
    Search arXiv for academic papers.

    Args:
        query: Search query (e.g., "attention mechanisms transformers")
        max_results: Maximum papers to return
        sort_by: Sort order (Relevance, LastUpdatedDate, SubmittedDate)

    Returns:
        List of paper dicts with keys:
        - title, authors, abstract, published, pdf_url, entry_id

    Example:
        papers = search_arxiv("RAG systems", max_results=5)
        for paper in papers:
            print(f"{paper['title']} by {paper['authors'][0]}")
    """
    logger.info(f"Searching arXiv for: {query}")

    try:
        # Create search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by
        )

        # Execute search
        results = []
        for result in search.results():
            results.append({
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'published': result.published.isoformat(),
                'pdf_url': result.pdf_url,
                'entry_id': result.entry_id,
                'categories': result.categories,
                'primary_category': result.primary_category
            })

        logger.info(f"Found {len(results)} papers on arXiv")
        return results

    except Exception as e:
        logger.error(f"arXiv search failed: {e}")
        return []
```

**Usage in agents:**
```python
# src/agents/researcher_agent.py

def run(self, state: Dict) -> Dict:
    topic = state['topic']
    depth = state['depth']

    # Determine how many papers based on depth
    max_results = 20 if depth == 'comprehensive' else 10

    # Search arXiv
    papers = search_arxiv(topic, max_results=max_results)

    return {'research_papers': papers}
```

### Tool 2: Semantic Scholar Search

**Purpose:** Search papers with citation data and influence metrics.

**Code:**
```python
import requests

@retry_with_exponential_backoff(max_retries=3)
def search_semantic_scholar(
    query: str,
    max_results: int = 10,
    fields: List[str] = None
) -> List[Dict]:
    """
    Search Semantic Scholar for papers with citation data.

    Args:
        query: Search query
        max_results: Maximum papers to return
        fields: Fields to return (default: title, authors, abstract, citations)

    Returns:
        List of paper dicts

    API Docs: https://api.semanticscholar.org/
    """
    if fields is None:
        fields = ['title', 'authors', 'abstract', 'year', 'citationCount',
                  'influentialCitationCount', 'url', 'openAccessPdf']

    logger.info(f"Searching Semantic Scholar for: {query}")

    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': max_results,
            'fields': ','.join(fields)
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        papers = data.get('data', [])

        logger.info(f"Found {len(papers)} papers on Semantic Scholar")
        return papers

    except requests.RequestException as e:
        logger.error(f"Semantic Scholar search failed: {e}")
        return []
```

**Why use both arXiv and Semantic Scholar?**
- **arXiv**: More recent papers, full PDFs, preprints
- **Semantic Scholar**: Citation counts, influence metrics, broader coverage

**Combining results:**
```python
def search_comprehensive(query: str) -> List[Dict]:
    """Search both sources and merge."""

    # Parallel searches
    arxiv_papers = search_arxiv(query, max_results=10)
    ss_papers = search_semantic_scholar(query, max_results=10)

    # Merge and deduplicate by title
    all_papers = arxiv_papers + ss_papers
    unique_papers = deduplicate_papers(all_papers, key='title')

    # Sort by citations (Semantic Scholar) or recency (arXiv)
    sorted_papers = sort_papers_by_relevance(unique_papers)

    return sorted_papers[:15]  # Top 15
```

### Tool 3: Extract Key Findings

**Purpose:** Parse papers to extract actionable insights.

**Code:**
```python
def extract_key_findings(papers: List[Dict]) -> Dict[str, Any]:
    """
    Extract key findings from research papers.

    Returns:
        {
            'concepts': List[str],       # Main concepts discussed
            'methodologies': List[str],  # Approaches used
            'experiments': List[Dict],   # Experimental setups
            'results': List[str]         # Key results
        }
    """
    logger.info(f"Extracting findings from {len(papers)} papers")

    # Aggregate all abstracts
    abstracts = [p['abstract'] for p in papers if p.get('abstract')]

    # Use LLM to extract structured findings
    prompt = f"""
    Analyze these research paper abstracts and extract:

    1. Key Concepts: Main ideas/techniques discussed
    2. Methodologies: Approaches used (supervised, unsupervised, etc.)
    3. Experiments: Experimental setups described
    4. Results: Key findings and improvements

    Abstracts:
    {json.dumps(abstracts, indent=2)}

    Return JSON with keys: concepts, methodologies, experiments, results
    """

    response = llm.generate(prompt)
    findings = parse_json_with_fallback(response)

    logger.info(f"Extracted {len(findings.get('concepts', []))} concepts")

    return findings
```

---

## 💻 Code Tools Deep-Dive

### Tool: Code Execution

**Purpose:** Execute Python code in isolated environment.

**Modes:**
1. **Sync**: Direct subprocess execution (fast, for trusted code)
2. **Async**: Celery worker execution (safe, for untrusted code)

**Sync Implementation:**
```python
# src/tools/code_tools.py

def _execute_sync(code: str, timeout: int) -> Dict:
    """
    Execute code in subprocess with timeout.

    Returns:
        {
            'success': bool,
            'stdout': str,
            'stderr': str,
            'execution_time': float
        }
    """
    import subprocess
    import tempfile
    import time

    # Write code to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        code_file = f.name

    try:
        start_time = time.time()

        # Execute with timeout
        result = subprocess.run(
            ['python', code_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )

        execution_time = time.time() - start_time

        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': execution_time
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Execution timed out after {timeout}s',
            'execution_time': timeout
        }

    finally:
        # Cleanup
        import os
        os.unlink(code_file)
```

**Async Implementation (Celery):**
```python
# src/tasks/code_execution_tasks.py

@celery_app.task(bind=True, soft_time_limit=90, time_limit=120)
def execute_code_async(self, code: str, timeout: int = 30) -> Dict:
    """
    Execute code in Celery worker with resource limits.

    Resource Limits:
    - Memory: 1GB
    - CPU time: 120s
    - File size: 100MB
    - Process count: 10
    """
    import resource

    # Set resource limits
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, -1))  # 1GB memory
    resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))       # CPU time

    # Execute code (similar to sync version)
    # ... execution logic ...

    return result
```

**Usage:**
```python
# Sync mode (fast, for our generated code)
result = CodeTools.execute_code(code, async_mode=False)

# Async mode (safe, for user code)
task = CodeTools.execute_code(code, async_mode=True, wait_for_result=False)
task_id = task['task_id']

# Check later
result = CodeTools.get_task_result(task_id)
```

### Code Validation

**Purpose:** Validate code before execution.

**Checks:**
```python
def validate_code(code: str) -> Dict[str, Any]:
    """
    Validate code without executing.

    Checks:
    1. Syntax errors (AST parsing)
    2. Forbidden imports (os, subprocess, etc.)
    3. Dangerous functions (eval, exec, __import__)
    4. File operations
    """
    import ast

    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check 1: Syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        validation_result['valid'] = False
        validation_result['errors'].append(f"Syntax error: {e}")
        return validation_result  # Can't continue if syntax is invalid

    # Check 2: Forbidden imports
    tree = ast.parse(code)
    forbidden_modules = {'os', 'subprocess', 'sys', 'shutil', '__builtin__'}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    validation_result['warnings'].append(
                        f"Potentially dangerous import: {alias.name}"
                    )

    # Check 3: Dangerous functions
    dangerous_funcs = {'eval', 'exec', '__import__', 'compile'}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in dangerous_funcs:
                    validation_result['warnings'].append(
                        f"Potentially dangerous function: {node.func.id}"
                    )

    return validation_result
```

---

## 🧠 Memory Tools Deep-Dive

### Tool: Vector Storage

**Purpose:** Store and retrieve semantic information.

**Implementation:**
```python
# src/memory/memory_manager.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class MemoryManager:
    """
    FAISS-based vector store for semantic memory.
    """

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model)
        self.dimension = 384  # Model output dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []

    def store(self, content: str, metadata: Dict = None):
        """
        Store content in vector store.

        Args:
            content: Text to store
            metadata: Associated metadata (source, paper_id, etc.)
        """
        # Generate embedding
        embedding = self.model.encode([content])[0]

        # Add to FAISS index
        self.index.add(np.array([embedding], dtype=np.float32))

        # Store document and metadata
        self.documents.append(content)
        self.metadata.append(metadata or {})

    def query(self, query: str, k: int = 5) -> List[Dict]:
        """
        Semantic search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of dicts with keys: content, metadata, score
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]

        # Search FAISS
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            k
        )

        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(dist)  # L2 distance (lower = more similar)
                })

        return results
```

**Usage in agents:**
```python
# Researcher stores findings
def researcher_node(state: AgentState) -> Dict:
    papers = search_papers(state['topic'])

    memory = get_memory_manager()
    for paper in papers:
        memory.store(
            content=f"{paper['title']}: {paper['abstract']}",
            metadata={
                'type': 'paper',
                'paper_id': paper['id'],
                'citations': paper.get('citationCount', 0)
            }
        )

    return {'research_papers': papers}

# Coder retrieves relevant context
def coder_node(state: AgentState) -> Dict:
    memory = get_memory_manager()

    # Query for implementation details
    relevant_papers = memory.query(
        "How to implement attention mechanism?",
        k=3
    )

    # Generate code using retrieved context
    context = "\n\n".join([r['content'] for r in relevant_papers])
    code = llm.generate(f"Generate code based on:\n{context}")

    return {'generated_code': {'attention': code}}
```

---

## 🔗 Tool Integration Patterns

### Pattern 1: Retry with Exponential Backoff

**Problem:** API calls can fail transiently (network issues, rate limits).

**Solution:** Automatic retry with increasing delays.

**Implementation:**
```python
# src/utils/retry.py

import time
import random
from functools import wraps

def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Multiplier for delay (2.0 = double each time)
        jitter: Add randomness to prevent thundering herd

    Example:
        @retry_with_exponential_backoff(max_retries=3)
        def fetch_data():
            return requests.get(url)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    retries += 1

                    if retries > max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries")
                        raise

                    # Calculate delay
                    delay = min(base_delay * (exponential_base ** (retries - 1)), max_delay)

                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"{func.__name__} failed (attempt {retries}/{max_retries}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )

                    time.sleep(delay)

            return None  # Should never reach here

        return wrapper
    return decorator
```

**Usage:**
```python
@retry_with_exponential_backoff(max_retries=3)
def search_arxiv(query: str) -> List[Dict]:
    # If this raises an exception, it will retry up to 3 times
    # with delays: 1s, 2s, 4s (plus jitter)
    return arxiv.Search(query=query).results()
```

### Pattern 2: Graceful Degradation

**Problem:** Tool failure shouldn't crash entire workflow.

**Solution:** Fallback strategies.

**Implementation:**
```python
def search_papers_with_fallback(query: str) -> List[Dict]:
    """
    Search with fallback chain:
    1. Try arXiv
    2. If fails, try Semantic Scholar
    3. If fails, query memory cache
    4. If fails, return empty (don't crash)
    """
    # Try primary source
    try:
        papers = search_arxiv(query, max_results=10)
        if papers:
            return papers
    except Exception as e:
        logger.warning(f"arXiv failed: {e}, trying fallback")

    # Try secondary source
    try:
        papers = search_semantic_scholar(query, max_results=10)
        if papers:
            return papers
    except Exception as e:
        logger.warning(f"Semantic Scholar failed: {e}, trying memory")

    # Try memory cache
    try:
        memory = get_memory_manager()
        results = memory.query(query, k=10)
        if results:
            # Convert memory results to paper format
            papers = [{'title': r['content'][:100], 'source': 'memory'}
                      for r in results]
            return papers
    except Exception as e:
        logger.error(f"All sources failed: {e}")

    # Final fallback: empty list
    logger.error("No papers found from any source")
    return []
```

### Pattern 3: Result Caching

**Problem:** Same search query called multiple times wastes API quota.

**Solution:** Cache results.

**Implementation:**
```python
from functools import lru_cache
import hashlib
import json

class CachedSearcher:
    """Search with persistent cache."""

    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, query: str, params: Dict) -> str:
        """Generate cache key from query and params."""
        data = json.dumps({'query': query, 'params': params}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def search_arxiv_cached(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search with file-based cache."""
        cache_key = self._cache_key(query, {'max_results': max_results})
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        # Check cache
        if os.path.exists(cache_file):
            # Check if cache is fresh (< 1 day old)
            if time.time() - os.path.getmtime(cache_file) < 86400:
                logger.info(f"Cache hit for query: {query}")
                with open(cache_file, 'r') as f:
                    return json.load(f)

        # Cache miss, fetch from API
        logger.info(f"Cache miss for query: {query}, fetching from arXiv")
        results = search_arxiv(query, max_results=max_results)

        # Write to cache
        with open(cache_file, 'w') as f:
            json.dump(results, f)

        return results
```

---

## 🧪 Testing Tools

### Test Tool Functionality

```python
# tests/test_research_tools.py

def test_arxiv_search():
    """Test arXiv search returns valid papers."""
    from src.tools.research_tools import search_arxiv

    papers = search_arxiv("attention mechanisms", max_results=5)

    assert len(papers) > 0
    assert len(papers) <= 5

    # Check paper structure
    required_keys = {'title', 'authors', 'abstract', 'published'}
    for paper in papers:
        assert required_keys.issubset(paper.keys())
        assert isinstance(paper['authors'], list)
```

### Test Tool Integration

```python
def test_researcher_uses_search_tool():
    """Test that ResearcherAgent uses search tools."""
    from src.agents.researcher_agent import ResearcherAgent

    agent = ResearcherAgent()
    state = {'topic': 'transformers', 'depth': 'basic'}

    result = agent.run(state)

    # Assert: Agent used search tool
    assert 'research_papers' in result
    assert len(result['research_papers']) > 0
```

### Mock Tools for Testing

```python
import pytest
from unittest.mock import patch

@patch('src.tools.research_tools.search_arxiv')
def test_with_mocked_search(mock_search):
    """Test agent with mocked search tool."""
    # Mock returns fake papers
    mock_search.return_value = [
        {'title': 'Fake Paper', 'abstract': 'Fake abstract'}
    ]

    from src.agents.researcher_agent import ResearcherAgent
    agent = ResearcherAgent()
    result = agent.run({'topic': 'test'})

    # Assert: Agent processed mocked data
    assert len(result['research_papers']) == 1
    assert result['research_papers'][0]['title'] == 'Fake Paper'

    # Assert: Tool was called correctly
    mock_search.assert_called_once()
```

---

## 💡 Best Practices

### 1. Always Add Timeouts

```python
# Bad
response = requests.get(url)  # Can hang forever

# Good
response = requests.get(url, timeout=30)  # 30s timeout
```

### 2. Handle Errors Gracefully

```python
# Bad
def search(query):
    return api.search(query)  # Crashes if API down

# Good
def search(query):
    try:
        return api.search(query)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []  # Return empty, don't crash
```

### 3. Log Tool Usage

```python
def search_tool(query: str):
    logger.info(f"[Tool] Searching for: {query}")
    results = api.search(query)
    logger.info(f"[Tool] Found {len(results)} results")
    return results
```

### 4. Validate Tool Outputs

```python
def search_tool(query: str) -> List[Dict]:
    results = api.search(query)

    # Validate
    if not isinstance(results, list):
        logger.error(f"Invalid results type: {type(results)}")
        return []

    # Filter invalid entries
    valid_results = [r for r in results if 'title' in r and 'abstract' in r]

    return valid_results
```

### 5. Document Tool Interfaces

```python
def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search arXiv for academic papers.

    Args:
        query: Search query (e.g., "attention mechanisms")
        max_results: Maximum papers to return (default: 10)

    Returns:
        List of paper dicts with keys:
        - title (str): Paper title
        - authors (List[str]): Author names
        - abstract (str): Paper abstract
        - published (str): Publication date (ISO format)
        - pdf_url (str): URL to PDF

    Raises:
        APIError: If arXiv API is unavailable

    Example:
        >>> papers = search_arxiv("transformers", max_results=5)
        >>> print(papers[0]['title'])
        'Attention Is All You Need'
    """
    # Implementation...
```

---

## 🚀 Key Takeaways

1. **Tools extend LLM capabilities**
   - LLMs generate text, tools take action
   - Research tools = real data, not hallucinations
   - Code tools = actual execution, not just generation

2. **Always retry with backoff**
   - Network calls fail
   - Rate limits happen
   - Exponential backoff + jitter prevents thundering herd

3. **Fail gracefully**
   - Fallback chains
   - Return empty, not crash
   - Log errors for debugging

4. **Cache aggressively**
   - Same queries happen repeatedly
   - Save API quota
   - Faster response times

5. **Test tools independently**
   - Unit test each tool
   - Mock external APIs
   - Test error handling

---

## 🚀 Next Steps

**Next:** `08_MEMORY_AND_CONTEXT.md` → How to manage long-term memory and context windows

**Exercise:** Add a new research tool that searches Google Scholar. Follow the same pattern as `search_arxiv()`.

**Advanced Exercise:** Implement a caching layer that works across all research tools. Cache should:
- Store results by query hash
- Expire after 24 hours
- Track cache hit rate
- Support cache invalidation

---

**Key Insight:** The quality of your tools determines the quality of your agents. An agent with access to reliable, well-designed tools will outperform a more sophisticated agent with poor tools every time. Invest in tool quality.
