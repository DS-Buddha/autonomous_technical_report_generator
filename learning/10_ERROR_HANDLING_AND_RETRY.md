# Error Handling and Retry Logic: Building Resilient Systems

## 🎯 Why Error Handling Matters

### Production Reality

**In development:**
```python
papers = search_arxiv("transformers")  # Works perfectly
```

**In production:**
```python
papers = search_arxiv("transformers")
# ❌ Network timeout
# ❌ API rate limit exceeded
# ❌ Service temporarily unavailable
# ❌ Invalid response format
# ❌ Partial data corruption
```

**Murphy's Law applies:** If something can fail, it will fail. In production.

### The Cost of No Error Handling

**Scenario:** User submits expensive report generation request.

**Without error handling:**
```python
1. Planner runs (succeeds) ✓
2. Researcher runs (succeeds) ✓
3. Coder runs (succeeds) ✓
4. Tester runs (succeeds) ✓
5. Critic runs (API timeout) ❌
→ Entire workflow crashes
→ User gets error message
→ All previous work lost
→ Must restart from scratch
```

**With error handling:**
```python
5. Critic runs (API timeout) ❌
→ Retry with exponential backoff
→ Succeeds on 2nd attempt ✓
→ Workflow continues
→ User gets report
→ No manual intervention needed
```

---

## 🛡️ Our Error Handling Strategy

### Three-Layer Defense

**Layer 1: Retry Logic (Automatic recovery)**
- Transient failures (network issues, rate limits)
- Exponential backoff with jitter
- Configurable max attempts

**Layer 2: Graceful Degradation (Fallback strategies)**
- Primary source fails → try secondary source
- API unavailable → use cached data
- Code execution fails → return error without crashing workflow

**Layer 3: Fail-Fast (Stop execution when recovery impossible)**
- Invalid API keys
- Malformed input data
- Critical component unavailable

---

## 🔄 Retry Logic Implementation

### The Retry Decorator

**Core implementation:**

```python
# src/utils/retry.py

import time
import random
import logging
from functools import wraps
from typing import Callable, Type, Tuple

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Retry decorator with exponential backoff.

    The delay between retries grows exponentially:
    - Attempt 1: base_delay * 1 = 1s
    - Attempt 2: base_delay * 2 = 2s
    - Attempt 3: base_delay * 4 = 4s
    - ...

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds (cap)
        exponential_base: Multiplier for delay (2.0 = double each time)
        jitter: Add randomness to prevent thundering herd
        exceptions: Tuple of exception types to retry on

    Example:
        @retry_with_exponential_backoff(max_retries=3)
        def fetch_data():
            return requests.get(url)

        # Will retry up to 3 times with delays: ~1s, ~2s, ~4s
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    retries += 1

                    # Max retries exceeded
                    if retries > max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise  # Re-raise the exception

                    # Calculate delay
                    delay = min(
                        base_delay * (exponential_base ** (retries - 1)),
                        max_delay
                    )

                    # Add jitter (random factor between 0.5 and 1.5)
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"{func.__name__} failed (attempt {retries}/{max_retries}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )

                    time.sleep(delay)

            # Should never reach here
            return None

        return wrapper
    return decorator
```

### Usage Examples

**Example 1: arXiv Search**
```python
# src/tools/research_tools.py

@retry_with_exponential_backoff(
    max_retries=3,
    base_delay=1.0,
    exceptions=(requests.RequestException, TimeoutError)
)
def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search arXiv with automatic retry.

    If network fails, retries with delays: ~1s, ~2s, ~4s
    """
    search = arxiv.Search(query=query, max_results=max_results)
    return list(search.results())
```

**Example 2: LLM API Call**
```python
# src/agents/base_agent.py

@retry_with_exponential_backoff(
    max_retries=5,  # More retries for LLM (rate limits common)
    base_delay=2.0,
    max_delay=120.0,
    exceptions=(
        google.api_core.exceptions.ResourceExhausted,  # Rate limit
        google.api_core.exceptions.ServiceUnavailable,  # Service down
        requests.Timeout
    )
)
def call_llm(prompt: str) -> str:
    """
    Call LLM with retry logic.

    Handles:
    - Rate limits (429 errors)
    - Service unavailable (503 errors)
    - Network timeouts
    """
    response = model.generate_content(prompt)
    return response.text
```

**Example 3: Code Execution (Celery Task)**
```python
# src/tasks/code_execution_tasks.py

@celery_app.task(
    bind=True,
    autoretry_for=(OSError, MemoryError),  # Retry on these exceptions
    retry_kwargs={'max_retries': 3},
    retry_backoff=True,  # Exponential backoff
    retry_backoff_max=600,  # Max 10 minutes
    retry_jitter=True  # Add jitter
)
def execute_code_async(self, code: str, timeout: int = 30) -> Dict:
    """
    Execute code with Celery's built-in retry.

    Celery handles retries automatically for specified exceptions.
    """
    return run_code(code, timeout)
```

---

## 🎯 Graceful Degradation Patterns

### Pattern 1: Fallback Chain

**Concept:** Try multiple sources in order of preference.

```python
def search_papers_with_fallbacks(query: str) -> List[Dict]:
    """
    Search with fallback chain:
    1. Try arXiv (most recent, high quality)
    2. If fails, try Semantic Scholar (broader coverage)
    3. If fails, query local memory cache
    4. If fails, return empty list (don't crash)
    """
    logger.info(f"Searching for: {query}")

    # Try primary source: arXiv
    try:
        papers = search_arxiv(query, max_results=10)
        if papers:
            logger.info(f"Found {len(papers)} papers from arXiv")
            return papers
    except Exception as e:
        logger.warning(f"arXiv search failed: {e}, trying Semantic Scholar")

    # Try secondary source: Semantic Scholar
    try:
        papers = search_semantic_scholar(query, max_results=10)
        if papers:
            logger.info(f"Found {len(papers)} papers from Semantic Scholar")
            return papers
    except Exception as e:
        logger.warning(f"Semantic Scholar failed: {e}, trying memory cache")

    # Try tertiary source: Memory cache
    try:
        memory = get_memory_manager()
        results = memory.query(query, k=10)
        if results:
            # Convert memory results to paper format
            papers = [
                {
                    'title': r['content'][:100],
                    'abstract': r['content'],
                    'source': 'memory_cache'
                }
                for r in results
            ]
            logger.info(f"Found {len(papers)} papers from memory cache")
            return papers
    except Exception as e:
        logger.error(f"Memory cache query failed: {e}")

    # All sources failed
    logger.error(f"All sources failed for query: {query}")
    return []  # Return empty, don't crash
```

### Pattern 2: Partial Success

**Concept:** Return partial results instead of failing completely.

```python
def search_multiple_sources(query: str) -> Dict[str, List[Dict]]:
    """
    Search multiple sources, return whatever succeeds.

    Returns:
        {
            'arxiv': [...],  # May be empty if failed
            'semantic_scholar': [...],  # May be empty if failed
            'success_count': int,
            'failure_count': int
        }
    """
    results = {
        'arxiv': [],
        'semantic_scholar': [],
        'success_count': 0,
        'failure_count': 0
    }

    # Try arXiv (don't fail if it fails)
    try:
        results['arxiv'] = search_arxiv(query)
        results['success_count'] += 1
    except Exception as e:
        logger.warning(f"arXiv search failed: {e}")
        results['failure_count'] += 1

    # Try Semantic Scholar (independent of arXiv)
    try:
        results['semantic_scholar'] = search_semantic_scholar(query)
        results['success_count'] += 1
    except Exception as e:
        logger.warning(f"Semantic Scholar failed: {e}")
        results['failure_count'] += 1

    # Return whatever we got
    return results
```

### Pattern 3: Default Values

**Concept:** Use sensible defaults when data unavailable.

```python
def get_paper_metadata(paper_id: str) -> Dict:
    """
    Get paper metadata with defaults.

    If API fails, returns partial metadata instead of crashing.
    """
    try:
        # Try to fetch from API
        metadata = api.get_paper(paper_id)
        return metadata

    except Exception as e:
        logger.warning(f"Failed to fetch metadata for {paper_id}: {e}")

        # Return default metadata
        return {
            'paper_id': paper_id,
            'title': 'Unknown',
            'authors': [],
            'year': None,
            'citations': 0,
            'abstract': '',
            'source': 'fallback',
            'error': str(e)
        }
```

### Pattern 4: Cached Fallback

**Concept:** Use cached data when live data unavailable.

```python
import pickle
import os
from datetime import datetime, timedelta

def search_with_cache(query: str, cache_ttl_hours: int = 24) -> List[Dict]:
    """
    Search with file-based cache fallback.

    Returns cached results if:
    1. API call fails, OR
    2. Cache is fresh (< cache_ttl_hours old)
    """
    cache_key = hashlib.md5(query.encode()).hexdigest()
    cache_file = f"data/cache/{cache_key}.pkl"

    # Check cache freshness
    if os.path.exists(cache_file):
        cache_age = datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(cache_file)
        )
        if cache_age < timedelta(hours=cache_ttl_hours):
            logger.info(f"Using fresh cache for: {query}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

    # Try API call
    try:
        results = search_api(query)

        # Update cache
        os.makedirs('data/cache', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)

        return results

    except Exception as e:
        logger.warning(f"API failed: {e}, using stale cache if available")

        # Fallback to stale cache
        if os.path.exists(cache_file):
            logger.info(f"Using stale cache for: {query}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # No cache available
        logger.error(f"No cache available for: {query}")
        return []
```

---

## ⚠️ Error Classification

### Transient Errors (Retry)

**Network issues:**
```python
requests.exceptions.ConnectionError
requests.exceptions.Timeout
socket.timeout
```

**Rate limits:**
```python
google.api_core.exceptions.ResourceExhausted  # 429 Too Many Requests
requests.exceptions.HTTPError  # When status_code == 429
```

**Service unavailable:**
```python
google.api_core.exceptions.ServiceUnavailable  # 503 Service Unavailable
requests.exceptions.HTTPError  # When status_code == 503
```

**Retry strategy:**
```python
@retry_with_exponential_backoff(
    max_retries=5,
    base_delay=2.0,
    exceptions=(
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable
    )
)
```

### Permanent Errors (Don't Retry)

**Authentication errors:**
```python
google.api_core.exceptions.Unauthenticated  # 401 Unauthorized
google.api_core.exceptions.PermissionDenied  # 403 Forbidden
```

**Invalid input:**
```python
ValueError  # Invalid parameters
TypeError  # Wrong type
KeyError  # Missing required field
```

**Resource not found:**
```python
google.api_core.exceptions.NotFound  # 404 Not Found
FileNotFoundError
```

**Don't retry these:**
```python
def api_call(params):
    try:
        return api.request(params)
    except google.api_core.exceptions.Unauthenticated:
        logger.error("Invalid API key, retrying won't help")
        raise  # Don't retry, fail fast
    except ValueError as e:
        logger.error(f"Invalid parameters: {e}, retrying won't help")
        raise  # Don't retry, fail fast
```

---

## 🧪 Testing Error Handling

### Test Retry Logic

```python
# tests/test_retry.py

import pytest
from unittest.mock import Mock, patch
from src.utils.retry import retry_with_exponential_backoff

def test_retry_succeeds_on_second_attempt():
    """Test that retry works when function succeeds on retry."""
    # Create mock that fails once, then succeeds
    mock_func = Mock(side_effect=[Exception("Fail"), "Success"])

    @retry_with_exponential_backoff(max_retries=2, base_delay=0.1)
    def wrapper():
        return mock_func()

    result = wrapper()

    assert result == "Success"
    assert mock_func.call_count == 2  # Failed once, succeeded on retry


def test_retry_fails_after_max_retries():
    """Test that retry gives up after max attempts."""
    mock_func = Mock(side_effect=Exception("Always fails"))

    @retry_with_exponential_backoff(max_retries=3, base_delay=0.1)
    def wrapper():
        return mock_func()

    with pytest.raises(Exception, match="Always fails"):
        wrapper()

    assert mock_func.call_count == 4  # Initial + 3 retries
```

### Test Fallback Logic

```python
def test_fallback_chain():
    """Test that fallback chain works correctly."""
    with patch('src.tools.research_tools.search_arxiv') as mock_arxiv, \
         patch('src.tools.research_tools.search_semantic_scholar') as mock_ss:

        # arXiv fails
        mock_arxiv.side_effect = Exception("arXiv down")

        # Semantic Scholar succeeds
        mock_ss.return_value = [{'title': 'Paper from SS'}]

        results = search_papers_with_fallbacks("test query")

        # Should have used Semantic Scholar
        assert len(results) == 1
        assert results[0]['title'] == 'Paper from SS'


def test_all_fallbacks_fail():
    """Test graceful handling when all sources fail."""
    with patch('src.tools.research_tools.search_arxiv') as mock_arxiv, \
         patch('src.tools.research_tools.search_semantic_scholar') as mock_ss, \
         patch('src.memory.memory_manager.get_memory_manager') as mock_memory:

        # All sources fail
        mock_arxiv.side_effect = Exception("Fail")
        mock_ss.side_effect = Exception("Fail")
        mock_memory.return_value.query.side_effect = Exception("Fail")

        results = search_papers_with_fallbacks("test query")

        # Should return empty list, not crash
        assert results == []
```

---

## 📊 Error Monitoring

### Logging Errors

**Structured logging:**
```python
import logging

logger = logging.getLogger(__name__)

def search_with_monitoring(query: str) -> List[Dict]:
    """Search with comprehensive error logging."""
    logger.info(f"Starting search for: {query}")

    try:
        results = search_api(query)
        logger.info(f"Search succeeded: {len(results)} results")
        return results

    except requests.exceptions.Timeout as e:
        logger.warning(
            f"Search timed out for query: {query}",
            extra={
                'error_type': 'timeout',
                'query': query,
                'exception': str(e)
            }
        )
        raise

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning(
                f"Rate limit hit for query: {query}",
                extra={
                    'error_type': 'rate_limit',
                    'query': query,
                    'status_code': e.response.status_code
                }
            )
        else:
            logger.error(
                f"HTTP error for query: {query}",
                extra={
                    'error_type': 'http_error',
                    'query': query,
                    'status_code': e.response.status_code,
                    'response': e.response.text
                }
            )
        raise

    except Exception as e:
        logger.error(
            f"Unexpected error for query: {query}",
            extra={
                'error_type': 'unexpected',
                'query': query,
                'exception_type': type(e).__name__,
                'exception': str(e)
            },
            exc_info=True  # Include stack trace
        )
        raise
```

### Error Metrics

**Track error rates:**
```python
from collections import defaultdict
from datetime import datetime

class ErrorTracker:
    """Track error rates for monitoring."""

    def __init__(self):
        self.errors = defaultdict(int)
        self.successes = defaultdict(int)

    def record_error(self, function_name: str, error_type: str):
        """Record an error."""
        key = f"{function_name}:{error_type}"
        self.errors[key] += 1

    def record_success(self, function_name: str):
        """Record a success."""
        self.successes[function_name] += 1

    def get_error_rate(self, function_name: str) -> float:
        """Get error rate for a function."""
        total = self.successes[function_name] + sum(
            count for key, count in self.errors.items()
            if key.startswith(function_name)
        )
        if total == 0:
            return 0.0

        errors = sum(
            count for key, count in self.errors.items()
            if key.startswith(function_name)
        )
        return errors / total

# Global tracker
error_tracker = ErrorTracker()

# Usage
def monitored_search(query: str):
    try:
        results = search_api(query)
        error_tracker.record_success('search_api')
        return results
    except Exception as e:
        error_tracker.record_error('search_api', type(e).__name__)
        raise
```

---

## 💡 Best Practices

### 1. Fail Fast for Unrecoverable Errors

```python
# Good: Check auth upfront
def initialize_api():
    try:
        api.authenticate(api_key)
    except AuthenticationError:
        logger.error("Invalid API key, cannot proceed")
        sys.exit(1)  # Fail fast, don't retry

# Bad: Keep retrying auth errors
@retry_with_exponential_backoff(max_retries=10)  # Wasteful!
def api_call():
    return api.request()  # Will keep failing if key is invalid
```

### 2. Use Specific Exception Types

```python
# Good: Catch specific exceptions
try:
    result = api.call()
except requests.exceptions.Timeout:
    # Handle timeout specifically
    return cached_result
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        # Handle rate limit
        time.sleep(60)
        return retry()
    raise

# Bad: Catch everything
try:
    result = api.call()
except Exception:
    # What went wrong? Can't tell!
    return None
```

### 3. Log Context

```python
# Good: Log what you were doing
logger.error(f"Failed to search arXiv for query: {query}, error: {e}")

# Bad: Log just the error
logger.error(str(e))  # What was the query? What function?
```

### 4. Set Reasonable Timeouts

```python
# Good: Explicit timeout
response = requests.get(url, timeout=30)

# Bad: No timeout (can hang forever)
response = requests.get(url)
```

### 5. Don't Suppress Errors Silently

```python
# Good: Log and handle
try:
    result = operation()
except Exception as e:
    logger.warning(f"Operation failed: {e}, using fallback")
    result = fallback()

# Bad: Silent suppression
try:
    result = operation()
except:
    pass  # Error disappeared, good luck debugging!
```

---

## 🚀 Key Takeaways

1. **Three-layer defense**
   - Retry (automatic recovery)
   - Fallback (graceful degradation)
   - Fail-fast (stop when hopeless)

2. **Classify errors**
   - Transient → Retry
   - Permanent → Fail fast
   - Unknown → Log and investigate

3. **Exponential backoff prevents hammering**
   - Delays grow: 1s, 2s, 4s, 8s, ...
   - Add jitter to prevent thundering herd
   - Cap maximum delay

4. **Fallback chains provide resilience**
   - Primary source → Secondary → Cache → Empty
   - Partial success better than total failure

5. **Monitor and log everything**
   - Track error rates
   - Log context, not just messages
   - Alert on anomalies

---

## 🚀 Next Steps

**Next:** `11_MONITORING_AND_OBSERVABILITY.md` → How to monitor production systems

**Exercise:** Add retry logic to a function that currently has none. Test with mocked failures.

**Advanced Exercise:** Implement a circuit breaker pattern that:
- Tracks failure rate for an API
- "Opens" circuit (stops calling) if failure rate > 50%
- Periodically tries again ("half-open")
- "Closes" circuit when service recovers

---

**Key Insight:** Error handling is not defensive programming - it's production readiness. A system without error handling is a prototype. A system with comprehensive error handling is production-grade. The difference is 90% of the engineering effort.
