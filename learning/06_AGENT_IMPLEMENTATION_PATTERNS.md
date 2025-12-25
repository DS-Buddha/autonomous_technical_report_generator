# Agent Implementation Patterns

## 🎯 What You'll Learn

How to **implement agents from scratch** following production-grade patterns:
1. Agent design principles
2. System prompt engineering
3. Tool integration patterns
4. Error handling strategies
5. Testing and debugging

**Time:** 4-6 hours

---

## 🏗️ Anatomy of a Production Agent

### The Perfect Agent Structure

```python
"""
Agent: [Name]
Role: [One-sentence description]
Responsibilities: [Bulleted list]
"""

from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger
from src.utils.retry import retry_with_exponential_backoff

logger = get_logger(__name__)

# System prompt (module-level for testability)
AGENT_PROMPT = """You are a [Role] with expertise in [Domain].

Your responsibilities:
1. [Responsibility 1]
2. [Responsibility 2]
3. [Responsibility 3]

Approach:
- [Guideline 1]
- [Guideline 2]

Output format:
[Expected JSON/text structure]

Quality criteria:
- [Criterion 1]
- [Criterion 2]
"""


class MyAgent(BaseAgent):
    """
    [Agent name] for [purpose].

    This agent [detailed description of what it does].
    """

    def __init__(self):
        super().__init__(
            name="MyAgent",
            system_prompt=AGENT_PROMPT,
            temperature=0.7  # Adjust based on task
        )

    def run(self, state: Dict, **kwargs) -> Dict:
        """
        Main execution logic.

        Args:
            state: Workflow state
            **kwargs: Additional parameters

        Returns:
            State updates dictionary

        Raises:
            ValueError: If required state fields missing
            RuntimeError: If critical operations fail
        """
        # 1. Extract inputs
        required_field = state.get('field')
        if not required_field:
            raise ValueError("Missing required field")

        logger.info(f"[{self.name}] Starting with input: {required_field}")

        try:
            # 2. Perform work
            result = self._do_work(required_field)

            # 3. Validate output
            if not self._validate(result):
                logger.warning("Output validation failed")
                # Handle gracefully

            # 4. Return state updates
            logger.info(f"[{self.name}] Completed successfully")
            return {
                'output_field': result,
                'messages': [{
                    'role': 'assistant',
                    'content': f"{self.name}: Processed successfully"
                }]
            }

        except Exception as e:
            logger.error(f"[{self.name}] Failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'messages': [{
                    'role': 'assistant',
                    'content': f"{self.name}: Failed with error"
                }]
            }

    def _do_work(self, input_data: Any) -> Any:
        """
        Internal work logic.

        Separated for testability.
        """
        pass

    def _validate(self, output: Any) -> bool:
        """
        Validate output quality.

        Returns:
            True if valid, False otherwise
        """
        pass
```

---

## 🎨 Pattern 1: Research Agent

### Use Case
Search external data sources, extract structured information.

### Implementation

```python
class ResearcherAgent(BaseAgent):
    """
    Research agent that searches academic papers and extracts findings.

    Pattern: Search → Filter → Extract → Store
    """

    def run(self, state: Dict, **kwargs) -> Dict:
        topic = state['topic']
        depth = state['depth']

        logger.info(f"Researching: {topic} (depth: {depth})")

        # Pattern: Generate queries first
        queries = self._generate_queries(topic, depth)

        # Pattern: Parallel search with retry
        papers = self._search_papers_parallel(queries)

        # Pattern: Filter by quality
        papers = self._filter_by_quality(papers)

        # Pattern: Extract structured data
        findings = self._extract_findings(papers)

        # Pattern: Store in memory for later retrieval
        self._store_in_memory(papers)

        return {
            'research_papers': papers,
            'key_findings': findings,
            'messages': [{
                'role': 'assistant',
                'content': f"Researched {len(papers)} papers, extracted {len(findings)} findings"
            }]
        }

    def _generate_queries(self, topic: str, depth: str) -> List[str]:
        """
        Generate search queries using LLM.

        Pattern: Prompt LLM for structured output
        """
        prompt = f"""Generate {3 if depth == 'basic' else 5} search queries for: {topic}

Return as JSON array: ["query1", "query2", ...]"""

        response = self.generate_response(prompt)

        try:
            import json
            queries = json.loads(response)
            return queries
        except:
            # Fallback: Use topic as single query
            return [topic]

    @retry_with_exponential_backoff(max_retries=3)
    def _search_papers_parallel(self, queries: List[str]) -> List[Dict]:
        """
        Search papers with retry on failure.

        Pattern: Parallel execution with ThreadPoolExecutor
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.tools.research_tools import search_arxiv, search_semantic_scholar

        papers = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            futures = []
            for query in queries:
                futures.append(executor.submit(search_arxiv, query))
                futures.append(executor.submit(search_semantic_scholar, query))

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    papers.extend(result)
                except Exception as e:
                    logger.warning(f"Search failed: {e}")

        # Remove duplicates by title
        seen = set()
        unique_papers = []
        for paper in papers:
            if paper['title'] not in seen:
                seen.add(paper['title'])
                unique_papers.append(paper)

        return unique_papers

    def _filter_by_quality(self, papers: List[Dict]) -> List[Dict]:
        """
        Filter papers by quality metrics.

        Pattern: Multi-criteria filtering
        """
        filtered = []

        for paper in papers:
            # Criterion 1: Recent papers preferred
            year = paper.get('year', 2000)
            if year < 2018:
                continue

            # Criterion 2: Minimum citations (if available)
            citations = paper.get('citations', 0)
            if citations < 10 and year < 2023:
                continue

            # Criterion 3: Has abstract
            if not paper.get('abstract'):
                continue

            filtered.append(paper)

        # Sort by relevance (citations + recency)
        filtered.sort(
            key=lambda p: p.get('citations', 0) + (p.get('year', 2000) - 2000),
            reverse=True
        )

        return filtered[:20]  # Top 20

    def _extract_findings(self, papers: List[Dict]) -> List[str]:
        """
        Extract key findings using LLM.

        Pattern: Batch processing with chunking
        """
        findings = []

        # Process in batches to avoid context limits
        batch_size = 5
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]

            # Build context from batch
            context = "\n\n".join([
                f"Paper: {p['title']}\nAbstract: {p['abstract'][:500]}"
                for p in batch
            ])

            prompt = f"""Extract 3-5 key findings from these papers:

{context}

Return as JSON array: ["finding1", "finding2", ...]"""

            response = self.generate_response(prompt)

            try:
                import json
                batch_findings = json.loads(response)
                findings.extend(batch_findings)
            except:
                logger.warning("Failed to parse findings")

        return findings

    def _store_in_memory(self, papers: List[Dict]):
        """
        Store in FAISS for later retrieval.

        Pattern: Async storage (fire-and-forget)
        """
        from src.memory.memory_manager import get_memory_manager

        memory = get_memory_manager()

        for paper in papers:
            try:
                memory.store(
                    content=f"{paper['title']}: {paper['abstract']}",
                    metadata={
                        'type': 'paper',
                        'title': paper['title'],
                        'url': paper.get('url'),
                        'citations': paper.get('citations', 0)
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to store paper: {e}")
```

**Key patterns used:**
1. ✅ Query generation with LLM
2. ✅ Parallel execution with ThreadPoolExecutor
3. ✅ Retry on failure
4. ✅ Multi-criteria filtering
5. ✅ Batch processing to avoid context limits
6. ✅ Async storage (fire-and-forget)

---

## ⚙️ Pattern 2: Code Generation Agent

### Use Case
Generate code from requirements, ensure quality.

### Implementation

```python
class CoderAgent(BaseAgent):
    """
    Code generation agent with validation.

    Pattern: Generate → Validate → Refine → Test
    """

    def run(self, state: Dict, **kwargs) -> Dict:
        topic = state['topic']
        research_papers = state['research_papers']

        logger.info(f"Generating code for: {topic}")

        # Pattern: Retrieve relevant context from memory
        context = self._get_relevant_context(topic)

        # Pattern: Generate code with examples
        code = self._generate_code(topic, context)

        # Pattern: Validate syntax
        is_valid, error = self._validate_syntax(code)

        if not is_valid:
            # Pattern: Self-correction
            code = self._fix_syntax(code, error)

        # Pattern: Format code
        code = self._format_code(code)

        # Pattern: Add documentation
        code = self._add_docstrings(code)

        return {
            'generated_code': {'main': code},
            'messages': [{
                'role': 'assistant',
                'content': f"Generated {len(code)} lines of code"
            }]
        }

    def _get_relevant_context(self, topic: str) -> str:
        """
        Retrieve relevant code patterns from memory.

        Pattern: Semantic search
        """
        from src.memory.memory_manager import get_memory_manager

        memory = get_memory_manager()

        results = memory.query(
            query=f"Code examples for {topic}",
            k=3,
            filter_metadata={'type': 'code'}
        )

        return "\n\n".join([r['content'] for r in results])

    def _generate_code(self, topic: str, context: str) -> str:
        """
        Generate code using LLM.

        Pattern: Few-shot prompting
        """
        prompt = f"""Generate production-quality Python code for: {topic}

Reference implementations:
{context}

Requirements:
- Use type hints
- Add docstrings
- Handle errors gracefully
- Include example usage

Return ONLY the code, no explanations."""

        code = self.generate_response(prompt)

        # Extract code from markdown if wrapped
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]

        return code.strip()

    def _validate_syntax(self, code: str) -> tuple:
        """
        Validate Python syntax.

        Pattern: AST parsing
        """
        import ast

        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def _fix_syntax(self, code: str, error: str) -> str:
        """
        Attempt to fix syntax errors.

        Pattern: Self-correction with LLM
        """
        prompt = f"""Fix this Python code:

```python
{code}
```

Error: {error}

Return corrected code only."""

        fixed_code = self.generate_response(prompt)

        if "```python" in fixed_code:
            fixed_code = fixed_code.split("```python")[1].split("```")[0]

        return fixed_code.strip()

    def _format_code(self, code: str) -> str:
        """
        Format code with black.

        Pattern: Standard formatter
        """
        try:
            import black
            return black.format_str(code, mode=black.Mode())
        except:
            return code

    def _add_docstrings(self, code: str) -> str:
        """
        Add docstrings to functions.

        Pattern: AST transformation
        """
        # Implementation left as exercise
        return code
```

**Key patterns:**
1. ✅ Context retrieval from memory
2. ✅ Few-shot prompting
3. ✅ Syntax validation with AST
4. ✅ Self-correction loop
5. ✅ Code formatting
6. ✅ Documentation generation

---

## 🎯 Pattern 3: Critic Agent (Quality Evaluation)

### Use Case
Evaluate output quality, provide actionable feedback.

### Implementation

```python
class CriticAgent(BaseAgent):
    """
    Quality evaluation agent.

    Pattern: Evaluate → Score → Provide Feedback → Route
    """

    def run(self, state: Dict, **kwargs) -> Dict:
        research_papers = state.get('research_papers', [])
        generated_code = state.get('generated_code', {})
        iteration = state.get('iteration_count', 0)

        logger.info(f"Evaluating quality (iteration {iteration})")

        # Pattern: Multi-dimensional scoring
        scores = self._evaluate_quality(state)

        # Pattern: Generate actionable feedback
        feedback = self._generate_feedback(scores, state)

        # Pattern: Determine if revision needed
        needs_revision = min(scores.values()) < 7.0 and iteration < 3

        return {
            'quality_scores': scores,
            'feedback': feedback,
            'needs_revision': needs_revision,
            'messages': [{
                'role': 'assistant',
                'content': f"Quality: {sum(scores.values())/len(scores):.1f}/10"
            }]
        }

    def _evaluate_quality(self, state: Dict) -> Dict[str, float]:
        """
        Evaluate on multiple dimensions.

        Pattern: Rubric-based scoring
        """
        scores = {}

        # Dimension 1: Research quality
        papers = state.get('research_papers', [])
        scores['research_quality'] = min(10, len(papers) / 2)  # 10 papers = 10/10

        # Dimension 2: Code quality
        code = state.get('generated_code', {})
        if code:
            scores['code_quality'] = self._evaluate_code_quality(code)
        else:
            scores['code_quality'] = 0

        # Dimension 3: Completeness
        findings = state.get('key_findings', [])
        scores['completeness'] = min(10, len(findings) / 1.5)

        # Dimension 4: Accuracy (check citations)
        scores['accuracy'] = self._check_citations(state)

        return scores

    def _evaluate_code_quality(self, code: Dict[str, str]) -> float:
        """
        Evaluate code quality.

        Pattern: Static analysis
        """
        score = 10.0

        for name, code_str in code.items():
            # Check: Has docstrings
            if '"""' not in code_str:
                score -= 1

            # Check: Has type hints
            if '->' not in code_str:
                score -= 1

            # Check: Not too long
            if len(code_str.split('\n')) > 100:
                score -= 0.5

        return max(0, score)

    def _check_citations(self, state: Dict) -> float:
        """
        Verify citations are accurate.

        Pattern: Cross-reference with papers
        """
        # Simplified: Just check if papers exist
        papers = state.get('research_papers', [])
        if len(papers) >= 5:
            return 10.0
        return len(papers) * 2

    def _generate_feedback(self, scores: Dict[str, float], state: Dict) -> Dict[str, str]:
        """
        Generate actionable feedback.

        Pattern: Targeted improvement suggestions
        """
        feedback = {}

        for dimension, score in scores.items():
            if score < 7.0:
                feedback[dimension] = self._get_improvement_suggestion(
                    dimension, score, state
                )

        return feedback

    def _get_improvement_suggestion(self, dimension: str, score: float, state: Dict) -> str:
        """
        Generate specific improvement suggestion.

        Pattern: Template-based feedback
        """
        suggestions = {
            'research_quality': f"Add more papers (current: {len(state.get('research_papers', []))}, target: 10+)",
            'code_quality': "Add docstrings and type hints to all functions",
            'completeness': f"Extract more findings (current: {len(state.get('key_findings', []))}, target: 15+)",
            'accuracy': "Verify all citations are from actual papers"
        }

        return suggestions.get(dimension, "Improve this dimension")
```

**Key patterns:**
1. ✅ Multi-dimensional scoring rubric
2. ✅ Static code analysis
3. ✅ Cross-referencing for accuracy
4. ✅ Actionable feedback generation
5. ✅ Threshold-based revision decisions

---

## 🧪 Testing Agents

### Unit Testing

```python
import pytest
from src.agents.researcher_agent import ResearcherAgent

def test_researcher_generates_queries():
    """Test query generation."""
    agent = ResearcherAgent()

    queries = agent._generate_queries("transformers", "basic")

    assert len(queries) >= 3
    assert all(isinstance(q, str) for q in queries)
    assert any("transformer" in q.lower() for q in queries)


def test_researcher_filters_by_quality():
    """Test paper filtering."""
    agent = ResearcherAgent()

    papers = [
        {'title': 'Old paper', 'year': 2010, 'citations': 100, 'abstract': 'Test'},
        {'title': 'Recent paper', 'year': 2023, 'citations': 10, 'abstract': 'Test'},
        {'title': 'No abstract', 'year': 2023, 'citations': 50},
    ]

    filtered = agent._filter_by_quality(papers)

    # Should keep recent paper, filter out old and no-abstract
    assert len(filtered) == 1
    assert filtered[0]['title'] == 'Recent paper'
```

### Integration Testing

```python
def test_researcher_full_run():
    """Test full researcher workflow."""
    agent = ResearcherAgent()

    state = {
        'topic': 'attention mechanisms',
        'depth': 'basic'
    }

    result = agent.run(state)

    assert 'research_papers' in result
    assert 'key_findings' in result
    assert len(result['research_papers']) > 0
    assert len(result['key_findings']) > 0
```

---

## 📖 Key Takeaways

1. **Structure matters:** Follow the standard pattern for consistency
2. **Error handling is not optional:** Every external call can fail
3. **Logging is your friend:** Log state transitions, not just errors
4. **Validation prevents cascading failures:** Validate inputs and outputs
5. **Separation of concerns:** Keep `run()` high-level, delegate to private methods
6. **Testing agents is testing logic:** Mock LLM calls, test the orchestration

---

## ➡️ Next Steps

**Practice:** Implement one of these agents from scratch:
1. **SummarizerAgent**: Condense long text
2. **ValidatorAgent**: Check if requirements met
3. **OptimizerAgent**: Improve existing code

**Next guide:** `07_TOOL_INTEGRATION.md` - Building and integrating tools

**Time estimate:** 2-3 hours to implement a full agent
