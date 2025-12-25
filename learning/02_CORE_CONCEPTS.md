# Core Concepts: The Building Blocks

## 🎯 What You'll Learn

This document explains the **fundamental concepts** that make this system work:
1. LangGraph state machines
2. Agent design patterns
3. Tool integration
4. State management
5. Control flow (conditional routing)

**Time to complete:** 3-4 hours of focused study

---

## 🔧 Concept 1: LangGraph State Machines

### What is LangGraph?

**Simple explanation:** A framework for building multi-agent workflows as **state machines**.

**Staff engineer explanation:** LangGraph is an abstraction over state machines that:
- Manages state transitions
- Handles conditional routing
- Provides checkpointing/persistence
- Enables streaming
- Integrates with LangChain tools

### Core Abstraction: Graph = Nodes + Edges

```python
from langgraph.graph import StateGraph, END

# Define state schema
class AgentState(TypedDict):
    topic: str
    messages: list
    research_papers: list
    # ... more fields

# Create graph
workflow = StateGraph(AgentState)

# Add nodes (agents)
workflow.add_node("planner", planner_function)
workflow.add_node("researcher", researcher_function)

# Add edges (control flow)
workflow.add_edge("planner", "researcher")      # Direct edge
workflow.add_conditional_edges(                  # Conditional edge
    "researcher",
    should_continue,
    {"continue": "coder", "end": END}
)

# Compile
app = workflow.compile()

# Execute
result = app.invoke({"topic": "transformers"})
```

---

### Key Insight: Nodes are Pure Functions

**Critical concept:** Each node is a **pure function** that takes state and returns state updates.

```python
def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Pure function:
    - Takes current state
    - Returns state updates (not full state!)
    - No side effects (logging is OK)
    """
    topic = state['topic']

    # Do work
    plan = create_plan(topic)

    # Return ONLY updates
    return {
        'plan': plan,
        'messages': [{'role': 'assistant', 'content': f'Created plan with {len(plan)} steps'}]
    }
```

**Why pure functions?**
- Testable: Easy to unit test
- Composable: Can reuse in different workflows
- Debuggable: No hidden state
- Parallel-safe: Can run multiple nodes concurrently

---

### State Reducers: How Updates are Merged

**Problem:** Multiple nodes might update the same field. How do we merge?

**Solution:** Reducers

```python
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    topic: str                    # Last write wins
    messages: Annotated[          # Append (don't replace)
        list,
        operator.add              # Reducer: add (concatenate)
    ]
    research_papers: Annotated[
        list,
        operator.add
    ]
```

**Example:**
```python
# Initial state
state = {'messages': [{'role': 'user', 'content': 'Hi'}]}

# Planner updates
update1 = {'messages': [{'role': 'assistant', 'content': 'Planning...'}]}

# Researcher updates
update2 = {'messages': [{'role': 'assistant', 'content': 'Researching...'}]}

# After merging with operator.add
final_state = {
    'messages': [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Planning...'},
        {'role': 'assistant', 'content': 'Researching...'}
    ]
}
```

**Why this matters:** Without reducers, messages would be overwritten, losing history.

---

### Conditional Routing: The Smart Part

**Concept:** Edges can be conditional based on state.

```python
def route_after_critic(state: AgentState) -> str:
    """
    Routing logic:
    - If quality good → synthesize
    - If research weak → revise_research
    - If code weak → revise_code
    """
    scores = state['quality_scores']

    if min(scores.values()) >= 7.0:
        return "synthesize"

    # Find weakest dimension
    lowest = min(scores, key=scores.get)

    if lowest in ['accuracy', 'completeness']:
        return "revise_research"
    else:
        return "revise_code"

# Use in workflow
workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "synthesize": "synthesizer",
        "revise_research": "researcher",
        "revise_code": "coder"
    }
)
```

**This creates the reflection loop!**

---

## 🤖 Concept 2: Agent Design Pattern

### What is an Agent? (Our Definition)

**Naive:** "AI that does stuff"

**Our definition:** An agent is:
1. **Specialized function:** Does one thing well
2. **Tool-enabled:** Can call external APIs
3. **LLM-powered:** Uses language model for reasoning
4. **State-aware:** Reads from and writes to shared state

### Agent Anatomy

```python
class BaseAgent:
    """
    Base class for all agents.

    Components:
    - system_prompt: Defines agent's role and behavior
    - model: LLM to use (Gemini, GPT-4, etc.)
    - temperature: Randomness (0 = deterministic, 1 = creative)
    - tools: External functions agent can call
    """

    def __init__(self, name: str, system_prompt: str, temperature: float):
        self.name = name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model = self._initialize_model()

    def generate_response(self, prompt: str) -> str:
        """Call LLM with system prompt + user prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = self.model.generate_content(messages)
        return response.text

    def run(self, state: Dict, **kwargs) -> Dict:
        """
        Main entry point.
        Override in subclasses for agent-specific logic.
        """
        raise NotImplementedError
```

### Example: Researcher Agent

```python
class ResearcherAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Researcher",
            system_prompt="""You are a research specialist.
            Search academic papers, extract key findings,
            and provide comprehensive literature reviews.""",
            temperature=0.3  # Low = focused, factual
        )

    def run(self, state: Dict, **kwargs) -> Dict:
        topic = state['topic']
        depth = state['depth']

        # 1. Generate search queries
        queries = self._generate_queries(topic)

        # 2. Search papers (parallel)
        papers = []
        for query in queries:
            papers.extend(search_arxiv(query))
            papers.extend(search_semantic_scholar(query))

        # 3. Extract findings
        findings = self._extract_findings(papers)

        # 4. Store in memory
        for paper in papers:
            memory.store(paper)

        # 5. Return state updates
        return {
            'research_papers': papers,
            'key_findings': findings,
            'messages': [{
                'role': 'assistant',
                'content': f"Researched {len(papers)} papers"
            }]
        }
```

**Key insights:**
- Agent is **autonomous:** Decides queries, searches, extracts
- Agent uses **tools:** `search_arxiv`, `search_semantic_scholar`, `memory.store`
- Agent is **stateless:** All state is in `state` dict
- Agent returns **updates:** Not full state

---

### System Prompts: The Secret Sauce

**Why system prompts matter:** They define agent behavior.

**Bad prompt:**
```python
system_prompt = "You are a helpful assistant."
```

**Good prompt:**
```python
system_prompt = """You are a Research Specialist with expertise in ML/AI.

Your role:
1. Generate 3-5 targeted search queries for the topic
2. Search academic papers using provided tools
3. Extract key findings (concepts, methods, results)
4. Cite sources accurately

Quality criteria:
- Prioritize recent papers (2020+)
- Include seminal works if relevant
- Focus on implementation details, not just theory
- Provide concrete examples

Output format:
Return JSON with:
{
  "queries": [...],
  "papers": [...],
  "findings": [...]
}"""
```

**What makes it good:**
- Clear role and responsibilities
- Specific quality criteria
- Structured output format
- Examples of good behavior

---

## 🛠️ Concept 3: Tool Integration

### What are Tools?

**Definition:** Functions that agents can call to interact with external systems.

**Examples:**
- `search_arxiv(query)` → Search academic papers
- `execute_code(code)` → Run Python code
- `memory.store(content)` → Save to vector store
- `web_search(query)` → Google search

### Tool Pattern

```python
def search_arxiv(
    query: str,
    max_results: int = 10
) -> List[Dict]:
    """
    Tool: Search arXiv for papers.

    Args:
        query: Search query
        max_results: Number of results

    Returns:
        List of papers with metadata
    """
    import arxiv

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in search.results():
        papers.append({
            'title': result.title,
            'authors': [a.name for a in result.authors],
            'abstract': result.summary,
            'url': result.entry_id,
            'published': result.published.isoformat(),
        })

    return papers
```

**Tool characteristics:**
- **Idempotent:** Same input → same output
- **Documented:** Clear docstrings
- **Error-handled:** Graceful failures
- **Typed:** Type hints for safety

---

### Retry Pattern for Tools

**Problem:** External APIs fail (rate limits, network issues, etc.)

**Solution:** Exponential backoff with retries

```python
from src.utils.retry import retry_with_exponential_backoff

@retry_with_exponential_backoff(
    max_retries=3,
    initial_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0
)
def search_semantic_scholar(query: str) -> List[Dict]:
    """Search with automatic retry on failure."""
    response = requests.get(
        "https://api.semanticscholar.org/search",
        params={'query': query}
    )
    response.raise_for_status()
    return response.json()['papers']
```

**Retry logic:**
- 1st try: Immediate
- 2nd try: After 1 second
- 3rd try: After 2 seconds
- 4th try: After 4 seconds
- Then give up, raise exception

**Why exponential:** Avoids thundering herd, gives API time to recover

---

## 📦 Concept 4: State Management

### State Schema Design

**Bad state (flat, untyped):**
```python
state = {
    'data': {...},  # What's in here?
    'stuff': [...], # What type?
    'x': True       # What does this mean?
}
```

**Good state (typed, structured):**
```python
class AgentState(TypedDict):
    # Input
    topic: str
    depth: Literal['basic', 'moderate', 'comprehensive']
    report_mode: Literal['staff_ml_engineer', 'research_innovation']

    # Work products
    plan: Dict[str, Any]
    research_papers: Annotated[List[Dict], operator.add]
    key_findings: Annotated[List[str], operator.add]
    generated_code: Dict[str, str]
    executable_code: Dict[str, str]

    # Quality control
    quality_scores: Dict[str, float]
    feedback: Dict[str, str]
    iteration_count: int

    # Output
    final_report: str
    report_metadata: Dict[str, Any]
```

**Benefits:**
- **Type safety:** Catch errors at design time
- **Documentation:** Schema is self-documenting
- **IDE support:** Autocomplete, type checking
- **Validation:** Can validate state shape

---

### State Evolution (Trace Through Workflow)

```python
# 1. Initial state (user input)
state = {
    'topic': 'transformers',
    'depth': 'comprehensive',
    'messages': [],
    'research_papers': [],
    'iteration_count': 0
}

# 2. After Planner
state = {
    ...
    'plan': {'subtasks': [...], 'dependencies': {...}},
    'messages': [{'role': 'assistant', 'content': 'Created plan'}]
}

# 3. After Researcher
state = {
    ...
    'research_papers': [paper1, paper2, ...],  # Added 15 papers
    'key_findings': ['Attention mechanism', 'Multi-head attention', ...],
    'messages': [..., {'role': 'assistant', 'content': 'Found 15 papers'}]
}

# 4. After Coder
state = {
    ...
    'generated_code': {'attention': '...', 'transformer': '...'},
    'messages': [..., {'role': 'assistant', 'content': 'Generated code'}]
}

# 5. After Critic (needs revision)
state = {
    ...
    'quality_scores': {'accuracy': 8.0, 'completeness': 6.5, ...},
    'feedback': {'completeness': 'Add more details on positional encoding'},
    'iteration_count': 1  # Increment
}

# 6. After Researcher (2nd iteration)
state = {
    ...
    'research_papers': [..., paper16, paper17],  # Added 2 more
    'key_findings': [..., 'Positional encoding methods'],
    'iteration_count': 1
}

# 7. After Synthesizer (final)
state = {
    ...
    'final_report': '# Transformers\n\n## Introduction\n...',
    'report_metadata': {'word_count': 4500, 'code_examples': 3}
}
```

**Key insight:** State accumulates information. Nothing is lost (thanks to `operator.add` reducers).

---

## 🔀 Concept 5: Control Flow Patterns

### Pattern 1: Sequential

```python
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "coder")
workflow.add_edge("coder", "tester")
```

**Use when:** Order matters, no decisions needed

---

### Pattern 2: Conditional Branching

```python
def should_test_code(state):
    if state['generated_code']:
        return "test"
    return "skip"

workflow.add_conditional_edges(
    "coder",
    should_test_code,
    {"test": "tester", "skip": "critic"}
)
```

**Use when:** Next step depends on current state

---

### Pattern 3: Reflection Loop

```python
def should_revise(state):
    scores = state['quality_scores']
    iterations = state['iteration_count']

    if min(scores.values()) >= 7.0 or iterations >= 3:
        return "done"

    # Route to weakest component
    if scores['research'] < 7.0:
        return "revise_research"
    return "revise_code"

workflow.add_conditional_edges(
    "critic",
    should_revise,
    {
        "done": "synthesizer",
        "revise_research": "researcher",
        "revise_code": "coder"
    }
)
```

**Use when:** Need iterative improvement with quality gates

---

### Pattern 4: Parallel Execution

```python
from langgraph.prebuilt import parallel

# Execute multiple agents in parallel
workflow.add_node(
    "parallel_research",
    parallel(
        domain_research=researcher_domain,
        cross_domain_research=researcher_cross_domain
    )
)
```

**Use when:** Tasks are independent and can run concurrently

---

## 🧪 Hands-On Exercise: Build a Simple Workflow

### Challenge: Create a "Paper Summarizer" Workflow

**Requirements:**
1. Input: Paper URL
2. Agent 1 (Fetcher): Download paper text
3. Agent 2 (Summarizer): Generate summary
4. Agent 3 (Critic): Check if summary is good (>100 words)
5. If not good → loop back to Summarizer
6. Output: Final summary

**Starter Code:**
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 1. Define state
class SummaryState(TypedDict):
    paper_url: str
    paper_text: Annotated[str, lambda x, y: y]  # Replace
    summary: str
    quality_score: float
    iteration: int

# 2. Define nodes
def fetch_node(state):
    # TODO: Implement paper fetching
    return {'paper_text': "Downloaded text..."}

def summarize_node(state):
    # TODO: Implement summarization
    return {'summary': "Summary..."}

def critic_node(state):
    # TODO: Implement quality check
    word_count = len(state['summary'].split())
    score = 10 if word_count >= 100 else 5
    return {'quality_score': score}

# 3. Define routing
def should_revise(state):
    if state['quality_score'] >= 7.0 or state['iteration'] >= 2:
        return "done"
    return "revise"

# 4. Build workflow
workflow = StateGraph(SummaryState)
workflow.add_node("fetcher", fetch_node)
workflow.add_node("summarizer", summarize_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("fetcher")
workflow.add_edge("fetcher", "summarizer")
workflow.add_edge("summarizer", "critic")
workflow.add_conditional_edges(
    "critic",
    should_revise,
    {"done": END, "revise": "summarizer"}
)

app = workflow.compile()

# 5. Test
result = app.invoke({'paper_url': 'https://arxiv.org/abs/1706.03762'})
print(result['summary'])
```

**Try it:** Implement the TODOs and run the workflow.

---

## 📖 Key Takeaways

### 1. LangGraph is a State Machine Framework
- Nodes = functions that update state
- Edges = control flow
- State = TypedDict with reducers

### 2. Agents are Specialized, Autonomous, Tool-Enabled
- Each agent has a clear responsibility
- Agents use tools to interact with external world
- System prompts define behavior

### 3. State Management is Critical
- Well-designed schema prevents bugs
- Reducers enable accumulation (messages, papers)
- State is the single source of truth

### 4. Control Flow Enables Intelligence
- Sequential: A → B → C
- Conditional: If X then Y else Z
- Loops: Critic → revise → Critic
- Parallel: Do X and Y simultaneously

### 5. Tools Need Retry Logic
- External APIs fail
- Exponential backoff is the pattern
- Graceful degradation when retries exhausted

---

## ➡️ Next Steps

You now understand the **core concepts**. Next, dive into:

**Next:** `03_LANGGRAPH_DEEP_DIVE.md` → Advanced LangGraph patterns, checkpointing, streaming

**Before moving on, make sure you can:**
- ✅ Explain what a LangGraph node does
- ✅ Write a simple conditional routing function
- ✅ Design a state schema with proper reducers
- ✅ Understand why we use tools vs hardcoded logic
- ✅ Trace state evolution through a workflow

**Practice:** Implement the paper summarizer exercise above. It should take ~30-60 minutes.

---

**Remember:** These concepts are **universal** to multi-agent systems. Master them here, apply them everywhere.
