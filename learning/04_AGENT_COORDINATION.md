# Agent Coordination: How Multi-Agent Systems Collaborate

## 🎯 The Coordination Problem

### Why Coordination Matters

**Single agent:**
```python
result = agent.run("Generate report on RAG")
# Works, but limited by single agent's capabilities
```

**Multiple independent agents:**
```python
research = researcher.run("Search papers on RAG")
code = coder.run("Generate RAG code")
# Problem: Coder doesn't know what Researcher found!
```

**Coordinated agents:**
```python
research = researcher.run("Search papers on RAG")
code = coder.run("Generate RAG code", context=research)
# Coder builds on Researcher's findings
```

**The challenge:** How do agents share knowledge, avoid duplicate work, and build on each other's output?

---

## 🏗️ Coordination Patterns

### Pattern 1: Sequential Pipeline

**Concept:** Each agent hands off to the next in a fixed order.

```
Agent A → Agent B → Agent C → Result
```

**Example from our system:**
```
Planner → Researcher → Coder → Tester → Critic → Synthesizer
```

**Implementation:**
```python
# src/graph/workflow.py

workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "coder")
workflow.add_edge("coder", "tester")
# ... and so on
```

**When to use:**
- Dependencies are clear (B needs A's output)
- Order matters
- No branching needed

**Pros:**
- ✅ Simple to understand
- ✅ Easy to debug (linear flow)
- ✅ Predictable execution

**Cons:**
- ❌ No parallelism
- ❌ Rigid (can't skip steps)
- ❌ Slowest pattern

---

### Pattern 2: Supervisor Pattern

**Concept:** Central coordinator delegates tasks to specialist agents.

```
        Supervisor
       /    |    \
   Agent1 Agent2 Agent3
       \    |    /
        Supervisor
```

**Example:**
```python
def supervisor_node(state: AgentState) -> str:
    """Route to appropriate specialist."""

    # Analyze task
    task_type = classify_task(state['task'])

    # Route to specialist
    if task_type == 'research':
        return 'researcher'
    elif task_type == 'coding':
        return 'coder'
    elif task_type == 'testing':
        return 'tester'

workflow.add_conditional_edges(
    "supervisor",
    supervisor_node,
    {
        "researcher": "researcher_node",
        "coder": "coder_node",
        "tester": "tester_node"
    }
)

# Each specialist routes back to supervisor
workflow.add_edge("researcher_node", "supervisor")
workflow.add_edge("coder_node", "supervisor")
workflow.add_edge("tester_node", "supervisor")
```

**When to use:**
- Task decomposition needed
- Multiple specialists
- Dynamic routing based on state

**Pros:**
- ✅ Flexible routing
- ✅ Clear responsibility separation
- ✅ Easy to add new specialists

**Cons:**
- ❌ Supervisor is bottleneck
- ❌ More LLM calls (supervisor decisions)
- ❌ Complex state management

---

### Pattern 3: Reflection Loop

**Concept:** Critic evaluates output and routes back for improvement.

```
Worker → Critic → Evaluate
   ↑               ↓
   └─── Revise ────┘
```

**Our implementation:**
```python
def should_revise(state: AgentState) -> str:
    """Evaluate quality and decide to revise or continue."""

    scores = state['quality_scores']
    iteration = state['iteration_count']

    # Stop conditions
    if iteration >= state['max_iterations']:
        logger.info("Max iterations reached")
        return "synthesize"

    if min(scores.values()) >= 7.0:
        logger.info(f"Quality threshold met: {min(scores.values())}")
        return "synthesize"

    # Identify weakest dimension
    lowest_dim = min(scores, key=scores.get)
    logger.info(f"Revising {lowest_dim} (score: {scores[lowest_dim]})")

    # Route to appropriate agent
    if lowest_dim in ['accuracy', 'completeness']:
        return "revise_research"
    elif lowest_dim in ['code_quality', 'executability']:
        return "revise_code"
    else:
        return "synthesize"

workflow.add_conditional_edges(
    "critic",
    should_revise,
    {
        "revise_research": "researcher",
        "revise_code": "coder",
        "synthesize": "synthesizer"
    }
)
```

**Quality scoring:**
```python
# src/agents/critic_agent.py

def evaluate_quality(state: AgentState) -> Dict[str, float]:
    """Evaluate on 5 dimensions."""

    prompt = f"""
    Evaluate the following work on a scale of 1-10:

    Research papers: {state['research_papers']}
    Generated code: {state['generated_code']}

    Rate on:
    1. Accuracy (citations correct, facts verified)
    2. Completeness (covers all aspects of topic)
    3. Code Quality (clean, documented, follows best practices)
    4. Clarity (easy to understand, well-explained)
    5. Executability (code runs without errors)

    Return JSON: {{"accuracy": 8.5, "completeness": 7.0, ...}}
    """

    response = llm.generate(prompt)
    scores = parse_json(response)

    return scores
```

**When to use:**
- Quality improvement needed
- Have objective evaluation criteria
- Can afford multiple iterations

**Pros:**
- ✅ Iterative improvement
- ✅ Quality-driven execution
- ✅ Self-correcting

**Cons:**
- ❌ Slower (multiple passes)
- ❌ More expensive (more LLM calls)
- ❌ Risk of infinite loops (need max iterations)

---

### Pattern 4: Parallel Specialists

**Concept:** Multiple agents work independently, then merge results.

```
        Splitter
       /    |    \
   Agent1 Agent2 Agent3
       \    |    /
         Merger
```

**Example:**
```python
async def parallel_research_node(state: AgentState) -> Dict:
    """Search multiple sources in parallel."""

    async def search_arxiv():
        return await arxiv_search(state['topic'])

    async def search_semantic_scholar():
        return await semantic_scholar_search(state['topic'])

    async def search_pubmed():
        return await pubmed_search(state['topic'])

    # Execute in parallel
    results = await asyncio.gather(
        search_arxiv(),
        search_semantic_scholar(),
        search_pubmed(),
        return_exceptions=True  # Don't fail if one source fails
    )

    # Merge results
    all_papers = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Search failed: {result}")
        else:
            all_papers.extend(result)

    # Deduplicate by DOI
    seen_dois = set()
    unique_papers = []
    for paper in all_papers:
        doi = paper.get('doi')
        if doi and doi not in seen_dois:
            seen_dois.add(doi)
            unique_papers.append(paper)

    return {'research_papers': unique_papers}
```

**When to use:**
- Tasks are independent
- Want faster execution
- Can merge results easily

**Pros:**
- ✅ Fast (parallel execution)
- ✅ Fault-tolerant (one failure doesn't block others)
- ✅ Scalable

**Cons:**
- ❌ Requires async implementation
- ❌ Complex error handling
- ❌ Merge logic can be tricky

---

## 📡 Communication Mechanisms

### Mechanism 1: Shared State

**How it works:** All agents read/write to common state dictionary.

```python
# Agent A writes
def agent_a(state: AgentState) -> Dict:
    findings = do_research()
    return {'research_papers': findings}

# Agent B reads
def agent_b(state: AgentState) -> Dict:
    papers = state['research_papers']  # Reads A's output
    code = generate_code_from_papers(papers)
    return {'generated_code': code}
```

**Pros:**
- ✅ Simple
- ✅ Built into LangGraph
- ✅ Automatic merging with reducers

**Cons:**
- ❌ State can grow large
- ❌ No namespacing (field name conflicts)

---

### Mechanism 2: Shared Memory (FAISS)

**How it works:** Agents write to vector store, others query for relevant context.

```python
# Agent A stores research
def researcher_node(state: AgentState) -> Dict:
    papers = search_papers(state['topic'])

    # Store in vector memory
    memory = get_memory_manager()
    for paper in papers:
        memory.store(
            content=f"{paper['title']}: {paper['abstract']}",
            metadata={'source': 'research', 'paper_id': paper['id']}
        )

    return {'research_papers': papers}

# Agent B retrieves relevant context
def coder_node(state: AgentState) -> Dict:
    memory = get_memory_manager()

    # Query for relevant context
    relevant_research = memory.query(
        "How do RAG systems work?",
        k=5  # Top 5 most relevant
    )

    # Generate code using retrieved context
    code = llm.generate(
        f"Generate code based on:\n{relevant_research}"
    )

    return {'generated_code': {'rag_example': code}}
```

**Pros:**
- ✅ Keeps state small (store references, not full content)
- ✅ Semantic search (get relevant info, not everything)
- ✅ Persistent across workflow runs

**Cons:**
- ❌ External dependency (FAISS)
- ❌ Adds latency (vector search)
- ❌ No guarantees on what's retrieved

---

### Mechanism 3: Message Passing

**How it works:** Agents append to message history, others see full conversation.

```python
# src/graph/state.py
messages: Annotated[List[Dict], operator.add]

# Agent A
def agent_a(state: AgentState) -> Dict:
    return {
        'messages': [{
            'role': 'assistant',
            'content': 'I found 10 papers on RAG systems',
            'name': 'researcher'
        }]
    }

# Agent B (sees A's message)
def agent_b(state: AgentState) -> Dict:
    messages = state['messages']
    # Can read full conversation history

    return {
        'messages': [{
            'role': 'assistant',
            'content': 'I generated code based on those papers',
            'name': 'coder'
        }]
    }
```

**Pros:**
- ✅ Full context visible to all agents
- ✅ Audit trail (see who did what)
- ✅ Works with LangChain's ChatModel interface

**Cons:**
- ❌ Grows unbounded (context limit issues)
- ❌ Redundant information
- ❌ Agents see irrelevant messages

---

## 🎭 Role Specialization

### How We Assign Roles

Each agent has a **specific responsibility** defined by:
1. **System prompt** (defines expertise)
2. **Tool access** (what APIs they can call)
3. **Workflow position** (when they execute)

**Example: ResearcherAgent**
```python
# src/agents/researcher_agent.py

RESEARCHER_PROMPT = """
You are a Research Specialist with expertise in academic literature review.

Your responsibilities:
- Search academic databases (arXiv, Semantic Scholar)
- Extract key findings from papers
- Identify methodologies and experiments
- Summarize research trends

You DO NOT:
- Generate code (that's Coder's job)
- Evaluate quality (that's Critic's job)
- Write final reports (that's Synthesizer's job)

Focus on your core competency: RESEARCH.
"""

class ResearcherAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ResearcherAgent",
            system_prompt=RESEARCHER_PROMPT,
            temperature=0.7
        )

    def run(self, state: Dict) -> Dict:
        # Only research-related logic
        papers = self._search_papers(state['topic'])
        findings = self._extract_findings(papers)

        return {
            'research_papers': papers,
            'key_findings': findings
        }
```

**Example: CoderAgent**
```python
# src/agents/coder_agent.py

CODER_PROMPT = """
You are a Code Generation Specialist with expertise in production ML systems.

Your responsibilities:
- Generate working Python code based on research
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include type hints
- Write self-contained, executable examples

You DO NOT:
- Do research (that's Researcher's job)
- Test code (that's Tester's job)
- Evaluate code quality (that's Critic's job)

Focus on your core competency: CODE GENERATION.
"""

class CoderAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CoderAgent",
            system_prompt=CODER_PROMPT,
            temperature=0.3  # Lower temp for code (more deterministic)
        )

    def run(self, state: Dict) -> Dict:
        # Only code generation logic
        papers = state['research_papers']
        code = self._generate_code(papers)

        return {'generated_code': code}
```

### Why Specialization Works

**Without specialization (single agent does everything):**
```python
GENERIC_PROMPT = """
You are an AI assistant. Generate a report with research and code.
"""

# Result: Does everything poorly
# - Hallucinated citations
# - Broken code
# - No depth
```

**With specialization (each agent has one job):**
```python
# Researcher: Optimized for search and synthesis
# Coder: Optimized for code generation
# Critic: Optimized for evaluation

# Result: Each task done well
# - Real citations (Researcher searches APIs)
# - Working code (Coder specializes in codegen)
# - Quality assurance (Critic evaluates objectively)
```

---

## 🔄 Coordination Strategies in Our System

### Strategy 1: Planner-Driven Coordination

**Planner breaks down task into subtasks:**
```python
# src/agents/planner_agent.py

def run(self, state: Dict) -> Dict:
    topic = state['topic']
    depth = state['depth']

    prompt = f"""
    Create an execution plan for: {topic}

    Break down into subtasks for:
    1. Research (what to search, what sources)
    2. Code generation (what examples to create)
    3. Testing (what to validate)

    Return JSON plan.
    """

    plan = llm.generate(prompt)

    return {
        'plan': parse_json(plan),
        'subtasks': plan['subtasks']
    }
```

**Other agents follow the plan:**
```python
# Researcher reads plan
def researcher_node(state: AgentState) -> Dict:
    plan = state['plan']
    research_tasks = plan['research_tasks']

    # Execute plan
    results = []
    for task in research_tasks:
        results.extend(search_papers(task['query']))

    return {'research_papers': results}
```

---

### Strategy 2: Critic-Driven Coordination

**Critic identifies weakest component:**
```python
def should_revise(state: AgentState) -> str:
    scores = state['quality_scores']

    # Find weakest dimension
    weakest = min(scores, key=scores.get)

    # Route to agent responsible for that dimension
    routing_map = {
        'accuracy': 'researcher',      # Researcher improves accuracy
        'completeness': 'researcher',  # Researcher adds more details
        'code_quality': 'coder',       # Coder improves code
        'executability': 'tester',     # Tester fixes execution issues
        'clarity': 'synthesizer'       # Synthesizer improves clarity
    }

    target_agent = routing_map[weakest]
    return f"revise_{target_agent}"
```

**Feedback is specific:**
```python
# Critic provides actionable feedback
feedback = {
    'dimension': 'completeness',
    'score': 6.5,
    'issue': 'Missing discussion of hybrid RAG approaches',
    'suggestion': 'Add section on BM25 + vector search combination'
}

state['feedback'] = feedback

# Researcher uses feedback
def researcher_node(state: AgentState) -> Dict:
    if 'feedback' in state:
        # Address specific feedback
        additional_research = search_papers(state['feedback']['suggestion'])
        # ...
```

---

### Strategy 3: Context Passing

**Explicit context passing between agents:**
```python
# Researcher → Coder (explicit context)
def coder_node(state: AgentState) -> Dict:
    # Get Researcher's output
    papers = state['research_papers']
    findings = state['key_findings']

    # Build rich context for code generation
    context = {
        'papers': papers[:5],  # Top 5 most relevant
        'key_concepts': findings['concepts'],
        'methodologies': findings['methodologies']
    }

    code = generate_code_with_context(context)

    return {'generated_code': code}
```

**Implicit context via memory:**
```python
# Researcher stores in memory
memory.store(content, metadata={'agent': 'researcher'})

# Coder queries memory
relevant = memory.query("how to implement", k=3)
# Gets Researcher's findings automatically
```

---

## 🧪 Testing Coordination

### Test 1: Sequential Dependencies

```python
def test_sequential_coordination():
    """Test that Coder uses Researcher's output."""

    # Mock researcher output
    state = {
        'topic': 'RAG',
        'research_papers': [
            {'title': 'RAG paper', 'abstract': 'RAG uses retrieval...'}
        ]
    }

    # Execute coder
    from src.graph.nodes import coder_node
    result = coder_node(state)

    # Assert: Code references research findings
    code = result['generated_code']
    assert 'retrieval' in code.lower()
```

### Test 2: Reflection Loop

```python
def test_reflection_loop():
    """Test that low quality triggers revision."""

    state = {
        'quality_scores': {
            'accuracy': 5.0,  # Below threshold
            'completeness': 8.0,
            'code_quality': 8.0
        },
        'iteration_count': 1,
        'max_iterations': 3
    }

    from src.graph.edges import should_revise
    next_node = should_revise(state)

    # Assert: Routes back to researcher (accuracy is weakest)
    assert next_node == "revise_research"
```

### Test 3: Memory Coordination

```python
def test_memory_coordination():
    """Test that agents can share via memory."""

    memory = get_memory_manager()

    # Researcher stores
    memory.store(
        "RAG uses vector retrieval",
        metadata={'agent': 'researcher', 'topic': 'RAG'}
    )

    # Coder queries
    results = memory.query("how does RAG work", k=1)

    # Assert: Coder can retrieve Researcher's findings
    assert len(results) > 0
    assert 'retrieval' in results[0]['content'].lower()
```

---

## 💡 Best Practices

### 1. Clear Responsibilities

**Bad:**
```python
class GeneralAgent(BaseAgent):
    """Does research, coding, and testing."""
    # Problem: Too many responsibilities
```

**Good:**
```python
class ResearcherAgent(BaseAgent):
    """Only does research."""

class CoderAgent(BaseAgent):
    """Only generates code."""
```

### 2. Explicit Context

**Bad:**
```python
def agent(state):
    # Implicitly relies on state having certain fields
    code = generate_code(state['research_papers'])  # KeyError if missing!
```

**Good:**
```python
def agent(state):
    # Explicitly check for required context
    if 'research_papers' not in state or not state['research_papers']:
        logger.error("No research papers found")
        return {'error': 'Missing research context'}

    code = generate_code(state['research_papers'])
    return {'generated_code': code}
```

### 3. Iteration Limits

**Bad:**
```python
def should_continue(state):
    if state['quality_score'] < 7.0:
        return "revise"  # Infinite loop possible!
```

**Good:**
```python
def should_continue(state):
    if state['iteration_count'] >= state['max_iterations']:
        logger.warning("Max iterations reached, stopping")
        return "done"

    if state['quality_score'] < 7.0:
        return "revise"

    return "done"
```

### 4. Graceful Degradation

**Bad:**
```python
def agent(state):
    papers = search_api(state['topic'])  # Crashes if API down
    return {'papers': papers}
```

**Good:**
```python
def agent(state):
    try:
        papers = search_api(state['topic'])
    except APIError as e:
        logger.error(f"API failed: {e}")
        # Fallback: Use memory cache
        papers = memory.query(state['topic'], k=10)

    if not papers:
        # Fallback: Return empty with warning
        logger.warning("No papers found from any source")
        papers = []

    return {'papers': papers}
```

---

## 🎯 Key Takeaways

1. **Coordination is not communication**
   - Communication = agents exchange data
   - Coordination = agents work toward common goal

2. **Use the right pattern for the problem**
   - Sequential: When order matters
   - Supervisor: When routing is complex
   - Reflection: When quality matters
   - Parallel: When speed matters

3. **Specialization beats generalization**
   - One agent, one job
   - Clear boundaries prevent conflicts

4. **State is the coordination backbone**
   - All agents share state
   - Reducers enable merging
   - Memory enables selective retrieval

5. **Always have exit conditions**
   - Max iterations
   - Quality thresholds
   - Timeout limits

---

## 🚀 Next Steps

**Next:** `05_STATE_MANAGEMENT.md` → Deep dive into state design and reducers

**Exercise:** Modify the workflow to add a "Validator" agent that runs after Coder but before Tester. It should check for:
- Code has docstrings
- Code has type hints
- Code follows PEP 8

Route to Coder for fixes if validation fails.

**Advanced Exercise:** Implement a parallel research pattern that searches 3 sources simultaneously, then merges results before passing to Coder.

---

**Key Insight:** Good coordination is invisible. The system should work seamlessly, with each agent doing its job and handing off cleanly. If you're manually passing data around or writing complex merge logic, your coordination pattern is wrong.
