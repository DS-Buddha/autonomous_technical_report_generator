# LangGraph Deep-Dive: State Machines for AI Agents

## 🎯 What is LangGraph?

**TL;DR:** LangGraph is a framework for building stateful, multi-agent workflows as directed graphs.

**Think of it as:**
- State machine + Workflow engine + Agent orchestrator
- React/Redux for AI agents (state management + routing)
- Airflow/Prefect but optimized for LLM workflows

**Core metaphor:**
```
Your agents = Nodes in a graph
Your workflow = Edges connecting nodes
Your data = State flowing through the graph
```

---

## 🏗️ Core Architecture

### The Three Pillars

```python
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# 1. STATE: What data flows through the graph
class MyState(TypedDict):
    messages: list
    result: str

# 2. NODES: What actions happen (your agents)
def researcher_node(state: MyState):
    # Do research
    return {'result': 'findings...'}

# 3. EDGES: How nodes connect (your workflow)
graph = StateGraph(MyState)
graph.add_node("researcher", researcher_node)
graph.add_edge("researcher", END)
```

**Every LangGraph workflow has these three components. Master them, and you've mastered LangGraph.**

---

## 📊 State: The Data Container

### What is State?

State is a **typed dictionary** that flows through your workflow. Every node reads state, does work, and returns updates.

**Analogy:** State is like a shopping cart that gets passed between store clerks. Each clerk adds items (data) to the cart, then passes it to the next clerk.

### Our State Schema

```python
# src/graph/state.py

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add

class AgentState(TypedDict):
    # ===== Input Fields (user provides) =====
    topic: str                    # "RAG systems"
    depth: str                    # "basic" | "comprehensive"
    report_mode: str              # "staff_ml_engineer" | "research_innovation"

    # ===== Work Products (agents populate) =====
    plan: Dict                    # Planner output
    research_papers: Annotated[List[Dict], add]  # Researcher output
    key_findings: List[str]       # Researcher output
    generated_code: Dict[str, str]  # Coder output
    executable_code: Dict[str, str]  # Tester output
    quality_scores: Dict[str, float]  # Critic output
    feedback: str                 # Critic output
    final_report: str             # Synthesizer output

    # ===== Metadata (tracking) =====
    iteration_count: int          # Current iteration
    max_iterations: int           # Stop condition
    messages: Annotated[List[Dict], add]  # LLM message history
```

### Key Insight: Annotated with Reducers

**Without reducer (default behavior):**
```python
messages: list
# When node returns: {'messages': [new_msg]}
# Result: state['messages'] = [new_msg]  # OVERWRITES!
```

**With reducer (merge behavior):**
```python
messages: Annotated[list, operator.add]
# When node returns: {'messages': [new_msg]}
# Result: state['messages'] = [old_msg1, old_msg2, new_msg]  # APPENDS!
```

**Common reducers:**
```python
import operator

# Concatenate lists
field: Annotated[list, operator.add]

# Add numbers
field: Annotated[int, operator.add]

# Last write wins (default)
field: Annotated[str, lambda x, y: y]

# Custom: Keep highest score
field: Annotated[float, lambda x, y: max(x, y)]

# Custom: Merge dicts
field: Annotated[dict, lambda x, y: {**x, **y}]
```

---

## 🔷 Nodes: The Workers

### What is a Node?

A node is a **function** that:
1. Receives current state
2. Does work (call LLM, search API, run code)
3. Returns state updates

**Signature:**
```python
def my_node(state: AgentState) -> Dict[str, Any]:
    # Read from state
    topic = state['topic']

    # Do work
    result = do_something(topic)

    # Return updates (will be merged into state)
    return {
        'my_field': result,
        'messages': [{'role': 'assistant', 'content': 'Done!'}]
    }
```

### Node Patterns in Our Codebase

#### Pattern 1: Agent Wrapper Node

```python
# src/graph/nodes.py

def researcher_node(state: AgentState) -> Dict[str, Any]:
    """Wrapper that delegates to ResearcherAgent."""
    from src.agents.researcher_agent import ResearcherAgent

    logger.info("=== RESEARCHER NODE ===")

    # Progress tracking
    streamer.start_agent("researcher", "Searching academic papers")

    try:
        # Delegate to agent
        agent = ResearcherAgent()
        result = agent.run(state=state)

        streamer.complete_agent("researcher", f"Found {len(result['research_papers'])} papers")
        return result

    except Exception as e:
        streamer.fail_agent("researcher", str(e))
        logger.error(f"Researcher failed: {e}")
        raise
```

**Why this pattern?**
- **Separation of concerns:** Node handles LangGraph integration, agent handles business logic
- **Testability:** Agent can be tested without LangGraph
- **Reusability:** Same agent can be used in different workflows

#### Pattern 2: Direct Implementation Node

```python
def simple_node(state: AgentState) -> Dict[str, Any]:
    """Node that does work directly (no agent class)."""
    topic = state['topic']

    # Simple work that doesn't need a full agent
    formatted_topic = topic.upper()

    return {'formatted_topic': formatted_topic}
```

**When to use:**
- Simple transformations
- Routing logic
- Validation checks

#### Pattern 3: Conditional Exit Node

```python
def should_continue(state: AgentState) -> str:
    """Routing node that returns next node name."""
    if state['iteration_count'] >= state['max_iterations']:
        return "synthesize"  # Stop iterating

    scores = state['quality_scores']
    if min(scores.values()) < 7.0:
        return "revise"  # Need improvement

    return "synthesize"  # Quality met
```

**Used with conditional edges (see next section).**

---

## ➡️ Edges: The Connections

### Types of Edges

#### 1. Sequential Edges (Unconditional)

```python
workflow.add_edge("node_a", "node_b")
# Always: A → B
```

**Example from our workflow:**
```python
workflow.add_edge("planner", "researcher")
# Planner always leads to Researcher
```

#### 2. Conditional Edges (Routing)

```python
workflow.add_conditional_edges(
    source="node_a",
    path=routing_function,
    path_map={
        "option_1": "node_b",
        "option_2": "node_c",
        "option_3": END
    }
)
```

**Example from our workflow:**
```python
def route_after_critic(state: AgentState) -> str:
    """Route based on quality scores."""
    scores = state['quality_scores']
    iteration = state['iteration_count']

    # Stop if max iterations
    if iteration >= state['max_iterations']:
        return "synthesize"

    # Stop if quality threshold met
    if min(scores.values()) >= 7.0:
        return "synthesize"

    # Identify weakest dimension
    lowest_dim = min(scores, key=scores.get)

    # Route to appropriate agent
    if lowest_dim in ['accuracy', 'completeness']:
        return "revise_research"
    elif lowest_dim == 'code_quality':
        return "revise_code"
    else:
        return "revise_synthesis"

workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "revise_research": "researcher",
        "revise_code": "coder",
        "revise_synthesis": "synthesizer",
        "synthesize": "synthesizer"
    }
)
```

#### 3. Start Edge (Entry Point)

```python
workflow.set_entry_point("planner")
# Workflow starts at planner node
```

#### 4. End Edge (Exit Point)

```python
from langgraph.graph import END

workflow.add_edge("synthesizer", END)
# Workflow terminates after synthesizer
```

---

## 🔄 Workflow Execution Model

### How LangGraph Executes

```
1. Initialize state with user input
2. Execute entry point node
3. Merge node's return value into state
4. Follow edge to next node
5. Repeat 2-4 until END is reached
6. Return final state
```

**Example execution trace:**

```python
# Initial state
state = {'topic': 'RAG', 'iteration_count': 0, 'max_iterations': 3}

# Execute planner
planner_result = planner_node(state)  # {'plan': {...}}
state = {**state, **planner_result}  # Merge

# Execute researcher
researcher_result = researcher_node(state)  # {'research_papers': [...]}
state = {**state, **researcher_result}  # Merge

# ... continue until END
```

### Streaming vs Invoke

**Invoke (blocking):**
```python
final_state = workflow.invoke(initial_state)
# Returns only when workflow completes
```

**Stream (real-time updates):**
```python
for event in workflow.stream(initial_state):
    print(f"Node {event['node']} completed")
    print(f"State update: {event['state']}")
# Get updates as each node completes
```

**Our usage:**
```python
# app.py
async for event in app.astream_events(
    {"topic": topic, "depth": depth},
    version="v1"
):
    # Send progress to UI via SSE
    yield f"data: {json.dumps(event)}\n\n"
```

---

## 🎨 Advanced Patterns

### Pattern 1: Reflection Loop

**Problem:** Need iterative improvement until quality threshold met.

**Solution:**
```python
# Create loop with conditional exit
workflow.add_node("worker", worker_node)
workflow.add_node("critic", critic_node)

def should_continue(state):
    if state['iteration_count'] >= 3:
        return "done"
    if state['quality_score'] >= 8.0:
        return "done"
    return "revise"

workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "revise": "worker",  # Loop back
        "done": END
    }
)
```

**Our implementation:**
```
Researcher → Coder → Tester → Critic
     ↑        ↑                  ↓
     └────────┴──────────────────┘
     (Loop back based on feedback)
```

### Pattern 2: Parallel Execution

**Problem:** Need to run multiple independent tasks simultaneously.

**LangGraph doesn't have built-in parallel nodes**, but you can parallelize within a node:

```python
import asyncio

async def parallel_searches_node(state: AgentState):
    """Run multiple searches in parallel."""

    async def search_arxiv():
        return await async_arxiv_search(state['topic'])

    async def search_semantic_scholar():
        return await async_semantic_scholar_search(state['topic'])

    # Run in parallel
    results = await asyncio.gather(
        search_arxiv(),
        search_semantic_scholar()
    )

    return {
        'arxiv_papers': results[0],
        'semantic_papers': results[1]
    }
```

### Pattern 3: Supervisor Pattern

**Problem:** Need central coordinator to assign work.

**Implementation:**
```python
def supervisor_node(state: AgentState) -> str:
    """Decide which specialist to route to."""

    llm_decision = llm.generate(
        f"Given task: {state['task']}, which agent should handle this? "
        f"Options: researcher, coder, tester"
    )

    return llm_decision  # Returns agent name

workflow.add_conditional_edges(
    "supervisor",
    supervisor_node,
    {
        "researcher": "researcher_node",
        "coder": "coder_node",
        "tester": "tester_node"
    }
)
```

### Pattern 4: Human-in-the-Loop

**Problem:** Need user approval before continuing.

```python
from langgraph.checkpoint import MemorySaver

# Enable checkpointing
memory = MemorySaver()
workflow = StateGraph(AgentState).compile(checkpointer=memory)

# Add interrupt before critical node
workflow.add_node("awaiting_approval", human_approval_node)
workflow.add_edge("draft_report", "awaiting_approval")

# Execution pauses at awaiting_approval
config = {"configurable": {"thread_id": "user_123"}}
for event in workflow.stream(state, config):
    if event['node'] == 'awaiting_approval':
        # Show draft to user
        user_approved = get_user_input()

        if user_approved:
            # Resume workflow
            workflow.update_state(config, {'approved': True})
```

---

## 🔧 Our Workflow Architecture

### Staff ML Engineer Mode

```python
START
  ↓
Planner (breaks down task)
  ↓
Researcher (searches papers)
  ↓
Coder (generates code)
  ↓
Tester (validates code)
  ↓
Critic (evaluates quality)
  ↓
  ├─ Quality < 7.0 AND iterations < 3
  │   ↓
  │   └→ Route back to weakest component
  │      (Researcher, Coder, or Tester)
  │
  └─ Quality >= 7.0 OR iterations >= 3
      ↓
  Synthesizer (creates final report)
      ↓
     END
```

**Code:**
```python
# src/graph/workflow.py

def create_workflow():
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("tester", tester_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Entry point
    workflow.set_entry_point("planner")

    # Main path
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "coder")
    workflow.add_edge("coder", "tester")
    workflow.add_edge("tester", "critic")

    # Reflection loop
    workflow.add_conditional_edges(
        "critic",
        should_revise_or_synthesize,
        {
            "revise_research": "researcher",
            "revise_code": "coder",
            "synthesize": "synthesizer"
        }
    )

    # Exit
    workflow.add_edge("synthesizer", END)

    return workflow.compile()
```

### Research Innovation Mode

```
START → Planner
         ↓
      Researcher
         ↓
   CrossDomainAnalyst (finds cross-domain insights)
         ↓
  InnovationSynthesizer (generates innovation report)
         ↓
  ImplementationResearcher (researches how to implement)
         ↓
  ImplementationSynthesizer (generates implementation report)
         ↓
        END

Output: TWO reports (Innovation + Implementation)
```

---

## 🐛 Common Mistakes

### Mistake 1: Forgetting to Return State Updates

```python
# WRONG
def bad_node(state: AgentState):
    result = do_work()
    # Returns None → No state update!

# RIGHT
def good_node(state: AgentState) -> Dict:
    result = do_work()
    return {'result': result}  # State updated
```

### Mistake 2: Mutating State Directly

```python
# WRONG
def bad_node(state: AgentState):
    state['messages'].append(new_msg)  # Mutates in place!
    return state

# RIGHT
def good_node(state: AgentState) -> Dict:
    return {'messages': [new_msg]}  # Reducer will merge
```

### Mistake 3: Missing END Connection

```python
# WRONG
workflow.add_edge("last_node", "nonexistent_node")
# Error: Expected END, got 'nonexistent_node'

# RIGHT
from langgraph.graph import END
workflow.add_edge("last_node", END)
```

### Mistake 4: Conditional Edge with Invalid Return

```python
# WRONG
def bad_router(state):
    return "invalid_option"  # Not in path_map!

workflow.add_conditional_edges(
    "node",
    bad_router,
    {"option_a": "next_node"}  # "invalid_option" will error
)

# RIGHT
def good_router(state):
    if condition:
        return "option_a"
    return "option_b"  # Always return valid key

workflow.add_conditional_edges(
    "node",
    good_router,
    {"option_a": "next_node", "option_b": END}
)
```

---

## 🧪 Testing LangGraph Workflows

### Unit Test Individual Nodes

```python
# tests/test_nodes.py

def test_researcher_node():
    from src.graph.nodes import researcher_node

    # Mock state
    state = {
        'topic': 'transformers',
        'depth': 'basic',
        'messages': []
    }

    # Execute node
    result = researcher_node(state)

    # Assertions
    assert 'research_papers' in result
    assert len(result['research_papers']) > 0
    assert all('title' in paper for paper in result['research_papers'])
```

### Integration Test Full Workflow

```python
def test_full_workflow():
    from src.graph.workflow import create_workflow

    workflow = create_workflow()

    initial_state = {
        'topic': 'RAG systems',
        'depth': 'basic',
        'report_mode': 'staff_ml_engineer',
        'iteration_count': 0,
        'max_iterations': 1  # Limit iterations for faster test
    }

    # Execute workflow
    final_state = workflow.invoke(initial_state)

    # Assertions
    assert 'final_report' in final_state
    assert len(final_state['final_report']) > 1000
    assert final_state['iteration_count'] <= 1
```

### Test Conditional Routing

```python
def test_should_revise_routing():
    from src.graph.edges import should_revise

    # Test: Quality met
    state_good = {
        'quality_scores': {
            'accuracy': 8.0,
            'completeness': 8.5,
            'code_quality': 9.0
        },
        'iteration_count': 1,
        'max_iterations': 3
    }
    assert should_revise(state_good) == "synthesize"

    # Test: Quality low, more iterations available
    state_bad = {
        'quality_scores': {
            'accuracy': 6.0,  # Below threshold
            'completeness': 8.0,
            'code_quality': 8.0
        },
        'iteration_count': 1,
        'max_iterations': 3
    }
    assert should_revise(state_bad) == "revise_research"
```

---

## 💡 Key Takeaways

1. **LangGraph = State + Nodes + Edges**
   - State: Typed dictionary with reducers
   - Nodes: Functions that transform state
   - Edges: Workflow connections (sequential or conditional)

2. **State is immutable**
   - Nodes return updates, not mutations
   - Reducers merge updates into state

3. **Nodes are testable in isolation**
   - Just functions with input/output
   - No need for full workflow to test

4. **Conditional edges enable intelligence**
   - Routing based on state
   - Enables reflection loops, supervisor patterns

5. **Streaming provides real-time feedback**
   - Don't block users waiting for completion
   - Show progress as workflow executes

---

## 🚀 Next Steps

Now that you understand LangGraph mechanics, dive into:

**Next:** `04_AGENT_COORDINATION.md` → How agents communicate and collaborate

**Exercise:** Try modifying `src/graph/workflow.py` to:
1. Add a new node that logs the current state
2. Route to it after the Researcher
3. Run the workflow and observe the logs

**Advanced Exercise:** Implement a simple approval loop:
- After Coder generates code, add a node that checks code length
- If code > 200 lines, route to a "simplify" node
- Otherwise, continue to Tester

---

**Key Insight:** LangGraph isn't magic. It's a clean abstraction over state machines that makes AI workflows composable, testable, and observable. Master the fundamentals, and complex workflows become simple.
