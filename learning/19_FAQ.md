# FAQ & Troubleshooting

## 🎯 Common Questions

### General Understanding

#### Q: What's the difference between an "agent" and a "node"?

**A:** In this codebase:
- **Agent** = Python class with `run()` method (e.g., `ResearcherAgent`)
- **Node** = Wrapper function that calls an agent (e.g., `researcher_node()`)

**Why the separation?**
- Agents are testable in isolation
- Nodes handle LangGraph integration (state updates, progress streaming)
- Clean separation of concerns

**Example:**
```python
# Agent (can be tested without LangGraph)
class ResearcherAgent(BaseAgent):
    def run(self, state: Dict) -> Dict:
        # Do work
        return {'research_papers': [...]}

# Node (LangGraph wrapper)
def researcher_node(state: AgentState) -> Dict:
    agent = ResearcherAgent()
    return agent.run(state)  # Delegates to agent
```

---

#### Q: Why do we use TypedDict instead of Pydantic models for state?

**A:** LangGraph requires `TypedDict` for state schemas.

**Pros:**
- ✅ Native Python (no dependencies)
- ✅ Runtime overhead is minimal
- ✅ Works with LangGraph reducers

**Cons:**
- ❌ No runtime validation (only static type checking)
- ❌ Less feature-rich than Pydantic

**For settings**, we use Pydantic (`src/config/settings.py`) because we want validation.

---

#### Q: What's `operator.add` doing in the state schema?

**A:** It's a **reducer** that tells LangGraph how to merge state updates.

**Without reducer:**
```python
messages: list  # Last write wins (overwrites)
```

**With reducer:**
```python
messages: Annotated[list, operator.add]  # Concatenates
```

**Example:**
```python
# Initial
state = {'messages': [msg1]}

# Agent 1 updates
{'messages': [msg2]}

# After merge (with operator.add)
state = {'messages': [msg1, msg2]}  # Concatenated!
```

**Other reducers:**
- `operator.add` - Concatenate lists/add numbers
- `lambda x, y: y` - Last write wins (default)
- Custom function for complex logic

---

#### Q: Why Google Gemini instead of OpenAI GPT-4?

**A:** **Cost vs quality trade-off.**

**Gemini Pros:**
- ✅ Free tier (2M tokens/day)
- ✅ Fast (gemini-flash)
- ✅ Long context (2M tokens)
- ✅ Good enough for most tasks

**GPT-4 Pros:**
- ✅ Better reasoning
- ✅ Better code generation
- ✅ More reliable

**For production:** Use GPT-4 for critical agents (Critic, Synthesizer), Gemini for bulk work (Researcher).

**Switching is easy:**
```python
# src/agents/base_agent.py
def _initialize_model(self):
    if self.name in CRITICAL_AGENTS:
        return GPT4()  # Expensive but better
    return Gemini()  # Cheap and fast
```

---

### Architecture Questions

#### Q: Why not just use one LLM call with a mega-prompt?

**A:** **Context limits and specialization.**

**Naive approach (doesn't work):**
```python
prompt = """Search papers, generate code, test it, write report"""
report = llm.generate(prompt)  # 🔥 Too much for one call
```

**Problems:**
1. **Context limit:** Can't fit 20 papers + code + tests in 128K tokens
2. **No specialization:** One model does everything poorly
3. **No iteration:** Can't improve based on feedback
4. **No tool use:** Can't actually search APIs or run code

**Our approach (works):**
```python
papers = Researcher.search()  # Specialized for search
code = Coder.generate(papers)  # Specialized for coding
quality = Critic.evaluate(code)  # Specialized for evaluation
if quality < threshold:
    code = Coder.improve(feedback)  # Iteration!
```

---

#### Q: What's the difference between the two report modes?

**A:**

**Staff ML Engineer Mode:**
- Focus: Production best practices
- Agents: Planner → Researcher → Coder → Tester → Critic → Synthesizer
- Output: 1 report with code examples
- Use case: "How do I implement X in production?"

**Research Innovation Mode:**
- Focus: Novel research ideas
- Agents: Planner → Researcher → CrossDomainAnalyst → InnovationSynthesizer → ImplementationResearcher → ImplementationSynthesizer
- Output: 2 reports (Innovation + Implementation)
- Use case: "What are novel research directions for X?"

**Different workflows for different goals.**

---

### Technical Issues

#### Q: Why is my workflow stuck/not progressing?

**Debug checklist:**

1. **Check logs:**
   ```bash
   tail -f outputs/app.log
   ```

2. **Common causes:**
   - API rate limit hit (wait or add retry)
   - Infinite loop (check routing logic)
   - Exception swallowed (add more logging)
   - Deadlock (check for circular dependencies)

3. **Add debug logging:**
   ```python
   logger.info(f"Current state: {state.keys()}")
   logger.info(f"Routing to: {next_node}")
   ```

4. **Test node in isolation:**
   ```python
   def test_stuck_node():
       from src.graph.nodes import problematic_node
       state = {...}  # Minimal state
       result = problematic_node(state)
       print(result)
   ```

---

#### Q: "ModuleNotFoundError: No module named 'langgraph'"

**A:** You haven't installed dependencies.

```bash
# Activate virtual environment first
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt
```

**Pro tip:** Use `pip list` to verify installed packages.

---

#### Q: "google.api_core.exceptions.ResourceExhausted: 429 Resource exhausted"

**A:** You've hit Google API rate limits.

**Solutions:**

1. **Free tier limits:**
   - Gemini Flash: 15 requests/minute
   - Gemini Pro: 2 requests/minute

2. **Wait and retry** (automatic with our retry logic)

3. **Reduce requests:**
   ```python
   # src/config/settings.py
   MAX_CONCURRENT_SEARCHES = 3  # Lower this
   ```

4. **Upgrade to paid tier** (500 requests/minute)

5. **Use caching:**
   ```python
   @cache
   def search_papers(query):
       # Results cached, won't hit API again
   ```

---

#### Q: Code execution times out

**A:** Default timeout is 30s. Some code takes longer.

**Solution 1: Increase timeout**
```python
# src/config/settings.py
CODE_EXECUTION_TIMEOUT = 120  # 2 minutes
```

**Solution 2: Use async execution**
```python
result = CodeTools.execute_code(
    code,
    async_mode=True,  # Runs in Celery worker
    timeout=300  # 5 minutes
)
```

**Solution 3: Optimize the code**
- Use the Critic agent to suggest optimizations
- Add early stopping conditions
- Reduce dataset size in examples

---

#### Q: FAISS index not found

**A:** Vector store hasn't been initialized.

```bash
# Initialize vector store
python -c "from src.memory.memory_manager import get_memory_manager; get_memory_manager()"
```

**This creates:** `data/vector_store/faiss_index.pkl`

**Alternatively:** Delete and it will auto-create on first use.

---

### Workflow Questions

#### Q: How do I add a new tool (e.g., GitHub API)?

**Step-by-step:**

1. **Create tool function:**
   ```python
   # src/tools/github_tools.py
   def search_github_repos(query: str) -> List[Dict]:
       """Search GitHub repositories."""
       import requests
       response = requests.get(
           "https://api.github.com/search/repositories",
           params={'q': query}
       )
       return response.json()['items']
   ```

2. **Add retry logic:**
   ```python
   from src.utils.retry import retry_with_exponential_backoff

   @retry_with_exponential_backoff(max_retries=3)
   def search_github_repos(query: str) -> List[Dict]:
       # ... same code
   ```

3. **Use in agent:**
   ```python
   # src/agents/researcher_agent.py
   from src.tools.github_tools import search_github_repos

   def run(self, state):
       repos = search_github_repos(state['topic'])
       # Process repos...
   ```

4. **Test:**
   ```python
   def test_github_tool():
       repos = search_github_repos("machine learning")
       assert len(repos) > 0
   ```

---

#### Q: How do I modify the reflection loop logic?

**Current logic:**
```python
# src/graph/edges.py
def should_revise(state):
    scores = state['quality_scores']
    if min(scores.values()) < 7.0 and iterations < 3:
        return "revise"
    return "done"
```

**To customize:**

1. **Change threshold:**
   ```python
   if min(scores.values()) < 8.0:  # Stricter
   ```

2. **Change max iterations:**
   ```python
   if ... and iterations < 5:  # More attempts
   ```

3. **Route to specific agent:**
   ```python
   lowest_dimension = min(scores, key=scores.get)
   if lowest_dimension == 'code_quality':
       return "revise_code"
   elif lowest_dimension == 'research':
       return "revise_research"
   ```

---

#### Q: How do I debug which agent is failing?

**Method 1: Progress logs**
```bash
# Watch real-time progress
tail -f outputs/app.log | grep "agent_started\|agent_completed\|agent_failed"
```

**Method 2: Add breakpoints**
```python
# src/graph/nodes.py
def researcher_node(state):
    import pdb; pdb.set_trace()  # Debugger will stop here
    ...
```

**Method 3: Test agent in isolation**
```python
from src.agents.researcher_agent import ResearcherAgent

agent = ResearcherAgent()
state = {'topic': 'test'}

try:
    result = agent.run(state)
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
```

---

### Performance Questions

#### Q: Generation is too slow (10+ minutes). How to optimize?

**Optimization strategies:**

1. **Use faster model:**
   ```python
   # Switch to gemini-flash
   temperature=0.3  # Lower = faster
   ```

2. **Reduce research depth:**
   ```python
   depth = "basic"  # vs "comprehensive"
   ```

3. **Parallel execution:**
   ```python
   # Already implemented for searches
   # Check: MAX_CONCURRENT_SEARCHES in settings
   ```

4. **Cache results:**
   ```python
   # Memory stores papers automatically
   # Subsequent runs are faster
   ```

5. **Profile to find bottleneck:**
   ```python
   import cProfile
   cProfile.run('app.invoke(state)', 'profile.stats')
   ```

---

#### Q: High API costs. How to reduce?

**Cost reduction:**

1. **Use Gemini (free tier)**
   - Already default

2. **Cache aggressively:**
   ```python
   # Don't re-search same queries
   # Use FAISS memory
   ```

3. **Reduce max iterations:**
   ```python
   max_iterations = 1  # vs 3
   ```

4. **Use smaller context windows:**
   ```python
   # Don't send all 20 papers to LLM
   # Send top 5 most relevant
   ```

5. **Monitor token usage:**
   ```python
   # Check: src/utils/token_tracker.py
   tracker.get_statistics()
   ```

---

### Development Questions

#### Q: How do I run tests?

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_agents.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/
```

---

#### Q: How do I add a new route to the API?

**Example: Add "/api/stats" endpoint**

```python
# app.py
@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    from src.utils.token_tracker import get_token_tracker

    tracker = get_token_tracker()
    stats = tracker.get_statistics()

    return {
        'total_requests': stats['total_requests'],
        'total_tokens': stats['total_tokens'],
        'total_cost': stats['total_cost']
    }
```

**Test:**
```bash
curl http://localhost:8001/api/stats
```

---

#### Q: How do I deploy this to production?

**Quick deployment (Docker Compose):**

```bash
# Build
docker-compose -f docker-compose.async.yml build

# Run
docker-compose -f docker-compose.async.yml up -d

# Check logs
docker-compose logs -f api
```

**Production deployment (Kubernetes):**

See: `docs/DEPLOYMENT.md` (if you create it)

**Considerations:**
- [ ] Set up Redis for Celery
- [ ] Configure environment variables
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure autoscaling
- [ ] Set up logging aggregation
- [ ] Add authentication
- [ ] Configure CORS properly
- [ ] Set up HTTPS

---

## 🐛 Common Errors and Solutions

### Error: "Expected END, got ..."

**Cause:** Workflow routing issue. An edge is missing.

**Solution:**
```python
# Make sure all paths lead to END
workflow.add_edge("last_node", END)

# OR use conditional routing
workflow.add_conditional_edges(
    "last_node",
    lambda state: "end",
    {"end": END}
)
```

---

### Error: "KeyError: 'research_papers'"

**Cause:** Trying to access state field that doesn't exist.

**Solution 1: Use .get() with default**
```python
papers = state.get('research_papers', [])  # Returns [] if missing
```

**Solution 2: Check before accessing**
```python
if 'research_papers' in state:
    papers = state['research_papers']
else:
    logger.warning("No papers in state")
```

---

### Error: "MemoryError" or "RuntimeError: out of memory"

**Cause:** FAISS index too large or vector dimension mismatch.

**Solution:**
```bash
# Delete and rebuild index
rm data/vector_store/*

# Reduce vector dimension in settings
# src/config/settings.py
EMBEDDING_DIMENSION = 384  # Lower from 768
```

---

## 💡 Best Practices

### When to Use Which Pattern?

**Use sequential edges when:**
- Order always matters
- No decisions needed
- Example: Planner → Researcher

**Use conditional edges when:**
- Next step depends on state
- Example: Critic → (revise research | revise code | done)

**Use parallel execution when:**
- Tasks are independent
- Want faster execution
- Example: Multiple searches simultaneously

**Use reflection loops when:**
- Need iterative improvement
- Have quality threshold
- Example: Critic → refine → Critic

---

### State Management Tips

1. **Keep state flat** (no deep nesting)
2. **Use Annotated with reducers** for lists
3. **Document state fields** in schema
4. **Validate state in nodes**
5. **Log state transitions**

---

### Agent Design Tips

1. **Single Responsibility:** Each agent does ONE thing
2. **Stateless:** All state is in `state` dict
3. **Testable:** Can test without LangGraph
4. **Error handling:** Never let exceptions bubble up
5. **Logging:** Log inputs, outputs, decisions

---

## 📚 Additional Resources

**Official Docs:**
- LangGraph: https://langchain-ai.github.io/langgraph/
- Google Gemini: https://ai.google.dev/docs
- FAISS: https://github.com/facebookresearch/faiss

**Learning:**
- LangChain tutorials: https://python.langchain.com/docs/
- Multi-agent patterns: This repo's learning guides

**Community:**
- LangChain Discord: https://discord.gg/langchain
- GitHub Issues: This repo

---

## 🆘 Still Stuck?

1. **Check logs:** `outputs/app.log`
2. **Enable debug mode:** Set `LOG_LEVEL=DEBUG` in `.env`
3. **Test in isolation:** Test each component separately
4. **Search issues:** GitHub issues for this repo
5. **Ask specific questions:** "When I do X, I get Y, but expect Z"

**Good question format:**
```
Problem: [What's not working]
Expected: [What should happen]
Actual: [What actually happens]
Code: [Minimal code to reproduce]
Logs: [Relevant error messages]
Tried: [What you've already attempted]
```

---

**Remember:** Every expert was once a beginner. Keep debugging, keep learning! 🚀
