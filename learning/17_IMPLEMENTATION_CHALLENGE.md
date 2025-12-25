# Implementation Challenge: Build Your Own Report Mode

## 🎯 The Challenge

**Goal:** Implement a new report mode from scratch to prove you've mastered the system.

**Time:** 2-4 days (part-time) or 1 day (full-time focus)

**Difficulty:** ⭐⭐⭐⭐ (Staff-level challenge)

---

## 🏆 Challenge Options (Pick One)

### Option 1: Security Audit Report Mode ⭐⭐⭐⭐⭐

**What it does:**
Generates comprehensive security audit reports for ML systems/code.

**Requirements:**
1. Analyzes code for security vulnerabilities
2. Checks for common ML-specific issues (data poisoning, model stealing, adversarial attacks)
3. Reviews dependencies for known CVEs
4. Generates security recommendations with severity ratings
5. Provides remediation code examples

**Output:**
```markdown
# Security Audit: [System Name]

## Executive Summary
[High-level findings, risk score]

## Vulnerability Analysis
### Critical (3)
- SQL Injection in data loader
- Unvalidated user input in model inference
- Hardcoded API keys

### High (5)
- Missing input sanitization
- Outdated dependencies with known CVEs
...

## Detailed Findings
### Finding 1: SQL Injection
**Severity:** Critical
**Location:** `src/data/loader.py:45`
**Description:** [Detailed explanation]
**Proof of Concept:** [Code showing exploit]
**Remediation:** [Fixed code example]

## ML-Specific Risks
### Data Poisoning Vulnerability
...

## Compliance Check
- ✅ OWASP Top 10
- ❌ PCI DSS (missing encryption at rest)
...
```

**Agents needed:**
1. **SecurityAnalyzer**: Scans code for vulnerabilities
2. **CVEChecker**: Checks dependencies against CVE databases
3. **MLSecurityAuditor**: Checks ML-specific risks
4. **ComplianceChecker**: Verifies against security standards
5. **RemediationGenerator**: Creates fix examples
6. **SecuritySynthesizer**: Generates final report

**Complexity:** Very high (requires security knowledge)

---

### Option 2: Performance Optimization Report ⭐⭐⭐⭐

**What it does:**
Analyzes ML models/code and suggests performance optimizations.

**Requirements:**
1. Profiles code to find bottlenecks
2. Analyzes model architecture for optimization opportunities
3. Suggests quantization/pruning strategies
4. Estimates latency/throughput improvements
5. Provides optimized code examples

**Output:**
```markdown
# Performance Optimization Report: [Model Name]

## Current Performance Baseline
- Latency: 450ms (p50), 850ms (p99)
- Throughput: 120 req/s
- Memory: 2.3GB
- Cost: $1,200/month

## Optimization Opportunities
### High Impact (3)
1. Model Quantization (INT8)
   - Expected latency reduction: 60%
   - Memory reduction: 75%
   - Accuracy impact: -1.2%

2. Batch Processing
   - Expected throughput increase: 300%
   - Latency increase: +50ms

...

## Detailed Analysis
### Bottleneck 1: Tokenization
**Current:** 120ms per request
**Issue:** Inefficient string operations
**Solution:** [Optimized code]
**Expected gain:** 80% faster

## Implementation Roadmap
1. Week 1: Implement quantization
2. Week 2: Optimize data pipeline
...
```

**Agents needed:**
1. **Profiler**: Profiles code execution
2. **ArchitectureAnalyzer**: Analyzes model architecture
3. **OptimizationExplorer**: Suggests optimization strategies
4. **BenchmarkRunner**: Tests optimizations
5. **ROICalculator**: Estimates cost/benefit
6. **OptimizationSynthesizer**: Generates report

**Complexity:** High (requires ML performance knowledge)

---

### Option 3: Research Literature Survey (Easier) ⭐⭐⭐

**What it does:**
Generates comprehensive literature surveys on any topic.

**Requirements:**
1. Searches multiple academic databases
2. Categorizes papers by methodology
3. Identifies research trends over time
4. Finds research gaps
5. Suggests future research directions

**Output:**
```markdown
# Literature Survey: [Topic]

## Overview
- Papers reviewed: 87
- Date range: 2018-2024
- Key themes: [themes]

## Research Categories
### 1. Foundational Work (15 papers)
[Seminal papers with summaries]

### 2. Recent Advances (32 papers)
[2022-2024 papers]

...

## Timeline Analysis
[Visualization of research progress over time]

## Methodology Breakdown
- Supervised: 45%
- Self-supervised: 30%
- Reinforcement learning: 25%

## Research Gaps
1. Limited work on [gap 1]
2. No benchmarks for [gap 2]
...

## Future Directions
[Suggested research questions]
```

**Agents needed:**
1. **MultiSourceResearcher**: Searches multiple databases
2. **CategoryAnalyzer**: Categorizes papers
3. **TrendAnalyzer**: Identifies trends
4. **GapIdentifier**: Finds research gaps
5. **FutureDirector**: Suggests future work
6. **SurveySynthesizer**: Creates report

**Complexity:** Medium (good starting point)

---

## 📋 Implementation Steps

### Phase 1: Planning (2-4 hours)

**Step 1: Define the workflow**
```
START → Planner
         ↓
    [Agent 1] → [Agent 2] → [Agent 3]
         ↓           ↓           ↓
                 Synthesizer → END
```

**Exercise:** Draw your workflow on paper first.

**Questions to answer:**
- What are the distinct responsibilities?
- What's the critical path?
- Where do you need reflection loops?
- What external tools do you need?

---

### Phase 2: State Schema (1-2 hours)

**Step 2: Define state schema**

```python
# src/graph/state.py

class AgentState(TypedDict):
    # ... existing fields ...

    # Your new mode fields
    security_vulnerabilities: Annotated[List[Dict], operator.add]
    risk_score: float
    remediation_code: Dict[str, str]
    # ... add more as needed
```

**Exercise:** List all state fields you'll need.

**Checklist:**
- [ ] Input fields (from user)
- [ ] Intermediate work products (from agents)
- [ ] Output fields (final report)
- [ ] Metadata fields (tracking, stats)

---

### Phase 3: Implement Agents (8-12 hours)

**Step 3: Create agent files**

```bash
# Create new agent files
touch src/agents/your_agent_1.py
touch src/agents/your_agent_2.py
touch src/agents/your_synthesizer.py
```

**Template for each agent:**

```python
"""
[Agent Name]
[Purpose]
"""

from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

AGENT_PROMPT = """[Your system prompt]"""


class YourAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="YourAgent",
            system_prompt=AGENT_PROMPT,
            temperature=0.7
        )

    def run(self, state: Dict, **kwargs) -> Dict:
        logger.info(f"[{self.name}] Starting")

        # TODO: Implement your logic

        return {
            'your_field': result,
            'messages': [{
                'role': 'assistant',
                'content': f"{self.name}: Completed"
            }]
        }
```

**Exercise:** Implement all agents for your chosen challenge.

**Testing tip:** Test each agent independently before integrating.

```python
# Test agent in isolation
def test_your_agent():
    agent = YourAgent()
    state = {'topic': 'test'}
    result = agent.run(state)
    assert 'your_field' in result
```

---

### Phase 4: Create Workflow Nodes (2-3 hours)

**Step 4: Add node functions**

```python
# src/graph/nodes.py

def your_agent_node(state: AgentState) -> Dict[str, Any]:
    """Your agent node."""
    from src.agents.your_agent import YourAgent

    logger.info("=== YOUR AGENT NODE ===")
    streamer.start_agent("your_agent", "Your task description")

    try:
        agent = YourAgent()
        result = agent.run(state=state)

        streamer.complete_agent("your_agent", "Success message")
        return result

    except Exception as e:
        streamer.fail_agent("your_agent", str(e))
        raise
```

**Exercise:** Create node functions for all your agents.

---

### Phase 5: Wire Up Workflow (2-3 hours)

**Step 5: Update workflow routing**

```python
# src/graph/workflow.py

# Add your nodes
workflow.add_node("your_agent_1", your_agent_1_node)
workflow.add_node("your_agent_2", your_agent_2_node)
workflow.add_node("your_synthesizer", your_synthesizer_node)

# Add routing logic
def route_after_planning(state: AgentState) -> str:
    report_mode = state.get('report_mode')

    if report_mode == 'your_new_mode':
        return "your_mode_path"
    # ... existing routing

workflow.add_conditional_edges(
    "planner",
    route_after_planning,
    {
        "your_mode_path": "your_agent_1",
        # ... existing routes
    }
)

# Wire your agents together
workflow.add_edge("your_agent_1", "your_agent_2")
workflow.add_edge("your_agent_2", "your_synthesizer")
workflow.add_edge("your_synthesizer", END)
```

---

### Phase 6: Update UI (1-2 hours)

**Step 6: Add mode to dropdown**

```html
<!-- templates/index.html -->

<select id="reportMode" name="reportMode">
    <option value="staff_ml_engineer">Staff ML Engineer</option>
    <option value="research_innovation">Research Innovation</option>
    <option value="your_new_mode">Your New Mode</option>
</select>
```

**Step 7: Add description**

```javascript
// static/script.js

const descriptions = {
    'staff_ml_engineer': '...',
    'research_innovation': '...',
    'your_new_mode': 'Your mode description here'
};
```

---

### Phase 7: Testing (2-4 hours)

**Step 8: Unit tests**

```python
# tests/test_your_mode.py

def test_your_agent_1():
    from src.agents.your_agent_1 import YourAgent1

    agent = YourAgent1()
    state = {'topic': 'test topic'}
    result = agent.run(state)

    assert 'expected_field' in result
    # Add more assertions


def test_workflow_routing():
    from src.graph.workflow import create_workflow

    workflow = create_workflow()
    state = {
        'topic': 'test',
        'report_mode': 'your_new_mode'
    }

    # Test that it routes correctly
    # (This requires understanding LangGraph testing)
```

**Step 9: Integration test**

```bash
# Test full workflow
python main.py "test topic" --report-mode your_new_mode --depth basic
```

**Step 10: Manual testing checklist**

- [ ] UI dropdown shows new mode
- [ ] Description updates correctly
- [ ] Workflow executes all agents
- [ ] Progress updates in real-time
- [ ] Final report generates correctly
- [ ] No errors in logs
- [ ] Edge cases handled (empty results, API failures, etc.)

---

## 🎯 Success Criteria

### Minimum Viable Implementation (Pass)
- ✅ New report mode appears in UI
- ✅ Workflow executes without errors
- ✅ Generates a basic report
- ✅ At least 3 custom agents implemented

### Good Implementation (Senior Level)
- ✅ All of above PLUS:
- ✅ Comprehensive error handling
- ✅ Progress tracking with phase indicators
- ✅ Quality validation and iteration
- ✅ Well-structured, documented code
- ✅ Unit tests for all agents

### Excellent Implementation (Staff Level)
- ✅ All of above PLUS:
- ✅ Advanced features (parallel execution, caching, etc.)
- ✅ Performance optimizations
- ✅ Comprehensive testing (unit + integration)
- ✅ Production-ready error handling
- ✅ Monitoring/logging integration
- ✅ Could be merged to main branch

---

## 💡 Hints and Tips

### Hint 1: Start Small
Don't try to implement everything at once. Build one agent, test it, then add the next.

**Incremental approach:**
1. Implement Agent 1 → Test
2. Add Agent 2 → Test both
3. Add Synthesizer → Test full flow
4. Add refinements (error handling, validation, etc.)

---

### Hint 2: Reuse Existing Patterns
Study the existing modes:
- How does Staff ML Engineer mode work?
- How does Research Innovation mode structure its report?
- Copy the pattern, adapt for your use case

---

### Hint 3: Mock External APIs During Development
Don't wait for real API calls during development.

```python
# Development mode
USE_MOCK_DATA = True

if USE_MOCK_DATA:
    papers = [{'title': 'Mock paper 1', ...}]
else:
    papers = search_arxiv(query)
```

---

### Hint 4: Log Everything
You'll spend 50% of time debugging. Good logs save hours.

```python
logger.info(f"[Agent] Input: {input_data}")
logger.info(f"[Agent] Processing...")
logger.info(f"[Agent] Output: {len(result)} items")
```

---

## 🐛 Common Pitfalls

### Pitfall 1: Not Updating State Schema
**Symptom:** KeyError when accessing new state fields

**Fix:** Always update `src/graph/state.py` first

---

### Pitfall 2: Forgetting to Register Nodes
**Symptom:** "Node not found" error

**Fix:** Remember to both:
1. Create node function in `nodes.py`
2. Add node to workflow in `workflow.py`

---

### Pitfall 3: Infinite Loops
**Symptom:** Workflow never completes

**Fix:** Add iteration limits and proper exit conditions

```python
if iteration_count >= MAX_ITERATIONS:
    return END
```

---

### Pitfall 4: Not Handling Empty Results
**Symptom:** Crashes when no data found

**Fix:** Always check for empty results

```python
papers = search_papers(query)
if not papers:
    logger.warning("No papers found")
    return {'research_papers': []}  # Return empty, don't crash
```

---

## 📚 Resources

**Code to study:**
- `src/agents/cross_domain_analyst.py` - Good example of structured output
- `src/agents/implementation_researcher.py` - Good example of experiment extraction
- `src/graph/workflow.py` - Study routing logic

**Documentation:**
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- Your own learning guides in this folder

---

## ✅ Submission Checklist

Before you consider it "done":

- [ ] Code compiles without errors
- [ ] All agents implemented
- [ ] Workflow routes correctly
- [ ] UI updated
- [ ] Basic testing completed
- [ ] Error handling implemented
- [ ] Logging added
- [ ] README updated (document your new mode)
- [ ] Example output generated
- [ ] Code formatted (black, isort)

---

## 🎓 What You'll Learn

By completing this challenge, you'll gain:

1. **Deep understanding** of the system architecture
2. **Hands-on experience** with LangGraph
3. **Agent design** skills
4. **Debugging** multi-agent systems
5. **Production patterns** for real-world systems

Most importantly: **The confidence to build similar systems from scratch.**

---

## 🚀 After You Complete

### Next Steps

1. **Write a blog post** about your implementation
2. **Present it** to your team (great staff promo material)
3. **Open a PR** to contribute back
4. **Build another mode** (get faster with practice)

### Career Impact

This challenge demonstrates:
- ✅ System design skills
- ✅ Implementation ability
- ✅ Testing practices
- ✅ Production mindset
- ✅ Independent problem solving

**Perfect for:** Staff engineer promotions, interviews, portfolio

---

## 🆘 Getting Stuck?

### Debugging Checklist

1. Check logs: `outputs/app.log`
2. Add more logging to narrow down issue
3. Test agents individually
4. Simplify: Remove features until it works
5. Compare with working modes

### Where to Get Help

1. Search this repo's issues
2. Check LangGraph docs
3. Review learning guides in this folder
4. Ask specific questions (not "it doesn't work")

---

**Good luck! You've got this. 🚀**

**Remember:** The goal isn't perfection. It's learning and proving mastery.

Start with Option 3 (Literature Survey) if you want an easier entry point, then tackle Options 1-2 for the full challenge.
