# Token Usage Tracking & Cost Monitoring Guide

## Overview

The Token Tracking system provides real-time monitoring of LLM API costs across all agent executions. It automatically tracks every API call, calculates costs, and alerts when budgets are approaching limits.

**Key Benefits**:
- 📊 Real-time cost tracking across all agents
- 💰 Budget alerts and warnings
- 📈 Detailed per-agent and per-model breakdowns
- 🎯 Optimize costs by identifying expensive operations
- 📋 Export usage data for analysis

---

## Quick Start

### Basic Usage

Token tracking is **automatically enabled** - no setup required! Just run your workflow:

```python
from src.graph.workflow import run_workflow

# Execute workflow (token tracking happens automatically)
final_state = run_workflow(
    topic="Transformer architectures",
    stream_output=True
)

# Access token statistics from final state
stats = final_state['token_usage_stats']
print(f"Total cost: ${stats['total_cost_usd']:.4f}")
print(f"Total tokens: {stats['total_tokens']:,}")
```

### With Budget Limits

Set a budget limit to get alerts when costs approach your limit:

```python
from src.graph.workflow import run_workflow

# Set $0.50 budget limit
final_state = run_workflow(
    topic="RAG systems",
    budget_limit_usd=0.50,  # 50 cents
    stream_output=True
)

# Budget status is automatically monitored
# Warnings at 90% utilization, errors when exceeded
```

**Console output with budget tracking**:
```
[workflow_started] Starting workflow: RAG systems
[agent_started] planner: Breaking down task into subtasks
[agent_completed] planner: Created 5 subtasks
[agent_started] researcher: Searching 5 queries across arXiv and Semantic Scholar
[agent_completed] researcher: Found 12 papers with 8 key findings
⚠️  [budget_warning] Budget warning: $0.47 / $0.50 (94.0%)
...
```

---

## Detailed Usage

### Accessing Token Statistics

```python
from src.utils.token_tracker import get_token_tracker

# Get global tracker instance
tracker = get_token_tracker()

# Get comprehensive statistics
stats = tracker.get_statistics()

print(f"Total calls: {stats['total_calls']}")
print(f"Total tokens: {stats['total_tokens']:,}")
print(f"Total cost: ${stats['total_cost_usd']:.4f}")
print(f"Average cost per call: ${stats['average_cost_per_call']:.6f}")
```

**Example Output**:
```json
{
  "total_calls": 7,
  "total_tokens": 28450,
  "total_cost_usd": 0.0423,
  "average_tokens_per_call": 4064,
  "average_cost_per_call": 0.006043,
  "by_agent": {
    "planner": {
      "calls": 1,
      "tokens": 2340,
      "cost_usd": 0.002106
    },
    "researcher": {
      "calls": 2,
      "tokens": 8920,
      "cost_usd": 0.006318
    }
    // ... more agents
  },
  "by_model": {
    "gemini-1.5-flash": {
      "calls": 4,
      "tokens": 15200,
      "cost_usd": 0.01824
    },
    "gemini-1.5-pro": {
      "calls": 3,
      "tokens": 13250,
      "cost_usd": 0.02406
    }
  }
}
```

### Human-Readable Cost Report

```python
from src.utils.token_tracker import get_token_tracker

tracker = get_token_tracker()

# Generate formatted report
print(tracker.get_cost_breakdown_report())
```

**Example Output**:
```
============================================================
TOKEN USAGE & COST REPORT
============================================================

📊 OVERALL STATISTICS
  Total LLM Calls:        7
  Total Tokens:           28,450
  Total Cost:             $0.0423
  Average Cost/Call:      $0.006043
  Session Duration:       45.2s

🤖 COST BY AGENT
  researcher            $0.0156 (36.9%) | 10,240 tokens | 2 calls
  coder                 $0.0121 (28.6%) | 6,800 tokens | 2 calls
  synthesizer           $0.0089 (21.0%) | 5,120 tokens | 1 call
  planner               $0.0032 ( 7.6%) | 2,340 tokens | 1 call
  tester                $0.0015 ( 3.5%) | 1,950 tokens | 1 call
  critic                $0.0010 ( 2.4%) | 2,000 tokens | 1 call

⚙️  COST BY MODEL
  gemini-1.5-pro        $0.0241 (57.0%) | 13,250 tokens | 3 calls
  gemini-1.5-flash      $0.0182 (43.0%) | 15,200 tokens | 4 calls

============================================================
```

---

## Cost Analysis Examples

### Typical Report Generation Costs

Based on real-world usage:

| Report Type | Tokens Used | Estimated Cost | Details |
|------------|-------------|----------------|---------|
| **Basic** (no code) | ~15K | $0.02 - $0.05 | Simple research + synthesis |
| **Moderate** (1-2 code examples) | ~30K | $0.04 - $0.08 | Research + code + testing |
| **Comprehensive** (3+ code examples) | ~50K | $0.08 - $0.15 | Full workflow with iterations |

**With Model Tiering** (Researcher + Tester use Flash):
- Cost savings: **~40%**
- Typical comprehensive report: **$0.05 - $0.09** (down from $0.08 - $0.15)

### Cost Breakdown by Agent

Average per agent (comprehensive report):

```
Researcher:    $0.015 - $0.025  (30-40% of total)
Coder:         $0.020 - $0.035  (35-45% of total)
Synthesizer:   $0.008 - $0.015  (15-20% of total)
Planner:       $0.002 - $0.005  (3-8% of total)
Tester:        $0.001 - $0.003  (2-5% of total)
Critic:        $0.001 - $0.003  (2-5% of total)
```

**Insight**: Researcher and Coder are the most expensive agents. Using model tiering for simple tasks drastically reduces costs.

---

## Budget Management

### Setting Budget Limits

```python
from src.graph.workflow import run_workflow

# Method 1: Pass budget to workflow
final_state = run_workflow(
    topic="Topic",
    budget_limit_usd=1.00  # $1 limit
)

# Method 2: Set on tracker directly
from src.utils.token_tracker import TokenTracker

tracker = TokenTracker(budget_limit_usd=0.50)
```

### Budget Alerts

The system automatically emits warnings:

1. **90% Warning**: When cost reaches 90% of budget
   ```
   ⚠️  Budget warning: $0.45 / $0.50 (90.0%)
   ```

2. **Budget Exceeded**: When cost exceeds budget
   ```
   ⚠️  BUDGET EXCEEDED: $0.52 / $0.50
   ```

**Note**: Budget alerts are **warnings only** - execution continues. To enforce hard limits, check the budget in your code:

```python
tracker = get_token_tracker()
stats = tracker.get_statistics()

if stats['budget_utilization_percent'] and stats['budget_utilization_percent'] > 100:
    raise RuntimeError(f"Budget exceeded: ${stats['total_cost_usd']:.4f}")
```

---

## Advanced Features

### Export Usage Data

Export token usage to JSON for external analysis:

```python
tracker = get_token_tracker()

# Export to JSON file
tracker.export_to_json('outputs/token_usage_2025_01_15.json')
```

**Output Format**:
```json
{
  "statistics": {
    "total_calls": 7,
    "total_tokens": 28450,
    "total_cost_usd": 0.0423,
    // ... full stats
  },
  "usage_history": [
    {
      "agent_name": "planner",
      "model": "gemini-1.5-flash",
      "prompt_tokens": 1200,
      "completion_tokens": 1140,
      "total_tokens": 2340,
      "estimated_cost_usd": 0.002106,
      "timestamp": 1705334400.123,
      "datetime": "2025-01-15T10:00:00.123Z"
    }
    // ... all calls
  ]
}
```

### Custom Analysis

```python
tracker = get_token_tracker()

# Find most expensive agent
stats = tracker.get_statistics()
most_expensive = max(
    stats['by_agent'].items(),
    key=lambda x: x[1]['cost_usd']
)

print(f"Most expensive: {most_expensive[0]} (${most_expensive[1]['cost_usd']:.4f})")

# Calculate cost per paper (for research-heavy tasks)
papers_found = 12  # from workflow
cost_per_paper = stats['total_cost_usd'] / papers_found
print(f"Cost per paper: ${cost_per_paper:.4f}")

# Identify optimization opportunities
for agent, data in stats['by_agent'].items():
    avg_tokens = data['tokens'] / data['calls']
    if avg_tokens > 5000:
        print(f"⚠️  {agent} uses {avg_tokens:.0f} tokens/call - consider compression")
```

---

## Pricing Reference

### Google Gemini Pricing (2025)

| Model | Prompt Tokens | Completion Tokens | Use Case |
|-------|---------------|-------------------|----------|
| **gemini-1.5-flash** | $0.075 / 1M | $0.30 / 1M | Simple tasks, high volume |
| **gemini-1.5-pro** | $1.25 / 1M | $5.00 / 1M | Complex reasoning, analysis |

**Cost Comparison** (for 10K prompt + 5K completion):
- Flash: $0.0022 per call (baseline)
- Pro: $0.0375 per call (**17x more expensive**)

**Recommendation**: Use Flash for Researcher and Tester agents (80% cost savings).

---

## Cost Optimization Tips

### 1. Use Model Tiering

Already implemented! Researcher and Tester use Flash:

```python
# src/agents/researcher_agent.py
researcher = ResearcherAgent(model_tier="fast")  # Uses Flash

# src/agents/coder_agent.py
coder = CoderAgent(model_tier="standard")  # Uses Pro
```

**Savings**: ~40% reduction on typical reports

### 2. Monitor State Compression

State compression reduces token usage in later agents:

```python
# State grows from 10K → 50K tokens during revisions
# Compression reduces to 15K (70% reduction)

# Check compression effectiveness
from src.utils.token_tracker import get_token_tracker

tracker = get_token_tracker()
stats = tracker.get_statistics()

# Compare synthesizer tokens with/without compression
# With compression: ~5K tokens
# Without: ~15K tokens (3x more expensive)
```

### 3. Limit Revision Iterations

Each revision loop adds ~30% to costs:

```python
# Conservative budget
run_workflow(topic="Topic", max_iterations=2)  # vs default 3

# Cost impact:
# - 3 iterations: ~$0.08
# - 2 iterations: ~$0.06 (25% savings)
```

### 4. Adjust Research Depth

```python
# Basic depth = fewer papers = lower cost
run_workflow(
    topic="Topic",
    requirements={'depth': 'basic'}  # vs 'comprehensive'
)

# Cost impact:
# - Comprehensive: 10-15 papers, $0.08
# - Basic: 3-5 papers, $0.04 (50% savings)
```

---

## Troubleshooting

### Issue: Token counts are 0

**Cause**: Google GenAI API may not return usage_metadata

**Solution**: Check response object:
```python
# In BaseAgent.generate_response()
if hasattr(response, 'usage_metadata'):
    print(response.usage_metadata)  # Verify it exists
```

### Issue: Costs seem incorrect

**Verify pricing**:
```python
tracker = TokenTracker()

# Check pricing table
print(tracker.PRICING)

# Manually calculate
prompt_cost = (tokens / 1_000_000) * pricing['prompt']
completion_cost = (tokens / 1_000_000) * pricing['completion']
```

### Issue: Budget alerts not showing

**Check configuration**:
```python
# Ensure budget is set
tracker = get_token_tracker()
print(f"Budget: {tracker.budget_limit_usd}")

# Verify cost exceeds threshold
stats = tracker.get_statistics()
print(f"Utilization: {stats.get('budget_utilization_percent')}%")
```

---

## Integration Examples

### With FastAPI

```python
from fastapi import FastAPI, HTTPException
from src.graph.workflow import run_workflow
from src.utils.token_tracker import get_token_tracker

app = FastAPI()

@app.post("/generate-report")
async def generate_report(topic: str, budget: float = 1.0):
    """Generate report with budget enforcement."""

    # Reset tracker for this request
    tracker = get_token_tracker()
    tracker.reset()

    try:
        final_state = run_workflow(
            topic=topic,
            budget_limit_usd=budget,
            stream_output=False
        )

        stats = tracker.get_statistics()

        # Enforce budget (hard limit)
        if stats['total_cost_usd'] > budget:
            raise HTTPException(
                status_code=402,
                detail=f"Budget exceeded: ${stats['total_cost_usd']:.4f} > ${budget:.4f}"
            )

        return {
            'report_path': final_state['report_metadata']['output_path'],
            'cost_usd': stats['total_cost_usd'],
            'tokens_used': stats['total_tokens'],
            'breakdown': stats['by_agent']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### With Monitoring Dashboard

```python
from src.utils.token_tracker import get_token_tracker
import matplotlib.pyplot as plt

def plot_cost_breakdown():
    """Create cost breakdown visualization."""
    tracker = get_token_tracker()
    stats = tracker.get_statistics()

    # Plot by agent
    agents = list(stats['by_agent'].keys())
    costs = [data['cost_usd'] for data in stats['by_agent'].values()]

    plt.figure(figsize=(10, 6))
    plt.bar(agents, costs)
    plt.xlabel('Agent')
    plt.ylabel('Cost (USD)')
    plt.title('Cost Breakdown by Agent')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/cost_breakdown.png')
```

---

## Next Steps

1. **Monitor Your Costs**: Run a few reports and check the cost breakdown
2. **Set Budgets**: Add budget limits to production workflows
3. **Export Data**: Export usage data monthly for trend analysis
4. **Optimize**: Use the breakdown to identify expensive operations
5. **Automate Alerts**: Integrate with monitoring systems (Prometheus, DataDog, etc.)

For more advanced usage, see:
- `src/utils/token_tracker.py` - Full implementation
- `tests/integration/test_token_tracking.py` - Usage examples
- `ENHANCEMENT_ROADMAP.md` - Future improvements

---

**Questions or Issues?**
Check logs in `outputs/app.log` for detailed token tracking information.
