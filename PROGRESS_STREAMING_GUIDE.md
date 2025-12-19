# Real-Time Progress Streaming Guide

## Overview

The Progress Streaming system provides live updates as agents execute, giving you real-time visibility into workflow execution. No more black box - see exactly what's happening as it happens.

**Key Benefits**:
- 👀 Real-time visibility into agent execution
- 🎯 Early error detection and debugging
- 📊 Track iteration counts and validation results
- ⏱️ Monitor execution duration
- 🔔 Subscribe to events for custom integrations

---

## Quick Start

### Basic Usage

Progress streaming is **automatically enabled** when `stream_output=True`:

```python
from src.graph.workflow import run_workflow

# Execute with real-time progress
final_state = run_workflow(
    topic="Transformer architectures",
    stream_output=True  # Enables console output
)
```

**Console Output**:
```
[workflow_started] Starting workflow: Transformer architectures
[agent_started] planner: Breaking down task into subtasks
[agent_completed] planner: Created 5 subtasks
[agent_started] researcher: Searching 5 queries across arXiv and Semantic Scholar
[agent_completed] researcher: Found 12 papers with 8 key findings
[agent_started] coder: Generating 3 code implementations
[agent_completed] coder: Generated 3 code blocks
[agent_started] tester: Testing 3 code blocks
[agent_completed] tester: 3/3 blocks passed (85.0% coverage)
[agent_started] critic: Evaluating quality across 5 dimensions
[agent_completed] critic: Score: 8.5/10 - Approved
[agent_started] synthesizer: Creating final markdown report
[agent_completed] synthesizer: Report generated (2450 words) → outputs/transformer_architectures_20250115_100523.md
[workflow_completed] Workflow completed in 45.2s
```

### Silent Mode

Disable console output but still track events:

```python
from src.graph.workflow import run_workflow
from src.utils.progress_streamer import get_progress_streamer

# Run workflow without console output
final_state = run_workflow(
    topic="RAG systems",
    stream_output=False
)

# Access event history
streamer = get_progress_streamer()
history = streamer.get_history()

print(f"Total events: {len(history)}")
for event in history:
    print(f"  {event}")
```

---

## Event Types

The system emits 14 different event types:

### Workflow Events

| Event Type | When Emitted | Example |
|-----------|--------------|---------|
| `workflow_started` | Workflow begins | "Starting workflow: Topic" |
| `workflow_completed` | Workflow finishes successfully | "Workflow completed in 45.2s" |
| `workflow_failed` | Workflow fails with error | "Workflow failed after 23.1s: API error" |

### Agent Events

| Event Type | When Emitted | Example |
|-----------|--------------|---------|
| `agent_started` | Agent begins execution | "planner: Breaking down task" |
| `agent_completed` | Agent finishes successfully | "planner: Created 5 subtasks" |
| `agent_failed` | Agent fails with error | "coder: Failed: Syntax error" |

### Validation Events

| Event Type | When Emitted | Example |
|-----------|--------------|---------|
| `validation_started` | Validation begins | "Validating research quality" |
| `validation_passed` | Validation succeeds | "Research validated: 10 papers found" |
| `validation_failed` | Validation fails | "Insufficient research: only 2 papers" |

### Iteration & Budget Events

| Event Type | When Emitted | Example |
|-----------|--------------|---------|
| `iteration_increment` | Critic requests revision | "Revision needed - iteration 2/3" |
| `budget_warning` | Cost reaches 90% of budget | "Budget warning: $0.45 / $0.50" |
| `budget_exceeded` | Cost exceeds budget | "Budget exceeded: $0.52 / $0.50" |

### Other Events

| Event Type | When Emitted | Example |
|-----------|--------------|---------|
| `compression_started` | State compression begins | "Compressing state (50K → 15K tokens)" |
| `compression_completed` | Compression finishes | "State compressed: 70% reduction" |
| `custom_status` | Custom status message | Any custom message |

---

## Subscribing to Events

### Custom Event Handlers

Subscribe to events for custom processing:

```python
from src.utils.progress_streamer import get_progress_streamer, ProgressEvent

# Get streamer
streamer = get_progress_streamer()

# Define callback
def on_progress(event: ProgressEvent):
    print(f"[{event.event_type.value}] {event.agent_name}: {event.message}")

    # Custom logic
    if event.event_type.value == 'agent_failed':
        send_alert(f"Agent {event.agent_name} failed!")

# Subscribe
streamer.subscribe(on_progress)

# Now run workflow - callback will be called for each event
from src.graph.workflow import run_workflow
run_workflow(topic="Test", stream_output=False)
```

### Multiple Subscribers

You can have multiple subscribers for different purposes:

```python
from src.utils.progress_streamer import get_progress_streamer

streamer = get_progress_streamer()

# Subscriber 1: Log to database
def log_to_db(event):
    db.insert('events', event.to_dict())

# Subscriber 2: Send to monitoring
def send_to_metrics(event):
    if event.event_type.value.startswith('agent_'):
        metrics.track('agent_event', {'agent': event.agent_name})

# Subscriber 3: Update UI
def update_ui(event):
    websocket.send_json(event.to_dict())

# Subscribe all
streamer.subscribe(log_to_db)
streamer.subscribe(send_to_metrics)
streamer.subscribe(update_ui)
```

### Unsubscribing

```python
streamer = get_progress_streamer()

def my_callback(event):
    print(event)

streamer.subscribe(my_callback)

# ... later ...

streamer.unsubscribe(my_callback)
```

---

## Event Structure

Each event is a `ProgressEvent` object:

```python
@dataclass
class ProgressEvent:
    event_type: ProgressEventType  # Type of event
    agent_name: Optional[str]       # Agent name (if applicable)
    message: str                    # Human-readable message
    metadata: Dict[str, Any]        # Additional data
    timestamp: float                # Unix timestamp
```

### Accessing Event Data

```python
streamer = get_progress_streamer()

# Run workflow
from src.graph.workflow import run_workflow
run_workflow(topic="Test", stream_output=False)

# Get event history
history = streamer.get_history()

for event in history:
    print(f"Type: {event.event_type.value}")
    print(f"Agent: {event.agent_name}")
    print(f"Message: {event.message}")
    print(f"Metadata: {event.metadata}")
    print(f"Time: {event.timestamp}")
    print("---")
```

### Event Serialization

```python
# Convert to dict
event_dict = event.to_dict()

# Example output:
{
    'event_type': 'agent_completed',
    'agent_name': 'planner',
    'message': 'Created 5 subtasks',
    'metadata': {'subtask_count': 5},
    'timestamp': 1705334400.123,
    'datetime': '2025-01-15T10:00:00.123Z'
}

# Convert to JSON
json_str = event.to_json()
```

---

## Progress Statistics

### Get Summary Statistics

```python
streamer = get_progress_streamer()

summary = streamer.get_summary()

print(summary)
```

**Example Output**:
```python
{
    'total_events': 24,
    'event_counts': {
        'workflow_started': 1,
        'workflow_completed': 1,
        'agent_started': 6,
        'agent_completed': 6,
        'validation_passed': 1,
        'iteration_increment': 2
    },
    'agents_executed': ['planner', 'researcher', 'coder', 'tester', 'critic', 'synthesizer'],
    'workflow_duration_seconds': 45.2
}
```

### Export Event History

```python
streamer = get_progress_streamer()

# Export to JSON file
streamer.export_to_json('outputs/progress_history_20250115.json')
```

**Output Format**:
```json
{
  "summary": {
    "total_events": 24,
    "event_counts": { "agent_started": 6, ... },
    "agents_executed": ["planner", "researcher", ...],
    "workflow_duration_seconds": 45.2
  },
  "events": [
    {
      "event_type": "workflow_started",
      "agent_name": null,
      "message": "Starting workflow: Transformers",
      "metadata": {"task": "Transformers"},
      "timestamp": 1705334400.0,
      "datetime": "2025-01-15T10:00:00.000Z"
    },
    {
      "event_type": "agent_started",
      "agent_name": "planner",
      "message": "Breaking down task",
      "metadata": {},
      "timestamp": 1705334401.5,
      "datetime": "2025-01-15T10:00:01.500Z"
    }
    // ... all events
  ]
}
```

---

## Advanced Usage

### Real-Time Progress Bar

```python
from src.utils.progress_streamer import get_progress_streamer, ProgressEventType
from tqdm import tqdm

streamer = get_progress_streamer()

# Estimated total steps
total_steps = 6  # planner, researcher, coder, tester, critic, synthesizer
completed_steps = 0

# Create progress bar
pbar = tqdm(total=total_steps, desc="Generating report")

def update_progress(event):
    global completed_steps

    if event.event_type == ProgressEventType.AGENT_COMPLETED:
        completed_steps += 1
        pbar.update(1)
        pbar.set_description(f"Completed: {event.agent_name}")

streamer.subscribe(update_progress)

# Run workflow
from src.graph.workflow import run_workflow
run_workflow(topic="Test", stream_output=False)

pbar.close()
```

**Output**:
```
Completed: synthesizer: 100%|███████████████████| 6/6 [00:45<00:00,  7.53s/step]
```

### WebSocket Integration

```python
from fastapi import FastAPI, WebSocket
from src.utils.progress_streamer import get_progress_streamer

app = FastAPI()

@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    streamer = get_progress_streamer()

    # Subscribe to events
    async def send_event(event):
        await websocket.send_json(event.to_dict())

    streamer.subscribe(send_event)

    try:
        # Run workflow in background
        from src.graph.workflow import run_workflow
        final_state = run_workflow(topic="Test", stream_output=False)

        # Send completion
        await websocket.send_json({'status': 'completed', 'result': final_state})

    except Exception as e:
        await websocket.send_json({'status': 'failed', 'error': str(e)})

    finally:
        streamer.unsubscribe(send_event)
```

### Monitoring Dashboard

```python
from src.utils.progress_streamer import get_progress_streamer, ProgressEventType

class MonitoringDashboard:
    """Real-time monitoring dashboard."""

    def __init__(self):
        self.streamer = get_progress_streamer()
        self.streamer.subscribe(self.handle_event)

        self.current_agent = None
        self.start_time = None

    def handle_event(self, event):
        """Handle progress events."""

        if event.event_type == ProgressEventType.WORKFLOW_STARTED:
            self.start_time = event.timestamp
            self.display_header(event.message)

        elif event.event_type == ProgressEventType.AGENT_STARTED:
            self.current_agent = event.agent_name
            self.display_agent_start(event)

        elif event.event_type == ProgressEventType.AGENT_COMPLETED:
            elapsed = self.get_elapsed()
            self.display_agent_complete(event, elapsed)

        elif event.event_type == ProgressEventType.AGENT_FAILED:
            self.display_error(event)

        elif event.event_type == ProgressEventType.BUDGET_WARNING:
            self.display_warning(event)

        elif event.event_type == ProgressEventType.WORKFLOW_COMPLETED:
            total_time = self.get_elapsed()
            self.display_completion(total_time)

    def get_elapsed(self):
        """Get elapsed time since workflow start."""
        import time
        if self.start_time:
            return time.time() - self.start_time
        return 0

    def display_header(self, message):
        print("\n" + "=" * 70)
        print(f"🚀 {message}")
        print("=" * 70)

    def display_agent_start(self, event):
        print(f"\n▶️  {event.agent_name.upper()}")
        print(f"   {event.message}")

    def display_agent_complete(self, event, elapsed):
        print(f"✅ {event.agent_name}: {event.message} ({elapsed:.1f}s)")

    def display_error(self, event):
        print(f"❌ {event.agent_name}: {event.message}")

    def display_warning(self, event):
        print(f"⚠️  WARNING: {event.message}")

    def display_completion(self, total_time):
        print("\n" + "=" * 70)
        print(f"✅ Workflow completed in {total_time:.1f}s")
        print("=" * 70)


# Usage
dashboard = MonitoringDashboard()

from src.graph.workflow import run_workflow
run_workflow(topic="Transformers", stream_output=False)
```

**Output**:
```
======================================================================
🚀 Starting workflow: Transformers
======================================================================

▶️  PLANNER
   Breaking down task into subtasks
✅ planner: Created 5 subtasks (2.3s)

▶️  RESEARCHER
   Searching 5 queries across arXiv and Semantic Scholar
✅ researcher: Found 12 papers with 8 key findings (15.7s)

▶️  CODER
   Generating 3 code implementations
✅ coder: Generated 3 code blocks (12.4s)

... (more agents)

======================================================================
✅ Workflow completed in 45.2s
======================================================================
```

---

## Debugging with Progress Events

### Track Down Slowdowns

```python
streamer = get_progress_streamer()

# Run workflow
from src.graph.workflow import run_workflow
run_workflow(topic="Test", stream_output=False)

# Analyze agent execution times
history = streamer.get_history()

agent_times = {}
current_agent = None
start_time = None

for event in history:
    if event.event_type.value == 'agent_started':
        current_agent = event.agent_name
        start_time = event.timestamp

    elif event.event_type.value == 'agent_completed':
        if current_agent and start_time:
            duration = event.timestamp - start_time
            agent_times[current_agent] = duration

# Find slowest agent
slowest = max(agent_times.items(), key=lambda x: x[1])
print(f"Slowest agent: {slowest[0]} ({slowest[1]:.1f}s)")

# Find agents taking >10s
slow_agents = {k: v for k, v in agent_times.items() if v > 10}
print(f"Agents taking >10s: {slow_agents}")
```

### Detect Failures Early

```python
streamer = get_progress_streamer()

def alert_on_failure(event):
    """Send alert when agents fail."""
    if event.event_type.value in ['agent_failed', 'validation_failed', 'workflow_failed']:
        send_alert(
            title=f"{event.event_type.value.upper()}",
            message=f"{event.agent_name}: {event.message}",
            severity='high'
        )

streamer.subscribe(alert_on_failure)
```

---

## Integration Examples

### Slack Notifications

```python
from slack_sdk import WebClient
from src.utils.progress_streamer import get_progress_streamer, ProgressEventType

slack_client = WebClient(token="your-token")
streamer = get_progress_streamer()

def notify_slack(event):
    """Send important events to Slack."""

    # Only send key events
    important_events = [
        ProgressEventType.WORKFLOW_COMPLETED,
        ProgressEventType.WORKFLOW_FAILED,
        ProgressEventType.BUDGET_EXCEEDED
    ]

    if event.event_type in important_events:
        slack_client.chat_postMessage(
            channel='#reports',
            text=f"[{event.event_type.value}] {event.message}"
        )

streamer.subscribe(notify_slack)
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram
from src.utils.progress_streamer import get_progress_streamer, ProgressEventType

# Define metrics
workflow_completed = Counter('workflow_completed_total', 'Total workflows completed')
workflow_failed = Counter('workflow_failed_total', 'Total workflows failed')
agent_duration = Histogram('agent_duration_seconds', 'Agent execution time', ['agent_name'])

streamer = get_progress_streamer()

def export_metrics(event):
    """Export events to Prometheus."""

    if event.event_type == ProgressEventType.WORKFLOW_COMPLETED:
        workflow_completed.inc()

    elif event.event_type == ProgressEventType.WORKFLOW_FAILED:
        workflow_failed.inc()

    elif event.event_type == ProgressEventType.AGENT_COMPLETED:
        if 'duration' in event.metadata:
            agent_duration.labels(agent_name=event.agent_name).observe(
                event.metadata['duration']
            )

streamer.subscribe(export_metrics)
```

---

## Troubleshooting

### Issue: No events appearing

**Check**: Is streaming enabled?

```python
# Verify streamer is initialized
from src.utils.progress_streamer import get_progress_streamer

streamer = get_progress_streamer()
print(f"History: {len(streamer.get_history())} events")
```

### Issue: Duplicate events

**Cause**: Multiple subscribers or not resetting streamer

**Solution**:
```python
streamer = get_progress_streamer()
streamer.reset()  # Clear history before new workflow
```

### Issue: Subscriber errors crashing workflow

**Good news**: Subscriber errors are caught automatically and logged, they won't crash the workflow.

```python
# This won't crash the workflow
def bad_subscriber(event):
    raise ValueError("Oops!")

streamer.subscribe(bad_subscriber)
# Workflow continues, error is logged
```

---

## Next Steps

1. **Try It Out**: Run a workflow with `stream_output=True`
2. **Subscribe to Events**: Create a custom event handler
3. **Export Event History**: Export events for analysis
4. **Integrate**: Connect to monitoring/alerting systems
5. **Build Dashboard**: Create a real-time monitoring dashboard

For more advanced usage, see:
- `src/utils/progress_streamer.py` - Full implementation
- `tests/integration/test_progress_streaming.py` - Usage examples
- `TOKEN_TRACKING_GUIDE.md` - Related cost tracking feature

---

**Questions or Issues?**
Check logs in `outputs/app.log` for detailed event information.
