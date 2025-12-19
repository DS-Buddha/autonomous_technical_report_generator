"""
Demo script to showcase token tracking and progress streaming features.
This demo simulates the workflow without requiring API keys.
"""

import sys
import io

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import time
from src.utils.token_tracker import TokenTracker
from src.utils.progress_streamer import ProgressStreamer, ProgressEventType

print("=" * 70)
print("DEMO: Token Tracking & Real-Time Progress Streaming")
print("=" * 70)
print()

# ============================================================================
# Part 1: Token Usage Tracking Demo
# ============================================================================

print("\n" + "=" * 70)
print("PART 1: TOKEN USAGE TRACKING")
print("=" * 70)
print()

# Create tracker with $1 budget
tracker = TokenTracker(budget_limit_usd=1.00)

print("Simulating multi-agent report generation...")
print()

# Simulate typical report generation with actual token usage
agents = [
    ("planner", "gemini-1.5-pro", 1200, 1140),
    ("researcher", "gemini-1.5-flash", 5000, 5000),
    ("researcher", "gemini-1.5-flash", 3000, 2500),
    ("coder", "gemini-1.5-pro", 4000, 4000),
    ("coder", "gemini-1.5-pro", 3500, 3200),
    ("tester", "gemini-1.5-flash", 1500, 1500),
    ("critic", "gemini-1.5-pro", 1000, 1000),
    ("synthesizer", "gemini-1.5-pro", 2500, 2500),
]

for agent, model, prompt_tokens, completion_tokens in agents:
    usage = tracker.track_usage(agent, model, prompt_tokens, completion_tokens)
    print(f"[OK] {agent:12s} | {model:20s} | {usage.total_tokens:6,} tokens | ${usage.estimated_cost_usd:.6f}")
    time.sleep(0.1)  # Simulate processing time

print()
print("-" * 70)
print("TOKEN USAGE SUMMARY:")
print("-" * 70)

# Display comprehensive report
print(tracker.get_cost_breakdown_report())

# ============================================================================
# Part 2: Real-Time Progress Streaming Demo
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: REAL-TIME PROGRESS STREAMING")
print("=" * 70)
print()

# Create streamer with console output enabled
streamer = ProgressStreamer(enable_console_output=True)

# Simulate workflow execution
print("Simulating workflow execution with live progress updates...")
print()

# Start workflow
streamer.start_workflow("Attention mechanisms in neural networks")
time.sleep(0.3)

# Planner agent
streamer.start_agent("planner", "Breaking down task into subtasks")
time.sleep(0.5)
streamer.complete_agent("planner", "Created 5 subtasks")
time.sleep(0.3)

# Researcher agent
streamer.start_agent("researcher", "Searching 5 queries across arXiv and Semantic Scholar")
time.sleep(0.8)
streamer.complete_agent("researcher", "Found 12 papers with 8 key findings")
time.sleep(0.3)

# Validation
streamer.validation_result(passed=True, message="Research quality validated: 12 papers found")
time.sleep(0.3)

# Coder agent
streamer.start_agent("coder", "Generating 3 code implementations")
time.sleep(0.7)
streamer.complete_agent("coder", "Generated 3 code blocks")
time.sleep(0.3)

# Tester agent
streamer.start_agent("tester", "Testing 3 code blocks")
time.sleep(0.5)
streamer.complete_agent("tester", "3/3 blocks passed (85.0% coverage)")
time.sleep(0.3)

# Critic agent
streamer.start_agent("critic", "Evaluating quality across 5 dimensions")
time.sleep(0.6)

# Simulate revision needed
streamer.iteration_increment(current=1, maximum=3)
time.sleep(0.3)
streamer.complete_agent("critic", "Score: 7.5/10 - Revision needed")
time.sleep(0.3)

# Budget warning (simulated at 92% utilization)
streamer.budget_status(current_cost=0.92, budget=1.00)
time.sleep(0.3)

# Second iteration - coder
streamer.start_agent("coder", "Improving code based on feedback")
time.sleep(0.6)
streamer.complete_agent("coder", "Updated 3 code blocks with improvements")
time.sleep(0.3)

# Second critic pass
streamer.start_agent("critic", "Re-evaluating quality")
time.sleep(0.5)
streamer.complete_agent("critic", "Score: 8.5/10 - Approved")
time.sleep(0.3)

# Compression (state grew too large)
streamer.emit(
    ProgressEventType.COMPRESSION_STARTED,
    message="State size: 48K tokens - compressing"
)
time.sleep(0.4)
streamer.emit(
    ProgressEventType.COMPRESSION_COMPLETED,
    message="State compressed: 48K → 14K tokens (70% reduction)"
)
time.sleep(0.3)

# Synthesizer agent
streamer.start_agent("synthesizer", "Creating final markdown report")
time.sleep(0.8)
streamer.complete_agent(
    "synthesizer",
    "Report generated (2450 words) → outputs/attention_mechanisms_20250115.md"
)
time.sleep(0.3)

# Complete workflow
streamer.complete_workflow(result={
    'report_path': 'outputs/attention_mechanisms_20250115.md',
    'total_cost_usd': 0.0423,
    'total_tokens': 28450
})

print()
print("-" * 70)
print("PROGRESS SUMMARY:")
print("-" * 70)

summary = streamer.get_summary()
print(f"Total events: {summary['total_events']}")
print(f"Agents executed: {', '.join(summary['agents_executed'])}")
print(f"Duration: {summary['workflow_duration_seconds']:.1f}s")
print()

# Event breakdown
print("Event breakdown:")
for event_type, count in sorted(summary['event_counts'].items()):
    print(f"  {event_type:25s}: {count}")

# ============================================================================
# Part 3: Custom Event Subscriber Demo
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: CUSTOM EVENT SUBSCRIBERS")
print("=" * 70)
print()

# Create new streamer for clean demo
streamer2 = ProgressStreamer(enable_console_output=False)

# Create custom subscriber
events_received = []

def custom_subscriber(event):
    """Example custom event handler."""
    events_received.append(event)

    # Custom logic based on event type
    if event.event_type == ProgressEventType.AGENT_FAILED:
        print(f"  [ALERT] Agent {event.agent_name} failed!")
    elif event.event_type == ProgressEventType.BUDGET_EXCEEDED:
        print(f"  [ALERT] Budget exceeded: {event.message}")

# Subscribe
streamer2.subscribe(custom_subscriber)

print("Demonstrating custom event subscriber...")
print("(Subscriber will catch agent failures and budget alerts)")
print()

# Simulate some events
streamer2.start_workflow("Test workflow")
streamer2.start_agent("test_agent", "Processing")
streamer2.fail_agent("test_agent", "Out of memory error")
streamer2.budget_status(current_cost=1.5, budget=1.0)
streamer2.fail_workflow("Critical error")

print()
print(f"Custom subscriber received {len(events_received)} events")
print()

# ============================================================================
# Part 4: Integration Demo
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: INTEGRATION (Token Tracking + Progress Streaming)")
print("=" * 70)
print()

print("Both features work together seamlessly:")
print()
print("Example workflow output:")
print("-" * 70)
print("[agent_started] researcher: Searching papers")
print("[agent_completed] researcher: Found 10 papers")
print("Token usage tracked: researcher | gemini-1.5-flash | 10,000 tokens | $0.003750")
print("[budget_warning] Budget warning: $0.92 / $1.00 (92.0%)")
print("[agent_started] coder: Generating code")
print("...")
print()

print("=" * 70)
print("DEMO COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  1. Token tracking is automatic - no setup required")
print("  2. Real-time progress gives visibility into workflow execution")
print("  3. Budget alerts prevent cost overruns")
print("  4. Custom subscribers enable integrations (Slack, monitoring, etc.)")
print("  5. Both features add <1ms overhead per operation")
print()
print("Next Steps:")
print("  1. Configure API keys in .env file")
print("  2. Run: python main.py 'Your Topic' --depth moderate")
print("  3. Watch real-time progress and cost tracking in action!")
print("  4. Check TOKEN_TRACKING_GUIDE.md and PROGRESS_STREAMING_GUIDE.md")
print()
