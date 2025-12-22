"""
Real-time progress streaming for workflow execution.

Provides live updates as agents execute, enabling:
- Real-time status monitoring
- Progress bars for long-running tasks
- Early error detection
- Better user experience
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProgressEventType(Enum):
    """Types of progress events."""

    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"

    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"

    VALIDATION_STARTED = "validation_started"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"

    COMPRESSION_STARTED = "compression_started"
    COMPRESSION_COMPLETED = "compression_completed"

    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"

    ITERATION_INCREMENT = "iteration_increment"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"

    CUSTOM_STATUS = "custom_status"


@dataclass
class ProgressEvent:
    """A single progress event."""

    event_type: ProgressEventType
    agent_name: Optional[str] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'event_type': self.event_type.value,
            'agent_name': self.agent_name,
            'message': self.message,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def __str__(self) -> str:
        """Human-readable event description."""
        parts = [f"[{self.event_type.value}]"]

        if self.agent_name:
            parts.append(f"{self.agent_name}:")

        if self.message:
            parts.append(self.message)

        return " ".join(parts)


class ProgressStreamer:
    """
    Real-time progress streaming for workflow execution.

    Usage:
        # Create streamer
        streamer = ProgressStreamer()

        # Subscribe to events
        def on_progress(event: ProgressEvent):
            print(f"Progress: {event}")

        streamer.subscribe(on_progress)

        # Emit events
        streamer.emit(
            ProgressEventType.AGENT_STARTED,
            agent_name="planner",
            message="Breaking down task into subtasks"
        )

        # Get event history
        history = streamer.get_history()
    """

    def __init__(self, enable_console_output: bool = True):
        """
        Initialize progress streamer.

        Args:
            enable_console_output: Whether to print events to console
        """
        self.enable_console_output = enable_console_output
        self._subscribers: List[Callable[[ProgressEvent], None]] = []
        self._history: List[ProgressEvent] = []
        self._workflow_start_time: Optional[float] = None
        self._current_agent: Optional[str] = None

        logger.info("Progress streamer initialized")

    def subscribe(self, callback: Callable[[ProgressEvent], None]) -> None:
        """
        Subscribe to progress events.

        Args:
            callback: Function to call with each event
        """
        self._subscribers.append(callback)
        logger.debug(f"Added subscriber (total: {len(self._subscribers)})")

    def unsubscribe(self, callback: Callable[[ProgressEvent], None]) -> None:
        """
        Unsubscribe from progress events.

        Args:
            callback: Function to remove from subscribers
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.debug(f"Removed subscriber (remaining: {len(self._subscribers)})")

    def emit(
        self,
        event_type: ProgressEventType,
        agent_name: Optional[str] = None,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProgressEvent:
        """
        Emit a progress event.

        Args:
            event_type: Type of event
            agent_name: Name of agent (if applicable)
            message: Human-readable message
            metadata: Additional event data

        Returns:
            The emitted ProgressEvent
        """
        event = ProgressEvent(
            event_type=event_type,
            agent_name=agent_name or self._current_agent,
            message=message,
            metadata=metadata or {}
        )

        # Add to history
        self._history.append(event)

        # Console output
        if self.enable_console_output:
            self._print_event(event)

        # Notify subscribers
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")

        return event

    def _print_event(self, event: ProgressEvent) -> None:
        """
        Print event to console with formatting.

        Args:
            event: Event to print
        """
        # Color codes for different event types
        colors = {
            ProgressEventType.WORKFLOW_STARTED: "\033[94m",      # Blue
            ProgressEventType.WORKFLOW_COMPLETED: "\033[92m",    # Green
            ProgressEventType.WORKFLOW_FAILED: "\033[91m",       # Red
            ProgressEventType.AGENT_STARTED: "\033[96m",         # Cyan
            ProgressEventType.AGENT_COMPLETED: "\033[92m",       # Green
            ProgressEventType.AGENT_FAILED: "\033[91m",          # Red
            ProgressEventType.VALIDATION_FAILED: "\033[93m",     # Yellow
            ProgressEventType.PHASE_STARTED: "\033[95m",         # Magenta (bright)
            ProgressEventType.PHASE_COMPLETED: "\033[92m",       # Green
            ProgressEventType.BUDGET_WARNING: "\033[93m",        # Yellow
            ProgressEventType.BUDGET_EXCEEDED: "\033[91m",       # Red
        }
        reset = "\033[0m"

        color = colors.get(event.event_type, "")
        print(f"{color}{event}{reset}")

    def start_workflow(self, task: str) -> None:
        """
        Signal workflow start.

        Args:
            task: Task description
        """
        self._workflow_start_time = time.time()
        self.emit(
            ProgressEventType.WORKFLOW_STARTED,
            message=f"Starting workflow: {task}",
            metadata={'task': task}
        )

    def complete_workflow(self, result: Optional[Dict] = None) -> None:
        """
        Signal workflow completion.

        Args:
            result: Optional result metadata
        """
        duration = time.time() - self._workflow_start_time if self._workflow_start_time else 0

        self.emit(
            ProgressEventType.WORKFLOW_COMPLETED,
            message=f"Workflow completed in {duration:.1f}s",
            metadata={'duration_seconds': duration, 'result': result}
        )

    def fail_workflow(self, error: str) -> None:
        """
        Signal workflow failure.

        Args:
            error: Error message
        """
        duration = time.time() - self._workflow_start_time if self._workflow_start_time else 0

        self.emit(
            ProgressEventType.WORKFLOW_FAILED,
            message=f"Workflow failed after {duration:.1f}s: {error}",
            metadata={'duration_seconds': duration, 'error': error}
        )

    def start_agent(self, agent_name: str, task: str = "") -> None:
        """
        Signal agent execution start.

        Args:
            agent_name: Name of agent
            task: Optional task description
        """
        self._current_agent = agent_name

        self.emit(
            ProgressEventType.AGENT_STARTED,
            agent_name=agent_name,
            message=task or f"Executing {agent_name}",
            metadata={'task': task}
        )

    def complete_agent(
        self,
        agent_name: str,
        result_summary: Optional[str] = None
    ) -> None:
        """
        Signal agent execution completion.

        Args:
            agent_name: Name of agent
            result_summary: Optional result summary
        """
        message = result_summary or f"{agent_name} completed"

        self.emit(
            ProgressEventType.AGENT_COMPLETED,
            agent_name=agent_name,
            message=message
        )

        if self._current_agent == agent_name:
            self._current_agent = None

    def fail_agent(self, agent_name: str, error: str) -> None:
        """
        Signal agent execution failure.

        Args:
            agent_name: Name of agent
            error: Error message
        """
        self.emit(
            ProgressEventType.AGENT_FAILED,
            agent_name=agent_name,
            message=f"Failed: {error}",
            metadata={'error': error}
        )

        if self._current_agent == agent_name:
            self._current_agent = None

    def validation_result(self, passed: bool, message: str) -> None:
        """
        Signal validation result.

        Args:
            passed: Whether validation passed
            message: Validation message
        """
        event_type = (
            ProgressEventType.VALIDATION_PASSED if passed
            else ProgressEventType.VALIDATION_FAILED
        )

        self.emit(event_type, message=message, metadata={'passed': passed})

    def iteration_increment(self, current: int, maximum: int) -> None:
        """
        Signal iteration counter increment.

        Args:
            current: Current iteration
            maximum: Maximum iterations
        """
        self.emit(
            ProgressEventType.ITERATION_INCREMENT,
            message=f"Revision needed - iteration {current}/{maximum}",
            metadata={'current': current, 'maximum': maximum}
        )

    def budget_status(self, current_cost: float, budget: float) -> None:
        """
        Signal budget status.

        Args:
            current_cost: Current cost in USD
            budget: Budget limit in USD
        """
        utilization = (current_cost / budget * 100) if budget > 0 else 0

        if current_cost > budget:
            self.emit(
                ProgressEventType.BUDGET_EXCEEDED,
                message=f"Budget exceeded: ${current_cost:.4f} / ${budget:.4f}",
                metadata={'cost': current_cost, 'budget': budget, 'utilization': utilization}
            )
        elif utilization > 90:
            self.emit(
                ProgressEventType.BUDGET_WARNING,
                message=f"Budget warning: ${current_cost:.4f} / ${budget:.4f} ({utilization:.1f}%)",
                metadata={'cost': current_cost, 'budget': budget, 'utilization': utilization}
            )

    def custom_status(self, message: str, metadata: Optional[Dict] = None) -> None:
        """
        Emit custom status message.

        Args:
            message: Status message
            metadata: Optional metadata
        """
        self.emit(
            ProgressEventType.CUSTOM_STATUS,
            message=message,
            metadata=metadata or {}
        )

    def start_phase(self, phase_name: str, description: str = "") -> None:
        """
        Signal start of a new workflow phase.

        Args:
            phase_name: Name of the phase
            description: Optional phase description
        """
        self.emit(
            ProgressEventType.PHASE_STARTED,
            message=f"Starting {phase_name}",
            metadata={'phase_name': phase_name, 'description': description}
        )

    def complete_phase(self, phase_name: str, summary: str = "") -> None:
        """
        Signal completion of a workflow phase.

        Args:
            phase_name: Name of the phase
            summary: Optional result summary
        """
        self.emit(
            ProgressEventType.PHASE_COMPLETED,
            message=f"Completed {phase_name}",
            metadata={'phase_name': phase_name, 'summary': summary}
        )

    def get_history(self) -> List[ProgressEvent]:
        """
        Get event history.

        Returns:
            List of all events
        """
        return self._history.copy()

    def get_summary(self) -> Dict:
        """
        Get summary statistics.

        Returns:
            Dict with statistics
        """
        if not self._history:
            return {'total_events': 0}

        # Count events by type
        event_counts = {}
        for event in self._history:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Find agent executions
        agents_executed = set(
            e.agent_name for e in self._history
            if e.agent_name and e.event_type == ProgressEventType.AGENT_COMPLETED
        )

        # Calculate duration
        duration = None
        if self._workflow_start_time:
            duration = time.time() - self._workflow_start_time

        return {
            'total_events': len(self._history),
            'event_counts': event_counts,
            'agents_executed': list(agents_executed),
            'workflow_duration_seconds': duration
        }

    def export_to_json(self, filepath: str) -> None:
        """
        Export event history to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        data = {
            'summary': self.get_summary(),
            'events': [e.to_dict() for e in self._history]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Progress history exported to {filepath}")

    def reset(self) -> None:
        """Reset streamer state (useful for testing)."""
        self._history.clear()
        self._workflow_start_time = None
        self._current_agent = None
        logger.info("Progress streamer reset")


# Global singleton instance
_global_streamer: Optional[ProgressStreamer] = None


def get_progress_streamer(enable_console_output: bool = True) -> ProgressStreamer:
    """
    Get global progress streamer instance (singleton pattern).

    Args:
        enable_console_output: Whether to enable console output (only on first call)

    Returns:
        Global ProgressStreamer instance
    """
    global _global_streamer

    if _global_streamer is None:
        _global_streamer = ProgressStreamer(enable_console_output=enable_console_output)

    return _global_streamer
