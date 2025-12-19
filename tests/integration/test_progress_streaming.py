"""
Integration tests for real-time progress streaming.
"""

import pytest
from unittest.mock import Mock
from src.utils.progress_streamer import (
    ProgressStreamer,
    ProgressEvent,
    ProgressEventType,
    get_progress_streamer
)


class TestProgressEvent:
    """Test suite for ProgressEvent."""

    def test_event_creation(self):
        """Test creating a progress event."""
        event = ProgressEvent(
            event_type=ProgressEventType.AGENT_STARTED,
            agent_name="planner",
            message="Starting planning",
            metadata={'task': 'decompose'}
        )

        assert event.event_type == ProgressEventType.AGENT_STARTED
        assert event.agent_name == "planner"
        assert event.message == "Starting planning"
        assert event.metadata['task'] == 'decompose'

    def test_event_to_dict(self):
        """Test event serialization to dict."""
        event = ProgressEvent(
            event_type=ProgressEventType.AGENT_STARTED,
            agent_name="planner",
            message="Starting"
        )

        event_dict = event.to_dict()

        assert event_dict['event_type'] == 'agent_started'
        assert event_dict['agent_name'] == 'planner'
        assert event_dict['message'] == 'Starting'
        assert 'timestamp' in event_dict
        assert 'datetime' in event_dict

    def test_event_to_json(self):
        """Test event JSON serialization."""
        import json

        event = ProgressEvent(
            event_type=ProgressEventType.AGENT_COMPLETED,
            agent_name="researcher"
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed['event_type'] == 'agent_completed'
        assert parsed['agent_name'] == 'researcher'

    def test_event_string_representation(self):
        """Test event string formatting."""
        event = ProgressEvent(
            event_type=ProgressEventType.AGENT_STARTED,
            agent_name="planner",
            message="Breaking down task"
        )

        event_str = str(event)

        assert "[agent_started]" in event_str
        assert "planner:" in event_str
        assert "Breaking down task" in event_str


class TestProgressStreamer:
    """Test suite for ProgressStreamer."""

    def test_initialization(self):
        """Test streamer initialization."""
        streamer = ProgressStreamer(enable_console_output=False)

        assert len(streamer._history) == 0
        assert len(streamer._subscribers) == 0
        assert streamer.enable_console_output is False

    def test_emit_event(self):
        """Test emitting events."""
        streamer = ProgressStreamer(enable_console_output=False)

        event = streamer.emit(
            ProgressEventType.AGENT_STARTED,
            agent_name="planner",
            message="Starting"
        )

        assert len(streamer._history) == 1
        assert event.event_type == ProgressEventType.AGENT_STARTED
        assert event.agent_name == "planner"

    def test_subscribe_to_events(self):
        """Test subscribing to event stream."""
        streamer = ProgressStreamer(enable_console_output=False)

        events_received = []

        def callback(event: ProgressEvent):
            events_received.append(event)

        streamer.subscribe(callback)

        streamer.emit(ProgressEventType.AGENT_STARTED, agent_name="planner")
        streamer.emit(ProgressEventType.AGENT_COMPLETED, agent_name="planner")

        assert len(events_received) == 2
        assert events_received[0].event_type == ProgressEventType.AGENT_STARTED
        assert events_received[1].event_type == ProgressEventType.AGENT_COMPLETED

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        streamer = ProgressStreamer(enable_console_output=False)

        events_received = []

        def callback(event: ProgressEvent):
            events_received.append(event)

        streamer.subscribe(callback)
        streamer.emit(ProgressEventType.AGENT_STARTED, agent_name="test")

        streamer.unsubscribe(callback)
        streamer.emit(ProgressEventType.AGENT_COMPLETED, agent_name="test")

        # Should only receive first event (before unsubscribe)
        assert len(events_received) == 1

    def test_workflow_lifecycle(self):
        """Test complete workflow lifecycle events."""
        streamer = ProgressStreamer(enable_console_output=False)

        # Start workflow
        streamer.start_workflow("Test topic")

        # Execute agents
        streamer.start_agent("planner", "Planning")
        streamer.complete_agent("planner", "Created 5 subtasks")

        streamer.start_agent("researcher", "Researching")
        streamer.complete_agent("researcher", "Found 10 papers")

        # Complete workflow
        streamer.complete_workflow({'report_path': '/test.md'})

        history = streamer.get_history()

        # Should have: workflow_started, 2x agent_started, 2x agent_completed, workflow_completed
        assert len(history) == 6

        # Check event types
        event_types = [e.event_type for e in history]
        assert ProgressEventType.WORKFLOW_STARTED in event_types
        assert ProgressEventType.WORKFLOW_COMPLETED in event_types
        assert event_types.count(ProgressEventType.AGENT_STARTED) == 2
        assert event_types.count(ProgressEventType.AGENT_COMPLETED) == 2

    def test_workflow_failure(self):
        """Test workflow failure event."""
        streamer = ProgressStreamer(enable_console_output=False)

        streamer.start_workflow("Test")
        streamer.fail_workflow("API error")

        history = streamer.get_history()

        assert len(history) == 2
        assert history[1].event_type == ProgressEventType.WORKFLOW_FAILED
        assert "API error" in history[1].message

    def test_agent_failure(self):
        """Test agent failure event."""
        streamer = ProgressStreamer(enable_console_output=False)

        streamer.start_agent("coder", "Generating code")
        streamer.fail_agent("coder", "Syntax error")

        history = streamer.get_history()

        assert len(history) == 2
        assert history[1].event_type == ProgressEventType.AGENT_FAILED
        assert history[1].agent_name == "coder"

    def test_validation_events(self):
        """Test validation result events."""
        streamer = ProgressStreamer(enable_console_output=False)

        streamer.validation_result(passed=True, message="Research validated")
        streamer.validation_result(passed=False, message="Code quality failed")

        history = streamer.get_history()

        assert len(history) == 2
        assert history[0].event_type == ProgressEventType.VALIDATION_PASSED
        assert history[1].event_type == ProgressEventType.VALIDATION_FAILED

    def test_iteration_increment(self):
        """Test iteration counter events."""
        streamer = ProgressStreamer(enable_console_output=False)

        streamer.iteration_increment(current=1, maximum=3)
        streamer.iteration_increment(current=2, maximum=3)

        history = streamer.get_history()

        assert len(history) == 2
        assert all(e.event_type == ProgressEventType.ITERATION_INCREMENT for e in history)
        assert history[0].metadata['current'] == 1
        assert history[1].metadata['current'] == 2

    def test_budget_status_exceeded(self):
        """Test budget exceeded event."""
        streamer = ProgressStreamer(enable_console_output=False)

        streamer.budget_status(current_cost=1.5, budget=1.0)

        history = streamer.get_history()

        assert len(history) == 1
        assert history[0].event_type == ProgressEventType.BUDGET_EXCEEDED
        assert history[0].metadata['cost'] == 1.5
        assert history[0].metadata['budget'] == 1.0

    def test_budget_status_warning(self):
        """Test budget warning event (>90% utilization)."""
        streamer = ProgressStreamer(enable_console_output=False)

        streamer.budget_status(current_cost=0.95, budget=1.0)

        history = streamer.get_history()

        assert len(history) == 1
        assert history[0].event_type == ProgressEventType.BUDGET_WARNING
        assert history[0].metadata['utilization'] > 90

    def test_budget_status_ok(self):
        """Test no event when budget is fine."""
        streamer = ProgressStreamer(enable_console_output=False)

        streamer.budget_status(current_cost=0.5, budget=1.0)

        history = streamer.get_history()

        # Should not emit event for <90% utilization
        assert len(history) == 0

    def test_custom_status(self):
        """Test custom status messages."""
        streamer = ProgressStreamer(enable_console_output=False)

        streamer.custom_status("Processing data", metadata={'count': 100})

        history = streamer.get_history()

        assert len(history) == 1
        assert history[0].event_type == ProgressEventType.CUSTOM_STATUS
        assert history[0].message == "Processing data"
        assert history[0].metadata['count'] == 100

    def test_get_summary(self):
        """Test summary statistics generation."""
        streamer = ProgressStreamer(enable_console_output=False)

        streamer.start_workflow("Test")
        streamer.start_agent("planner", "Planning")
        streamer.complete_agent("planner", "Done")
        streamer.start_agent("researcher", "Researching")
        streamer.complete_agent("researcher", "Done")
        streamer.complete_workflow()

        summary = streamer.get_summary()

        assert summary['total_events'] == 6
        assert 'planner' in summary['agents_executed']
        assert 'researcher' in summary['agents_executed']
        assert 'workflow_duration_seconds' in summary
        assert 'event_counts' in summary

    def test_reset(self):
        """Test streamer reset."""
        streamer = ProgressStreamer(enable_console_output=False)

        streamer.emit(ProgressEventType.AGENT_STARTED, agent_name="test")
        assert len(streamer._history) == 1

        streamer.reset()

        assert len(streamer._history) == 0
        assert streamer._workflow_start_time is None
        assert streamer._current_agent is None

    def test_singleton_pattern(self):
        """Test global streamer singleton."""
        # Reset singleton
        import src.utils.progress_streamer as streamer_module
        streamer_module._global_streamer = None

        streamer1 = get_progress_streamer()
        streamer2 = get_progress_streamer()

        assert streamer1 is streamer2

        # Reset
        streamer_module._global_streamer = None

    def test_subscriber_error_handling(self):
        """Test that subscriber errors don't crash the streamer."""
        streamer = ProgressStreamer(enable_console_output=False)

        def bad_callback(event):
            raise ValueError("Subscriber error")

        streamer.subscribe(bad_callback)

        # Should not raise exception
        streamer.emit(ProgressEventType.AGENT_STARTED, agent_name="test")

        # Event should still be in history
        assert len(streamer._history) == 1


class TestProgressIntegration:
    """Integration tests with workflow nodes."""

    def test_node_emits_events(self, monkeypatch):
        """Test that nodes emit progress events during execution."""
        from src.graph.state import create_initial_state
        from src.graph.nodes import planner_node
        from src.utils.progress_streamer import get_progress_streamer

        # Reset singleton
        import src.utils.progress_streamer as streamer_module
        streamer_module._global_streamer = None

        # Mock planner agent to avoid real API call
        from unittest.mock import Mock
        import src.graph.nodes as nodes_module

        mock_planner = Mock()
        mock_planner.run.return_value = {
            'plan': {},
            'subtasks': [{'id': '1', 'description': 'test'}],
            'dependencies': {},
            'search_queries': ['test query']
        }

        monkeypatch.setattr(nodes_module, 'planner', mock_planner)

        # Create state
        state = create_initial_state(topic="Test topic")

        # Execute node
        planner_node(state)

        # Check streamer recorded events
        streamer = get_progress_streamer()
        history = streamer.get_history()

        # Should have agent_started and agent_completed
        assert len(history) >= 2
        event_types = [e.event_type for e in history]
        assert ProgressEventType.AGENT_STARTED in event_types
        assert ProgressEventType.AGENT_COMPLETED in event_types

        # Reset
        streamer_module._global_streamer = None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
