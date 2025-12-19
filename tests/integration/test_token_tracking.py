"""
Integration tests for token usage tracking system.
"""

import pytest
from unittest.mock import Mock, patch
from src.utils.token_tracker import TokenTracker, get_token_tracker, TokenUsage


class TestTokenTracker:
    """Test suite for TokenTracker functionality."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = TokenTracker()
        assert tracker.get_total_cost() == 0.0
        assert tracker.get_total_tokens() == 0
        assert len(tracker.usage_history) == 0

    def test_initialization_with_budget(self):
        """Test tracker initialization with budget limit."""
        tracker = TokenTracker(budget_limit_usd=1.0)
        assert tracker.budget_limit_usd == 1.0

    def test_track_usage_flash_model(self):
        """Test tracking usage for Gemini Flash model."""
        tracker = TokenTracker()

        usage = tracker.track_usage(
            agent_name="test_agent",
            model="gemini-1.5-flash",
            prompt_tokens=1000,
            completion_tokens=500
        )

        assert usage.total_tokens == 1500
        assert usage.agent_name == "test_agent"
        assert usage.model == "gemini-1.5-flash"

        # Cost calculation: (1000/1M * 0.075) + (500/1M * 0.30) = 0.000075 + 0.00015 = 0.000225
        expected_cost = 0.000225
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.000001

    def test_track_usage_pro_model(self):
        """Test tracking usage for Gemini Pro model."""
        tracker = TokenTracker()

        usage = tracker.track_usage(
            agent_name="test_agent",
            model="gemini-1.5-pro",
            prompt_tokens=1000,
            completion_tokens=500
        )

        # Cost calculation: (1000/1M * 1.25) + (500/1M * 5.00) = 0.00125 + 0.0025 = 0.00375
        expected_cost = 0.00375
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.000001

    def test_track_multiple_calls(self):
        """Test tracking multiple LLM calls."""
        tracker = TokenTracker()

        # Track 3 calls
        tracker.track_usage("agent1", "gemini-1.5-flash", 1000, 500)
        tracker.track_usage("agent2", "gemini-1.5-flash", 2000, 1000)
        tracker.track_usage("agent3", "gemini-1.5-pro", 1000, 500)

        assert len(tracker.usage_history) == 3
        assert tracker.get_total_tokens() == 6000

    def test_get_statistics(self):
        """Test statistics generation."""
        tracker = TokenTracker()

        tracker.track_usage("planner", "gemini-1.5-flash", 1000, 500)
        tracker.track_usage("researcher", "gemini-1.5-flash", 2000, 1000)
        tracker.track_usage("coder", "gemini-1.5-pro", 1000, 500)

        stats = tracker.get_statistics()

        assert stats['total_calls'] == 3
        assert stats['total_tokens'] == 6000
        assert 'by_agent' in stats
        assert 'by_model' in stats

        # Check per-agent breakdown
        assert 'planner' in stats['by_agent']
        assert stats['by_agent']['planner']['calls'] == 1
        assert stats['by_agent']['planner']['tokens'] == 1500

        # Check per-model breakdown
        assert 'gemini-1.5-flash' in stats['by_model']
        assert stats['by_model']['gemini-1.5-flash']['calls'] == 2

    def test_budget_warning(self, caplog):
        """Test budget warning when approaching limit."""
        tracker = TokenTracker(budget_limit_usd=0.001)

        # This should trigger warning (cost ~0.00375, 375% of budget)
        tracker.track_usage("test", "gemini-1.5-pro", 1000, 500)

        # Check for budget exceeded log
        assert "BUDGET EXCEEDED" in caplog.text

    def test_budget_under_limit(self, caplog):
        """Test no warning when under budget."""
        tracker = TokenTracker(budget_limit_usd=1.0)

        # Small usage - should not trigger warning
        tracker.track_usage("test", "gemini-1.5-flash", 100, 50)

        # Should not have budget warnings
        assert "BUDGET EXCEEDED" not in caplog.text
        assert "Budget warning" not in caplog.text

    def test_cost_breakdown_report(self):
        """Test human-readable cost report generation."""
        tracker = TokenTracker()

        tracker.track_usage("planner", "gemini-1.5-flash", 1000, 500)
        tracker.track_usage("researcher", "gemini-1.5-flash", 2000, 1000)

        report = tracker.get_cost_breakdown_report()

        assert "TOKEN USAGE & COST REPORT" in report
        assert "planner" in report
        assert "researcher" in report
        assert "gemini-1.5-flash" in report

    def test_reset(self):
        """Test tracker reset functionality."""
        tracker = TokenTracker()

        tracker.track_usage("test", "gemini-1.5-flash", 1000, 500)
        assert len(tracker.usage_history) == 1

        tracker.reset()

        assert len(tracker.usage_history) == 0
        assert tracker.get_total_cost() == 0.0
        assert tracker.get_total_tokens() == 0

    def test_singleton_pattern(self):
        """Test global tracker singleton pattern."""
        # Reset singleton for test
        import src.utils.token_tracker as tracker_module
        tracker_module._global_tracker = None

        tracker1 = get_token_tracker()
        tracker2 = get_token_tracker()

        assert tracker1 is tracker2

        # Reset for other tests
        tracker_module._global_tracker = None

    def test_unknown_model_fallback(self, caplog):
        """Test fallback to Pro pricing for unknown models."""
        tracker = TokenTracker()

        usage = tracker.track_usage(
            agent_name="test",
            model="unknown-model",
            prompt_tokens=1000,
            completion_tokens=500
        )

        # Should use Pro pricing as fallback
        expected_cost = 0.00375  # Pro model cost
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.000001

        # Should log warning
        assert "Unknown model" in caplog.text


class TestTokenUsageIntegration:
    """Integration tests with BaseAgent."""

    @patch('src.agents.base_agent.genai.Client')
    def test_agent_tracks_tokens_automatically(self, mock_genai_client):
        """Test that agents automatically track token usage."""
        from src.agents.base_agent import BaseAgent
        from src.utils.token_tracker import get_token_tracker

        # Reset singleton
        import src.utils.token_tracker as tracker_module
        tracker_module._global_tracker = None

        # Mock response with usage metadata
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 1000
        mock_response.usage_metadata.candidates_token_count = 500

        mock_client_instance = Mock()
        mock_client_instance.models.generate_content.return_value = mock_response
        mock_genai_client.return_value = mock_client_instance

        # Create agent
        agent = BaseAgent(
            name="test_agent",
            system_prompt="Test",
            model="gemini-1.5-flash"
        )

        # Generate response (should track tokens)
        response = agent.generate_response("Test prompt")

        # Check token tracker recorded usage
        tracker = get_token_tracker()
        assert tracker.get_total_tokens() == 1500
        assert len(tracker.usage_history) == 1
        assert tracker.usage_history[0].agent_name == "test_agent"

        # Reset
        tracker_module._global_tracker = None


class TestCostCalculations:
    """Test cost calculation accuracy."""

    def test_flash_pricing_accuracy(self):
        """Test Flash model pricing calculation."""
        tracker = TokenTracker()

        # 1 million tokens each
        usage = tracker.track_usage(
            "test", "gemini-1.5-flash",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000
        )

        # Should be $0.075 + $0.30 = $0.375
        assert abs(usage.estimated_cost_usd - 0.375) < 0.001

    def test_pro_pricing_accuracy(self):
        """Test Pro model pricing calculation."""
        tracker = TokenTracker()

        # 1 million tokens each
        usage = tracker.track_usage(
            "test", "gemini-1.5-pro",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000
        )

        # Should be $1.25 + $5.00 = $6.25
        assert abs(usage.estimated_cost_usd - 6.25) < 0.01

    def test_typical_report_cost(self):
        """Test cost for typical report generation."""
        tracker = TokenTracker()

        # Simulate typical report generation
        # Planner: ~2K tokens
        tracker.track_usage("planner", "gemini-1.5-flash", 1000, 1000)

        # Researcher: ~10K tokens
        tracker.track_usage("researcher", "gemini-1.5-flash", 5000, 5000)

        # Coder: ~8K tokens (Pro model)
        tracker.track_usage("coder", "gemini-1.5-pro", 4000, 4000)

        # Tester: ~3K tokens
        tracker.track_usage("tester", "gemini-1.5-flash", 1500, 1500)

        # Critic: ~2K tokens (Pro model)
        tracker.track_usage("critic", "gemini-1.5-pro", 1000, 1000)

        # Synthesizer: ~5K tokens
        tracker.track_usage("synthesizer", "gemini-1.5-flash", 2500, 2500)

        total_cost = tracker.get_total_cost()

        # Total ~30K tokens, should be < $0.10
        assert total_cost < 0.10
        assert total_cost > 0.01  # Should cost something


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
