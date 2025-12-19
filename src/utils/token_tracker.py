"""
Token usage tracking and cost estimation.

Monitors token consumption across all LLM calls to provide:
- Real-time cost tracking
- Per-agent usage statistics
- Budget alerts and warnings
- Cost optimization insights
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""

    agent_name: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'agent_name': self.agent_name,
            'model': self.model,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'estimated_cost_usd': self.estimated_cost_usd,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


class TokenTracker:
    """
    Centralized token usage tracking for cost monitoring.

    Usage:
        tracker = TokenTracker()

        # Track usage
        tracker.track_usage(
            agent_name="researcher",
            model="gemini-1.5-flash",
            prompt_tokens=1500,
            completion_tokens=800
        )

        # Get statistics
        stats = tracker.get_statistics()
        print(f"Total cost: ${stats['total_cost_usd']:.4f}")
    """

    # Pricing per 1M tokens (as of 2025)
    PRICING = {
        'gemini-1.5-flash': {
            'prompt': 0.075,      # $0.075 per 1M prompt tokens
            'completion': 0.30    # $0.30 per 1M completion tokens
        },
        'gemini-1.5-pro': {
            'prompt': 1.25,       # $1.25 per 1M prompt tokens
            'completion': 5.00    # $5.00 per 1M completion tokens
        },
        'gemini-2.0-flash-exp': {
            'prompt': 0.0,        # Free during experimental period
            'completion': 0.0     # Free during experimental period
        }
    }

    def __init__(self, budget_limit_usd: Optional[float] = None):
        """
        Initialize token tracker.

        Args:
            budget_limit_usd: Optional budget limit for alerts
        """
        self.usage_history: List[TokenUsage] = []
        self.budget_limit_usd = budget_limit_usd
        self._session_start = time.time()

        logger.info(f"Token tracker initialized (budget: ${budget_limit_usd or 'unlimited'})")

    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Calculate estimated cost for token usage.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        if model not in self.PRICING:
            logger.warning(f"Unknown model '{model}' - using Pro pricing as fallback")
            pricing = self.PRICING['gemini-1.5-pro']
        else:
            pricing = self.PRICING[model]

        # Calculate cost (pricing is per 1M tokens)
        prompt_cost = (prompt_tokens / 1_000_000) * pricing['prompt']
        completion_cost = (completion_tokens / 1_000_000) * pricing['completion']

        return prompt_cost + completion_cost

    def track_usage(
        self,
        agent_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> TokenUsage:
        """
        Track token usage for an LLM call.

        Args:
            agent_name: Name of the agent making the call
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            TokenUsage object with cost estimation
        """
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        usage = TokenUsage(
            agent_name=agent_name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost
        )

        self.usage_history.append(usage)

        # Log usage
        logger.info(
            f"Token usage tracked: {agent_name} | {model} | "
            f"{total_tokens:,} tokens | ${estimated_cost:.6f}"
        )

        # Check budget
        if self.budget_limit_usd:
            total_cost = self.get_total_cost()
            if total_cost > self.budget_limit_usd:
                logger.error(
                    f"⚠️  BUDGET EXCEEDED: ${total_cost:.4f} / ${self.budget_limit_usd:.4f}"
                )
            elif total_cost > self.budget_limit_usd * 0.9:
                logger.warning(
                    f"⚠️  Budget warning: ${total_cost:.4f} / ${self.budget_limit_usd:.4f} (90%+)"
                )

        return usage

    def get_total_cost(self) -> float:
        """Get total cost for all tracked usage."""
        return sum(u.estimated_cost_usd for u in self.usage_history)

    def get_total_tokens(self) -> int:
        """Get total tokens across all calls."""
        return sum(u.total_tokens for u in self.usage_history)

    def get_statistics(self) -> Dict:
        """
        Get comprehensive usage statistics.

        Returns:
            Dict with usage stats, costs, and breakdowns
        """
        if not self.usage_history:
            return {
                'total_calls': 0,
                'total_tokens': 0,
                'total_cost_usd': 0.0,
                'by_agent': {},
                'by_model': {}
            }

        # Overall stats
        total_calls = len(self.usage_history)
        total_tokens = self.get_total_tokens()
        total_cost = self.get_total_cost()

        # By agent breakdown
        by_agent = {}
        for usage in self.usage_history:
            if usage.agent_name not in by_agent:
                by_agent[usage.agent_name] = {
                    'calls': 0,
                    'tokens': 0,
                    'cost_usd': 0.0
                }

            by_agent[usage.agent_name]['calls'] += 1
            by_agent[usage.agent_name]['tokens'] += usage.total_tokens
            by_agent[usage.agent_name]['cost_usd'] += usage.estimated_cost_usd

        # By model breakdown
        by_model = {}
        for usage in self.usage_history:
            if usage.model not in by_model:
                by_model[usage.model] = {
                    'calls': 0,
                    'tokens': 0,
                    'cost_usd': 0.0
                }

            by_model[usage.model]['calls'] += 1
            by_model[usage.model]['tokens'] += usage.total_tokens
            by_model[usage.model]['cost_usd'] += usage.estimated_cost_usd

        # Calculate session duration
        session_duration_seconds = time.time() - self._session_start

        return {
            'total_calls': total_calls,
            'total_tokens': total_tokens,
            'total_cost_usd': total_cost,
            'session_duration_seconds': session_duration_seconds,
            'tokens_per_second': total_tokens / session_duration_seconds if session_duration_seconds > 0 else 0,
            'average_tokens_per_call': total_tokens / total_calls if total_calls > 0 else 0,
            'average_cost_per_call': total_cost / total_calls if total_calls > 0 else 0,
            'by_agent': by_agent,
            'by_model': by_model,
            'budget_limit_usd': self.budget_limit_usd,
            'budget_remaining_usd': (self.budget_limit_usd - total_cost) if self.budget_limit_usd else None,
            'budget_utilization_percent': (total_cost / self.budget_limit_usd * 100) if self.budget_limit_usd else None
        }

    def get_cost_breakdown_report(self) -> str:
        """
        Generate human-readable cost breakdown report.

        Returns:
            Formatted report string
        """
        stats = self.get_statistics()

        if stats['total_calls'] == 0:
            return "No token usage recorded yet."

        report = []
        report.append("=" * 60)
        report.append("TOKEN USAGE & COST REPORT")
        report.append("=" * 60)
        report.append("")

        # Overall stats
        report.append("📊 OVERALL STATISTICS")
        report.append(f"  Total LLM Calls:        {stats['total_calls']:,}")
        report.append(f"  Total Tokens:           {stats['total_tokens']:,}")
        report.append(f"  Total Cost:             ${stats['total_cost_usd']:.4f}")
        report.append(f"  Average Cost/Call:      ${stats['average_cost_per_call']:.6f}")
        report.append(f"  Session Duration:       {stats['session_duration_seconds']:.1f}s")
        report.append("")

        # Budget status
        if self.budget_limit_usd:
            report.append("💰 BUDGET STATUS")
            report.append(f"  Budget Limit:           ${self.budget_limit_usd:.4f}")
            report.append(f"  Budget Remaining:       ${stats['budget_remaining_usd']:.4f}")
            report.append(f"  Budget Utilization:     {stats['budget_utilization_percent']:.1f}%")
            report.append("")

        # By agent breakdown
        report.append("🤖 COST BY AGENT")
        for agent, data in sorted(
            stats['by_agent'].items(),
            key=lambda x: x[1]['cost_usd'],
            reverse=True
        ):
            pct = (data['cost_usd'] / stats['total_cost_usd'] * 100) if stats['total_cost_usd'] > 0 else 0
            report.append(
                f"  {agent:20s} ${data['cost_usd']:8.4f} ({pct:5.1f}%) | "
                f"{data['tokens']:,} tokens | {data['calls']} calls"
            )
        report.append("")

        # By model breakdown
        report.append("⚙️  COST BY MODEL")
        for model, data in sorted(
            stats['by_model'].items(),
            key=lambda x: x[1]['cost_usd'],
            reverse=True
        ):
            pct = (data['cost_usd'] / stats['total_cost_usd'] * 100) if stats['total_cost_usd'] > 0 else 0
            report.append(
                f"  {model:20s} ${data['cost_usd']:8.4f} ({pct:5.1f}%) | "
                f"{data['tokens']:,} tokens | {data['calls']} calls"
            )
        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def export_to_json(self, filepath: str) -> None:
        """
        Export usage history to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        data = {
            'statistics': self.get_statistics(),
            'usage_history': [u.to_dict() for u in self.usage_history]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Token usage exported to {filepath}")

    def reset(self) -> None:
        """Reset all tracked usage (useful for testing)."""
        self.usage_history.clear()
        self._session_start = time.time()
        logger.info("Token tracker reset")


# Global singleton instance
_global_tracker: Optional[TokenTracker] = None


def get_token_tracker(budget_limit_usd: Optional[float] = None) -> TokenTracker:
    """
    Get global token tracker instance (singleton pattern).

    Args:
        budget_limit_usd: Optional budget limit (only used on first call)

    Returns:
        Global TokenTracker instance
    """
    global _global_tracker

    if _global_tracker is None:
        _global_tracker = TokenTracker(budget_limit_usd=budget_limit_usd)

    return _global_tracker
