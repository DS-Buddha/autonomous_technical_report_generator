"""
Cross-agent voting mechanisms for consensus-based decision making.
"""

from typing import List, Dict, Any, Optional
from collections import Counter

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConsensusManager:
    """
    Manages cross-agent decision making through voting.
    """

    @staticmethod
    def majority_vote(votes: List[Any]) -> Any:
        """
        Simple majority voting.

        Args:
            votes: List of votes

        Returns:
            Most common vote
        """
        if not votes:
            return None

        counter = Counter(votes)
        result, count = counter.most_common(1)[0]

        logger.debug(f"Majority vote: {result} ({count}/{len(votes)} votes)")
        return result

    @staticmethod
    def weighted_vote(
        votes: Dict[str, Any],
        weights: Dict[str, float]
    ) -> Any:
        """
        Weighted voting based on agent expertise.

        Args:
            votes: Dict of agent_name -> vote
            weights: Dict of agent_name -> weight

        Returns:
            Vote with highest weighted support
        """
        weighted_votes = {}

        for agent, vote in votes.items():
            weight = weights.get(agent, 1.0)
            weighted_votes[vote] = weighted_votes.get(vote, 0.0) + weight

        if not weighted_votes:
            return None

        result = max(weighted_votes, key=weighted_votes.get)

        logger.debug(f"Weighted vote: {result} (weight: {weighted_votes[result]:.2f})")
        return result

    @staticmethod
    def consensus_threshold(
        votes: List[Any],
        threshold: float = 0.7
    ) -> Optional[Any]:
        """
        Require threshold agreement for consensus.

        Args:
            votes: List of votes
            threshold: Required agreement fraction (0-1)

        Returns:
            Consensus vote if threshold met, else None
        """
        if not votes:
            return None

        counter = Counter(votes)
        most_common, count = counter.most_common(1)[0]

        agreement = count / len(votes)

        if agreement >= threshold:
            logger.debug(f"Consensus reached: {most_common} ({agreement:.1%} agreement)")
            return most_common
        else:
            logger.debug(f"No consensus ({agreement:.1%} < {threshold:.1%})")
            return None

    @staticmethod
    def ranked_choice_voting(
        rankings: List[List[Any]]
    ) -> Any:
        """
        Ranked choice voting (instant runoff).

        Args:
            rankings: List of ranked preference lists

        Returns:
            Winning choice
        """
        if not rankings:
            return None

        # Count first choices
        first_choices = [r[0] for r in rankings if r]
        counter = Counter(first_choices)

        # Check for majority
        for choice, count in counter.most_common():
            if count > len(rankings) / 2:
                logger.debug(f"Ranked choice winner: {choice}")
                return choice

        # If no majority, return plurality winner
        result = counter.most_common(1)[0][0]
        logger.debug(f"Ranked choice plurality: {result}")
        return result
