"""
Quality validators for cross-agent validation gates.
"""

from typing import Dict
from src.graph.state import AgentState
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QualityValidator:
    """
    Validates outputs against quality criteria.
    """

    def __init__(self):
        self.settings = settings

    def validate_research(self, state: AgentState) -> bool:
        """
        Validate research phase outputs.

        Args:
            state: Workflow state

        Returns:
            True if research meets quality standards
        """
        papers = state.get('research_papers', [])
        findings = state.get('key_findings', [])

        papers_ok = len(papers) >= self.settings.min_research_papers
        findings_ok = len(findings) >= self.settings.min_key_findings

        valid = papers_ok and findings_ok

        logger.info(
            f"Research validation: {'✓' if valid else '✗'} "
            f"(papers: {len(papers)}/{self.settings.min_research_papers}, "
            f"findings: {len(findings)}/{self.settings.min_key_findings})"
        )

        return valid

    def validate_code(self, state: AgentState) -> bool:
        """
        Validate code quality.

        Args:
            state: Workflow state

        Returns:
            True if code meets quality standards
        """
        coverage = state.get('test_coverage', 0.0)
        errors = state.get('validation_errors', [])

        coverage_ok = coverage >= self.settings.min_test_coverage
        no_errors = len(errors) == 0

        valid = coverage_ok and no_errors

        logger.info(
            f"Code validation: {'✓' if valid else '✗'} "
            f"(coverage: {coverage:.1f}%/{self.settings.min_test_coverage}%, "
            f"errors: {len(errors)})"
        )

        return valid

    def validate_overall_quality(self, state: AgentState) -> bool:
        """
        Validate overall report quality.

        Args:
            state: Workflow state

        Returns:
            True if quality meets standards
        """
        scores = state.get('quality_scores', {})

        if not scores:
            logger.warning("No quality scores available")
            return False

        avg_score = sum(scores.values()) / len(scores)
        valid = avg_score >= self.settings.min_quality_score

        logger.info(
            f"Quality validation: {'✓' if valid else '✗'} "
            f"(avg score: {avg_score:.1f}/{self.settings.min_quality_score})"
        )

        return valid

    def validate_all(self, state: AgentState) -> Dict[str, bool]:
        """
        Run all validations.

        Args:
            state: Workflow state

        Returns:
            Dict of validation results
        """
        return {
            'research': self.validate_research(state),
            'code': self.validate_code(state),
            'quality': self.validate_overall_quality(state)
        }
