"""
Critic agent for quality evaluation and feedback.
"""

from typing import Dict, List
from src.agents.base_agent import BaseAgent
from src.config.prompts import CRITIC_PROMPT
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CriticAgent(BaseAgent):
    """
    Critic agent that evaluates quality and provides feedback.
    """

    def __init__(self):
        super().__init__(
            name="Critic",
            system_prompt=CRITIC_PROMPT,
            temperature=0.4
        )

    def run(self, state: Dict, **kwargs) -> Dict:
        """
        Evaluate overall quality of research, code, and outputs.

        Args:
            state: Current workflow state with all outputs

        Returns:
            Dict with quality scores, feedback, and revision flag
        """
        logger.info("Evaluating quality across all dimensions")

        # Define evaluation criteria
        criteria = ['accuracy', 'completeness', 'code_quality', 'clarity', 'executability']

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(state, criteria)

        # Get evaluation
        evaluation = self.generate_json_response(prompt)

        # Calculate overall score
        scores = evaluation.get('dimension_scores', {})
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0

        # Determine if revision needed
        needs_revision = (
            overall_score < settings.min_quality_score or
            any(score < 5.0 for score in scores.values())
        )

        result = {
            'quality_scores': scores,
            'overall_score': overall_score,
            'feedback': evaluation.get('feedback', {}),
            'needs_revision': needs_revision,
            'priority_issues': evaluation.get('priority_issues', [])
        }

        logger.info(f"Quality evaluation complete: {overall_score:.1f}/10.0")
        if needs_revision:
            logger.warning("Quality below threshold - revision needed")

        return result

    def _build_evaluation_prompt(self, state: Dict, criteria: List[str]) -> str:
        """Build evaluation prompt from state."""
        prompt_parts = [
            "Evaluate the following technical report components:\n",
            f"\n## Research ({len(state.get('research_papers', []))} papers)",
            f"Key findings: {len(state.get('key_findings', []))}",
            f"\n## Code ({len(state.get('generated_code', {}))} blocks)",
            f"Executable: {len(state.get('executable_code', {}))}",
            f"Test coverage: {state.get('test_coverage', 0):.1f}%",
            f"\n## Errors",
            f"Validation errors: {len(state.get('validation_errors', []))}",
        ]

        prompt_parts.append("\n\nEvaluate on these dimensions (0-10 scale):")
        prompt_parts.append("1. Accuracy: Factual correctness, proper citations")
        prompt_parts.append("2. Completeness: All requirements addressed")
        prompt_parts.append("3. Code Quality: Clean, documented, follows best practices")
        prompt_parts.append("4. Clarity: Clear explanations, logical flow")
        prompt_parts.append("5. Executability: Code runs without errors")

        prompt_parts.append("\n\nRespond with JSON:")
        prompt_parts.append("""{
    "dimension_scores": {"accuracy": X, "completeness": X, "code_quality": X, "clarity": X, "executability": X},
    "feedback": {"dimension": "specific feedback"},
    "priority_issues": ["issue1", "issue2"]
}""")

        return '\n'.join(prompt_parts)
