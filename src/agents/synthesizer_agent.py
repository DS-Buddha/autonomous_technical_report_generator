"""
Synthesizer agent for final report generation.
"""

from typing import Dict, List
from src.agents.base_agent import BaseAgent
from src.config.prompts import SYNTHESIZER_PROMPT
from src.tools.file_tools import FileTools
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SynthesizerAgent(BaseAgent):
    """
    Synthesizer agent that creates comprehensive markdown reports.
    """

    def __init__(self):
        super().__init__(
            name="Synthesizer",
            system_prompt=SYNTHESIZER_PROMPT,
            temperature=0.6
        )
        self.file_tools = FileTools()

    def run(self, state: Dict, **kwargs) -> Dict:
        """
        Synthesize all outputs into a final report.

        Args:
            state: Workflow state with all outputs

        Returns:
            Dict with final_report and metadata
        """
        logger.info("Synthesizing final report")

        # Extract components from state
        topic = state.get('topic', 'Technical Report')
        findings = state.get('key_findings', [])
        code_blocks = state.get('executable_code', {})
        lit_summary = state.get('literature_summary', '')

        # Generate report sections
        report_parts = []

        # Metadata header
        report_parts.append(self.file_tools.add_metadata_header(
            topic,
            metadata={
                'papers_reviewed': len(state.get('research_papers', [])),
                'code_examples': len(code_blocks),
                'quality_score': state.get('quality_scores', {}).get('overall', 'N/A')
            }
        ))

        # Title and abstract
        report_parts.append(f"# {topic}\n\n")
        report_parts.append(self._generate_abstract(topic, findings))

        # Introduction
        report_parts.append(self._generate_section("Introduction", topic, lit_summary))

        # Literature Review
        report_parts.append(self._generate_literature_review(findings))

        # Implementation
        if code_blocks:
            report_parts.append(self._generate_implementation_section(code_blocks, state))

        # Conclusion
        report_parts.append(self._generate_conclusion(topic, findings))

        # References
        report_parts.append(self.file_tools.format_reference_list(findings))

        # Combine report
        final_report = '\n'.join(report_parts)

        # Calculate metadata
        metadata = {
            'word_count': len(final_report.split()),
            'code_blocks': len(code_blocks),
            'references': len(findings),
            'sections': 6
        }

        logger.info(f"Report generated: {metadata['word_count']} words, {metadata['code_blocks']} code examples")

        return {
            'final_report': final_report,
            'report_metadata': metadata
        }

    def _generate_abstract(self, topic: str, findings: List[Dict]) -> str:
        """Generate abstract section."""
        prompt = f"""
Write a technical abstract (150-200 words) for a report on: {topic}

Based on these research findings:
{self._format_findings_brief(findings[:5])}

The abstract should summarize the topic, approach, and key insights.
"""
        abstract = self.generate_response(prompt)
        return f"## Abstract\n\n{abstract}\n\n"

    def _generate_section(self, title: str, topic: str, context: str) -> str:
        """Generate a report section."""
        prompt = f"""
Write a {title} section (300-400 words) for a technical report on: {topic}

Context: {context[:500]}

Make it informative and well-structured.
"""
        content = self.generate_response(prompt)
        return f"## {title}\n\n{content}\n\n"

    def _generate_literature_review(self, findings: List[Dict]) -> str:
        """Generate literature review section."""
        review_parts = ["## Literature Review\n\n"]

        for i, finding in enumerate(findings[:5], 1):
            title = finding.get('title', 'Untitled')
            authors = ', '.join(finding.get('authors', [])[:2])
            year = finding.get('year', 'n.d.')
            abstract = finding.get('abstract', '')[:300]

            review_parts.append(f"### {i}. {title}\n\n")
            review_parts.append(f"**Authors:** {authors} ({year})\n\n")
            review_parts.append(f"{abstract}...\n\n")

        return ''.join(review_parts)

    def _generate_implementation_section(self, code_blocks: Dict, state: Dict) -> str:
        """Generate implementation section with code."""
        impl_parts = ["## Implementation\n\n"]

        for code_id, code in list(code_blocks.items())[:3]:  # Max 3 examples
            # Get specification description if available
            specs = state.get('code_specifications', [])
            description = next(
                (s['description'] for s in specs if s.get('id') == code_id),
                "Implementation example"
            )

            impl_parts.append(f"### {description}\n\n")
            impl_parts.append(self.file_tools.format_code_block(code, 'python'))

        return ''.join(impl_parts)

    def _generate_conclusion(self, topic: str, findings: List[Dict]) -> str:
        """Generate conclusion section."""
        prompt = f"""
Write a conclusion (200-250 words) for a technical report on: {topic}

Summarize key takeaways and suggest future directions.
Base it on {len(findings)} research papers reviewed.
"""
        conclusion = self.generate_response(prompt)
        return f"## Conclusion\n\n{conclusion}\n\n"

    def _format_findings_brief(self, findings: List[Dict]) -> str:
        """Format findings briefly for prompts."""
        return '\n'.join([
            f"- {f.get('title', 'Untitled')} ({f.get('year', 'n.d.')})"
            for f in findings
        ])
