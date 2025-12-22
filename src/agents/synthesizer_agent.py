"""
Synthesizer agent for final report generation.
"""

from typing import Dict, List
from src.agents.base_agent import BaseAgent
from src.config.prompts import SYNTHESIZER_PROMPT, RESEARCH_INNOVATION_PROMPT
from src.tools.file_tools import FileTools
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SynthesizerAgent(BaseAgent):
    """
    Synthesizer agent that creates comprehensive markdown reports.
    Supports multiple report modes.
    """

    def __init__(self):
        super().__init__(
            name="Synthesizer",
            system_prompt=SYNTHESIZER_PROMPT,  # Default prompt
            temperature=0.6
        )
        self.file_tools = FileTools()
        self.prompts = {
            'staff_ml_engineer': SYNTHESIZER_PROMPT,
            'research_innovation': RESEARCH_INNOVATION_PROMPT
        }

    def run(self, state: Dict, **kwargs) -> Dict:
        """
        Synthesize all outputs into a comprehensive report.
        Supports multiple report modes.

        Args:
            state: Workflow state with all outputs

        Returns:
            Dict with final_report and metadata
        """
        # Extract components from state
        topic = state.get('topic', 'Technical Report')
        findings = state.get('key_findings', [])
        research_papers = state.get('research_papers', [])
        code_blocks = state.get('executable_code', {}) or state.get('generated_code', {})
        lit_summary = state.get('literature_summary', '')
        quality_scores = state.get('quality_scores', {})
        report_mode = state.get('report_mode', 'staff_ml_engineer')

        # Select appropriate prompt based on mode
        selected_prompt = self.prompts.get(report_mode, SYNTHESIZER_PROMPT)
        self.system_prompt = selected_prompt

        logger.info(f"Synthesizing report in '{report_mode}' mode")

        # Build mode-specific context prompt
        if report_mode == 'research_innovation':
            context_prompt = self._build_innovation_context(
                topic, research_papers, findings, lit_summary, code_blocks
            )
        else:  # staff_ml_engineer
            context_prompt = self._build_mentoring_context(
                topic, research_papers, findings, lit_summary, code_blocks
            )

        # Generate the complete report using the LLM
        logger.info(f"Generating {report_mode} report with LLM")
        final_report = self.generate_response(context_prompt)

        # Calculate metadata
        metadata = {
            'word_count': len(final_report.split()),
            'code_blocks': len(code_blocks),
            'references': len(research_papers),
            'format': report_mode,
            'quality_score': quality_scores.get('overall', 'N/A')
        }

        logger.info(
            f"Report generated: {metadata['word_count']} words, "
            f"{metadata['references']} papers, {metadata['code_blocks']} code examples"
        )

        return {
            'final_report': final_report,
            'report_metadata': metadata
        }

    def _build_mentoring_context(self, topic, research_papers, findings, lit_summary, code_blocks):
        """Build context prompt for Staff ML Engineer mentoring mode."""
        return f"""
Create a Staff ML Engineer mentoring report on: {topic}

Research Papers Reviewed ({len(research_papers)} papers):
{self._format_papers_for_context(research_papers[:10])}

Key Research Findings:
{self._format_findings_for_context(findings[:8])}

Literature Summary Context:
{lit_summary[:1000] if lit_summary else 'No summary available'}

Code Examples Available:
{list(code_blocks.keys()) if code_blocks else ['No code examples - focus on conceptual/architectural guidance']}

Your Task:
Write a comprehensive, production-focused mentoring report following the Staff ML Engineer format.
Focus on:
1. Common mistakes senior engineers make with this topic
2. How these mistakes fail in production (real symptoms, detection times, impact)
3. Production-grade fixes with code examples (use the code blocks provided as a starting point, but enhance them)
4. Preventive design principles and checklists
5. Hard-won lessons and war stories

The report should help senior engineers avoid production pitfalls.
"""

    def _build_innovation_context(self, topic, research_papers, findings, lit_summary, code_blocks):
        """Build context prompt for Research Innovation mode."""
        return f"""
Create a Research Innovation report on: {topic}

Research Papers Reviewed ({len(research_papers)} papers):
{self._format_papers_for_context(research_papers[:10])}

Key Research Findings:
{self._format_findings_for_context(findings[:8])}

Literature Summary Context:
{lit_summary[:1500] if lit_summary else 'No summary available'}

Code Examples Available:
{list(code_blocks.keys()) if code_blocks else ['No code examples available']}

Your Task:
Analyze the current research and generate novel research directions by drawing parallels with:
1. Neuroscience (neural networks, learning mechanisms, memory systems)
2. Quantum Physics (superposition, entanglement, uncertainty principles)
3. Biology (evolutionary algorithms, immune systems, cellular processes)
4. Complex Systems Theory (emergence, self-organization, network effects)
5. Other relevant scientific domains

Focus on:
1. Identifying deep parallels between {topic} and phenomena from other fields
2. Proposing novel research questions inspired by cross-domain insights
3. Suggesting concrete experiments to test these ideas
4. Explaining how these directions could advance the field
5. Being rigorous about where analogies hold and where they break down

Generate actionable, bold research directions that could open new avenues of investigation.
"""

    def _format_papers_for_context(self, papers: List[Dict]) -> str:
        """Format papers for LLM context."""
        formatted = []
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'Untitled')
            authors = ', '.join(paper.get('authors', [])[:3])
            year = paper.get('year', 'n.d.')
            abstract = paper.get('abstract', '')[:300]

            formatted.append(
                f"{i}. {title}\n"
                f"   Authors: {authors} ({year})\n"
                f"   Abstract: {abstract}...\n"
            )
        return '\n'.join(formatted)

    def _format_findings_for_context(self, findings: List[Dict]) -> str:
        """Format findings for LLM context."""
        formatted = []
        for i, finding in enumerate(findings, 1):
            title = finding.get('title', 'Untitled')
            summary = finding.get('summary', finding.get('abstract', ''))[:200]
            formatted.append(f"{i}. {title}\n   Key insight: {summary}...")
        return '\n'.join(formatted)

