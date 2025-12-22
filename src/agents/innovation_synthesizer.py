"""
Innovation Synthesizer agent for generating novel research ideas from cross-domain insights.
"""

from typing import Dict, List
from src.agents.base_agent import BaseAgent
from src.config.prompts import RESEARCH_INNOVATION_PROMPT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InnovationSynthesizer(BaseAgent):
    """
    Innovation synthesizer that generates state-of-the-art research ideas
    by combining domain research with cross-domain parallels.
    """

    def __init__(self):
        super().__init__(
            name="InnovationSynthesizer",
            system_prompt=RESEARCH_INNOVATION_PROMPT,
            temperature=0.7  # Higher creativity for innovation
        )

    def run(self, state: Dict, **kwargs) -> Dict:
        """
        Synthesize novel research directions from all gathered insights.

        Args:
            state: Workflow state with research and cross-domain analysis

        Returns:
            Dict with final innovation report
        """
        topic = state.get('topic', 'Technical Report')

        # Original domain research
        research_papers = state.get('research_papers', [])
        key_findings = state.get('key_findings', [])
        lit_summary = state.get('literature_summary', '')

        # Cross-domain analysis
        cross_domain_analysis = state.get('cross_domain_analysis', {})
        cross_domain_papers = state.get('cross_domain_papers', [])

        # Code examples
        code_blocks = state.get('executable_code', {}) or state.get('generated_code', {})

        # Quality metrics
        quality_scores = state.get('quality_scores', {})

        logger.info(f"Synthesizing innovation report for: {topic}")
        logger.info(f"Using {len(research_papers)} domain papers and {len(cross_domain_papers)} cross-domain papers")
        logger.info(f"Analyzing {len(cross_domain_analysis.get('cross_domain_parallels', []))} parallels")

        # Build comprehensive innovation context
        context_prompt = self._build_innovation_context(
            topic=topic,
            research_papers=research_papers,
            key_findings=key_findings,
            lit_summary=lit_summary,
            cross_domain_analysis=cross_domain_analysis,
            cross_domain_papers=cross_domain_papers,
            code_blocks=code_blocks
        )

        # Generate the innovation report
        logger.info("Generating comprehensive cross-domain innovation report")
        final_report = self.generate_response(context_prompt)

        # Calculate metadata
        metadata = {
            'word_count': len(final_report.split()),
            'code_blocks': len(code_blocks),
            'domain_papers': len(research_papers),
            'cross_domain_papers': len(cross_domain_papers),
            'parallels_analyzed': len(cross_domain_analysis.get('cross_domain_parallels', [])),
            'format': 'research_innovation',
            'quality_score': quality_scores.get('overall', 'N/A')
        }

        logger.info(
            f"Innovation report generated: {metadata['word_count']} words, "
            f"{metadata['domain_papers']} domain papers, "
            f"{metadata['cross_domain_papers']} cross-domain papers, "
            f"{metadata['parallels_analyzed']} parallels"
        )

        return {
            'final_report': final_report,
            'report_metadata': metadata
        }

    def _build_innovation_context(
        self,
        topic: str,
        research_papers: List[Dict],
        key_findings: List[Dict],
        lit_summary: str,
        cross_domain_analysis: Dict,
        cross_domain_papers: List[Dict],
        code_blocks: Dict
    ) -> str:
        """Build comprehensive context for innovation synthesis."""

        # Extract cross-domain parallels
        parallels = cross_domain_analysis.get('cross_domain_parallels', [])
        core_principles = cross_domain_analysis.get('core_principles', [])

        # Format domain papers
        domain_papers_formatted = self._format_papers(research_papers[:10], "ML/AI Domain")

        # Format cross-domain papers by domain
        cross_domain_formatted = self._format_cross_domain_papers(cross_domain_papers, parallels)

        # Format parallels analysis
        parallels_formatted = self._format_parallels(parallels, core_principles)

        # Format key findings
        findings_formatted = self._format_findings(key_findings[:10])

        return f"""
Create a STATE-OF-THE-ART Research Innovation report on: {topic}

# CORE PRINCIPLES IDENTIFIED:
{self._format_core_principles(core_principles)}

# ORIGINAL DOMAIN RESEARCH ({len(research_papers)} papers):
{domain_papers_formatted}

# KEY FINDINGS FROM DOMAIN RESEARCH:
{findings_formatted}

# LITERATURE SUMMARY:
{lit_summary[:1500] if lit_summary else 'No summary available'}

# CROSS-DOMAIN PARALLELS IDENTIFIED ({len(parallels)} parallels):
{parallels_formatted}

# CROSS-DOMAIN RESEARCH PAPERS ({len(cross_domain_papers)} papers):
{cross_domain_formatted}

# CODE IMPLEMENTATIONS AVAILABLE:
{list(code_blocks.keys()) if code_blocks else ['No code examples available']}

---

YOUR TASK:

You now have comprehensive research from BOTH the original ML/AI domain AND related fields (Neuroscience, Quantum Physics, Biology, etc.).

Generate a breakthrough innovation report that:

1. **Synthesizes Insights Across Domains**
   - Connect the ML/AI research with cross-domain findings
   - Identify where parallels reveal novel mechanisms
   - Explain how cross-domain research validates or challenges ML assumptions

2. **Propose State-of-the-Art Research Directions**
   - Generate 5-7 novel research questions inspired by cross-domain parallels
   - For each direction, explain:
     * Which parallel(s) inspired it
     * Why it could advance the field
     * What experiments would test it
     * Expected breakthroughs

3. **Design Concrete Innovations**
   - Propose new algorithms/architectures inspired by cross-domain mechanisms
   - Suggest novel applications combining insights from multiple fields
   - Include pseudocode or implementation sketches where relevant

4. **Map the Conceptual Bridges**
   - Create detailed mappings between ML concepts and cross-domain phenomena
   - Explain what each field can learn from the other
   - Identify transferable principles

5. **Provide Actionable Next Steps**
   - Immediate experiments (3-6 months)
   - Long-term investigations (1-2 years)
   - Interdisciplinary collaborations needed

CRITICAL: This should be a COMPREHENSIVE, INNOVATIVE report that goes beyond summarizing existing work.
Use the cross-domain insights to propose genuinely novel ideas that could advance the state-of-the-art.

Follow the Research Innovation format structure strictly.
"""

    def _format_papers(self, papers: List[Dict], category: str) -> str:
        """Format papers with category."""
        if not papers:
            return f"No {category} papers available."

        formatted = [f"## {category} Papers:\n"]
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'Untitled')
            authors = ', '.join(paper.get('authors', [])[:3])
            year = paper.get('year', 'n.d.')
            abstract = paper.get('abstract', '')[:300]

            formatted.append(
                f"{i}. **{title}**\n"
                f"   Authors: {authors} ({year})\n"
                f"   Abstract: {abstract}...\n"
            )
        return '\n'.join(formatted)

    def _format_cross_domain_papers(self, papers: List[Dict], parallels: List[Dict]) -> str:
        """Format cross-domain papers grouped by domain."""
        if not papers:
            return "No cross-domain papers available."

        # Group papers by domain if available
        by_domain = {}
        for paper in papers:
            domain = paper.get('domain', 'Other')
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(paper)

        formatted = []
        for domain, domain_papers in by_domain.items():
            formatted.append(f"\n### {domain} Research:")
            for i, paper in enumerate(domain_papers[:5], 1):
                title = paper.get('title', 'Untitled')
                authors = ', '.join(paper.get('authors', [])[:2])
                year = paper.get('year', 'n.d.')
                relevance = paper.get('relevance_note', '')

                formatted.append(
                    f"{i}. **{title}** ({authors}, {year})\n"
                    f"   Relevance: {relevance[:200] if relevance else 'Cross-domain insight'}..."
                )

        return '\n'.join(formatted)

    def _format_parallels(self, parallels: List[Dict], core_principles: List[Dict]) -> str:
        """Format cross-domain parallels analysis."""
        if not parallels:
            return "No parallels identified."

        formatted = []
        for i, parallel in enumerate(parallels, 1):
            domain = parallel.get('domain', 'Unknown')
            phenomenon = parallel.get('phenomenon', 'Unknown')
            connection = parallel.get('connection', '')
            strength = parallel.get('strength', 'unknown')
            insights = parallel.get('potential_insights', '')

            formatted.append(
                f"\n**Parallel {i}: {domain} - {phenomenon}** (Strength: {strength})\n"
                f"Connection: {connection}\n"
                f"Potential Insights: {insights}"
            )

        return '\n'.join(formatted)

    def _format_core_principles(self, principles: List[Dict]) -> str:
        """Format core principles."""
        if not principles:
            return "No core principles identified."

        formatted = []
        for i, principle in enumerate(principles, 1):
            name = principle.get('principle', 'Unknown')
            desc = principle.get('description', '')
            why = principle.get('why_important', '')

            formatted.append(
                f"{i}. **{name}**\n"
                f"   How it manifests: {desc}\n"
                f"   Why important: {why}"
            )

        return '\n'.join(formatted)

    def _format_findings(self, findings: List[Dict]) -> str:
        """Format key findings."""
        if not findings:
            return "No key findings available."

        formatted = []
        for i, finding in enumerate(findings, 1):
            title = finding.get('title', 'Untitled')
            summary = finding.get('summary', finding.get('abstract', ''))[:200]
            formatted.append(f"{i}. **{title}**\n   {summary}...")

        return '\n'.join(formatted)
