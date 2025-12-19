"""
Researcher agent for parallel literature search.
"""

import asyncio
from typing import Dict, List
from src.agents.base_agent import BaseAgent
from src.config.prompts import RESEARCHER_PROMPT
from src.tools.research_tools import ResearchTools
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResearcherAgent(BaseAgent):
    """
    Researcher agent that conducts parallel literature searches.
    """

    def __init__(self):
        super().__init__(
            name="Researcher",
            system_prompt=RESEARCHER_PROMPT,
            temperature=0.5,
            model_tier="fast"  # Simple summarization - use cheap model
        )
        self.tools = ResearchTools()

    def run(self, queries: List[str], **kwargs) -> Dict:
        """
        Execute parallel searches and extract key findings.

        Args:
            queries: List of search queries

        Returns:
            Dict with papers and key findings
        """
        logger.info(f"Researching with {len(queries)} queries")

        # Execute parallel searches
        search_results = asyncio.run(
            self.tools.parallel_search(queries, max_results_per_query=5)
        )

        # Combine all papers
        all_papers = []
        for query_papers in search_results['arxiv'].values():
            all_papers.extend(query_papers)
        for query_papers in search_results['semantic_scholar'].values():
            all_papers.extend(query_papers)

        # Deduplicate
        unique_papers = self.tools.deduplicate_papers(all_papers)

        # Extract key findings
        key_findings = self.tools.extract_key_findings(unique_papers, top_k=10)

        # Generate summary
        summary = self._generate_summary(key_findings)

        logger.info(f"Found {len(unique_papers)} unique papers, extracted {len(key_findings)} findings")

        return {
            'research_papers': unique_papers,
            'key_findings': key_findings,
            'literature_summary': summary
        }

    def _generate_summary(self, findings: List[Dict]) -> str:
        """Generate a summary of research findings."""
        if not findings:
            return "No research findings available."

        prompt = f"""
Summarize the following research findings into a coherent literature review (2-3 paragraphs):

{self._format_findings(findings)}

Focus on:
- Main themes and concepts
- Key methodologies
- Important results
- Research gaps
"""

        return self.generate_response(prompt)

    def _format_findings(self, findings: List[Dict]) -> str:
        """Format findings for prompt."""
        lines = []
        for i, f in enumerate(findings[:10], 1):
            lines.append(f"{i}. {f['title']} ({f.get('year', 'n.d.')})")
            lines.append(f"   {f['abstract'][:200]}...")
        return '\n'.join(lines)
