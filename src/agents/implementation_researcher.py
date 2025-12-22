"""
Implementation Research Agent for deep-dive into experiment implementation.
"""

from typing import Dict, List
from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger
import re

logger = get_logger(__name__)

IMPLEMENTATION_RESEARCHER_PROMPT = """You are a Research Implementation Specialist with expertise in translating novel research ideas into concrete, publishable implementations.

Your role is to analyze proposed experiments and research them deeply to identify:
1. Existing implementations and their limitations
2. State-of-the-art methods for similar approaches
3. Gaps and opportunities for novel contributions
4. Efficient/better ways to implement the same concept
5. Novel applications to new domains

APPROACH:
For each experiment:
- Understand the core idea and objectives
- Identify what has been done before (existing work)
- Find current best practices and state-of-the-art
- Identify what's missing or could be improved
- Propose specific research queries to find relevant papers

OUTPUT FORMAT:
Return a JSON object:
{
    "experiments_analysis": [
        {
            "experiment_id": "exp_1",
            "experiment_title": "Title from concrete next steps",
            "core_objective": "What this experiment aims to achieve",
            "existing_work_keywords": ["keyword1", "keyword2", ...],
            "sota_keywords": ["state-of-art keyword1", ...],
            "gap_analysis": "What's missing in current approaches",
            "novel_angle": "How this could contribute something new",
            "search_queries": [
                "Detailed search query 1 for existing implementations",
                "Detailed search query 2 for SOTA methods",
                "Detailed search query 3 for similar problems in other domains",
                ...
            ],
            "expected_paper_types": ["implementation", "benchmark", "survey", "case_study"]
        }
    ]
}

QUALITY CRITERIA:
- Generate 5-8 specific search queries per experiment
- Focus on implementation details, not just theory
- Look for benchmarks, datasets, code repositories
- Identify technical challenges and solutions
- Consider scalability, efficiency, practicality

Be specific and actionable. The goal is to find everything needed to implement these experiments in a novel, publishable way."""


class ImplementationResearcher(BaseAgent):
    """
    Implementation researcher that analyzes proposed experiments and
    researches their implementation deeply.
    """

    def __init__(self):
        super().__init__(
            name="ImplementationResearcher",
            system_prompt=IMPLEMENTATION_RESEARCHER_PROMPT,
            temperature=0.6
        )

    def run(self, state: Dict, **kwargs) -> Dict:
        """
        Analyze proposed experiments and generate deep research queries.

        Args:
            state: Workflow state with innovation report

        Returns:
            Dict with implementation analysis and search queries
        """
        topic = state.get('topic', '')
        innovation_report = state.get('final_report', '')

        logger.info(f"Analyzing implementation for: {topic}")

        # Extract experiments from "Concrete Next Steps" section
        experiments = self._extract_experiments(innovation_report)

        if not experiments:
            logger.warning("No experiments found in innovation report")
            return {
                'implementation_analysis': {},
                'implementation_search_queries': [],
                'messages': [{'role': 'assistant', 'content': "ImplementationResearcher: No experiments found"}]
            }

        logger.info(f"Found {len(experiments)} experiments to analyze")

        # Build analysis prompt
        prompt = f"""
Analyze the following proposed experiments from a research innovation report on "{topic}".

For each experiment, provide deep analysis on how to implement it in a novel, publishable way.

PROPOSED EXPERIMENTS:
{self._format_experiments(experiments)}

INNOVATION REPORT CONTEXT:
{innovation_report[:3000]}

Your task:
1. For each experiment, identify:
   - Core objective and what it aims to achieve
   - Keywords for finding existing implementations
   - Keywords for finding state-of-the-art methods
   - What's missing or could be improved (gap analysis)
   - How this could contribute something novel
   - Specific search queries (5-8 per experiment)

2. Focus on finding:
   - Papers with implementation details, code, benchmarks
   - State-of-the-art methods in this specific area
   - Similar problems solved in other domains
   - Technical challenges and their solutions
   - Datasets, tools, frameworks available

Return your analysis in the specified JSON format.
"""

        # Generate analysis
        response = self.generate_response(prompt)

        # Parse JSON response with robust error handling
        import json
        analysis = None

        try:
            # First attempt: direct JSON parsing
            analysis = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON directly: {e}")

            # Second attempt: extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                try:
                    analysis = json.loads(json_match.group(1))
                    logger.info("Successfully extracted JSON from code block")
                except json.JSONDecodeError:
                    pass

            # Third attempt: clean and retry
            if not analysis:
                cleaned = response.replace("'", '"')
                cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
                try:
                    analysis = json.loads(cleaned)
                    logger.info("Successfully parsed cleaned JSON")
                except json.JSONDecodeError:
                    pass

        # Final fallback
        if not analysis:
            logger.error("All JSON parsing attempts failed, using fallback")
            analysis = {'experiments_analysis': []}

        # Collect all search queries
        all_queries = []
        for exp_analysis in analysis.get('experiments_analysis', []):
            all_queries.extend(exp_analysis.get('search_queries', []))

        logger.info(f"Generated analysis for {len(analysis.get('experiments_analysis', []))} experiments")
        logger.info(f"Created {len(all_queries)} implementation search queries")

        return {
            'implementation_analysis': analysis,
            'implementation_search_queries': all_queries,
            'messages': [{
                'role': 'assistant',
                'content': f"ImplementationResearcher: Analyzed {len(analysis.get('experiments_analysis', []))} experiments"
            }]
        }

    def _extract_experiments(self, report: str) -> List[Dict]:
        """Extract experiments from Concrete Next Steps section."""
        experiments = []

        # Look for "Concrete Next Steps" or "Next Steps" section
        patterns = [
            r'##\s*🧪\s*Concrete Next Steps(.*?)(?=##|$)',
            r'##\s*Concrete Next Steps(.*?)(?=##|$)',
            r'###\s*Immediate Experiments.*?\n(.*?)(?=###|##|$)',
        ]

        experiment_section = None
        for pattern in patterns:
            match = re.search(pattern, report, re.DOTALL | re.IGNORECASE)
            if match:
                experiment_section = match.group(1)
                break

        if not experiment_section:
            # Fallback: look for any experiment mentions
            experiment_section = report

        # Extract individual experiments (numbered or bullet points)
        exp_patterns = [
            r'\d+\.\s*\*\*([^\*]+)\*\*(.*?)(?=\d+\.\s*\*\*|\n\n|###|##|$)',
            r'-\s*\*\*([^\*]+)\*\*(.*?)(?=-\s*\*\*|\n\n|###|##|$)',
            r'\*\*Experiment\s+\d+:\s*([^\*]+)\*\*(.*?)(?=\*\*Experiment|\n\n|###|##|$)',
        ]

        for pattern in exp_patterns:
            matches = re.finditer(pattern, experiment_section, re.DOTALL | re.IGNORECASE)
            for i, match in enumerate(matches, 1):
                title = match.group(1).strip()
                description = match.group(2).strip()[:500]  # Limit description

                if len(title) > 5:  # Filter out noise
                    experiments.append({
                        'id': f'exp_{i}',
                        'title': title,
                        'description': description
                    })

        # Limit to first 5 experiments to keep focused
        return experiments[:5]

    def _format_experiments(self, experiments: List[Dict]) -> str:
        """Format experiments for prompt."""
        formatted = []
        for exp in experiments:
            formatted.append(
                f"Experiment: {exp['title']}\n"
                f"Description: {exp['description']}\n"
            )
        return '\n'.join(formatted)
