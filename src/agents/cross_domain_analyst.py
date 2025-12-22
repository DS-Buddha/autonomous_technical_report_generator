"""
Cross-Domain Analyst agent for identifying parallels across scientific fields.
"""

from typing import Dict, List
from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

CROSS_DOMAIN_ANALYST_PROMPT = """You are a Cross-Domain Research Analyst with expertise spanning Machine Learning, Neuroscience, Quantum Physics, Biology, Chemistry, Network Theory, and Complex Systems.

Your role is to identify meaningful parallels between a given ML/AI concept and phenomena from other scientific domains.

APPROACH:
1. Deeply understand the core mechanisms of the ML/AI concept
2. Identify fundamental principles (e.g., optimization, information flow, adaptation, emergence)
3. Find analogous phenomena in other fields that share these principles
4. Ensure parallels are substantive, not superficial

OUTPUT FORMAT:
Return a JSON object with the following structure:
{
    "core_principles": [
        {
            "principle": "Name of fundamental principle",
            "description": "How it manifests in the ML/AI concept",
            "why_important": "Why this principle matters"
        }
    ],
    "cross_domain_parallels": [
        {
            "domain": "Neuroscience|Quantum Physics|Biology|Network Theory|Complex Systems|Other",
            "phenomenon": "Specific phenomenon or concept from that domain",
            "connection": "Detailed explanation of how it relates to the ML/AI concept",
            "strength": "strong|moderate|speculative",
            "research_keywords": ["keyword1", "keyword2", "keyword3"],
            "potential_insights": "What studying this parallel might reveal"
        }
    ],
    "search_queries": [
        "Specific literature search query 1",
        "Specific literature search query 2",
        ...
    ]
}

QUALITY CRITERIA:
- Identify 5-8 high-quality parallels across diverse domains
- Each parallel should suggest concrete research directions
- Include both well-established and speculative connections
- Provide specific keywords for literature search
- Focus on mechanisms, not superficial similarities

Be rigorous but creative. The goal is to find connections that could genuinely advance research."""


class CrossDomainAnalyst(BaseAgent):
    """
    Cross-domain analyst that identifies parallels across scientific fields.
    """

    def __init__(self):
        super().__init__(
            name="CrossDomainAnalyst",
            system_prompt=CROSS_DOMAIN_ANALYST_PROMPT,
            temperature=0.7  # Higher temperature for creative connections
        )

    def run(self, state: Dict, **kwargs) -> Dict:
        """
        Analyze the topic and identify cross-domain parallels.

        Args:
            state: Workflow state

        Returns:
            Dict with cross-domain analysis
        """
        topic = state.get('topic', '')
        research_papers = state.get('research_papers', [])
        key_findings = state.get('key_findings', [])

        logger.info(f"Analyzing cross-domain parallels for: {topic}")

        # Build context from existing research
        research_context = self._build_research_context(research_papers, key_findings)

        prompt = f"""
Analyze the following ML/AI topic and identify meaningful cross-domain parallels:

TOPIC: {topic}

EXISTING RESEARCH CONTEXT:
{research_context}

Your task:
1. Identify the core principles underlying this topic
2. Find 5-8 substantive parallels with phenomena from other scientific domains
3. For each parallel, provide specific keywords for literature search
4. Suggest how studying these parallels could advance research on {topic}

Focus on domains: Neuroscience, Quantum Physics, Biology, Chemistry, Network Theory, Complex Systems, and any other relevant fields.

Return your analysis in the specified JSON format.
"""

        # Generate analysis
        response = self.generate_response(prompt)

        # Parse JSON response
        import json
        import re

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

            # Third attempt: find any JSON-like structure
            if not analysis:
                json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', response, re.DOTALL)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group())
                        logger.info("Successfully extracted JSON pattern")
                    except json.JSONDecodeError:
                        pass

            # Fourth attempt: clean and retry
            if not analysis:
                # Remove common issues: trailing commas, single quotes
                cleaned = response.replace("'", '"')
                cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)  # Remove trailing commas
                try:
                    analysis = json.loads(cleaned)
                    logger.info("Successfully parsed cleaned JSON")
                except json.JSONDecodeError:
                    pass

        # Final fallback: create minimal structure
        if not analysis:
            logger.error("All JSON parsing attempts failed, using fallback structure")
            analysis = {
                "core_principles": [
                    {
                        "principle": "Information Processing",
                        "description": "Core mechanisms of the topic",
                        "why_important": "Fundamental to understanding the concept"
                    }
                ],
                "cross_domain_parallels": [
                    {
                        "domain": "Neuroscience",
                        "phenomenon": "Neural processing",
                        "connection": "Similar information processing patterns",
                        "strength": "moderate",
                        "research_keywords": [topic, "neuroscience", "brain"],
                        "potential_insights": "Understanding biological parallels"
                    },
                    {
                        "domain": "Complex Systems",
                        "phenomenon": "Emergent behavior",
                        "connection": "System-level properties",
                        "strength": "moderate",
                        "research_keywords": [topic, "complex systems", "emergence"],
                        "potential_insights": "System dynamics understanding"
                    }
                ],
                "search_queries": [
                    f"{topic} neuroscience parallels",
                    f"{topic} biological inspiration",
                    f"{topic} complex systems theory",
                    f"{topic} quantum mechanics applications",
                    f"{topic} network theory"
                ]
            }

        logger.info(f"Identified {len(analysis.get('cross_domain_parallels', []))} cross-domain parallels")
        logger.info(f"Generated {len(analysis.get('search_queries', []))} search queries")

        return {
            'cross_domain_analysis': analysis,
            'cross_domain_search_queries': analysis.get('search_queries', []),
            'messages': [{
                'role': 'assistant',
                'content': f"CrossDomainAnalyst: Identified {len(analysis.get('cross_domain_parallels', []))} parallels"
            }]
        }

    def _build_research_context(self, papers: List[Dict], findings: List[Dict]) -> str:
        """Build context from existing research."""
        context_parts = []

        if papers:
            context_parts.append("Key Papers:")
            for i, paper in enumerate(papers[:5], 1):
                title = paper.get('title', 'Untitled')
                abstract = paper.get('abstract', '')[:200]
                context_parts.append(f"{i}. {title}\n   {abstract}...")

        if findings:
            context_parts.append("\nKey Findings:")
            for i, finding in enumerate(findings[:5], 1):
                summary = finding.get('summary', finding.get('abstract', ''))[:150]
                context_parts.append(f"{i}. {summary}...")

        return '\n'.join(context_parts) if context_parts else "No prior research context available."
