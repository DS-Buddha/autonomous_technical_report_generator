"""
Planner agent for hierarchical task decomposition.
"""

from typing import Dict, List
from src.agents.base_agent import BaseAgent
from src.config.prompts import PLANNER_PROMPT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PlannerAgent(BaseAgent):
    """
    Planner agent that decomposes research topics into hierarchical subtasks.
    """

    def __init__(self):
        super().__init__(
            name="Planner",
            system_prompt=PLANNER_PROMPT,
            temperature=0.7
        )

    def run(self, topic: str, requirements: Dict) -> Dict:
        """
        Decompose a research topic into actionable subtasks.

        Args:
            topic: Research topic
            requirements: User requirements dict

        Returns:
            Dict with plan, subtasks, and dependencies
        """
        logger.info(f"Planning for topic: {topic}")

        prompt = f"""
Decompose the following research topic into a comprehensive implementation plan:

Topic: {topic}

Requirements:
{self.format_context(requirements)}

Create a detailed plan with:
1. High-level research themes (2-5 major areas)
2. Specific search queries for literature review (5-10 queries)
3. Code implementation requirements (what needs to be built)
4. Validation criteria for success

Output as JSON with this structure:
{{
    "plan": {{
        "themes": ["theme1", "theme2", ...],
        "objectives": ["obj1", "obj2", ...]
    }},
    "subtasks": [
        {{
            "id": "task_1",
            "description": "what to do",
            "type": "research|code|test|synthesis",
            "priority": 1-10,
            "dependencies": []
        }}
    ],
    "dependencies": {{
        "task_id": ["dependency_ids"]
    }},
    "search_queries": ["query1", "query2", ...],
    "code_specifications": [
        {{
            "id": "code_1",
            "description": "what to implement",
            "requirements": "specific requirements"
        }}
    ]
}}
"""

        result = self.generate_json_response(prompt)

        logger.info(f"Generated plan with {len(result.get('subtasks', []))} subtasks")
        return result
