"""
Coder agent for Python code generation.
"""

from typing import Dict, List
from src.agents.base_agent import BaseAgent
from src.config.prompts import CODER_PROMPT
from src.tools.code_tools import CodeTools
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CoderAgent(BaseAgent):
    """
    Coder agent that generates Python implementations from research.
    """

    def __init__(self):
        super().__init__(
            name="Coder",
            system_prompt=CODER_PROMPT,
            temperature=0.3  # Lower temperature for more deterministic code
        )
        self.tools = CodeTools()

    def run(self, specifications: List[Dict], context: Dict = None) -> Dict:
        """
        Generate code from specifications.

        Args:
            specifications: List of code specs with id, description, requirements
            context: Optional research context

        Returns:
            Dict with generated code blocks and dependencies
        """
        logger.info(f"Generating code for {len(specifications)} specifications")

        generated_code = {}
        all_dependencies = set()

        for spec in specifications:
            code_id = spec['id']
            description = spec['description']
            requirements = spec.get('requirements', '')

            # Build prompt with context
            prompt = self._build_code_prompt(description, requirements, context)

            # Generate code
            code = self.generate_response(prompt)

            # Extract code from markdown if present
            code = self._extract_code_from_markdown(code)

            # Format code
            code = self.tools.format_code(code)

            # Extract dependencies
            deps = self.tools.extract_dependencies(code)
            all_dependencies.update(deps)

            generated_code[code_id] = code

            logger.info(f"Generated code for {code_id}")

        return {
            'generated_code': generated_code,
            'code_dependencies': list(all_dependencies)
        }

    def _build_code_prompt(self, description: str, requirements: str, context: Dict) -> str:
        """Build prompt for code generation."""
        prompt_parts = [
            f"Generate Python code for the following:",
            f"\nDescription: {description}",
            f"\nRequirements: {requirements}",
        ]

        if context and 'key_findings' in context:
            findings = context['key_findings'][:3]  # Use top 3 findings
            prompt_parts.append("\nResearch Context:")
            for f in findings:
                prompt_parts.append(f"- {f['title']}: {f['abstract'][:150]}...")

        prompt_parts.append("\nGenerate clean, well-documented Python code with:")
        prompt_parts.append("- Type hints")
        prompt_parts.append("- Docstrings (Google style)")
        prompt_parts.append("- Example usage")
        prompt_parts.append("- Error handling")

        return '\n'.join(prompt_parts)

    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        import re
        match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1)
        return text
