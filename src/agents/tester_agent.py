"""
Tester agent for code validation and execution.
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
from src.config.prompts import TESTER_PROMPT
from src.tools.code_tools import CodeTools
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TesterAgent(BaseAgent):
    """
    Tester agent that validates and executes code.
    """

    def __init__(self):
        super().__init__(
            name="Tester",
            system_prompt=TESTER_PROMPT,
            temperature=0.2,
            model_tier="fast"  # Deterministic validation - use cheap model
        )
        self.tools = CodeTools()

    def run(self, code_blocks: Dict[str, str], **kwargs) -> Dict:
        """
        Validate and test code blocks.

        Args:
            code_blocks: Dict of code_id -> code

        Returns:
            Dict with test results, errors, and executable code
        """
        logger.info(f"Testing {len(code_blocks)} code blocks")

        test_results = []
        validation_errors = []
        executable_code = {}

        for code_id, code in code_blocks.items():
            # Validate syntax
            is_valid, error = self.tools.validate_syntax(code)

            if not is_valid:
                validation_errors.append({
                    'code_id': code_id,
                    'error_type': 'syntax',
                    'error': error
                })
                logger.warning(f"Syntax error in {code_id}: {error}")
                continue

            # Execute code
            result = self.tools.execute_code(code)

            test_results.append({
                'code_id': code_id,
                'success': result['success'],
                'stdout': result['stdout'],
                'stderr': result['stderr'],
                'returncode': result['returncode']
            })

            if result['success']:
                executable_code[code_id] = code
                logger.info(f"✓ {code_id} executed successfully")
            else:
                validation_errors.append({
                    'code_id': code_id,
                    'error_type': 'runtime',
                    'error': result['stderr']
                })
                logger.warning(f"✗ {code_id} execution failed: {result['stderr'][:100]}")

        # Calculate coverage
        coverage = (len(executable_code) / len(code_blocks) * 100) if code_blocks else 0

        return {
            'test_results': test_results,
            'validation_errors': validation_errors,
            'executable_code': executable_code,
            'test_coverage': coverage
        }
