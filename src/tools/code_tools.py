"""
Code execution and validation tools.
Provides safe code execution, syntax checking, and formatting.
"""

import ast
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import black
import isort

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CodeTools:
    """
    Tools for code validation, execution, and formatting.

    Provides:
    - Syntax validation using AST parsing
    - Safe code execution in isolated environment
    - Code formatting with black and isort
    - Dependency extraction from imports
    """

    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax using AST parsing.

        Args:
            code: Python code string

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            logger.debug("Code syntax is valid")
            return True, None
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            logger.warning(f"Code syntax error: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error during syntax validation: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    def execute_code(
        code: str,
        timeout: int = None
    ) -> Dict:
        """
        Execute Python code in an isolated temporary environment.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (default from settings)

        Returns:
            Dict with execution results:
            - success: bool
            - stdout: str
            - stderr: str
            - returncode: int
        """
        timeout = timeout or settings.code_execution_timeout

        logger.info("Executing code in isolated environment")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            code_file = tmpdir_path / "temp_code.py"

            # Write code to temporary file
            try:
                code_file.write_text(code, encoding='utf-8')
            except Exception as e:
                logger.error(f"Failed to write code file: {e}")
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'Failed to write code file: {e}',
                    'returncode': -1
                }

            # Execute code
            try:
                result = subprocess.run(
                    [sys.executable, str(code_file)],
                    capture_output=True,
                    timeout=timeout,
                    text=True,
                    cwd=tmpdir
                )

                success = result.returncode == 0
                logger.info(
                    f"Code execution {'succeeded' if success else 'failed'} "
                    f"(returncode: {result.returncode})"
                )

                return {
                    'success': success,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }

            except subprocess.TimeoutExpired:
                logger.warning(f"Code execution timed out after {timeout}s")
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'Execution timed out after {timeout} seconds',
                    'returncode': -1
                }
            except Exception as e:
                logger.error(f"Code execution error: {e}")
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'Execution error: {str(e)}',
                    'returncode': -1
                }

    @staticmethod
    def format_code(code: str) -> str:
        """
        Format code using black and isort.

        Args:
            code: Python code to format

        Returns:
            Formatted code string
        """
        try:
            # Apply black formatting
            formatted = black.format_str(code, mode=black.Mode())

            # Apply isort for import sorting
            formatted = isort.code(formatted)

            logger.debug("Code formatted successfully")
            return formatted

        except Exception as e:
            logger.warning(f"Code formatting failed: {e}. Returning original code.")
            return code

    @staticmethod
    def extract_dependencies(code: str) -> List[str]:
        """
        Extract import statements and identify external dependencies.

        Args:
            code: Python code string

        Returns:
            List of package names (excluding stdlib)
        """
        dependencies = set()

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get base package name
                        pkg = alias.name.split('.')[0]
                        dependencies.add(pkg)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Get base package name
                        pkg = node.module.split('.')[0]
                        dependencies.add(pkg)

        except SyntaxError:
            logger.warning("Could not parse code for dependency extraction")
            return []

        # Filter out standard library modules
        stdlib_modules = {
            'abc', 'argparse', 'array', 'ast', 'asyncio', 'base64', 'bisect',
            'collections', 'copy', 'csv', 'datetime', 'decimal', 'enum',
            'functools', 'glob', 'hashlib', 'heapq', 'io', 'itertools',
            'json', 'logging', 'math', 'operator', 'os', 'pathlib', 'pickle',
            'random', 're', 'shutil', 'socket', 'sqlite3', 'string', 'struct',
            'subprocess', 'sys', 'tempfile', 'threading', 'time', 'typing',
            'unittest', 'urllib', 'uuid', 'warnings', 'weakref', 'xml'
        }

        external_deps = [dep for dep in dependencies if dep not in stdlib_modules]

        logger.debug(f"Extracted {len(external_deps)} external dependencies")
        return sorted(external_deps)

    @staticmethod
    def add_docstrings(code: str, function_name: str, description: str) -> str:
        """
        Add a docstring to a function if it doesn't have one.

        Args:
            code: Python code string
            function_name: Name of function to document
            description: Docstring description

        Returns:
            Code with added docstring
        """
        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Check if docstring exists
                    if (not node.body or
                        not isinstance(node.body[0], ast.Expr) or
                        not isinstance(node.body[0].value, ast.Constant)):

                        # Add docstring
                        docstring = f'"""{description}"""'
                        lines = code.split('\n')

                        # Find function definition line
                        for i, line in enumerate(lines):
                            if f'def {function_name}' in line:
                                # Insert docstring after function definition
                                indent = len(line) - len(line.lstrip()) + 4
                                docstring_line = ' ' * indent + docstring
                                lines.insert(i + 1, docstring_line)
                                break

                        return '\n'.join(lines)

        except Exception as e:
            logger.warning(f"Could not add docstring: {e}")

        return code

    @staticmethod
    def check_code_quality(code: str) -> Dict[str, any]:
        """
        Perform basic code quality checks.

        Args:
            code: Python code string

        Returns:
            Dict with quality metrics
        """
        metrics = {
            'valid_syntax': False,
            'line_count': 0,
            'function_count': 0,
            'class_count': 0,
            'has_docstrings': False,
            'has_type_hints': False,
            'complexity_score': 0
        }

        # Check syntax
        is_valid, _ = CodeTools.validate_syntax(code)
        metrics['valid_syntax'] = is_valid

        if not is_valid:
            return metrics

        # Count lines
        metrics['line_count'] = len([l for l in code.split('\n') if l.strip()])

        try:
            tree = ast.parse(code)

            # Count functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics['function_count'] += 1

                    # Check for docstring
                    if (node.body and
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant)):
                        metrics['has_docstrings'] = True

                    # Check for type hints
                    if node.returns or any(arg.annotation for arg in node.args.args):
                        metrics['has_type_hints'] = True

                elif isinstance(node, ast.ClassDef):
                    metrics['class_count'] += 1

            # Simple complexity score (lower is better)
            # Based on control flow statements
            complexity = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    complexity += 1

            metrics['complexity_score'] = complexity

        except Exception as e:
            logger.warning(f"Quality check error: {e}")

        return metrics

    @staticmethod
    def safe_eval(expression: str, timeout: int = 5) -> Tuple[bool, any]:
        """
        Safely evaluate a Python expression.

        Args:
            expression: Python expression to evaluate
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, result)
        """
        code = f"result = {expression}\nprint(result)"

        result = CodeTools.execute_code(code, timeout=timeout)

        if result['success']:
            try:
                value = result['stdout'].strip()
                return True, value
            except Exception as e:
                return False, str(e)
        else:
            return False, result['stderr']
