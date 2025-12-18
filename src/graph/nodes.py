"""
LangGraph node functions for each agent.
Each node takes the state, executes an agent, and returns state updates.
"""

from typing import Dict, Any
from src.agents.planner_agent import PlannerAgent
from src.agents.researcher_agent import ResearcherAgent
from src.agents.coder_agent import CoderAgent
from src.agents.tester_agent import TesterAgent
from src.agents.critic_agent import CriticAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.memory.memory_manager import MemoryManager
from src.graph.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize agents (singleton pattern)
planner = PlannerAgent()
researcher = ResearcherAgent()
coder = CoderAgent()
tester = TesterAgent()
critic = CriticAgent()
synthesizer = SynthesizerAgent()
memory = MemoryManager()


def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Planner node: Decompose topic into hierarchical plan.

    Args:
        state: Current workflow state

    Returns:
        State updates with plan, subtasks, dependencies
    """
    logger.info("=== PLANNER NODE ===")

    result = planner.run(
        topic=state['topic'],
        requirements=state['requirements']
    )

    return {
        'plan': result.get('plan', {}),
        'subtasks': result.get('subtasks', []),
        'dependencies': result.get('dependencies', {}),
        'search_queries': result.get('search_queries', []),
        'code_specifications': result.get('code_specifications', []),
        'messages': [{'role': 'planner', 'content': f"Created plan with {len(result.get('subtasks', []))} subtasks"}],
        'status': 'planning_complete'
    }


def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Researcher node: Execute parallel literature searches.

    Args:
        state: Current workflow state

    Returns:
        State updates with research papers and findings
    """
    logger.info("=== RESEARCHER NODE ===")

    # Get search queries from plan
    queries = state.get('search_queries', [])

    if not queries:
        # Generate default queries from topic
        queries = [state['topic']]

    result = researcher.run(queries=queries)

    # Store findings in memory
    findings = result.get('key_findings', [])
    if findings:
        memory.add_research_findings(findings)

    return {
        'research_papers': result.get('research_papers', []),
        'key_findings': findings,
        'literature_summary': result.get('literature_summary', ''),
        'memory_context': [{'type': 'research', 'count': len(findings)}],
        'messages': [{'role': 'researcher', 'content': f"Found {len(result.get('research_papers', []))} papers with {len(findings)} key findings"}],
        'status': 'research_complete'
    }


def coder_node(state: AgentState) -> Dict[str, Any]:
    """
    Coder node: Generate code implementations.

    Args:
        state: Current workflow state

    Returns:
        State updates with generated code and dependencies
    """
    logger.info("=== CODER NODE ===")

    # Get code specifications from plan
    specifications = state.get('code_specifications', [])

    if not specifications:
        # Create default specification
        specifications = [{
            'id': 'example_1',
            'description': f"Example implementation for {state['topic']}",
            'requirements': 'Create a simple, educational example'
        }]

    # Get research context for code generation
    context = {
        'key_findings': state.get('key_findings', [])[:5],
        'literature_summary': state.get('literature_summary', '')
    }

    result = coder.run(specifications=specifications, context=context)

    # Store code patterns in memory
    code_blocks = result.get('generated_code', {})
    if code_blocks:
        memory.add_code_patterns([
            {'id': k, 'code': v, 'description': spec.get('description', '')}
            for k, v in code_blocks.items()
            for spec in specifications if spec.get('id') == k
        ])

    return {
        'generated_code': code_blocks,
        'code_dependencies': result.get('code_dependencies', []),
        'memory_context': [{'type': 'code', 'count': len(code_blocks)}],
        'messages': [{'role': 'coder', 'content': f"Generated {len(code_blocks)} code blocks"}],
        'status': 'coding_complete'
    }


def tester_node(state: AgentState) -> Dict[str, Any]:
    """
    Tester node: Validate and execute generated code.

    Args:
        state: Current workflow state

    Returns:
        State updates with test results and executable code
    """
    logger.info("=== TESTER NODE ===")

    code_blocks = state.get('generated_code', {})

    if not code_blocks:
        logger.warning("No code to test")
        return {
            'test_results': [],
            'validation_errors': [],
            'executable_code': {},
            'test_coverage': 0.0,
            'status': 'testing_skipped'
        }

    result = tester.run(code_blocks=code_blocks)

    return {
        'test_results': result.get('test_results', []),
        'validation_errors': result.get('validation_errors', []),
        'executable_code': result.get('executable_code', {}),
        'test_coverage': result.get('test_coverage', 0.0),
        'messages': [{'role': 'tester', 'content': f"Tested {len(code_blocks)} blocks, {len(result.get('executable_code', {}))} passed ({result.get('test_coverage', 0):.1f}% coverage)"}],
        'status': 'testing_complete'
    }


def critic_node(state: AgentState) -> Dict[str, Any]:
    """
    Critic node: Evaluate quality and provide feedback.

    Args:
        state: Current workflow state

    Returns:
        State updates with quality scores and feedback
    """
    logger.info("=== CRITIC NODE ===")

    result = critic.run(state=state)

    overall_score = result.get('overall_score', 0.0)
    needs_revision = result.get('needs_revision', False)

    logger.info(f"Quality score: {overall_score:.1f}/10.0, Needs revision: {needs_revision}")

    return {
        'quality_scores': result.get('quality_scores', {}),
        'feedback': result.get('feedback', {}),
        'needs_revision': needs_revision,
        'messages': [{'role': 'critic', 'content': f"Quality score: {overall_score:.1f}/10.0"}],
        'status': 'critique_complete'
    }


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizer node: Create final markdown report.

    Args:
        state: Current workflow state

    Returns:
        State updates with final report and metadata
    """
    logger.info("=== SYNTHESIZER NODE ===")

    result = synthesizer.run(state=state)

    final_report = result.get('final_report', '')
    metadata = result.get('report_metadata', {})

    # Save report to file
    from src.tools.file_tools import FileTools
    file_tools = FileTools()
    output_path = file_tools.save_markdown_report(
        content=final_report,
        topic=state['topic']
    )

    metadata['output_path'] = str(output_path)

    logger.info(f"Report generated: {output_path}")

    return {
        'final_report': final_report,
        'report_metadata': metadata,
        'messages': [{'role': 'synthesizer', 'content': f"Generated report ({metadata.get('word_count', 0)} words)"}],
        'status': 'synthesis_complete'
    }
