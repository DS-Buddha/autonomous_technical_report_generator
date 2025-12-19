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

    # CRITICAL FIX: Increment iteration counter when revision is needed
    iteration_count = state.get('iteration_count', 0)
    if needs_revision:
        iteration_count += 1
        logger.warning(f"Revision needed. Iteration count: {iteration_count}/{state.get('max_iterations', 3)}")

    logger.info(f"Quality score: {overall_score:.1f}/10.0, Needs revision: {needs_revision}")

    return {
        'quality_scores': result.get('quality_scores', {}),
        'feedback': result.get('feedback', {}),
        'needs_revision': needs_revision,
        'iteration_count': iteration_count,
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


def research_failure_node(state: AgentState) -> Dict[str, Any]:
    """
    Handle research failure with fallback strategies.

    Implements:
    1. Retry with broader queries (up to 2 attempts)
    2. Trigger human-in-the-loop (HITL) if retries exhausted
    3. Log failure for monitoring/alerting

    Args:
        state: Current workflow state

    Returns:
        State updates with retry instructions or HITL flag
    """
    logger.warning("=== RESEARCH FAILURE HANDLER ===")

    retry_count = state.get('research_retry_count', 0)
    max_retries = 2

    if retry_count < max_retries:
        logger.warning(f"Research failed, retrying ({retry_count + 1}/{max_retries})")

        # Broaden search queries by making them more general
        original_queries = state.get('search_queries', [])
        broader_queries = []

        for query in original_queries:
            # Remove restrictive operators
            broader = query.replace(' AND ', ' OR ')
            broader = broader.replace('"', '')  # Remove exact phrase matching
            broader_queries.append(broader)

            # Also add a very general version
            topic_words = state['topic'].split()
            if len(topic_words) > 2:
                broader_queries.append(' '.join(topic_words[:2]))  # First 2 words only

        logger.info(f"Retrying with {len(broader_queries)} broader queries")

        return {
            'search_queries': broader_queries,
            'research_retry_count': retry_count + 1,
            'status': 'retrying_research',
            'messages': [{'role': 'system', 'content': f"Research retry {retry_count + 1}/{max_retries}"}],
            'next_agent': 'researcher'
        }
    else:
        # After max retries, require human intervention
        logger.critical(
            f"RESEARCH FAILURE: Insufficient papers after {max_retries} retries. "
            f"Topic: {state['topic']}, Papers found: {len(state.get('research_papers', []))}"
        )

        # Generate failure report
        failure_report = f"""# Report Generation Failed

## Topic
{state['topic']}

## Failure Reason
Insufficient research papers found after {max_retries} retry attempts.

## Papers Found
{len(state.get('research_papers', []))} papers (minimum 3 required)

## Action Required
1. Check if topic is too narrow or specialized
2. Try alternative search terms
3. Verify API access (arXiv, Semantic Scholar)
4. Consider manual research supplement

## System Status
- Search queries attempted: {len(state.get('search_queries', []))}
- Retry count: {retry_count}
- Timestamp: {__import__('datetime').datetime.now().isoformat()}
"""

        return {
            'status': 'research_failure_hitl_required',
            'error': f'Insufficient research papers found after {max_retries} attempts',
            'final_report': failure_report,
            'report_metadata': {
                'status': 'failed',
                'failure_reason': 'insufficient_research',
                'retry_count': retry_count,
                'papers_found': len(state.get('research_papers', []))
            },
            'messages': [{'role': 'system', 'content': '⚠️ HITL Required: Research failure'}]
        }


def state_compression_node(state: AgentState) -> Dict[str, Any]:
    """
    Compress state to prevent context window bloat.

    Called before revision loops to keep only essential information.
    This prevents "Lost in the Middle" phenomenon where LLMs can't find
    relevant info buried in massive context.

    Strategy:
    - Keep only top N most relevant papers (not all 15+)
    - Keep only passing code (remove failed attempts)
    - Clear old test results (keep summary only)
    - Prune memory context (keep last N entries)

    Token Reduction: Typically 50K → 15K tokens (70% reduction)

    Args:
        state: Current workflow state

    Returns:
        Compressed state updates
    """
    logger.info("=== STATE COMPRESSION NODE ===")

    compressed = {}

    # 1. COMPRESS RESEARCH: Keep top 5 most cited papers only
    papers = state.get('research_papers', [])
    if papers:
        # Sort by citations and keep top 5
        top_papers = sorted(
            papers,
            key=lambda p: p.get('citations', 0) + p.get('influential_citations', 0) * 2,
            reverse=True
        )[:5]

        # Keep only essential metadata
        compressed['research_papers'] = [
            {
                'title': p['title'],
                'authors': p['authors'][:2],  # First 2 authors only
                'year': p.get('year'),
                'abstract': p.get('abstract', '')[:200],  # First 200 chars
                'citations': p.get('citations', 0),
                'url': p.get('url') or p.get('pdf_url')
            }
            for p in top_papers
        ]

        logger.info(f"Compressed research: {len(papers)} → {len(compressed['research_papers'])} papers")

    # 2. COMPRESS CODE: Keep only passing code
    compressed['generated_code'] = state.get('executable_code', {})
    compressed['executable_code'] = state.get('executable_code', {})

    code_count = len(state.get('generated_code', {}))
    exec_count = len(state.get('executable_code', {}))
    if code_count > exec_count:
        logger.info(f"Compressed code: {code_count} → {exec_count} blocks (removed failed attempts)")

    # 3. COMPRESS TEST RESULTS: Keep summary only
    compressed['test_coverage'] = state.get('test_coverage', 0)
    compressed['validation_errors'] = []  # Clear old errors
    logger.info("Cleared old test results and validation errors")

    # 4. COMPRESS KEY FINDINGS: Keep top 5 only
    findings = state.get('key_findings', [])
    if findings:
        compressed['key_findings'] = findings[:5]
        if len(findings) > 5:
            logger.info(f"Compressed findings: {len(findings)} → 5")

    # 5. COMPRESS LITERATURE SUMMARY: Keep as is (already summarized)
    compressed['literature_summary'] = state.get('literature_summary', '')

    # 6. PRUNE MEMORY CONTEXT: Keep last 10 entries only
    memory = state.get('memory_context', [])
    if len(memory) > 10:
        compressed['memory_context'] = memory[-10:]
        logger.info(f"Pruned memory context: {len(memory)} → 10 entries")

    # 7. Keep quality scores and feedback from last iteration
    compressed['quality_scores'] = state.get('quality_scores', {})
    compressed['feedback'] = state.get('feedback', {})

    # Calculate compression ratio
    original_size = sum([
        len(str(state.get('research_papers', []))),
        len(str(state.get('generated_code', {}))),
        len(str(state.get('test_results', []))),
        len(str(state.get('memory_context', [])))
    ])
    compressed_size = sum([
        len(str(compressed.get('research_papers', []))),
        len(str(compressed.get('generated_code', {}))),
        len(str(compressed.get('memory_context', [])))
    ])

    if original_size > 0:
        compression_ratio = (1 - compressed_size / original_size) * 100
        logger.info(f"State compression complete: {compression_ratio:.1f}% size reduction")

    return compressed
