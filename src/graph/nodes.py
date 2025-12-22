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
from src.agents.cross_domain_analyst import CrossDomainAnalyst
from src.agents.innovation_synthesizer import InnovationSynthesizer
from src.agents.implementation_researcher import ImplementationResearcher
from src.agents.implementation_synthesizer import ImplementationSynthesizer
from src.memory.memory_manager import MemoryManager
from src.graph.state import AgentState
from src.utils.logger import get_logger
from src.utils.progress_streamer import get_progress_streamer, ProgressEventType

logger = get_logger(__name__)
streamer = get_progress_streamer()

# Initialize agents (singleton pattern)
planner = PlannerAgent()
researcher = ResearcherAgent()
coder = CoderAgent()
tester = TesterAgent()
critic = CriticAgent()
synthesizer = SynthesizerAgent()
cross_domain_analyst = CrossDomainAnalyst()
innovation_synthesizer = InnovationSynthesizer()
implementation_researcher = ImplementationResearcher()
implementation_synthesizer = ImplementationSynthesizer()
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
    streamer.start_agent("planner", "Breaking down task into subtasks")

    try:
        result = planner.run(
            topic=state['topic'],
            requirements=state['requirements']
        )

        streamer.complete_agent("planner", f"Created {len(result.get('subtasks', []))} subtasks")
    except Exception as e:
        streamer.fail_agent("planner", str(e))
        raise

    return {
        'plan': result.get('plan', {}),
        'subtasks': result.get('subtasks', []),
        'dependencies': result.get('dependencies', {}),
        'search_queries': result.get('search_queries', []),
        'code_specifications': result.get('code_specifications', []),
        'messages': [{'role': 'assistant', 'content': f"Planner: Created plan with {len(result.get('subtasks', []))} subtasks"}],
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

    streamer.start_agent("researcher", f"Searching {len(queries)} queries on arXiv")

    try:
        result = researcher.run(queries=queries)

        # Store findings in memory
        findings = result.get('key_findings', [])
        if findings:
            memory.add_research_findings(findings)

        papers_count = len(result.get('research_papers', []))
        streamer.complete_agent("researcher", f"Found {papers_count} papers with {len(findings)} key findings")
    except Exception as e:
        streamer.fail_agent("researcher", str(e))
        raise

    return {
        'research_papers': result.get('research_papers', []),
        'key_findings': findings,
        'literature_summary': result.get('literature_summary', ''),
        'memory_context': [{'type': 'research', 'count': len(findings)}],
        'messages': [{'role': 'assistant', 'content': f"Researcher: Found {len(result.get('research_papers', []))} papers with {len(findings)} key findings"}],
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

    streamer.start_agent("coder", f"Generating {len(specifications)} code implementations")

    try:
        result = coder.run(specifications=specifications, context=context)

        # Store code patterns in memory
        code_blocks = result.get('generated_code', {})
        if code_blocks:
            memory.add_code_patterns([
                {'id': k, 'code': v, 'description': spec.get('description', '')}
                for k, v in code_blocks.items()
                for spec in specifications if spec.get('id') == k
            ])

        streamer.complete_agent("coder", f"Generated {len(code_blocks)} code blocks")
    except Exception as e:
        streamer.fail_agent("coder", str(e))
        raise

    return {
        'generated_code': code_blocks,
        'code_dependencies': result.get('code_dependencies', []),
        'memory_context': [{'type': 'code', 'count': len(code_blocks)}],
        'messages': [{'role': 'assistant', 'content': f"Coder: Generated {len(code_blocks)} code blocks"}],
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

    streamer.start_agent("tester", f"Testing {len(code_blocks)} code blocks")

    try:
        result = tester.run(code_blocks=code_blocks)

        coverage = result.get('test_coverage', 0.0)
        passed_count = len(result.get('executable_code', {}))
        streamer.complete_agent("tester", f"{passed_count}/{len(code_blocks)} blocks passed ({coverage:.1f}% coverage)")
    except Exception as e:
        streamer.fail_agent("tester", str(e))
        raise

    return {
        'test_results': result.get('test_results', []),
        'validation_errors': result.get('validation_errors', []),
        'executable_code': result.get('executable_code', {}),
        'test_coverage': result.get('test_coverage', 0.0),
        'messages': [{'role': 'assistant', 'content': f"Tester: Tested {len(code_blocks)} blocks, {len(result.get('executable_code', {}))} passed ({result.get('test_coverage', 0):.1f}% coverage)"}],
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
    streamer.start_agent("critic", "Evaluating quality across 5 dimensions")

    try:
        result = critic.run(state=state)

        overall_score = result.get('overall_score', 0.0)
        needs_revision = result.get('needs_revision', False)

        # CRITICAL FIX: Increment iteration counter when revision is needed
        iteration_count = state.get('iteration_count', 0)
        if needs_revision:
            iteration_count += 1
            max_iterations = state.get('max_iterations', 3)
            logger.warning(f"Revision needed. Iteration count: {iteration_count}/{max_iterations}")
            streamer.iteration_increment(iteration_count, max_iterations)

        logger.info(f"Quality score: {overall_score:.1f}/10.0, Needs revision: {needs_revision}")

        status_msg = f"Score: {overall_score:.1f}/10 - {'Revision needed' if needs_revision else 'Approved'}"
        streamer.complete_agent("critic", status_msg)
    except Exception as e:
        streamer.fail_agent("critic", str(e))
        raise

    return {
        'quality_scores': result.get('quality_scores', {}),
        'feedback': result.get('feedback', {}),
        'needs_revision': needs_revision,
        'iteration_count': iteration_count,
        'messages': [{'role': 'assistant', 'content': f"Critic: Quality score: {overall_score:.1f}/10.0"}],
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
    streamer.start_agent("synthesizer", "Creating final markdown report")

    try:
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

        word_count = metadata.get('word_count', 0)
        streamer.complete_agent("synthesizer", f"Report generated ({word_count} words) -> {output_path}")
    except Exception as e:
        streamer.fail_agent("synthesizer", str(e))
        raise

    return {
        'final_report': final_report,
        'report_metadata': metadata,
        'messages': [{'role': 'assistant', 'content': f"Synthesizer: Generated report ({metadata.get('word_count', 0)} words)"}],
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


def cross_domain_analyst_node(state: AgentState) -> Dict[str, Any]:
    """
    Cross-Domain Analyst node: Identify parallels across scientific fields.

    Args:
        state: Current workflow state

    Returns:
        State updates with cross-domain analysis and search queries
    """
    logger.info("=== CROSS-DOMAIN ANALYST NODE ===")
    streamer.start_agent("cross_domain_analyst", "Identifying cross-domain parallels")

    try:
        result = cross_domain_analyst.run(state=state)

        analysis = result.get('cross_domain_analysis', {})
        parallels = analysis.get('cross_domain_parallels', [])
        search_queries = result.get('cross_domain_search_queries', [])

        streamer.complete_agent(
            "cross_domain_analyst",
            f"Identified {len(parallels)} parallels, generated {len(search_queries)} search queries"
        )
    except Exception as e:
        streamer.fail_agent("cross_domain_analyst", str(e))
        raise

    return {
        'cross_domain_analysis': result.get('cross_domain_analysis', {}),
        'cross_domain_search_queries': result.get('cross_domain_search_queries', []),
        'messages': result.get('messages', []),
        'status': 'cross_domain_analysis_complete'
    }


def cross_domain_researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Cross-Domain Researcher node: Search literature from other scientific fields.

    Args:
        state: Current workflow state

    Returns:
        State updates with cross-domain research papers
    """
    logger.info("=== CROSS-DOMAIN RESEARCHER NODE ===")

    # Get cross-domain search queries and parallels
    queries = state.get('cross_domain_search_queries', [])
    parallels = state.get('cross_domain_analysis', {}).get('cross_domain_parallels', [])

    if not queries:
        logger.warning("No cross-domain queries available, skipping")
        return {
            'cross_domain_papers': [],
            'messages': [{'role': 'assistant', 'content': "CrossDomainResearcher: No queries provided"}],
            'status': 'cross_domain_research_skipped'
        }

    streamer.start_agent("cross_domain_researcher", f"Searching {len(queries)} cross-domain queries")

    try:
        result = researcher.run_cross_domain(
            cross_domain_queries=queries,
            parallels=parallels
        )

        papers_count = len(result.get('cross_domain_papers', []))
        streamer.complete_agent(
            "cross_domain_researcher",
            f"Found {papers_count} cross-domain papers"
        )
    except Exception as e:
        streamer.fail_agent("cross_domain_researcher", str(e))
        raise

    return {
        'cross_domain_papers': result.get('cross_domain_papers', []),
        'messages': [{'role': 'assistant', 'content': f"CrossDomainResearcher: Found {len(result.get('cross_domain_papers', []))} papers"}],
        'status': 'cross_domain_research_complete'
    }


def innovation_synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Innovation Synthesizer node: Create comprehensive innovation report.

    Args:
        state: Current workflow state

    Returns:
        State updates with final innovation report
    """
    logger.info("=== INNOVATION SYNTHESIZER NODE ===")
    streamer.start_agent("innovation_synthesizer", "Synthesizing cross-domain innovation report")

    try:
        result = innovation_synthesizer.run(state=state)

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

        logger.info(f"Innovation report generated: {output_path}")

        word_count = metadata.get('word_count', 0)
        parallels = metadata.get('parallels_analyzed', 0)
        streamer.complete_agent(
            "innovation_synthesizer",
            f"Report generated ({word_count} words, {parallels} parallels) -> {output_path}"
        )

        # Signal Phase 1 completion
        streamer.complete_phase(
            "Phase 1: Cross-Domain Innovation",
            f"Generated innovation report with {parallels} cross-domain parallels identified"
        )

    except Exception as e:
        streamer.fail_agent("innovation_synthesizer", str(e))
        raise

    return {
        'final_report': final_report,
        'report_metadata': metadata,
        'messages': [{'role': 'assistant', 'content': f"InnovationSynthesizer: Generated report ({metadata.get('word_count', 0)} words)"}],
        'status': 'innovation_synthesis_complete'
    }


def implementation_researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Implementation Researcher node: Analyze experiments and research implementations.

    Args:
        state: Current workflow state

    Returns:
        State updates with implementation analysis
    """
    logger.info("=== IMPLEMENTATION RESEARCHER NODE ===")

    # Signal Phase 2 start
    streamer.start_phase(
        "Phase 2: Implementation Research",
        "Deep-dive analysis of publishable implementations for each experiment"
    )

    streamer.start_agent("implementation_researcher", "Analyzing proposed experiments for implementation")

    try:
        result = implementation_researcher.run(state=state)

        analysis = result.get('implementation_analysis', {})
        experiments = analysis.get('experiments_analysis', [])
        queries = result.get('implementation_search_queries', [])

        streamer.complete_agent(
            "implementation_researcher",
            f"Analyzed {len(experiments)} experiments, generated {len(queries)} queries"
        )
    except Exception as e:
        streamer.fail_agent("implementation_researcher", str(e))
        raise

    return {
        'implementation_analysis': result.get('implementation_analysis', {}),
        'implementation_search_queries': result.get('implementation_search_queries', []),
        'messages': result.get('messages', []),
        'status': 'implementation_analysis_complete'
    }


def implementation_literature_node(state: AgentState) -> Dict[str, Any]:
    """
    Implementation Literature node: Deep research on how to implement experiments.

    Args:
        state: Current workflow state

    Returns:
        State updates with implementation papers
    """
    logger.info("=== IMPLEMENTATION LITERATURE NODE ===")

    # Get implementation search queries
    queries = state.get('implementation_search_queries', [])

    if not queries:
        logger.warning("No implementation queries available, skipping")
        return {
            'implementation_papers': [],
            'messages': [{'role': 'assistant', 'content': "ImplementationLiterature: No queries provided"}],
            'status': 'implementation_literature_skipped'
        }

    streamer.start_agent("implementation_literature", f"Searching {len(queries)} implementation queries")

    try:
        # Use regular researcher for implementation papers
        result = researcher.run(queries=queries)

        # Rename papers to implementation_papers
        papers = result.get('research_papers', [])

        streamer.complete_agent(
            "implementation_literature",
            f"Found {len(papers)} implementation papers"
        )
    except Exception as e:
        streamer.fail_agent("implementation_literature", str(e))
        raise

    return {
        'implementation_papers': papers,
        'messages': [{'role': 'assistant', 'content': f"ImplementationLiterature: Found {len(papers)} papers"}],
        'status': 'implementation_literature_complete'
    }


def implementation_synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Implementation Synthesizer node: Generate detailed implementation report.

    Args:
        state: Current workflow state

    Returns:
        State updates with implementation report
    """
    logger.info("=== IMPLEMENTATION SYNTHESIZER NODE ===")
    streamer.start_agent("implementation_synthesizer", "Generating detailed implementation report")

    try:
        result = implementation_synthesizer.run(state=state)

        implementation_report = result.get('implementation_report', '')
        metadata = result.get('implementation_metadata', {})

        # Save implementation report to file
        from src.tools.file_tools import FileTools
        file_tools = FileTools()

        # Create special filename for implementation report
        topic = state['topic']
        filename = f"{file_tools._sanitize_filename(topic)}_IMPLEMENTATION.md"
        output_path = file_tools.save_markdown_report(
            content=implementation_report,
            topic=filename.replace('.md', '')  # Remove .md as it's added by save function
        )

        metadata['output_path'] = str(output_path)

        logger.info(f"Implementation report generated: {output_path}")

        word_count = metadata.get('word_count', 0)
        experiments = metadata.get('experiments_analyzed', 0)
        streamer.complete_agent(
            "implementation_synthesizer",
            f"Report generated ({word_count} words, {experiments} experiments) -> {output_path}"
        )

        # Signal Phase 2 completion
        streamer.complete_phase(
            "Phase 2: Implementation Research",
            f"Generated detailed implementation report with {experiments} experiments analyzed"
        )

    except Exception as e:
        streamer.fail_agent("implementation_synthesizer", str(e))
        raise

    return {
        'implementation_report': implementation_report,
        'implementation_metadata': metadata,
        'messages': [{'role': 'assistant', 'content': f"ImplementationSynthesizer: Generated report ({metadata.get('word_count', 0)} words)"}],
        'status': 'implementation_complete'
    }
