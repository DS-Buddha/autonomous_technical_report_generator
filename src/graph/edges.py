"""
Conditional edge functions for workflow routing.
Determines which node to execute next based on state.
"""

from src.graph.state import AgentState
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def should_continue_research(state: AgentState) -> str:
    """
    Determine if more research is needed.

    Args:
        state: Current workflow state

    Returns:
        "continue" or "proceed"
    """
    papers = state.get('research_papers', [])
    findings = state.get('key_findings', [])

    min_papers = settings.min_research_papers
    min_findings = settings.min_key_findings

    if len(papers) < min_papers or len(findings) < min_findings:
        logger.info(f"Research insufficient ({len(papers)} papers, {len(findings)} findings) - continuing research")
        return "continue"

    logger.info(f"Research sufficient ({len(papers)} papers, {len(findings)} findings) - proceeding")
    return "proceed"


def should_test_code(state: AgentState) -> str:
    """
    Determine if code needs fixes or can proceed.

    Args:
        state: Current workflow state

    Returns:
        "fix_code" or "review"
    """
    errors = state.get('validation_errors', [])

    if errors:
        logger.info(f"{len(errors)} validation errors found - fixing code")
        return "fix_code"

    logger.info("No validation errors - proceeding to review")
    return "review"


def should_revise(state: AgentState) -> str:
    """
    Determine what needs revision based on critic feedback.

    Args:
        state: Current workflow state

    Returns:
        "revise_research", "revise_code", or "synthesize"
    """
    iteration_count = state.get('iteration_count', 0)
    max_iterations = state.get('max_iterations', 3)

    # Check iteration limit
    if iteration_count >= max_iterations:
        logger.info(f"Max iterations ({max_iterations}) reached - proceeding to synthesis")
        return "synthesize"

    # Check if revision needed
    needs_revision = state.get('needs_revision', False)

    if not needs_revision:
        logger.info("Quality approved - proceeding to synthesis")
        return "synthesize"

    # Determine what to revise based on lowest score
    scores = state.get('quality_scores', {})

    if not scores:
        logger.warning("No quality scores available - proceeding to synthesis")
        return "synthesize"

    # Find lowest scoring dimension
    lowest_dimension = min(scores, key=scores.get)
    lowest_score = scores[lowest_dimension]

    logger.info(f"Lowest score: {lowest_dimension}={lowest_score:.1f}")

    # Route based on dimension
    if lowest_dimension in ['accuracy', 'completeness']:
        logger.info("Routing to research revision")
        return "revise_research"
    elif lowest_dimension in ['code_quality', 'executability']:
        logger.info("Routing to code revision")
        return "revise_code"
    else:
        logger.info("Routing to synthesis")
        return "synthesize"


def route_after_planning(state: AgentState) -> str:
    """
    Route after planning based on plan complexity.

    Args:
        state: Current workflow state

    Returns:
        "researcher"
    """
    subtasks = state.get('subtasks', [])
    logger.info(f"Plan created with {len(subtasks)} subtasks - routing to researcher")
    return "researcher"


def validate_research_quality(state: AgentState) -> str:
    """
    Validate research meets minimum quality before proceeding.
    Triggers HITL (Human-in-the-Loop) if insufficient.

    This is a CRITICAL safety check to prevent hallucinated reports
    when research APIs fail or return insufficient results.

    Args:
        state: Current workflow state

    Returns:
        "research_approved" or "research_failed"
    """
    papers = state.get('research_papers', [])
    findings = state.get('key_findings', [])

    # Hard minimum thresholds
    min_papers = 3  # Absolute minimum for credible research
    min_findings = 2  # At least some insights extracted

    # Check paper count
    if len(papers) < min_papers:
        logger.error(
            f"INSUFFICIENT RESEARCH: Only {len(papers)} papers found "
            f"(minimum {min_papers} required)"
        )
        return "research_failed"

    # Check findings count
    if len(findings) < min_findings:
        logger.error(f"INSUFFICIENT FINDINGS: Only {len(findings)} extracted")
        return "research_failed"

    # Check that papers have sufficient detail (not just titles)
    papers_with_abstracts = sum(
        1 for p in papers
        if p.get('abstract') and len(p['abstract']) > 100
    )

    if papers_with_abstracts < min_papers // 2:
        logger.error(
            f"Research papers lack sufficient detail: "
            f"Only {papers_with_abstracts}/{len(papers)} have abstracts"
        )
        return "research_failed"

    # All checks passed
    logger.info(
        f"Research quality validated: {len(papers)} papers, "
        f"{len(findings)} findings, {papers_with_abstracts} with abstracts"
    )
    return "research_approved"
