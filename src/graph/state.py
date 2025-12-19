"""
LangGraph state schema for the hybrid agentic system.
Defines the shared state that flows between all agents.
"""

from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
from langgraph.graph import add_messages
import operator


def bounded_memory_add(existing: List, new: List) -> List:
    """
    Custom reducer for memory_context that prevents unbounded growth.

    Keeps only the last 50 entries to prevent context window bloat.
    This is critical for production as memory can grow indefinitely
    with operator.add, causing token costs to explode.

    Args:
        existing: Current memory entries
        new: New entries to add

    Returns:
        Combined list limited to last 50 entries
    """
    combined = existing + new
    max_entries = 50

    if len(combined) > max_entries:
        return combined[-max_entries:]
    return combined


class AgentState(TypedDict):
    """
    Shared state schema for the multi-agent workflow.

    This state is passed between all agents and contains:
    - Input parameters
    - Planning outputs
    - Research results
    - Generated code
    - Quality assessments
    - Final report

    State updates use reducers to handle concurrent modifications:
    - operator.add: Appends to lists
    - add_messages: Handles message deduplication
    - Default: Last write wins
    """

    # ============ INPUT ============
    topic: str  # User's research topic
    requirements: Dict[str, Any]  # Specific requirements (depth, code_examples, etc.)

    # ============ PLANNING PHASE ============
    plan: Dict[str, Any]  # Hierarchical task structure
    subtasks: List[Dict[str, Any]]  # Actionable subtasks
    dependencies: Dict[str, List[str]]  # Task dependency graph
    current_subtask: Optional[Dict[str, Any]]  # Currently executing subtask

    # ============ RESEARCH PHASE ============
    search_queries: List[str]  # Generated search queries
    research_papers: List[Dict[str, Any]]  # Retrieved papers with metadata
    key_findings: List[Dict[str, Any]]  # Extracted insights
    literature_summary: str  # Aggregated research summary
    research_retry_count: int  # Number of research retry attempts

    # ============ CODE GENERATION PHASE ============
    code_specifications: List[Dict[str, Any]]  # Code requirements from plan
    generated_code: Dict[str, str]  # Code blocks by identifier
    code_dependencies: List[str]  # Required Python packages
    executable_code: Dict[str, str]  # Only validated, executable code

    # ============ TESTING PHASE ============
    test_results: List[Dict[str, Any]]  # Code execution results
    validation_errors: List[Dict[str, Any]]  # Errors found during validation
    test_coverage: float  # Percentage of code that passed tests

    # ============ QUALITY ASSURANCE PHASE ============
    quality_scores: Dict[str, float]  # Evaluation metrics by dimension
    feedback: Dict[str, str]  # Improvement suggestions by dimension
    iteration_count: int  # Current reflection loop iteration
    needs_revision: bool  # Flag to trigger re-execution
    max_iterations: int  # Maximum allowed reflection loops

    # ============ SYNTHESIS PHASE ============
    report_outline: Dict[str, Any]  # Document structure
    final_report: str  # Complete markdown document
    report_metadata: Dict[str, Any]  # Statistics and information

    # ============ SHARED MEMORY CONTEXT ============
    # Use bounded_memory_add to prevent unbounded growth (max 50 entries)
    memory_context: Annotated[List[Dict[str, Any]], bounded_memory_add]

    # Use add_messages for agent communication (handles deduplication)
    messages: Annotated[Sequence[Dict[str, str]], add_messages]

    # ============ CONTROL FLOW ============
    next_agent: Optional[str]  # For swarm handoff pattern
    error: Optional[str]  # Error messages if any
    status: str  # Current workflow status (planning, researching, etc.)


# Type aliases for cleaner code
ResearchPaper = Dict[str, Any]
CodeBlock = Dict[str, Any]
Subtask = Dict[str, Any]
QualityScore = Dict[str, float]


def create_initial_state(
    topic: str,
    requirements: Optional[Dict[str, Any]] = None,
    max_iterations: int = 3
) -> AgentState:
    """
    Create initial state for a new workflow execution.

    Args:
        topic: Research topic for report generation
        requirements: Optional requirements dict
        max_iterations: Maximum reflection loop iterations

    Returns:
        Initialized AgentState
    """
    if requirements is None:
        requirements = {
            'depth': 'comprehensive',
            'code_examples': True
        }

    return {
        # Input
        'topic': topic,
        'requirements': requirements,

        # Planning
        'plan': {},
        'subtasks': [],
        'dependencies': {},
        'current_subtask': None,

        # Research
        'search_queries': [],
        'research_papers': [],
        'key_findings': [],
        'literature_summary': '',
        'research_retry_count': 0,

        # Code
        'code_specifications': [],
        'generated_code': {},
        'code_dependencies': [],
        'executable_code': {},

        # Testing
        'test_results': [],
        'validation_errors': [],
        'test_coverage': 0.0,

        # Quality
        'quality_scores': {},
        'feedback': {},
        'iteration_count': 0,
        'needs_revision': False,
        'max_iterations': max_iterations,

        # Synthesis
        'report_outline': {},
        'final_report': '',
        'report_metadata': {},

        # Shared
        'memory_context': [],
        'messages': [],

        # Control
        'next_agent': None,
        'error': None,
        'status': 'initialized'
    }
