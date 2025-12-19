"""
Complete LangGraph workflow orchestration.
Defines the multi-agent graph with conditional routing and reflection loops.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import AgentState, create_initial_state
from src.graph.nodes import (
    planner_node,
    researcher_node,
    coder_node,
    tester_node,
    critic_node,
    synthesizer_node,
    research_failure_node,
    state_compression_node
)
from src.graph.edges import (
    should_continue_research,
    should_test_code,
    should_revise,
    route_after_planning,
    validate_research_quality
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_workflow(with_checkpoints: bool = True):
    """
    Create the complete LangGraph workflow.

    Workflow structure:
    START → Planner → Researcher → Coder → Tester → Critic → Synthesizer → END
                         ↑            ↑        ↓        ↓
                         └────────────┴────────┴────────┘
                              (reflection loops)

    Args:
        with_checkpoints: Enable state persistence (default: True)

    Returns:
        Compiled LangGraph application
    """
    logger.info("Creating workflow graph")

    # Initialize workflow with state schema
    workflow = StateGraph(AgentState)

    # Add nodes (each node is an agent)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("research_failure_handler", research_failure_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("tester", tester_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("compress_state", state_compression_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Define edges and conditional routing

    # Planner → Researcher (always)
    workflow.add_edge("planner", "researcher")

    # Researcher → Validation (CRITICAL: prevent hallucinated reports)
    workflow.add_conditional_edges(
        "researcher",
        validate_research_quality,
        {
            "research_approved": "coder",            # Sufficient research, proceed
            "research_failed": "research_failure_handler"  # Insufficient, handle failure
        }
    )

    # Research Failure Handler → Researcher (retry) or END (HITL required)
    # The handler returns 'next_agent' in state to control routing
    def route_after_failure(state: AgentState) -> str:
        next_agent = state.get('next_agent', 'END')
        if next_agent == 'researcher':
            return 'retry'
        return 'end'

    workflow.add_conditional_edges(
        "research_failure_handler",
        route_after_failure,
        {
            "retry": "researcher",  # Retry with broader queries
            "end": END             # HITL required, stop workflow
        }
    )

    # Coder → Tester (always)
    workflow.add_edge("coder", "tester")

    # Tester → Critic (conditional: may loop back to coder)
    workflow.add_conditional_edges(
        "tester",
        should_test_code,
        {
            "fix_code": "coder",     # Tests failed, regenerate code
            "review": "critic"        # Tests passed, proceed to review
        }
    )

    # Critic → Compression (if revision needed) or Synthesizer (if approved)
    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise_research": "compress_state",   # Compress before research revision
            "revise_code": "compress_state",       # Compress before code revision
            "synthesize": "synthesizer"            # Quality approved, no compression needed
        }
    )

    # Compression → Route to appropriate revision agent
    def route_after_compression(state: AgentState) -> str:
        """Route to appropriate agent after compression based on feedback."""
        feedback = state.get('feedback', {})
        scores = state.get('quality_scores', {})

        if not scores:
            return "researcher"  # Default to research

        # Find lowest scoring dimension
        lowest_dimension = min(scores, key=scores.get)

        # Route based on dimension
        if lowest_dimension in ['accuracy', 'completeness']:
            logger.info("Routing to research after compression")
            return "research"
        elif lowest_dimension in ['code_quality', 'executability']:
            logger.info("Routing to code after compression")
            return "code"
        else:
            return "research"  # Default

    workflow.add_conditional_edges(
        "compress_state",
        route_after_compression,
        {
            "research": "researcher",
            "code": "coder"
        }
    )

    # Synthesizer → END (always)
    workflow.add_edge("synthesizer", END)

    # Add checkpointer for state persistence
    if with_checkpoints:
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
    else:
        app = workflow.compile()

    logger.info("Workflow graph created successfully")
    return app


def run_workflow(
    topic: str,
    requirements: dict = None,
    max_iterations: int = 3,
    stream_output: bool = True
):
    """
    Execute the complete workflow for a topic.

    Args:
        topic: Research topic
        requirements: Optional requirements dict
        max_iterations: Maximum reflection loop iterations
        stream_output: Print intermediate outputs

    Returns:
        Final state with report
    """
    logger.info(f"Starting workflow for topic: {topic}")

    # Create initial state
    initial_state = create_initial_state(
        topic=topic,
        requirements=requirements,
        max_iterations=max_iterations
    )

    # Create workflow
    app = create_workflow(with_checkpoints=False)

    # Execute workflow
    if stream_output:
        # Stream mode: print each step
        final_state = None
        for step_output in app.stream(initial_state):
            # step_output is a dict with node name as key
            for node_name, node_output in step_output.items():
                logger.info(f"Completed: {node_name}")
                if 'messages' in node_output:
                    for msg in node_output['messages']:
                        logger.info(f"  {msg['role']}: {msg['content']}")

            final_state = node_output

    else:
        # Non-streaming mode: execute all at once
        final_state = app.invoke(initial_state)

    logger.info("Workflow completed successfully")
    return final_state


# Export main functions
__all__ = ['create_workflow', 'run_workflow']
