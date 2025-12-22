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
    state_compression_node,
    cross_domain_analyst_node,
    cross_domain_researcher_node,
    innovation_synthesizer_node,
    implementation_researcher_node,
    implementation_literature_node,
    implementation_synthesizer_node
)
from src.graph.edges import (
    should_continue_research,
    should_test_code,
    should_revise,
    route_after_planning,
    validate_research_quality
)
from src.utils.logger import get_logger
from src.utils.progress_streamer import get_progress_streamer
from src.utils.token_tracker import get_token_tracker

logger = get_logger(__name__)


def create_workflow(with_checkpoints: bool = True):
    """
    Create the complete LangGraph workflow with dual modes.

    STANDARD MODE (staff_ml_engineer):
    START → Planner → Researcher → Coder → Tester → Critic → Synthesizer → END
                         ↑            ↑        ↓        ↓
                         └────────────┴────────┴────────┘
                              (reflection loops)

    INNOVATION MODE (research_innovation):
    START → Planner → Researcher (Domain) → CrossDomainAnalyst → CrossDomainResearcher
            → InnovationSynthesizer → ImplementationResearcher → ImplementationLiterature
            → ImplementationSynthesizer → END
                ↑
                └────────(retry on failure)

    Innovation mode generates TWO reports:
    1. Innovation Report: Cross-domain parallels and novel research directions
    2. Implementation Report: Deep dive on how to implement each experiment publishably

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

    # Innovation mode nodes
    workflow.add_node("cross_domain_analyst", cross_domain_analyst_node)
    workflow.add_node("cross_domain_researcher", cross_domain_researcher_node)
    workflow.add_node("innovation_synthesizer", innovation_synthesizer_node)

    # Implementation research nodes (Phase 2 of innovation mode)
    workflow.add_node("implementation_researcher", implementation_researcher_node)
    workflow.add_node("implementation_literature", implementation_literature_node)
    workflow.add_node("implementation_synthesizer", implementation_synthesizer_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Define edges and conditional routing

    # Planner → Researcher (always)
    workflow.add_edge("planner", "researcher")

    # Researcher → Validation (CRITICAL: prevent hallucinated reports)
    def route_after_research_validation(state: AgentState) -> str:
        """Route after research validation based on quality and report mode."""
        quality_result = validate_research_quality(state)

        if quality_result == "research_failed":
            return "research_failed"

        # Research approved - check report mode
        report_mode = state.get('report_mode', 'staff_ml_engineer')

        if report_mode == 'research_innovation':
            # Innovation mode: go to cross-domain analysis
            return "innovation_path"
        else:
            # Standard mode: go to coder
            return "standard_path"

    workflow.add_conditional_edges(
        "researcher",
        route_after_research_validation,
        {
            "standard_path": "coder",            # Standard mode: proceed to coding
            "innovation_path": "cross_domain_analyst",  # Innovation mode: analyze cross-domain parallels
            "research_failed": "research_failure_handler"  # Insufficient research
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

    # ============ INNOVATION MODE WORKFLOW ============
    # Cross-Domain Analyst → Cross-Domain Researcher (always)
    workflow.add_edge("cross_domain_analyst", "cross_domain_researcher")

    # Cross-Domain Researcher → Innovation Synthesizer (always)
    workflow.add_edge("cross_domain_researcher", "innovation_synthesizer")

    # Innovation Synthesizer → Implementation Researcher (Phase 2: Deep implementation research)
    workflow.add_edge("innovation_synthesizer", "implementation_researcher")

    # Implementation Researcher → Implementation Literature (always)
    workflow.add_edge("implementation_researcher", "implementation_literature")

    # Implementation Literature → Implementation Synthesizer (always)
    workflow.add_edge("implementation_literature", "implementation_synthesizer")

    # Implementation Synthesizer → END (final)
    workflow.add_edge("implementation_synthesizer", END)

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
    stream_output: bool = True,
    budget_limit_usd: float = None
):
    """
    Execute the complete workflow for a topic.

    Args:
        topic: Research topic
        requirements: Optional requirements dict
        max_iterations: Maximum reflection loop iterations
        stream_output: Print intermediate outputs
        budget_limit_usd: Optional budget limit for token usage

    Returns:
        Final state with report
    """
    logger.info(f"Starting workflow for topic: {topic}")

    # Initialize progress streamer and token tracker
    streamer = get_progress_streamer(enable_console_output=stream_output)
    tracker = get_token_tracker(budget_limit_usd=budget_limit_usd)

    # Emit workflow start event
    streamer.start_workflow(topic)

    try:
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

                # Check budget status after each step
                if budget_limit_usd:
                    current_cost = tracker.get_total_cost()
                    streamer.budget_status(current_cost, budget_limit_usd)

        else:
            # Non-streaming mode: execute all at once
            final_state = app.invoke(initial_state)

        logger.info("Workflow completed successfully")

        # Add token usage stats to final state
        final_state['token_usage_stats'] = tracker.get_statistics()

        # Emit workflow completion event
        streamer.complete_workflow(result={
            'report_path': final_state.get('report_metadata', {}).get('output_path'),
            'total_cost_usd': tracker.get_total_cost(),
            'total_tokens': tracker.get_total_tokens()
        })

        # Print token usage summary if streaming
        if stream_output:
            print("\n" + tracker.get_cost_breakdown_report())
            print("\n" + "=" * 60)
            print("PROGRESS SUMMARY")
            print("=" * 60)
            summary = streamer.get_summary()
            print(f"Total events: {summary['total_events']}")
            print(f"Agents executed: {', '.join(summary['agents_executed'])}")
            if summary['workflow_duration_seconds']:
                print(f"Duration: {summary['workflow_duration_seconds']:.1f}s")

        return final_state

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)

        # Emit workflow failure event
        streamer.fail_workflow(str(e))

        raise


# Export main functions
__all__ = ['create_workflow', 'run_workflow']
