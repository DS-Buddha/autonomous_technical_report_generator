"""
Integration tests for Phase 1 production fixes.

Tests verify:
1. Iteration counter prevents infinite loops
2. Research validation prevents hallucinated reports
3. Critic enforces quality standards with negative constraints
"""

import pytest
from unittest.mock import patch, MagicMock
from src.graph.state import create_initial_state
from src.graph.nodes import critic_node, research_failure_node
from src.graph.edges import validate_research_quality, should_revise
from src.graph.workflow import create_workflow


class TestIterationCounterFix:
    """Test that iteration counter prevents infinite loops."""

    def test_iteration_counter_increments_on_revision(self):
        """Verify iteration counter increments when critic requests revision."""
        # Create initial state
        state = create_initial_state("Test Topic", max_iterations=3)
        state['iteration_count'] = 0

        # Mock critic to request revision
        with patch('src.graph.nodes.critic') as mock_critic:
            mock_critic.run.return_value = {
                'quality_scores': {
                    'accuracy': 6.0,
                    'completeness': 5.5,
                    'code_quality': 6.5,
                    'clarity': 6.0,
                    'executability': 6.0
                },
                'overall_score': 6.0,
                'feedback': {'accuracy': 'Needs more papers'},
                'needs_revision': True,
                'priority_issues': ['Insufficient research depth']
            }

            # Run critic node
            result = critic_node(state)

            # Verify iteration counter incremented
            assert result['iteration_count'] == 1, "Iteration counter should increment"
            assert result['needs_revision'] is True

    def test_iteration_counter_does_not_increment_on_approval(self):
        """Verify iteration counter doesn't increment when approved."""
        state = create_initial_state("Test Topic")
        state['iteration_count'] = 0

        # Mock critic to approve
        with patch('src.graph.nodes.critic') as mock_critic:
            mock_critic.run.return_value = {
                'quality_scores': {
                    'accuracy': 8.0,
                    'completeness': 7.5,
                    'code_quality': 8.5,
                    'clarity': 8.0,
                    'executability': 9.0
                },
                'overall_score': 8.2,
                'feedback': {},
                'needs_revision': False,
                'priority_issues': []
            }

            result = critic_node(state)

            # Iteration counter should not increment when approved
            assert result['iteration_count'] == 0

    def test_workflow_stops_after_max_iterations(self):
        """Verify workflow stops after reaching max iterations."""
        state = create_initial_state("Test Topic", max_iterations=3)
        state['iteration_count'] = 3  # Already at max

        # Should route to synthesis despite low quality
        state['needs_revision'] = True
        state['quality_scores'] = {'accuracy': 5.0}

        result = should_revise(state)

        assert result == "synthesize", "Should synthesize after max iterations"


class TestResearchValidation:
    """Test research validation prevents hallucinated reports."""

    def test_validation_passes_with_sufficient_research(self):
        """Verify validation passes with good research."""
        state = create_initial_state("Test Topic")
        state['research_papers'] = [
            {
                'title': 'Paper 1',
                'abstract': 'A' * 150,
                'authors': ['Author 1']
            },
            {
                'title': 'Paper 2',
                'abstract': 'B' * 150,
                'authors': ['Author 2']
            },
            {
                'title': 'Paper 3',
                'abstract': 'C' * 150,
                'authors': ['Author 3']
            }
        ]
        state['key_findings'] = [
            {'title': 'Finding 1'},
            {'title': 'Finding 2'}
        ]

        result = validate_research_quality(state)

        assert result == "research_approved", "Should approve sufficient research"

    def test_validation_fails_with_insufficient_papers(self):
        """Verify validation fails with too few papers."""
        state = create_initial_state("Test Topic")
        state['research_papers'] = [
            {'title': 'Only Paper', 'abstract': 'Short abstract'}
        ]
        state['key_findings'] = []

        result = validate_research_quality(state)

        assert result == "research_failed", "Should fail with insufficient papers"

    def test_validation_fails_with_no_abstracts(self):
        """Verify validation fails if papers lack abstracts."""
        state = create_initial_state("Test Topic")
        state['research_papers'] = [
            {'title': 'Paper 1', 'abstract': ''},
            {'title': 'Paper 2', 'abstract': 'short'},
            {'title': 'Paper 3', 'abstract': None}
        ]
        state['key_findings'] = [{'title': 'Finding'}]

        result = validate_research_quality(state)

        assert result == "research_failed", "Should fail without quality abstracts"

    def test_research_failure_handler_retries(self):
        """Verify failure handler attempts retry with broader queries."""
        state = create_initial_state("Machine Learning Topic")
        state['search_queries'] = [
            'query1 AND specific',
            '"exact phrase" AND narrow'
        ]
        state['research_retry_count'] = 0
        state['research_papers'] = []

        result = research_failure_node(state)

        # Should retry
        assert result['research_retry_count'] == 1
        assert result['status'] == 'retrying_research'
        assert result['next_agent'] == 'researcher'

        # Queries should be broader
        broader_queries = result['search_queries']
        assert any(' OR ' in q for q in broader_queries), "Should use OR instead of AND"

    def test_research_failure_handler_triggers_hitl_after_retries(self):
        """Verify HITL triggered after max retries."""
        state = create_initial_state("Test Topic")
        state['research_retry_count'] = 2  # Max retries reached
        state['research_papers'] = []

        result = research_failure_node(state)

        # Should trigger HITL
        assert result['status'] == 'research_failure_hitl_required'
        assert 'error' in result
        assert 'final_report' in result  # Should include failure report
        assert 'failed' in result['final_report'].lower()


class TestCriticEnforcement:
    """Test critic enforces negative constraints."""

    def test_critic_rejects_work_without_docstrings(self):
        """Verify critic flags code without docstrings."""
        state = create_initial_state("Test Topic")
        state['generated_code'] = {
            'example1': 'def func(x):\n    return x * 2'  # No docstring
        }
        state['executable_code'] = state['generated_code']
        state['test_coverage'] = 100.0
        state['research_papers'] = [
            {'title': f'Paper {i}', 'abstract': 'A' * 200}
            for i in range(5)
        ]
        state['key_findings'] = [{'title': f'Finding {i}'} for i in range(3)]

        # Mock the critic's LLM response to simulate it flagging missing docstrings
        from src.agents.critic_agent import CriticAgent
        critic = CriticAgent()

        with patch.object(critic, 'generate_json_response') as mock_generate:
            # Simulate critic identifying code quality issue
            mock_generate.return_value = {
                'dimension_scores': {
                    'accuracy': 7.5,
                    'completeness': 7.5,
                    'code_quality': 6.0,  # Low due to missing docstrings
                    'clarity': 7.0,
                    'executability': 8.0
                },
                'feedback': {
                    'code_quality': 'Code lacks docstrings and type hints'
                },
                'priority_issues': [
                    'Missing docstrings in code blocks',
                    'No type hints provided'
                ]
            }

            result = critic.run(state)

            # Should flag code quality issues
            code_quality = result['quality_scores'].get('code_quality', 10)
            priority_issues = result.get('priority_issues', [])

            assert (
                code_quality < 8.0 or
                any('code' in issue.lower() or 'docstring' in issue.lower()
                    for issue in priority_issues)
            ), "Critic should flag missing docstrings"

    def test_critic_prevents_approval_without_finding_issues(self):
        """Verify critic must find at least one issue before approving."""
        # This test verifies the re-evaluation logic
        state = create_initial_state("Test Topic")
        state['research_papers'] = [
            {'title': f'Paper {i}', 'abstract': 'A' * 200}
            for i in range(5)
        ]
        state['key_findings'] = [{'title': f'Finding {i}'} for i in range(3)]
        state['generated_code'] = {'example': 'print("hello")'}
        state['executable_code'] = state['generated_code']
        state['test_coverage'] = 100.0

        from src.agents.critic_agent import CriticAgent
        critic = CriticAgent()

        # Mock the LLM to initially approve everything
        with patch.object(critic, 'generate_json_response') as mock_generate:
            # First call: all high scores
            mock_generate.return_value = {
                'dimension_scores': {
                    'accuracy': 9.0,
                    'completeness': 9.0,
                    'code_quality': 9.0,
                    'clarity': 9.0,
                    'executability': 9.0
                },
                'feedback': {},
                'priority_issues': []
            }

            result = critic.run(state)

            # Should have called generate_json_response twice (re-evaluation)
            assert mock_generate.call_count == 2, "Should re-evaluate if no issues found"

    def test_critic_enforces_priority_issues_threshold(self):
        """Verify >2 priority issues triggers revision."""
        state = create_initial_state("Test Topic")

        from src.agents.critic_agent import CriticAgent
        critic = CriticAgent()

        with patch.object(critic, 'generate_json_response') as mock_generate:
            mock_generate.return_value = {
                'dimension_scores': {
                    'accuracy': 7.5,
                    'completeness': 7.5,
                    'code_quality': 7.5,
                    'clarity': 7.5,
                    'executability': 7.5
                },
                'feedback': {},
                'priority_issues': [
                    'Issue 1: Missing tests',
                    'Issue 2: Incomplete docs',
                    'Issue 3: Code not optimized'
                ]
            }

            result = critic.run(state)

            # Despite score >= 7.0, should need revision due to 3 issues
            assert result['needs_revision'] is True, "Should revise with >2 priority issues"


class TestWorkflowIntegration:
    """Test complete workflow with fixes integrated."""

    def test_workflow_includes_research_validation(self):
        """Verify workflow has research validation edges."""
        app = create_workflow(with_checkpoints=False)

        # Check that research_failure_handler node exists
        assert 'research_failure_handler' in app.nodes, \
            "Workflow should include failure handler"

    def test_iteration_limit_in_workflow_state(self):
        """Verify max_iterations is properly initialized."""
        state = create_initial_state("Test Topic", max_iterations=5)

        assert state['max_iterations'] == 5
        assert state['iteration_count'] == 0
        assert state['research_retry_count'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
