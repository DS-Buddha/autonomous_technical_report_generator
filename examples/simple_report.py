"""
Simple example of using the Hybrid Agentic System programmatically.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.workflow import run_workflow
from src.config.settings import settings
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)


def main():
    """Generate a sample technical report."""

    print("=" * 60)
    print("Hybrid Agentic System - Example Usage")
    print("=" * 60)

    # Define topic
    topic = "Attention Mechanisms in Neural Networks"

    print(f"\nGenerating report on: {topic}")
    print(f"Using model: {settings.google_ai_model}\n")

    # Configure requirements
    requirements = {
        'depth': 'moderate',  # basic, moderate, or comprehensive
        'code_examples': True
    }

    try:
        # Run workflow
        final_state = run_workflow(
            topic=topic,
            requirements=requirements,
            max_iterations=2,
            stream_output=True
        )

        # Display results
        print("\n" + "=" * 60)
        print("Report Generation Complete!")
        print("=" * 60)

        metadata = final_state['report_metadata']
        print(f"\nStatistics:")
        print(f"  - Output: {metadata.get('output_path')}")
        print(f"  - Word Count: {metadata.get('word_count')}")
        print(f"  - Code Examples: {metadata.get('code_blocks')}")
        print(f"  - References: {metadata.get('references')}")

        # Display quality scores
        if 'quality_scores' in final_state:
            print(f"\nQuality Scores:")
            for dimension, score in final_state['quality_scores'].items():
                print(f"  - {dimension.capitalize()}: {score:.1f}/10.0")

        print("\nSuccess! Check the outputs/reports directory for your report.")

    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"Report generation failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
