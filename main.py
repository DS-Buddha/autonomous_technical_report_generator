"""
Main CLI entry point for the Hybrid Agentic System.
Generates technical reports autonomously using multi-agent workflow.
"""

import argparse
import sys
from pathlib import Path

from src.graph.workflow import run_workflow
from src.config.settings import settings
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__, log_file=Path('outputs/app.log'))


def generate_report(
    topic: str,
    depth: str = 'comprehensive',
    code_examples: bool = True,
    max_iterations: int = 3
) -> Path:
    """
    Generate a technical report for the given topic.

    Args:
        topic: Research topic for report generation
        depth: Depth of research ('basic', 'moderate', 'comprehensive')
        code_examples: Whether to include code examples
        max_iterations: Maximum reflection loop iterations

    Returns:
        Path to generated report
    """
    logger.info(f"Starting report generation for topic: {topic}")
    logger.info(f"Settings: depth={depth}, code_examples={code_examples}, max_iterations={max_iterations}")

    # Build requirements
    requirements = {
        'depth': depth,
        'code_examples': code_examples
    }

    try:
        # Execute workflow
        final_state = run_workflow(
            topic=topic,
            requirements=requirements,
            max_iterations=max_iterations,
            stream_output=True
        )

        # Get output path
        output_path = Path(final_state['report_metadata']['output_path'])

        logger.info(f"Report generated successfully: {output_path}")
        logger.info(f"Statistics: {final_state['report_metadata']}")

        return output_path

    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Technical Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Transformer architectures in NLP"
  %(prog)s "RAG systems" --depth moderate --max-iterations 2
  %(prog)s "Attention mechanisms" --no-code

For more information, see README.md
        """
    )

    parser.add_argument(
        'topic',
        type=str,
        help='Research topic for report generation'
    )

    parser.add_argument(
        '--depth',
        choices=['basic', 'moderate', 'comprehensive'],
        default='comprehensive',
        help='Depth of research and analysis (default: comprehensive)'
    )

    parser.add_argument(
        '--no-code',
        action='store_true',
        help='Disable code example generation'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='Maximum reflection loop iterations (default: 3)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Hybrid Agentic System v1.0.0'
    )

    args = parser.parse_args()

    # Display banner
    print("=" * 60)
    print("🤖 Hybrid Agentic System for Technical Report Generation")
    print("=" * 60)
    print(f"\nTopic: {args.topic}")
    print(f"Depth: {args.depth}")
    print(f"Code Examples: {not args.no_code}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"\nModel: {settings.google_ai_model}")
    print("\nStarting workflow...\n")

    try:
        # Generate report
        output_path = generate_report(
            topic=args.topic,
            depth=args.depth,
            code_examples=not args.no_code,
            max_iterations=args.max_iterations
        )

        # Success message
        print("\n" + "=" * 60)
        print("✅ Report Generation Complete!")
        print("=" * 60)
        print(f"\nReport saved to: {output_path}")
        print("\nYou can now:")
        print(f"  - Read the report: {output_path}")
        print(f"  - Check logs: outputs/app.log")
        print("\nThank you for using Hybrid Agentic System!")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Report generation interrupted by user")
        logger.warning("Process interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        print("\nPlease check:")
        print("  1. Your .env file has valid API keys")
        print("  2. You have internet connectivity")
        print("  3. All dependencies are installed (pip install -r requirements.txt)")
        print(f"\nSee outputs/app.log for details")
        return 1


if __name__ == '__main__':
    sys.exit(main())
