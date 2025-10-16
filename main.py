"""
QCEW Employment Forecasting Pipeline - Main Entry Point

This is the main entry point for the employment forecasting pipeline.
Supports both command-line arguments and interactive menu mode.

Usage:
    python main.py                    # Launch interactive menu
    python main.py --cli              # Use command-line mode
    python main.py --stage explore    # Run specific stage (CLI mode)
    python main.py --help             # Show help

Interactive Menu:
    User-friendly menu-driven interface for running pipeline stages.

Command-Line Mode:
    Supports all original --stage, --skip-plots, --force-rebuild options.
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from logging_config import setup_logging
from pipeline_orchestrator import QCEWPipeline

# Setup logging
logger = setup_logging()


def run_interactive():
    """Launch the interactive menu interface."""
    from interactive_menu import run_interactive_menu
    run_interactive_menu()


def run_cli(args):
    """Run in command-line mode with arguments."""
    # Build configuration
    config = {
        'skip_plots': args.skip_plots,
        'force_rebuild': args.force_rebuild,
        'launch_interface': args.launch_interface
    }

    # Initialize pipeline
    pipeline = QCEWPipeline(config)

    # Run pipeline or specific stage
    if args.stage:
        pipeline.run_stage(args.stage)
    else:
        pipeline.run_full_pipeline()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='QCEW Employment Forecasting Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Launch interactive menu (default)
  python main.py --cli               # Run full pipeline in CLI mode
  python main.py --stage explore     # Run only exploration stage
  python main.py --stage train       # Run only training stage
  python main.py --skip-plots        # Skip plot generation
  python main.py --force-rebuild     # Force rebuild of consolidated data

Interactive Menu:
  The interactive menu provides a user-friendly interface to:
  - Run individual pipeline stages
  - Check pipeline status
  - View completed stages
  - No need to remember command-line options!
        """
    )

    parser.add_argument(
        '--cli',
        action='store_true',
        help='Use command-line mode instead of interactive menu'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['consolidate', 'explore', 'validate', 'features', 'preprocess', 'train', 'evaluate', 'predict'],
        help='Run a specific pipeline stage (automatically enables CLI mode)'
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip plot generation in exploration stage'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild of consolidated dataset'
    )
    parser.add_argument(
        '--launch-interface',
        action='store_true',
        help='Launch prediction interface after pipeline completes'
    )

    args = parser.parse_args()

    # If any CLI-specific arguments are provided, use CLI mode
    if args.cli or args.stage or args.skip_plots or args.force_rebuild or args.launch_interface:
        run_cli(args)
    else:
        # Default to interactive menu
        run_interactive()


if __name__ == "__main__":
    main()
