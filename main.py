#!/usr/bin/env python3
"""
VideoARM — command-line entry point.

Examples
--------
# Open-ended question
python main.py --video path/to/video.mp4 --question "What happens in this video?"

# Multiple-choice (letter format, default)
python main.py --video video.mp4 --question "A. ... B. ... C. ... D. ..." --multiple-choice

# Multiple-choice (number format, 0-4)
python main.py --video video.mp4 --question "..." --multiple-choice --choice-format number

# Override the controller model
python main.py --video video.mp4 --question "..." --model gpt-4o
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VideoARM: Agentic Reasoning-over-Hierarchical-Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--video", "-v", required=True, help="Path to local video file.")
    parser.add_argument("--question", "-q", required=True, help="Question to answer.")
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Override controller model (default: from config or VIDEOARM_MODEL_CONTROLLER).",
    )
    parser.add_argument(
        "--multiple-choice",
        "--mc",
        action="store_true",
        help="Multiple-choice mode — respond with a single letter or number.",
    )
    parser.add_argument(
        "--choice-format",
        choices=["letter", "number"],
        default="letter",
        help="Answer format for multiple-choice: 'letter' (A-D) or 'number' (0-4).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the QA trace to disk.",
    )
    parser.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        help="Show detailed execution output.",
    )

    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    from videoarm.core.agent import VideoARMAgent

    try:
        agent = VideoARMAgent(model_name=args.model)
        answer = agent.ask(
            args.video,
            args.question,
            is_multiple_choice=args.multiple_choice,
            choice_format=args.choice_format,
            save_result=not args.no_save,
        )
        print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
