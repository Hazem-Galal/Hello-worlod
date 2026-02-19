"""
CLI for the AI web search agent.
Supports interactive mode and single-shot mode with --query.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from agent import search

load_dotenv()


def _validate_api_keys() -> None:
    """Validate that required API keys are set. Exit with error if missing."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")

    missing = []
    if not openai_key or openai_key.strip() in ("", "your_openai_api_key_here"):
        missing.append("OPENAI_API_KEY")
    if not tavily_key or tavily_key.strip() in ("", "your_tavily_api_key_here"):
        missing.append("TAVILY_API_KEY")

    if missing:
        print(
            f"Error: Missing required API keys: {', '.join(missing)}\n"
            "Please set them in your .env file. Copy .env.example to .env and add your keys.\n"
            "Get Tavily API key at https://app.tavily.com/sign-in",
            file=sys.stderr,
        )
        sys.exit(1)


def _interactive_mode() -> None:
    """Run the agent in interactive mode: prompt for queries in a loop."""
    print("AI Web Search Agent - Interactive Mode")
    print("Enter your search query (or 'quit' / 'exit' to stop)\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nSearching...")
        try:
            result = search(query)
            print(f"\n{result}\n")
        except Exception as e:
            print(f"Error: {e}\n", file=sys.stderr)


def _single_shot_mode(query: str) -> None:
    """Run a single query and print the result."""
    try:
        result = search(query)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Web Search Agent - Search the web using AI"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Run a single query and exit (non-interactive)",
    )
    args = parser.parse_args()

    _validate_api_keys()

    if args.query is not None:
        _single_shot_mode(args.query)
    else:
        _interactive_mode()


if __name__ == "__main__":
    main()
