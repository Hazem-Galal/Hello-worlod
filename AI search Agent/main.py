import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from the script's directory (works regardless of cwd)
load_dotenv(Path(__file__).resolve().parent / ".env")

# LangSmith tracing: send traces to project "AI_Web_Search_Agent"
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "AI_Web_Search_Agent")

from datetime import datetime

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# Agent components: LLM for reasoning, TavilySearch for web research
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tavily_search = TavilySearch(max_results=5, topic="general")


class SearchResult(BaseModel):
    """Structured output schema for web search results."""

    answer: str = Field(description="Main response summarizing the findings")
    key_points: list[str] = Field(default_factory=list, description="Important facts or takeaways")
    sources: list[str] = Field(default_factory=list, description="URLs or references used")


def _build_system_prompt(today: str, output_format: str | None) -> str:
    base = """You are a helpful web search assistant with strong analytical skills.

When answering:
1. **Think step by step** - Break complex questions into smaller parts. Consider multiple angles before concluding.
2. **Search strategically** - Use web search for facts, recent events, or when you need up-to-date information. Search multiple times if the query has several aspects.
3. **Synthesize clearly** - Combine findings into a coherent answer. Cite your sources when possible.
4. **Stay conversational** - In multi-turn chats, refer to earlier context. Acknowledge follow-ups and build on previous answers.

Today's date is {today}."""
    if output_format == "sections":
        base += """

**Output format** - Structure your response with these sections (use markdown headers):
- **Answer:** Main response (2-4 sentences)
- **Key points:** Bullet list of important facts
- **Sources:** URLs or references used"""
    elif output_format == "pydantic":
        base += """

**Output format** - Structure your final answer as: answer (main response), key_points (list of important facts), sources (list of URLs/references)."""
    return base.format(today=today)


def _create_agent(output_format: str | None = None):
    today = datetime.today().strftime("%B %d, %Y")
    kwargs = {
        "model": llm,
        "tools": [tavily_search],
        "system_prompt": _build_system_prompt(today, output_format),
    }
    if output_format == "pydantic":
        kwargs["response_format"] = SearchResult
    return create_agent(**kwargs)


def _extract_json_from_response(text: str) -> str | None:
    """Extract JSON from LLM response that may be wrapped in markdown code blocks."""
    text = text.strip()
    # Try direct parse first
    if text.startswith("{") and text.endswith("}"):
        return text
    # Try to extract from ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    # Try to find JSON object between first { and last }
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _parse_pydantic_response(content: str) -> SearchResult | None:
    """Parse content into SearchResult, handling markdown-wrapped JSON."""
    # Try direct parse
    try:
        return SearchResult.model_validate_json(content)
    except Exception:
        pass
    # Try extracted JSON
    json_str = _extract_json_from_response(content)
    if json_str:
        try:
            return SearchResult.model_validate_json(json_str)
        except Exception:
            pass
    return None


def _extract_answer(messages: list) -> str | None:
    """Extract the final AI message content from the result."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            if isinstance(content, list):
                parts = [p.get("text", p) if isinstance(p, dict) else str(p) for p in content]
                return "".join(str(p) for p in parts) or None
            return str(content)
    return None


def _get_response(result: dict, output_format: str | None) -> tuple[str | SearchResult | None, bool]:
    """
    Get the response from agent result. Returns (content, is_structured).
    For pydantic format, checks structured_response first.
    """
    if output_format == "pydantic":
        structured = result.get("structured_response")
        if structured is not None and isinstance(structured, SearchResult):
            return (structured, True)
    answer = _extract_answer(result.get("messages", []))
    return (answer, False) if answer else (None, False)


def main():
    parser = argparse.ArgumentParser(description="AI Web Search Agent")
    parser.add_argument("--query", "-q", type=str, help="Single query (exits after one answer)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Multi-turn conversation mode")
    parser.add_argument(
        "--format", "-f",
        choices=["sections", "pydantic"],
        help="Structure output: 'sections' (Answer/Key points/Sources) or 'pydantic' (validated schema)",
    )
    args = parser.parse_args()

    run_agent = _create_agent(args.format)
    chat_history = []

    if args.query:
        # Single-shot mode
        chat_history.append(HumanMessage(content=args.query))
        result = run_agent.invoke({"messages": chat_history})
        content, is_structured = _get_response(result, args.format)
        if content is not None:
            if is_structured:
                print(content.model_dump_json(indent=2))
            elif args.format == "pydantic":
                result_obj = _parse_pydantic_response(content)
                print(result_obj.model_dump_json(indent=2) if result_obj else content)
            else:
                print(content)
        else:
            print(result)
        return

    # Interactive multi-turn mode
    print("AI Web Search Agent - Interactive mode")
    print("Ask anything. Type 'quit', 'exit', or 'q' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        chat_history.append(HumanMessage(content=query))
        result = run_agent.invoke({"messages": chat_history})
        content, is_structured = _get_response(result, args.format)
        if content is not None:
            if is_structured:
                print(f"\nAssistant:\n{content.model_dump_json(indent=2)}\n")
            elif args.format == "pydantic":
                result_obj = _parse_pydantic_response(str(content))
                print(f"\nAssistant:\n{(result_obj.model_dump_json(indent=2) if result_obj else content)}\n")
            else:
                print(f"\nAssistant: {content}\n")
            chat_history = result.get("messages", [])
        else:
            print(result)


if __name__ == "__main__":
    main()