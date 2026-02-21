"""Chatbot with Tavily web search and multiply tool."""

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'. Use for any multiplication of two numbers."""
    return x * y


WIDTH = 70
SEP = "─" * WIDTH


def _format_response(content: str) -> str:
    """Format assistant response with rich layout."""
    lines = content.strip().split("\n")
    formatted = []
    for line in lines:
        # Preserve bullets and numbering
        stripped = line.strip()
        if stripped.startswith(("-", "*", "•")) or (
            len(stripped) > 1 and stripped[0].isdigit() and stripped[1] in ".)"
        ):
            formatted.append(f"  {stripped}")
        elif stripped.startswith("#"):
            formatted.append(f"\n  {stripped.lstrip('#').strip()}\n")
        else:
            formatted.append(f"  {stripped}" if stripped else "")
    return "\n".join(formatted) if formatted else content


def _tools_used(messages: list) -> list[str]:
    """Extract tool names used from ToolMessages."""
    names = []
    for m in messages:
        if isinstance(m, ToolMessage):
            name = getattr(m, "name", None) or "tool"
            if name not in names:
                names.append(name)
    return names


def main():
    print("\n" + "═" * WIDTH)
    print("  CHATBOT — Tavily Search + Multiply")
    print("═" * WIDTH)
    print("  • Web search (Tavily) for news, weather, facts")
    print("  • multiply(x, y) for multiplication")
    print("  • Type 'quit' or 'exit' to stop")
    print(SEP + "\n")

    tools = [TavilySearch(max_results=5), multiply]
    llm = ChatOpenAI(model="gpt-4-turbo")

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant with access to:\n"
            "1) Tavily web search — use it for current info, weather, news, facts.\n"
            "2) multiply(x, y) — use it for multiplication of two numbers.\n"
            "Use the right tool for each task. Format responses with bullets or "
            "numbered lists when listing items. Be concise and accurate."
        ),
    )

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        result = agent.invoke({"messages": messages})

        # Update messages with full conversation
        messages = result["messages"]

        # Find last AI response
        last_ai_content = None
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                last_ai_content = m.content
                break

        if last_ai_content:
            # Tools used in this turn (messages after last HumanMessage)
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                m = messages[i]
                if isinstance(m, HumanMessage) or (isinstance(m, dict) and m.get("role") == "user"):
                    last_user_idx = i
                    break
            turn_messages = messages[last_user_idx + 1:] if last_user_idx >= 0 else messages
            tools_used = _tools_used(turn_messages)
            tools_str = f" [tools: {', '.join(tools_used)}]" if tools_used else ""

            print(f"\n{SEP}")
            print(f" Assistant{tools_str}")
            print(SEP)
            print(_format_response(last_ai_content))
            print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
