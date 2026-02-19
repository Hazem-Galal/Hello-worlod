"""
Core AI web search agent using LangChain, OpenAI, and Tavily.
"""

from datetime import datetime

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


def _create_search_agent(verbose: bool = False):
    """Create and return the web search agent."""
    tavily_search = TavilySearch(max_results=5, topic="general")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = (
        "You are a helpful web search assistant. Use web search to find accurate, "
        "up-to-date information. Synthesize the results into a clear, concise answer. "
        "When possible, cite your sources. "
        f"Today's date is {datetime.today().strftime('%B %d, %Y')}."
    )

    agent = create_agent(
        model=llm,
        tools=[tavily_search],
        system_prompt=system_prompt,
    )
    return agent


def search(query: str, verbose: bool = False) -> str:
    """
    Run a web search query through the agent and return the answer.

    Args:
        query: The search query from the user.
        verbose: If True, print agent execution details.

    Returns:
        The agent's synthesized answer as a string.
    """
    agent = _create_search_agent(verbose=verbose)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    # Extract the final assistant message content
    messages = response.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
        if isinstance(msg, dict) and msg.get("type") == "ai" and msg.get("content"):
            return str(msg["content"])

    return str(response)
