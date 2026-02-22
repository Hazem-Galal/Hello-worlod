"""
LangGraph agent with RAG ingestion tools.
Exposes run_ingestion, load_documents, and get_index_status as tools.
"""

from typing_extensions import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from copilotkit import CopilotKitState

from .ingestion import run_ingestion, load_documents, get_index_status


class AgentState(CopilotKitState):
    """Agent state with CopilotKit integration."""

    pass


@tool
def run_ingestion_tool(docs_path: str = "") -> str:
    """
    Run the full RAG document ingestion pipeline. Loads .txt files from the docs folder,
    splits them into chunks, and indexes them in Pinecone. Use this when the user wants
    to ingest documents, index documents, or update the knowledge base.
    Optionally provide docs_path to use a custom folder (default: ./docs).
    """
    path = docs_path.strip() if docs_path else None
    return run_ingestion(path)


@tool
def load_documents_tool(docs_path: str = "") -> str:
    """
    Load documents from the docs folder without indexing. Returns the count of
    documents found. Use this to check what documents are available before running ingestion.
    Optionally provide docs_path (default: ./docs).
    """
    path = docs_path.strip() if docs_path else None
    documents, msg = load_documents(path)
    return f"{msg} Found {len(documents)} documents."


@tool
def get_index_status_tool() -> str:
    """
    Check the status of the Pinecone vector index. Use this to verify the index exists
    and is ready before or after running ingestion.
    """
    return get_index_status()


tools = [run_ingestion_tool, load_documents_tool, get_index_status_tool]


async def chat_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["tool_node", "__end__"]]:
    """Chat node that routes to tools when needed."""

    model = ChatOpenAI(model="gpt-4o")
    model_with_tools = model.bind_tools(
        [*state["copilotkit"]["actions"], *tools],
        parallel_tool_calls=False,
    )

    system_message = SystemMessage(
        content="""You are a helpful RAG ingestion assistant. You help users manage their document
ingestion pipeline for a RAG (Retrieval Augmented Generation) system. You can:
- Run the full ingestion pipeline to load, chunk, and index documents in Pinecone
- Load documents to see what's available
- Check the status of the Pinecone index

When users ask to ingest documents, index documents, or update the knowledge base,
use the run_ingestion tool. Be concise and informative in your responses."""
    )

    response = await model_with_tools.ainvoke(
        [system_message, *state["messages"]], config
    )

    if isinstance(response, AIMessage) and response.tool_calls:
        actions = state["copilotkit"]["actions"]
        if not any(
            action.get("name") == response.tool_calls[0].get("name")
            for action in actions
        ):
            return Command(goto="tool_node", update={"messages": response})

    return Command(goto=END, update={"messages": response})


workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", ToolNode(tools=tools))
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

graph = workflow.compile()
