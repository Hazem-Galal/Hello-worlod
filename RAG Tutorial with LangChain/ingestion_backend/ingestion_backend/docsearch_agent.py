"""
DocSearch Copilot — LangGraph agent that answers questions using vector search.
"""

import json
from typing_extensions import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from copilotkit import CopilotKitState

from .retrieval import vector_search, fetch_document_chunk, list_sources


class DocSearchState(CopilotKitState):
    """Agent state with CopilotKit integration."""

    pass


@tool
def vector_search_tool(
    query: str,
    top_k: int = 8,
    filters: str | None = None,
) -> str:
    """
    Search the document index for relevant passages. ALWAYS call this before answering
    any document-based question. Use the user's exact phrasing plus synonyms and key
    entities (names, dates, system names). Returns passages with id, text, source_title,
    source_type, url_or_path, page, section, chunk_id, score.
    """
    f = json.loads(filters) if filters else None
    results = vector_search(query, top_k=top_k, filters=f)
    return json.dumps(results, indent=2)


@tool
def fetch_document_chunk_tool(chunk_id: str) -> str:
    """
    Fetch full chunk text by chunk_id. Use when you need the complete content of a
    specific passage returned by vector_search.
    """
    result = fetch_document_chunk(chunk_id)
    return json.dumps(result) if result else json.dumps({"error": "Chunk not found"})


@tool
def list_sources_tool() -> str:
    """
    List available document sources in the index. Returns titles, paths, and types.
    """
    sources = list_sources()
    return json.dumps(sources, indent=2)


docsearch_tools = [vector_search_tool, fetch_document_chunk_tool, list_sources_tool]

DOCSEARCH_SYSTEM_PROMPT = """You are DocSearch — a document search tool, NOT a chatbot. You find and return information from the user's private documents.

GOAL
- Help the user find and understand information inside their documents.
- Prefer grounded answers. If the documents don't contain enough evidence, say so clearly and propose the next best step.

RETRIEVAL RULES
1) Always call vector_search before answering any document-based question.
2) Use the user's exact phrasing + synonyms + key entities (names, dates, system names).
3) If results are weak (low scores, vague, or conflicting), do a second vector_search with:
   - narrower query (exact keywords)
   - broader query (synonyms / rephrased intent)
4) If the user asks "where is it?" or "show me", respond with the best matching passages and their sources.
5) Never invent citations. Only cite what is in retrieved passages.

ANSWER STYLE
- You are a search tool, not a conversational assistant. Focus on returning relevant information.
- Be concise and structured. Give the direct answer first, then supporting evidence.
- Always include citations when you use document content.
- Use bullets for multi-part answers.
- If uncertain, say what's missing. Do not engage in small talk or off-topic conversation.

CITATIONS FORMAT
When using any retrieved passage, cite it like:
- [Source Title — page X, section Y] or [Source Title — chunk_id: abc123]
If multiple sources support a claim, cite multiple.

CONFLICTS & AMBIGUITY
- If sources conflict, explain the conflict and show citations for both.
- Ask a targeted follow-up question only when absolutely necessary to proceed.

PRIVACY & SAFETY
- Treat documents as confidential.
- Do not reveal private data unless the user asked for it and it is necessary for the task.
- Do not output full documents. Quote only short, relevant snippets.

WORKFLOW
For each user question:
1) Rewrite the query for retrieval (keep meaning; include key entities).
2) Call vector_search with top_k 6–10.
3) Extract answerable facts and cite them.
4) If insufficient evidence, say "I couldn't find that in the documents" and propose what to search next.

UI BEHAVIOR (CopilotKit)
- When you cite sources, also provide a short "Sources" list at the end with:
  title, page/section, and url_or_path if available.
- If the user's question looks like a search request, suggest 2–3 clickable follow-up queries.
  Format them clearly so the UI can render them as chips (e.g., one per line or "Try: [query1] | [query2] | [query3]").
- If the user says "open it", return the best source url_or_path and the most relevant page/section."""


async def docsearch_chat_node(
    state: DocSearchState, config: RunnableConfig
) -> Command[Literal["tool_node", "__end__"]]:
    """Chat node that routes to tools when needed."""

    model = ChatOpenAI(model="gpt-4o")
    model_with_tools = model.bind_tools(
        [*state["copilotkit"]["actions"], *docsearch_tools],
        parallel_tool_calls=False,
    )

    response = await model_with_tools.ainvoke(
        [SystemMessage(content=DOCSEARCH_SYSTEM_PROMPT), *state["messages"]],
        config,
    )

    if isinstance(response, AIMessage) and response.tool_calls:
        actions = state["copilotkit"]["actions"]
        if not any(
            action.get("name") == response.tool_calls[0].get("name")
            for action in actions
        ):
            return Command(goto="tool_node", update={"messages": response})

    return Command(goto=END, update={"messages": response})


docsearch_workflow = StateGraph(DocSearchState)
docsearch_workflow.add_node("chat_node", docsearch_chat_node)
docsearch_workflow.add_node("tool_node", ToolNode(tools=docsearch_tools))
docsearch_workflow.add_edge("tool_node", "chat_node")
docsearch_workflow.set_entry_point("chat_node")

docsearch_graph = docsearch_workflow.compile()
