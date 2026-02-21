"""Agent loop using LangGraph StateGraph with LangChain tool-calling."""

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# --- Tools (LangChain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return float(prices.get(product, 0.0))


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- LangGraph Agent ---

SYSTEM_PROMPT = (
    "You are a helpful shopping assistant. "
    "You have access to a product catalog tool "
    "and a discount tool.\n\n"
    "STRICT RULES — you must follow these exactly:\n"
    "1. NEVER guess or assume any product price. "
    "You MUST call get_product_price first to get the real price.\n"
    "2. Only call apply_discount AFTER you have received "
    "a price from get_product_price. Pass the exact price "
    "returned by get_product_price — do NOT pass a made-up number.\n"
    "3. NEVER calculate discounts yourself using math. "
    "Always use the apply_discount tool.\n"
    "4. If the user does not specify a discount tier, "
    "ask them which tier to use — do NOT assume one."
)


def _build_agent_node(llm_with_tools):
    """Build the agent node that calls the LLM with tools."""

    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        ai_message = llm_with_tools.invoke(messages)
        return {"messages": [ai_message]}

    return agent_node


def _build_tools_node(tools_dict):
    """Build the tools node that executes only the FIRST tool call."""

    def tools_node(state: MessagesState):
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls or []
        if not tool_calls:
            return {"messages": []}

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"  [Tool Result] {observation}")

        return {
            "messages": [
                ToolMessage(content=str(observation), tool_call_id=tool_call_id)
            ]
        }

    return tools_node


def _should_continue(state: MessagesState) -> str:
    """Route to tools node if last message has tool_calls, else END."""
    messages = state["messages"]
    if not messages:
        return "end"
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


@traceable(name="LangGraph Agent Loop")
def run_agent(question: str) -> str | None:
    """Run the LangGraph agent on the given question."""
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", _build_agent_node(llm_with_tools))
    workflow.add_node("tools", _build_tools_node(tools_dict))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    print(f"Question: {question}")
    print("=" * 60)

    initial_messages = [HumanMessage(content=question)]

    result = app.invoke(
        {"messages": initial_messages},
        config={"recursion_limit": MAX_ITERATIONS},
    )

    messages = result["messages"]
    if not messages:
        print("ERROR: No messages in result")
        return None

    last_message = messages[-1]
    if hasattr(last_message, "content") and last_message.content:
        print(f"\nFinal Answer: {last_message.content}")
        return last_message.content

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangGraph Agent (StateGraph)!")
    print()
    run_agent("What is the price of a laptop after applying a gold discount?")
