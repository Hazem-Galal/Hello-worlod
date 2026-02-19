"""
AI Web Search Agent - Web interface.
Modern chat-style UI with Streamlit.
"""

import os
import sys
from pathlib import Path

# Load .env before any other imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "AI_Web_Search_Agent")

import streamlit as st

# Import agent logic from main (after env is loaded)
from langchain_core.messages import AIMessage, HumanMessage

from main import SearchResult, _create_agent, _get_response, _parse_pydantic_response


def _validate_api_keys() -> bool:
    """Check if required API keys are set."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    return bool(
        openai_key
        and openai_key.strip() not in ("", "your_openai_api_key_here")
        and tavily_key
        and tavily_key.strip() not in ("", "your_tavily_api_key_here")
    )


def _render_search_result(result: SearchResult) -> None:
    """Render a SearchResult object."""
    with st.container():
        st.markdown("**Answer**")
        st.info(result.answer)
        if result.key_points:
            st.markdown("**Key points**")
            for point in result.key_points:
                st.markdown(f"- {point}")
        if result.sources:
            st.markdown("**Sources**")
            for src in result.sources:
                st.markdown(f"- [{src}]({src})" if src.startswith("http") else f"- {src}")


def _render_response(content: str | SearchResult, output_format: str | None) -> None:
    """Render the assistant response with optional formatting."""
    if isinstance(content, SearchResult):
        _render_search_result(content)
        return
    content = str(content)
    # Try Pydantic format if requested or if content looks like our schema
    if output_format == "pydantic" or (content.strip().startswith("{") and '"answer"' in content):
        result_obj = _parse_pydantic_response(content)
        if result_obj:
            _render_search_result(result_obj)
        else:
            st.markdown(content)
    else:
        st.markdown(content)


def main() -> None:
    st.set_page_config(
        page_title="AI Web Search Agent",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize chat history first (before any access)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Custom CSS for a polished look
    st.markdown("""
        <style>
        .stApp { max-width: 900px; margin: 0 auto; }
        .chat-user { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 16px; border-radius: 18px 18px 4px 18px; margin: 8px 0; }
        .chat-assistant { background: #f1f5f9; padding: 12px 16px; border-radius: 18px 18px 18px 4px; margin: 8px 0; }
        h1 { font-weight: 700 !important; letter-spacing: -0.5px; }
        .stChatInput { padding-bottom: 2rem; }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        output_format = st.selectbox(
            "Output format",
            options=["default", "sections", "pydantic"],
            format_func=lambda x: {"default": "Conversational", "sections": "Structured (sections)", "pydantic": "Structured (Pydantic)"}[x],
            help="How to structure the response",
        )
        format_for_agent = None if output_format == "default" else output_format

        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("Powered by LangChain, OpenAI & Tavily")

    # Header
    st.title("🔍 AI Web Search Agent")
    st.markdown("Ask anything — I'll search the web and synthesize an answer.")

    # Example queries (only show when chat is empty)
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        examples = [
            "What is the capital of France?",
            "Latest news about AI",
            "Weather in Tokyo today",
        ]
        for col, example in zip(cols, examples):
            with col:
                if st.button(example[:30] + "..." if len(example) > 30 else example, key=example, use_container_width=True):
                    st.session_state.pending_prompt = example
                    st.rerun()
    st.divider()

    # Display chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            if role == "user":
                st.markdown(content)
            else:
                _render_response(content, format_for_agent if output_format != "default" else None)

    # Process pending prompt (from example buttons) or new chat input
    chat_val = st.chat_input("Ask a question...")
    prompt = st.session_state.pop("pending_prompt", None) or chat_val

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching the web..."):
                try:
                    agent = _create_agent(format_for_agent)
                    # Build message history for context
                    chat_history = [
                        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                        for m in st.session_state.messages[:-1]
                    ]
                    chat_history.append(HumanMessage(content=prompt))
                    result = agent.invoke({"messages": chat_history})
                    content, is_structured = _get_response(result, format_for_agent)
                    if content is not None:
                        if is_structured:
                            _render_response(content, format_for_agent)
                            st.session_state.messages.append({"role": "assistant", "content": content.model_dump_json()})
                        else:
                            text = str(content)
                            _render_response(text, format_for_agent)
                            st.session_state.messages.append({"role": "assistant", "content": text})
                    else:
                        st.error("Could not extract response.")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.rerun()


if __name__ == "__main__":
    if not _validate_api_keys():
        st.set_page_config(page_title="AI Web Search Agent", page_icon="🔍")
        st.error(
            "**Missing API keys.** Set OPENAI_API_KEY and TAVILY_API_KEY in your `.env` file. "
            "Copy `.env.example` to `.env` and add your keys. Get Tavily API key at "
            "[app.tavily.com](https://app.tavily.com/sign-in)."
        )
        st.stop()
    main()
