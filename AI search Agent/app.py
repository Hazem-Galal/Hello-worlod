"""
Streamlit web UI for the AI web search agent.
"""

import os
import sys

import streamlit as st

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()


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


def main() -> None:
    st.set_page_config(
        page_title="AI Web Search Agent",
        page_icon="🔍",
        layout="centered",
    )
    st.title("🔍 AI Web Search Agent")
    st.caption("Search the web using AI powered by LangChain, OpenAI, and Tavily")

    if not _validate_api_keys():
        st.error(
            "Missing API keys. Please set OPENAI_API_KEY and TAVILY_API_KEY in your .env file. "
            "Copy .env.example to .env and add your keys. Get Tavily API key at "
            "[tavily.com](https://app.tavily.com/sign-in)."
        )
        st.stop()

    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., What is the capital of France?",
        label_visibility="collapsed",
    )

    if st.button("Search", type="primary", use_container_width=True):
        if not query or not query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching the web..."):
                try:
                    from agent import search

                    result = search(query.strip())
                    st.markdown("### Answer")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
