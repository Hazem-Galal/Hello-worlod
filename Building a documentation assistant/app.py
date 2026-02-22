"""
LangChain Documentation Assistant - RAG Chatbot

Streamlit app that answers questions about LangChain using documentation
stored in Pinecone. Supports streaming responses and conversation memory.
"""

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pinecone import Pinecone

from retriever import PineconeRetriever

load_dotenv()

INDEX_NAME = "langchain-docs"
NAMESPACE = "langchain_documentation"
MAX_HISTORY_TURNS = 10  # Number of prior Q&A pairs to include for context
CHAT_HISTORY_FILE = Path(__file__).parent / ".chat_history.json"
MAX_PERSISTED_MESSAGES = 100  # Max messages to keep in persisted file
# Set to "o1-mini" or "o4-mini" to enable reasoning/thinking display (requires Responses API)
REASONING_MODEL = os.getenv("REASONING_MODEL", "")

RAG_PROMPT = """Answer the question based on the following context from the LangChain documentation and the conversation history. If the answer is not in the context, say so clearly. Use the chat history to understand follow-up questions. Include relevant code examples when appropriate.

Context:
{context}

Chat history:
{chat_history}

Current question: {question}

Answer:"""


@st.cache_resource
def get_retriever():
    """Initialize and cache the Pinecone retriever."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        st.error("PINECONE_API_KEY not set in .env")
        st.stop()

    pc = Pinecone(api_key=api_key)
    if not pc.has_index(INDEX_NAME):
        st.error(
            f"Index '{INDEX_NAME}' not found. Run `python ingestion.py` first to crawl and index the documentation."
        )
        st.stop()

    index = pc.Index(INDEX_NAME)
    return PineconeRetriever(
        index=index,
        namespace=NAMESPACE,
        top_k=5,
        use_rerank=True,
    )


def load_chat_history() -> list:
    """Load chat history from file if it exists."""
    if CHAT_HISTORY_FILE.exists():
        try:
            with open(CHAT_HISTORY_FILE, encoding="utf-8") as f:
                data = json.load(f)
                return data.get("messages", [])[-MAX_PERSISTED_MESSAGES:]
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_chat_history(messages: list) -> None:
    """Persist chat history to file."""
    try:
        to_save = messages[-MAX_PERSISTED_MESSAGES:]
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"messages": to_save}, f, indent=2)
    except OSError:
        pass


def format_chat_history(messages: list) -> str:
    """Format message list into a string for the prompt."""
    if not messages:
        return "(No prior messages)"
    lines = []
    for m in messages:
        role = "Human" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def format_docs(docs):
    """Format retrieved documents for display or prompt."""
    return "\n\n---\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
        for d in docs
    )


@st.cache_resource
def get_rag_chain():
    """Build and cache the RAG chain (standard model, no reasoning)."""
    retriever = get_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


@st.cache_resource
def get_reasoning_llm():
    """Build reasoning model (o1-mini, o4-mini) with Responses API for thinking display."""
    return ChatOpenAI(
        model=REASONING_MODEL,
        temperature=0,
        streaming=True,
        use_responses_api=True,
        model_kwargs={
            "reasoning": {"effort": "medium", "summary": "auto"},
        },
    )


def main():
    st.set_page_config(
        page_title="LangChain Documentation Assistant",
        page_icon="📚",
        layout="centered",
    )
    st.title("📚 LangChain Documentation Assistant")
    st.caption("Ask questions about LangChain. Answers are based on the official documentation. Supports streaming, conversation memory, context absorption, and optional reasoning (set REASONING_MODEL=o4-mini).")

    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about LangChain..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                history_messages = st.session_state.messages[:-1]
                history_messages = history_messages[-(MAX_HISTORY_TURNS * 2) :]
                chat_history_str = format_chat_history(history_messages)

                # 1. Absorption: retrieve and show context
                retriever = get_retriever()
                docs = retriever.invoke(prompt)
                with st.expander("📥 Context absorbed", expanded=False):
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get("source", "unknown")
                        st.caption(f"{i}. {source}")
                        st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

                # 2. Thinking + Answer: stream response
                if REASONING_MODEL:
                    llm = get_reasoning_llm()
                    prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT)
                    context_str = format_docs(docs)
                    messages = prompt_template.format_messages(
                        context=context_str,
                        chat_history=chat_history_str,
                        question=prompt,
                    )

                    thinking_placeholder = st.empty()
                    answer_placeholder = st.empty()

                    reasoning_by_index = {}
                    answer_parts = []

                    for chunk in llm.stream(messages):
                        reasoning = chunk.additional_kwargs.get("reasoning", {})
                        if reasoning and "summary" in reasoning:
                            for item in reasoning["summary"]:
                                if isinstance(item, dict):
                                    idx = item.get("index", 0)
                                    t = item.get("text", "")
                                    if idx not in reasoning_by_index:
                                        reasoning_by_index[idx] = ""
                                    reasoning_by_index[idx] += t
                            thinking_str = "\n\n".join(
                                f"**Step {i+1}**\n{v}"
                                for i, (_, v) in enumerate(sorted(reasoning_by_index.items()))
                                if v.strip()
                            )
                            if thinking_str:
                                thinking_placeholder.markdown(f"**💭 Thinking**\n\n{thinking_str}")

                        content = chunk.content
                        if content:
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        t = block.get("text", "")
                                        answer_parts.append(t)
                            elif isinstance(content, str):
                                answer_parts.append(content)
                            answer_placeholder.markdown("".join(answer_parts))

                    response = "".join(answer_parts)
                    if not response and reasoning_by_index:
                        response = "\n\n".join(v for _, v in sorted(reasoning_by_index.items()) if v.strip())
                else:
                    chain = get_rag_chain()

                    def stream_response():
                        for chunk in chain.stream({
                            "question": prompt,
                            "chat_history": chat_history_str,
                        }):
                            yield chunk

                    response = st.write_stream(stream_response())

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                save_chat_history(st.session_state.messages)
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {e}"}
                )
                save_chat_history(st.session_state.messages)


if __name__ == "__main__":
    main()
