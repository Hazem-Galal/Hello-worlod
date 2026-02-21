"""
LangChain Documentation Assistant - RAG Chatbot

Streamlit app that answers questions about LangChain using documentation
stored in Pinecone.
"""

import os

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

RAG_PROMPT = """Answer the question based only on the following context from the LangChain documentation. If the answer is not in the context, say so clearly. Include relevant code examples when appropriate.

Context:
{context}

Question: {question}

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


@st.cache_resource
def get_rag_chain():
    """Build and cache the RAG chain."""
    retriever = get_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
            for d in docs
        )

    extract_question = RunnableLambda(lambda x: x["question"])
    format_docs_runnable = RunnableLambda(format_docs)

    chain = (
        {
            "context": extract_question | retriever | format_docs_runnable,
            "question": extract_question,
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    st.set_page_config(
        page_title="LangChain Documentation Assistant",
        page_icon="📚",
        layout="centered",
    )
    st.title("📚 LangChain Documentation Assistant")
    st.caption("Ask questions about LangChain. Answers are based on the official documentation.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about LangChain..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                try:
                    chain = get_rag_chain()
                    response = chain.invoke({"question": prompt})
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {e}"}
                    )


if __name__ == "__main__":
    main()
