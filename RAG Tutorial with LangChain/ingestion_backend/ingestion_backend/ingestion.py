"""
Ingestion pipeline logic - shared with the original ingestion.py.
Loads documents, splits into chunks, and indexes in Pinecone.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Configuration (can be overridden via env)
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-tutorial")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
# Default docs path: prefer env, else ./docs relative to project root (RAG Tutorial folder)
_DEFAULT_DOCS = os.getenv("DOCS_PATH", "./docs")
DOCS_PATH = _DEFAULT_DOCS
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_DIM = 1536  # text-embedding-3-small


def get_pinecone_client():
    """Get Pinecone client. Raises if API key is missing."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in environment")
    return Pinecone(api_key=api_key)


def ensure_index_exists(pc: Pinecone) -> str:
    """Create Pinecone index if it does not exist. Returns status message."""
    if pc.has_index(INDEX_NAME):
        return f"Index '{INDEX_NAME}' already exists and is ready."
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    return f"Created index '{INDEX_NAME}'. Wait a few seconds for it to be ready before ingesting."


def get_index_status() -> str:
    """Check if Pinecone index exists and return status."""
    try:
        pc = get_pinecone_client()
        if pc.has_index(INDEX_NAME):
            return f"Index '{INDEX_NAME}' exists and is ready for ingestion."
        return f"Index '{INDEX_NAME}' does not exist. Run ingestion to create it."
    except ValueError as e:
        return str(e)


def load_documents(docs_path: str | None = None) -> tuple[list, str]:
    """
    Load documents from the docs folder.
    Returns (documents, status_message).
    """
    path = docs_path or DOCS_PATH
    path_obj = Path(path)

    if not path_obj.is_dir():
        path_obj.mkdir(parents=True, exist_ok=True)
        sample_path = path_obj / "sample.txt"
        sample_path.write_text(
            "This is a sample document for the RAG tutorial.\n"
            "Add more .txt files to the docs folder to index them.",
            encoding="utf-8",
        )
        return [], f"Created {path} with sample.txt. Add more .txt files and run ingestion again."

    loader = DirectoryLoader(
        str(path_obj),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    documents = loader.load()
    return documents, f"Loaded {len(documents)} documents from {path}."


def run_ingestion(docs_path: str | None = None) -> str:
    """
    Run the full ingestion pipeline: load, split, embed, index.
    Returns a status message.
    """
    try:
        pc = get_pinecone_client()
    except ValueError as e:
        return str(e)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return "Error: OPENAI_API_KEY not set in environment."

    ensure_index_exists(pc)

    documents, load_msg = load_documents(docs_path)
    if not documents:
        return load_msg

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    # Enrich metadata for DocSearch (source_title, chunk_id, url_or_path, text for fetch)
    ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        ids.append(chunk_id)
        source = chunk.metadata.get("source", "")
        path_obj = Path(source)
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["source_title"] = path_obj.name if source else "unknown"
        chunk.metadata["source_type"] = path_obj.suffix.lstrip(".") or "text"
        chunk.metadata["url_or_path"] = source
        chunk.metadata["page"] = chunk.metadata.get("page", 0)
        chunk.metadata["section"] = chunk.metadata.get("section", "")
        chunk.metadata["text"] = chunk.page_content[:40000]  # Pinecone metadata limit ~40KB

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE,
    )
    vectorstore.add_documents(chunks, ids=ids)

    return f"Ingestion complete. Indexed {len(chunks)} chunks from {len(documents)} documents into Pinecone index '{INDEX_NAME}'."
