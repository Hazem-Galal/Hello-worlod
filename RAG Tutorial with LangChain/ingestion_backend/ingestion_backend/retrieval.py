"""
Retrieval logic for DocSearch Copilot.
Provides vector_search, fetch_document_chunk, and list_sources.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from .ingestion import INDEX_NAME, NAMESPACE, get_pinecone_client


def _get_vectorstore() -> PineconeVectorStore:
    """Get Pinecone vector store for retrieval."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE,
    )


def _doc_to_passage(doc: Any, score: float) -> dict:
    """Convert LangChain Document to passage dict for DocSearch format."""
    meta = doc.metadata or {}
    return {
        "id": meta.get("chunk_id", ""),
        "text": doc.page_content,
        "source_title": meta.get("source_title", meta.get("source", "unknown")),
        "source_type": meta.get("source_type", "text"),
        "url_or_path": meta.get("url_or_path", meta.get("source", "")),
        "page": meta.get("page", 0),
        "section": meta.get("section", ""),
        "chunk_id": meta.get("chunk_id", ""),
        "score": float(score),
    }


def vector_search(query: str, top_k: int = 8, filters: dict | None = None) -> list[dict]:
    """
    Search the vector index for relevant passages.
    Returns list of passages with metadata: id, text, source_title, source_type,
    url_or_path, page, section, chunk_id, score.
    """
    try:
        vs = _get_vectorstore()
    except Exception as e:
        return [{"error": str(e), "text": ""}]

    # Pinecone filter format: {"metadata_field": {"$eq": "value"}}
    filter_dict = filters if filters else {}
    try:
        docs_with_scores = vs.similarity_search_with_score(
            query, k=top_k, filter=filter_dict if filter_dict else None
        )
    except Exception as e:
        return [{"error": str(e), "text": ""}]

    return [_doc_to_passage(doc, score) for doc, score in docs_with_scores]


def fetch_document_chunk(chunk_id: str) -> dict | None:
    """
    Fetch full chunk text by chunk_id (Pinecone record id).
    Returns chunk dict or None if not found.
    """
    try:
        pc = get_pinecone_client()
        index = pc.Index(INDEX_NAME)
        result = index.fetch(ids=[chunk_id], namespace=NAMESPACE)
        if not result.get("vectors") or chunk_id not in result["vectors"]:
            return None
        rec = result["vectors"][chunk_id]
        meta = rec.get("metadata", {})
        return {
            "id": chunk_id,
            "text": meta.get("text", ""),  # Pinecone may store text in metadata
            "source_title": meta.get("source_title", ""),
            "url_or_path": meta.get("url_or_path", ""),
            "page": meta.get("page", 0),
            "section": meta.get("section", ""),
        }
    except Exception:
        return None


def list_sources() -> list[dict]:
    """
    List available document sources from the index.
    Does a broad search and extracts unique sources from results.
    """
    passages = vector_search("document content", top_k=50)
    if passages and passages[0].get("error"):
        return [{"error": passages[0]["error"]}]

    seen = set()
    sources = []
    for p in passages:
        key = (p.get("source_title", ""), p.get("url_or_path", ""))
        if key not in seen and p.get("source_title"):
            seen.add(key)
            sources.append({
                "title": p.get("source_title", ""),
                "url_or_path": p.get("url_or_path", ""),
                "source_type": p.get("source_type", "text"),
            })
    return sources
