"""
Pinecone Retriever for LangChain RAG.

Wraps native Pinecone SDK search (with integrated embeddings) as a LangChain BaseRetriever.
"""

from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class PineconeRetriever(BaseRetriever):
    """LangChain retriever that queries Pinecone index with integrated embeddings."""

    index: Any = Field(exclude=True)  # Pinecone Index - excluded from serialization
    namespace: str = "langchain_documentation"
    top_k: int = 5
    use_rerank: bool = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        """Retrieve relevant documents from Pinecone for the given query."""
        candidates = self.top_k * 2 if self.use_rerank else self.top_k

        search_kwargs = {
            "namespace": self.namespace,
            "query": {
                "top_k": candidates,
                "inputs": {"text": query},
            },
        }
        if self.use_rerank:
            search_kwargs["rerank"] = {
                "model": "bge-reranker-v2-m3",
                "top_n": self.top_k,
                "rank_fields": ["content"],
            }

        result = self.index.search(**search_kwargs)

        documents = []
        hits = []
        if isinstance(result, dict):
            hits = result.get("result", {}).get("hits", result.get("hits", []))
        elif hasattr(result, "result"):
            hits = getattr(result.result, "hits", [])
        elif hasattr(result, "hits"):
            hits = result.hits

        for hit in hits[: self.top_k]:
            if isinstance(hit, dict):
                fields = hit.get("fields", {})
                content = fields.get("content", "")
                source = fields.get("source", "")
                doc_id = hit.get("_id", "")
            else:
                fields = getattr(hit, "fields", {})
                if hasattr(fields, "get"):
                    content = fields.get("content", "")
                    source = fields.get("source", "")
                else:
                    content = getattr(fields, "content", "")
                    source = getattr(fields, "source", "")
                doc_id = getattr(hit, "_id", "") or getattr(hit, "id", "")

            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": source, "id": doc_id},
                    )
                )

        return documents
