"""
CopilotKit FastAPI server for the RAG ingestion backend.
Exposes the ingestion agent at /copilotkit.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env: first from ingestion_backend, then from RAG Tutorial folder
load_dotenv()
# Parent of ingestion_backend/ingestion_backend/ is RAG Tutorial
rag_env = Path(__file__).resolve().parent.parent.parent / ".env"
if rag_env.exists():
    load_dotenv(rag_env)

from fastapi import FastAPI
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAGUIAgent

from .agent import graph
from .docsearch_agent import docsearch_graph


class IngestionAgent(LangGraphAGUIAgent):
    """Agent with dict_repr for CopilotKit runtime compatibility."""

    def dict_repr(self):
        return {
            "name": self.name,
            "description": self.description or "",
            "type": "langgraph",
        }


app = FastAPI(
    title="RAG Ingestion Backend",
    description="CopilotKit backend for document ingestion into Pinecone",
)

sdk = CopilotKitRemoteEndpoint(
    agents=[
        IngestionAgent(
            name="ingestion_agent",
            description="An agent that helps ingest documents into the RAG system. "
            "Can run the full ingestion pipeline, load documents, or check index status.",
            graph=graph,
        ),
        IngestionAgent(
            name="docsearch",
            description="Document search tool (not a chatbot). Searches indexed documents "
            "and returns grounded answers with citations.",
            graph=docsearch_graph,
        ),
    ],
)

add_fastapi_endpoint(app, sdk, "/copilotkit")


@app.get("/")
def root():
    """Root redirects to health and docs."""
    return {
        "service": "ingestion-backend",
        "docs": "http://127.0.0.1:8001/docs",
        "health": "http://127.0.0.1:8001/health",
        "copilotkit": "http://127.0.0.1:8001/copilotkit",
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "ingestion-backend"}


def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8001"))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run(
        "ingestion_backend.server:app",
        host=host,
        port=port,
        reload=True,
    )


if __name__ == "__main__":
    main()
