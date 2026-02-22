# RAG Ingestion CopilotKit Backend

A CopilotKit backend that exposes the RAG document ingestion pipeline as an AI agent. Users can interact via chat to ingest documents, check index status, and load documents.

## Features

- **run_ingestion** – Run the full pipeline: load .txt files from docs, split into chunks, index in Pinecone
- **load_documents** – Load and count documents without indexing
- **get_index_status** – Check if the Pinecone index exists

## Setup

### 1. Install dependencies

```bash
cd ingestion_backend
uv sync
# or: pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Add OPENAI_API_KEY and PINECONE_API_KEY
# Or use the parent RAG Tutorial .env
```

### 3. Start the backend

Run from the **RAG Tutorial** folder so `./docs` and `.env` resolve correctly:

```bash
cd "RAG Tutorial with LangChain"
cd ingestion_backend
uv run python -m ingestion_backend.server
```

The backend runs at `http://localhost:8001` (or PORT from .env).

## Connect a CopilotKit UI

Point your CopilotKit frontend to this backend:

- **REMOTE_ACTION_URL**: `http://localhost:8001/copilotkit`
- **Agent name**: `ingestion_agent`

Example prompts:
- "Run the ingestion pipeline"
- "Ingest my documents"
- "Check the index status"
- "How many documents are in the docs folder?"

## Project structure

```
ingestion_backend/
├── ingestion_backend/
│   ├── __init__.py
│   ├── agent.py      # LangGraph agent with ingestion tools
│   ├── ingestion.py  # Ingestion pipeline logic
│   └── server.py    # FastAPI + CopilotKit
├── pyproject.toml
├── .env.example
└── README.md
```
