# DocSearch UI

Next.js frontend for the RAG DocSearch Copilot. Connects to the ingestion backend for document search with citations and clickable follow-up suggestions.

## Prerequisites

1. **RAG ingestion backend** running on port 8001:
   ```bash
   cd "../RAG Tutorial with LangChain/ingestion_backend"
   pip install -e .
   python -m ingestion_backend.server
   ```

2. **Ingest documents** via the ingestion agent before searching.

## Run

```bash
pnpm install
pnpm dev
```

Open http://localhost:3000

## Build

```bash
pnpm build
pnpm start
```
