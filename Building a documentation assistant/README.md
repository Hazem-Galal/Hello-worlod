# LangChain Documentation Assistant

A RAG chatbot that answers questions about LangChain using documentation crawled via Tavily and stored in Pinecone.

## Setup

1. **Environment**: Ensure `.env` has:
   - `PINECONE_API_KEY`
   - `TAVILY_API_KEY`
   - `OPENAI_API_KEY`

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Ingest documentation (one-time or periodic)

```bash
python ingestion.py
```

This will:
- Create the Pinecone index `langchain-docs` if it doesn't exist
- Crawl https://python.langchain.com via TavilyCrawl
- Split into chunks and upsert to Pinecone

### 2. Start the chatbot

```bash
streamlit run app.py
```

Ask questions about LangChain; answers are retrieved from the indexed documentation.
