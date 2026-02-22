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

### Optional: LLM thinking and absorption

- **Context absorption**: Each response shows an expandable "📥 Context absorbed" section with the retrieved documents used to answer.
- **Reasoning/thinking**: To show the model's reasoning (o1, o3, o4-mini), set in `.env`:
  ```
  REASONING_MODEL=o4-mini
  ```
  This uses OpenAI's Responses API and displays thinking steps before the answer.

---

## AWS Alternative

An AWS-native version uses Amazon Bedrock (embeddings + LLM), Amazon OpenSearch (vector store), and a simple requests+BeautifulSoup crawler (no Tavily).

### AWS Setup

1. **Environment** – add to `.env`:
   - `OPENSEARCH_HOST` – OpenSearch domain host (e.g. `search-mydomain.us-east-1.es.amazonaws.com`)
   - `AWS_REGION` – e.g. `us-east-1`
   - AWS credentials (via `~/.aws/credentials` or env vars)

2. **OpenSearch** – Create an OpenSearch domain with k-NN enabled (or use OpenSearch Serverless with `OPENSEARCH_SERVERLESS=true`).

3. **Bedrock** – Enable access to Titan Embed and Claude in the Bedrock console.

### AWS Usage

```bash
# Ingest (crawls, embeds with Bedrock, upserts to OpenSearch)
python ingestion_aws.py

# Start chatbot (Bedrock + OpenSearch)
streamlit run app_aws.py
```
