"""
LangChain Documentation Ingestion Pipeline (AWS)

Crawls docs via requests+BeautifulSoup, embeds with Amazon Bedrock Titan,
and upserts to Amazon OpenSearch Service (vector search).
"""

import json
import os
import sys
import time
import uuid
from typing import List
from urllib.parse import urljoin, urlparse

# Ensure we're using the project venv
try:
    from dotenv import load_dotenv
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    print(
        "Error: Missing dependency. Run with the project venv activated:\n"
        "  python ingestion_aws.py\n"
        "If venv is active, run: pip install -r requirements.txt"
    )
    sys.exit(1)

try:
    import boto3
    import requests
    from bs4 import BeautifulSoup
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth
except ImportError as e:
    print(
        f"Error: Missing AWS dependency: {e}\n"
        "Run: pip install boto3 opensearch-py requests-aws4auth"
    )
    sys.exit(1)

from logger import log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configuration
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "langchain-docs")
DOCS_URL = os.getenv("DOCS_URL", "https://docs.langchain.com/")
BATCH_SIZE = 25  # Bedrock batch limit for embeddings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CRAWL_LIMIT = 50
CRAWL_MAX_DEPTH = 3
EMBEDDING_DIM = 1024
BEDROCK_MODEL = "amazon.titan-embed-text-v2:0"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def get_bedrock_client():
    """Return Bedrock runtime client."""
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def get_opensearch_client():
    """Return OpenSearch client (supports managed and serverless)."""
    host = os.getenv("OPENSEARCH_HOST")
    if not host:
        log_error("OPENSEARCH_HOST not set (e.g. search-xxx.us-east-1.es.amazonaws.com)")
        sys.exit(1)

    # Remove protocol if present
    host = host.replace("https://", "").replace("http://", "").rstrip("/")

    use_serverless = os.getenv("OPENSEARCH_SERVERLESS", "false").lower() == "true"
    if use_serverless:
        # OpenSearch Serverless uses SigV4
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            AWS_REGION,
            "aoss",
            session_token=credentials.token,
        )
        client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
    else:
        # Managed OpenSearch
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            AWS_REGION,
            "es",
            session_token=credentials.token,
        )
        client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
    return client


def ensure_index_exists(client: OpenSearch) -> None:
    """Create OpenSearch index with k-NN vector field if it does not exist."""
    if client.indices.exists(index=INDEX_NAME):
        log_info("Index '%s' already exists", INDEX_NAME)
        return

    log_header("Creating OpenSearch index: %s", INDEX_NAME)
    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "source": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "space_type": "cosinesimil",
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "parameters": {"ef_construction": 128, "m": 24},
                    },
                },
            }
        },
    }
    client.indices.create(index=INDEX_NAME, body=body)
    log_success("Index created. Waiting 5s for readiness...")
    time.sleep(5)


def crawl_docs() -> List[Document]:
    """Crawl documentation using requests + BeautifulSoup (same-site BFS)."""
    log_header("Crawling documentation from %s (limit=%d, max_depth=%d)", DOCS_URL, CRAWL_LIMIT, CRAWL_MAX_DEPTH)

    base_domain = urlparse(DOCS_URL).netloc
    scheme = urlparse(DOCS_URL).scheme
    base_url = f"{scheme}://{base_domain}"

    visited = set()
    to_visit = [(DOCS_URL, 0)]
    documents = []

    while to_visit and len(documents) < CRAWL_LIMIT:
        url, depth = to_visit.pop(0)
        if url in visited or depth > CRAWL_MAX_DEPTH:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; DocBot/1.0)"})
            resp.raise_for_status()
        except Exception as e:
            log_warning("Failed to fetch %s: %s", url, e)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script/style
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if len(text) > 100:
            documents.append(Document(page_content=text, metadata={"source": url}))
            log_info("  Crawled: %s (%d chars)", url, len(text))

        if depth < CRAWL_MAX_DEPTH:
            for a in soup.find_all("a", href=True):
                href = a["href"].split("#")[0].rstrip("/")
                if not href or href.startswith("mailto:") or href.startswith("javascript:"):
                    continue
                full_url = urljoin(url, href)
                parsed = urlparse(full_url)
                if parsed.netloc == base_domain and full_url not in visited:
                    to_visit.append((full_url, depth + 1))

    log_success("Crawled %d pages", len(documents))
    for i, doc in enumerate(documents, 1):
        log_info("  %d. %s", i, doc.metadata.get("source", "unknown"))
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks."""
    log_header("Splitting documents (chunk_size=%d, overlap=%d)", CHUNK_SIZE, CHUNK_OVERLAP)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    log_success("Created %d chunks", len(chunks))
    return chunks


def embed_texts(bedrock, texts: List[str]) -> List[List[float]]:
    """Get embeddings from Bedrock Titan for a batch of texts."""
    embeddings = []
    for text in texts:
        body = json.dumps({"inputText": text[:8000], "dimensions": EMBEDDING_DIM})
        resp = bedrock.invoke_model(modelId=BEDROCK_MODEL, body=body)
        out = json.loads(resp["body"].read())
        embeddings.append(out["embedding"])
    return embeddings


def upsert_to_opensearch(chunks: List[Document], client: OpenSearch, bedrock) -> None:
    """Upsert document chunks to OpenSearch with Bedrock embeddings."""
    log_header("Upserting %d chunks to OpenSearch (index=%s)", len(chunks), INDEX_NAME)

    total_upserted = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.page_content for c in batch]
        embeddings = embed_texts(bedrock, texts)

        for chunk, embedding in zip(batch, embeddings):
            doc_id = str(uuid.uuid4())
            source = chunk.metadata.get("source", "unknown")[:500]
            body = {
                "content": chunk.page_content,
                "source": source,
                "embedding": embedding,
            }
            client.index(index=INDEX_NAME, id=doc_id, body=body)

        total_upserted += len(batch)
        log_info("Upserted batch %d-%d (%d total)", i + 1, i + len(batch), total_upserted)

    client.indices.refresh(index=INDEX_NAME)
    log_success("Upserted %d records total", total_upserted)
    log_info("Waiting 5s for indexing before search is available...")
    time.sleep(5)


def main() -> None:
    """Run the full ingestion pipeline."""
    if not os.getenv("AWS_REGION"):
        os.environ.setdefault("AWS_REGION", AWS_REGION)

    if not os.getenv("OPENSEARCH_HOST"):
        log_error("OPENSEARCH_HOST not set in environment")
        log_info("Example: OPENSEARCH_HOST=search-mydomain.us-east-1.es.amazonaws.com")
        return

    log_header("Starting LangChain documentation ingestion (AWS)")

    bedrock = get_bedrock_client()
    client = get_opensearch_client()
    ensure_index_exists(client)

    documents = crawl_docs()
    log_info("Total pages crawled: %d", len(documents))
    if not documents:
        log_error("No documents to process. Exiting.")
        return

    chunks = split_documents(documents)
    if not chunks:
        log_error("No chunks created. Exiting.")
        return

    upsert_to_opensearch(chunks, client, bedrock)

    log_success("Ingestion complete. Run 'streamlit run app_aws.py' to start the chatbot.")


if __name__ == "__main__":
    main()
