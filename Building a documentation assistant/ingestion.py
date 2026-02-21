"""
LangChain Documentation Ingestion Pipeline

Crawls LangChain docs via Tavily, splits into chunks, and upserts to Pinecone.
"""

import os
import sys
import time
import uuid
from typing import List

# Ensure we're using the project venv (python3 on Windows often points elsewhere)
try:
    from dotenv import load_dotenv
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    print(
        "Error: Missing dependency. Run with the project venv activated:\n"
        "  python ingestion.py   (not python3)\n"
        "If venv is active, run: pip install -r requirements.txt"
    )
    sys.exit(1)
from langchain_tavily import TavilyCrawl
from pinecone import Pinecone, IndexEmbed, CloudProvider, AwsRegion

from logger import log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configuration
INDEX_NAME = "langchain-docs"
NAMESPACE = "langchain_documentation"
DOCS_URL = "https://docs.langchain.com/"
BATCH_SIZE = 96
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def ensure_index_exists(pc: Pinecone) -> None:
    """Create Pinecone index with integrated embeddings if it does not exist."""
    if pc.has_index(INDEX_NAME):
        log_info("Index '%s' already exists", INDEX_NAME)
        return

    log_header("Creating Pinecone index: %s", INDEX_NAME)
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_EAST_1,
        embed=IndexEmbed(
            model="llama-text-embed-v2",
            metric="cosine",
            field_map={"text": "content"},
        ),
    )
    log_success("Index created. Waiting 10s for readiness...")
    time.sleep(10)


def crawl_docs() -> List[Document]:
    """Crawl LangChain documentation using TavilyCrawl."""
    log_header("Crawling LangChain documentation from %s", DOCS_URL)

    tavily_crawl = TavilyCrawl()
    result = tavily_crawl.invoke({
        "url": DOCS_URL,
        "instructions": "Extract all documentation pages, tutorials, and API references. Include code examples and explanatory text.",
        "limit": 150,           # Total links to process (default 50). Increase for more pages.
        "max_depth": 5,         # Link levels from root (1–5). 5 = deepest.
        "max_breadth": 50,      # Links to follow per page (default 20, max 500).
        "search_depth": "advanced",
        "search_type": "site",
        "search_query": "LangChain documentation",
        "search_filters": {
            "language": "en",
            "country": "us",
            "category": "documentation",
        },
    })

    results = result.get("results", [])
    if not results:
        log_warning("No results returned from TavilyCrawl")
        return []

    documents = []
    for item in results:
        url = item.get("url", "unknown")
        raw_content = item.get("raw_content", "")
        if raw_content.strip():
            documents.append(
                Document(page_content=raw_content, metadata={"source": url})
            )

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


def upsert_to_pinecone(chunks: List[Document], index) -> None:
    """Upsert document chunks to Pinecone in batches of 96."""
    log_header("Upserting %d chunks to Pinecone (namespace=%s)", len(chunks), NAMESPACE)

    total_upserted = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        records = []
        for chunk in batch:
            record_id = str(uuid.uuid4())
            source = chunk.metadata.get("source", "unknown")
            records.append({
                "_id": record_id,
                "content": chunk.page_content,
                "source": source[:500],  # Pinecone metadata limit
            })
        index.upsert_records(NAMESPACE, records)
        total_upserted += len(records)
        log_info("Upserted batch %d-%d (%d total)", i + 1, i + len(batch), total_upserted)

    log_success("Upserted %d records total", total_upserted)
    log_info("Waiting 10s for indexing before search is available...")
    time.sleep(10)


def main() -> None:
    """Run the full ingestion pipeline."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        log_error("PINECONE_API_KEY not set in environment")
        return

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        log_error("TAVILY_API_KEY not set in environment")
        return

    log_header("Starting LangChain documentation ingestion")

    pc = Pinecone(api_key=api_key)
    ensure_index_exists(pc)

    documents = crawl_docs()
    pages_crawled = len(documents)
    log_info("Total pages crawled: %d", pages_crawled)
    if not documents:
        log_error("No documents to process. Exiting.")
        return

    chunks = split_documents(documents)
    if not chunks:
        log_error("No chunks created. Exiting.")
        return

    index = pc.Index(INDEX_NAME)
    upsert_to_pinecone(chunks, index)

    log_success("Ingestion complete. Run 'streamlit run app.py' to start the chatbot.")


if __name__ == "__main__":
    main()
