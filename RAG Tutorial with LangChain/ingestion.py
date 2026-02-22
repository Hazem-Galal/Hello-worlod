"""
RAG Tutorial - Document Ingestion Pipeline

Loads documents from a local folder, splits into chunks, and indexes them in Pinecone.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Configuration
INDEX_NAME = "rag-tutorial"
NAMESPACE = "default"
DOCS_PATH = "./docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_DIM = 1536  # text-embedding-3-small


def ensure_index_exists(pc: Pinecone) -> None:
    """Create Pinecone index if it does not exist."""
    if pc.has_index(INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists")
        return

    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index created. Wait a few seconds for it to be ready.")


def load_documents() -> list:
    """Load documents from the docs folder."""
    if not os.path.isdir(DOCS_PATH):
        print(f"Creating {DOCS_PATH} and adding a sample file. Add your .txt files there.")
        os.makedirs(DOCS_PATH, exist_ok=True)
        sample_path = os.path.join(DOCS_PATH, "sample.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(
                "This is a sample document for the RAG tutorial.\n"
                "Add more .txt files to the docs folder to index them."
            )
        print(f"Created {sample_path}")

    loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def main() -> None:
    """Run the ingestion pipeline."""
    api_key = os.getenv("PINECONE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: PINECONE_API_KEY not set in .env")
        sys.exit(1)
    if not openai_key:
        print("Error: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    print("Starting ingestion pipeline...")

    pc = Pinecone(api_key=api_key)
    ensure_index_exists(pc)

    documents = load_documents()
    if not documents:
        print("No documents to process. Add .txt files to ./docs")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
    )
    print(f"Ingestion complete. Indexed {len(chunks)} chunks in Pinecone.")
    print("You can now run your RAG application.")


if __name__ == "__main__":
    main()
