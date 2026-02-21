"""
OpenSearch Retriever for LangChain RAG (AWS).

Wraps OpenSearch k-NN search with Bedrock embeddings as a LangChain BaseRetriever.
"""

import json
import os
from typing import Any, List

import boto3
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from opensearchpy import OpenSearch, RequestsHttpConnection
from pydantic import Field
from requests_aws4auth import AWS4Auth

# Config (must match ingestion_aws.py)
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "langchain-docs")
BEDROCK_MODEL = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM = 1024
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def get_opensearch_client() -> OpenSearch:
    """Return OpenSearch client."""
    host = os.getenv("OPENSEARCH_HOST", "").replace("https://", "").replace("http://", "").rstrip("/")
    if not host:
        raise ValueError("OPENSEARCH_HOST not set")

    use_serverless = os.getenv("OPENSEARCH_SERVERLESS", "false").lower() == "true"
    service = "aoss" if use_serverless else "es"
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        AWS_REGION,
        service,
        session_token=credentials.token,
    )
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def get_embedding(text: str) -> List[float]:
    """Get embedding from Bedrock Titan."""
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    body = json.dumps({"inputText": text[:8000], "dimensions": EMBEDDING_DIM})
    resp = bedrock.invoke_model(modelId=BEDROCK_MODEL, body=body)
    out = json.loads(resp["body"].read())
    return out["embedding"]


class OpenSearchRetriever(BaseRetriever):
    """LangChain retriever that queries OpenSearch with Bedrock embeddings."""

    client: Any = Field(default=None, exclude=True)
    top_k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        """Retrieve relevant documents from OpenSearch for the given query."""
        query_vector = get_embedding(query)

        body = {
            "size": self.top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": self.top_k,
                    }
                }
            },
            "_source": ["content", "source"],
        }

        client = self.client or get_opensearch_client()
        result = client.search(index=INDEX_NAME, body=body)
        documents = []

        for hit in result.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            content = src.get("content", "")
            source = src.get("source", "")
            doc_id = hit.get("_id", "")
            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": source, "id": doc_id},
                    )
                )

        return documents
