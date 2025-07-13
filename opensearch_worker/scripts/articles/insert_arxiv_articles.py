import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from opensearchpy import OpenSearch, helpers
from tqdm import tqdm
import torch
from dotenv import load_dotenv
import logging

load_dotenv()

# Logging configuration
logging.basicConfig(
    filename="logs/opensearch_worker.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# OpenSearch configuration
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
INDEX_NAME = os.getenv("ARXIV_INDEX")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION")) # for 'all-MiniLM-L6-v2'

logging.info(f"Script started for index '{INDEX_NAME}'.")

# Initialize OpenSearch client
client = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    use_ssl=False,
    verify_certs=False
)

# Check and create index with hybrid (knn + keyword) mapping
def create_hybrid_index_if_not_exists(index_name):
    try:
        if not client.indices.exists(index=index_name):
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "number_of_shards": 1,
                        "number_of_replicas": 1
                    }
                },
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": EMBEDDING_DIMENSION,
                            "method": {
                                "name": "hnsw",
                                "engine": "nmslib",
                                "space_type": "cosinesimil"
                            }
                        },
                        "title": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "url": {"type": "keyword"},
                        "document_type": {"type": "keyword"},
                        "text_chunk": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "chunk_id": {"type": "integer"}
                    }
                }
            }
            client.indices.create(index=index_name, body=index_body)
            logging.info(f"Created hybrid index '{index_name}' with KNN and keyword mappings.")
        else:
            logging.info(f"Hybrid index '{index_name}' already exists.")
    except Exception as e:
        logging.error("Error creating index '%s': %s", index_name, str(e))
        raise

# Load embedding model (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': device}
)

# Setup text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " "]
)

# Embedding function with batching
def embed_chunks(chunks, batch_embed_size=8):
    embeddings = []
    for i in range(0, len(chunks), batch_embed_size):
        batch_chunks = chunks[i:i+batch_embed_size]
        embeddings.extend(embedding_model.embed_documents(batch_chunks))
    return embeddings

# Create the index if not already present
create_hybrid_index_if_not_exists(INDEX_NAME)

# Insert articles
try:
    with open('data/articles/arxiv.jsonl', 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing arxiv articles"):
            try:
                doc = json.loads(line)
                chunks = text_splitter.split_text(doc["content"])
                embeddings = embed_chunks(chunks, batch_embed_size=8)

                actions = [
                    {
                        "_index": INDEX_NAME,
                        "_source": {
                            "title": doc["title"],
                            "url": doc["url"],
                            "document_type": doc["document_type"],
                            "chunk_id": idx,
                            "text_chunk": chunk,
                            "embedding": embedding
                        }
                    }
                    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                ]

                helpers.bulk(client, actions, request_timeout=60)

            except Exception as doc_err:
                logging.error(
                    "Failed processing article '%s' (%s): %s",
                    doc.get("title", "unknown"),
                    doc.get("url", "unknown"),
                    str(doc_err)
                )
                
except FileNotFoundError as file_err:
    logging.error("Data file not found: %s", str(file_err))
except Exception as e:
    logging.error("Unexpected error: %s", str(e))

logging.info(f"Insertion completed for index '{INDEX_NAME}'.")
