import json
import os
import pandas as pd
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
INDEX_NAME = os.getenv("FINANCIAL_PERFORMANCE_INDEX")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))

logging.info(f"Script started for index '{INDEX_NAME}'.")

# Initialize OpenSearch client
client = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    use_ssl=False,
    verify_certs=False
)

# Create hybrid index (knn + keyword)
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
                        "text_chunk": {"type": "text", "analyzer": "standard"},
                        "Domain": {"type": "keyword"},
                        "Category": {"type": "keyword"},
                        "Subject": {"type": "keyword"},
                        "Country": {"type": "keyword"},
                        "Measurement": {"type": "keyword"},
                        "Unit": {"type": "keyword"},
                        "Reference_year": {"type": "integer"},
                        "Value": {"type": "float"},
                        "URL": {"type": "keyword"},
                        "Description": {"type": "text"}
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

# Load embedding model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': device}
)

# Text splitter setup
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

# Embedding function
def embed_chunks(chunks, batch_embed_size=8):
    embeddings = []
    for i in range(0, len(chunks), batch_embed_size):
        batch = chunks[i:i+batch_embed_size]
        embeddings.extend(embedding_model.embed_documents(batch))
    return embeddings

# Create index if not exists
create_hybrid_index_if_not_exists(INDEX_NAME)

# Load dataset
csv_file = 'data/numerical/bso/financial_performance.csv'

try:
    df = pd.read_csv(csv_file).fillna("")

    actions = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Financial Performance records"):
        try:
            description = row.get("Description", "")
            chunks = text_splitter.split_text(description if description else row["Subject"])

            embeddings = embed_chunks(chunks)

            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc = {
                    "text_chunk": chunk,
                    "embedding": embedding,
                    "Domain": row["Domain"],
                    "Category": row["Category"],
                    "Subject": row["Subject"],
                    "Measurement": row["Measurement"],
                    "Country": row["Country"],
                    "Unit": row["Unit"],
                    "Reference_year": int(row["Reference year"]) if row["Reference year"] else None,
                    "Value": float(row["Value"]) if row["Value"] else None,
                    "URL": row["URL"],
                    "Description": description
                }

                actions.append({
                    "_index": INDEX_NAME,
                    "_source": doc
                })

            if len(actions) >= 5000:
                helpers.bulk(client, actions, request_timeout=60)
                actions.clear()

        except Exception as record_err:
            logging.error("Error processing record with Subject '%s', Country '%s': %s",
                          row.get("Subject", "unknown"), row.get("Country", "unknown"), str(record_err))

    # Insert remaining data
    if actions:
        helpers.bulk(client, actions, request_timeout=60)

except FileNotFoundError as file_err:
    logging.error("Data file not found: %s", str(file_err))
except Exception as e:
    logging.error("Unexpected error during ingestion: %s", str(e))

logging.info(f"Insertion completed for index '{INDEX_NAME}'.")