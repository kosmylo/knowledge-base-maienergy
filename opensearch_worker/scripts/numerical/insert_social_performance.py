import json
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from opensearchpy import OpenSearch, helpers
from tqdm import tqdm
import torch
from dotenv import load_dotenv

load_dotenv()

# OpenSearch configuration
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
INDEX_NAME = os.getenv("SOCIAL_PERFORMANCE_INDEX")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))

# Initialize OpenSearch client
client = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    use_ssl=False,
    verify_certs=False
)

# Create hybrid index (knn + keyword)
def create_hybrid_index_if_not_exists(index_name):
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
        print(f"Created hybrid index '{index_name}' with KNN and keyword mappings.")
    else:
        print(f"Hybrid index '{index_name}' already exists.")

# Load embedding model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': device}
)

# Text splitter setup for descriptions
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

# Embedding function
def embed_text(texts, batch_embed_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_embed_size):
        batch = texts[i:i+batch_embed_size]
        embeddings.extend(embedding_model.embed_documents(batch))
    return embeddings

# Create index if not exists
create_hybrid_index_if_not_exists(INDEX_NAME)

# Load dataset
csv_file = 'data/numerical/bso/social_performance.csv'
df = pd.read_csv(csv_file).fillna("")

actions = []
for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Social Performance records"):
    description = row.get("Description", "")
    chunks = text_splitter.split_text(description if description else row["Subject"])

    embeddings = embed_text(chunks)

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

# Insert remaining data
if actions:
    helpers.bulk(client, actions, request_timeout=60)

print(f"Insertion completed for index '{INDEX_NAME}'.")