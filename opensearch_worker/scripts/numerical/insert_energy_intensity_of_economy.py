import json
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from opensearchpy import OpenSearch, helpers
from country_mapping import country_code_to_name
from tqdm import tqdm
import torch
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
INDEX_NAME = os.getenv("ENERGY_INTENSITY_ECONOMY_INDEX")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))

# OpenSearch client initialization
client = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    use_ssl=False,
    verify_certs=False
)

# Create hybrid index if not exists
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
                    "freq": {"type": "keyword"},
                    "nrg_bal": {"type": "keyword"},
                    "unit": {"type": "keyword"},
                    "country": {"type": "keyword"},
                    "period": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "value": {"type": "float"},
                    "URL": {"type": "keyword"},
                    "dataset_name": {"type": "keyword"},
                    "description": {"type": "text"}
                }
            }
        }
        client.indices.create(index=index_name, body=index_body)
        print(f"Created hybrid index '{index_name}'.")
    else:
        print(f"Hybrid index '{index_name}' already exists.")

# Load embedding model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': device}
)

# Text splitter configuration
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

# Load dataset and metadata
csv_file = 'data/numerical/eurostat/energy_intensity_of_economy.csv'
metadata_file = 'data/numerical/eurostat/energy_intensity_of_economy_metadata.json'

df = pd.read_csv(csv_file).fillna("")

with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

actions = []

for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Energy Intensity records"):
    description = metadata.get("description", "")
    geo_time_period = row["geo\\TIME_PERIOD"]
    country_name = country_code_to_name.get(geo_time_period, geo_time_period)
    subject = (f"Energy intensity of economy for balance code {row['nrg_bal']} "
               f"in country {country_name} during period {row['period']} ({row['year']}). "
               f"Unit: {row['unit']}, Value: {row['value']}.")

    full_text = f"{subject} {description}"
    chunks = text_splitter.split_text(full_text)

    embeddings = embed_text(chunks)

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc = {
            "text_chunk": chunk,
            "embedding": embedding,
            "freq": row["freq"],
            "nrg_bal": row["nrg_bal"],
            "unit": row["unit"],
            "country": country_name,
            "period": row["period"],
            "year": int(row["year"]) if row["year"] else None,
            "value": float(row["value"]) if row["value"] else None,
            "URL": metadata["url"],
            "dataset_name": metadata["dataset_name"],
            "description": description
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