import os
import json
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from tqdm import tqdm
from dotenv import load_dotenv
import logging

# Load environment variables from .env
load_dotenv()

# Logging configuration
logging.basicConfig(
    filename="logs/milvus_worker.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.info("Milvus insertion script for EPREL images started.")

# Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION")) 
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME")
DATA_DIR = "data/eprel"
COLLECTION_NAME = "eprel_images"

# Connect to Milvus
try:
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logging.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}.")
except Exception as conn_err:
    logging.error(f"Failed to connect to Milvus: {conn_err}")
    raise

# Check if collection exists; otherwise create it
try:
    if not utility.has_collection(COLLECTION_NAME):
        fields = [
            FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=500),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIMENSION),
            FieldSchema("title", DataType.VARCHAR, max_length=500),
            FieldSchema("url", DataType.VARCHAR, max_length=1000),
            FieldSchema("categories", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10, max_length=100),
            FieldSchema("dataset", DataType.VARCHAR, max_length=100),
            FieldSchema("subcategory", DataType.VARCHAR, max_length=100),
        ]

        schema = CollectionSchema(fields, description="EPREL product imagery")
        collection = Collection(name=COLLECTION_NAME, schema=schema)

        # Create HNSW index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logging.info(f"Collection '{COLLECTION_NAME}' created with HNSW index.")
    else:
        collection = Collection(COLLECTION_NAME)
        logging.info(f"Collection '{COLLECTION_NAME}' already exists.")
except Exception as coll_err:
    logging.error(f"Error checking/creating collection '{COLLECTION_NAME}': {coll_err}")
    raise

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    logging.info(f"CLIP model '{CLIP_MODEL_NAME}' loaded successfully on {device}.")
except Exception as model_err:
    logging.error(f"Failed to load CLIP model '{CLIP_MODEL_NAME}': {model_err}")
    raise

def generate_multimodal_embedding(image_path, text):
    image = Image.open(image_path).convert("RGB")
    # Process image separately
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    # Process text separately
    text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_emb = model.get_image_features(**image_inputs)
        text_emb = model.get_text_features(**text_inputs)
    # Normalize embeddings
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    # Combine embeddings (average)
    combined_emb = (image_emb + text_emb) / 2
    combined_emb = combined_emb.cpu().numpy()[0]
    # Final normalization
    combined_emb /= np.linalg.norm(combined_emb)
    return combined_emb.tolist()

# Prepare data entries
entries = []

try:
    # Traverse all subfolders and files
    for subcategory in tqdm(os.listdir(DATA_DIR), desc="EPREL Subcategories"):
        subcat_path = os.path.join(DATA_DIR, subcategory)
        if os.path.isdir(subcat_path):
            for file in os.listdir(subcat_path):
                if file.lower().endswith(".png"):
                    base_name = os.path.splitext(file)[0]
                    image_path = os.path.join(subcat_path, file)
                    json_path = os.path.join(subcat_path, f"{base_name}.json")

                    if not os.path.exists(json_path):
                        logging.warning(f"Metadata not found for {image_path}, skipping.")
                        continue

                    with open(json_path, 'r') as f:
                        metadata = json.load(f)

                    # Prepare descriptive text for embedding
                    descriptive_text = metadata.get("title", "") + " " + " ".join(metadata.get("categories", []))

                    # Generate combined embedding
                    embedding = generate_multimodal_embedding(image_path, descriptive_text)

                    entry_id = f"{subcategory}_{base_name}"

                    entry = [
                        entry_id,
                        embedding,
                        metadata.get("title", ""),
                        metadata.get("url", ""),
                        metadata.get("categories", []),
                        "eprel",
                        subcategory,
                    ]

                    entries.append(entry)

except Exception as dir_err:
    logging.error(f"Error traversing data directory '{DATA_DIR}': {dir_err}")
    raise

# Insert data into Milvus in batches
BATCH_SIZE = 100
try:
    for i in tqdm(range(0, len(entries), BATCH_SIZE), desc="Inserting batches"):
        batch = entries[i:i+BATCH_SIZE]
        collection.insert(list(zip(*batch)))
    collection.flush()
    logging.info(f"Inserted {len(entries)} EPREL images into Milvus (multimodal embeddings).")
except Exception as insert_err:
    logging.error(f"Error inserting data into Milvus: {insert_err}")
    raise

# Load collection into memory
try:
    collection.load()
    logging.info(f"Collection '{COLLECTION_NAME}' loaded into memory successfully.")
except Exception as load_err:
    logging.error(f"Error loading collection '{COLLECTION_NAME}' into memory: {load_err}")

logging.info("Milvus insertion script for EPREL images completed.")
