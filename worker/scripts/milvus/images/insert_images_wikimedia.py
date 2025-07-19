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

logging.info("Milvus insertion script for Wikimedia images started.")

# Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION"))
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME")
DATA_DIR = "data/wikimedia"
COLLECTION_NAME = "wikimedia_images"

# Connect to Milvus
try:
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logging.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}.")
except Exception as conn_err:
    logging.error(f"Failed to connect to Milvus: {conn_err}")
    raise

# Check if collection exists
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

        schema = CollectionSchema(fields, description="Wikimedia multimodal imagery")
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

def generate_multimodal_embedding(image_path, descriptive_text):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[descriptive_text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_emb = model.get_image_features(pixel_values=inputs['pixel_values'])
        text_emb = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        embedding = (image_emb + text_emb) / 2
        embedding = embedding.cpu().numpy()[0]
        embedding /= np.linalg.norm(embedding)  # Normalize
    return embedding.tolist()

entries = []

try:
    # Traverse Wikimedia data folders
    for subcategory in tqdm(os.listdir(DATA_DIR), desc="Wikimedia Subcategories"):
        subcat_path = os.path.join(DATA_DIR, subcategory)
        if os.path.isdir(subcat_path):
            for file in os.listdir(subcat_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif")):
                    base_name = os.path.splitext(file)[0]
                    image_path = os.path.join(subcat_path, file)
                    json_path = os.path.join(subcat_path, f"{base_name}.json")

                    if not os.path.exists(json_path):
                        logging.warning(f"Metadata not found for {image_path}, skipping.")
                        continue

                    try:
                        with open(json_path, 'r') as f:
                            metadata = json.load(f)

                        descriptive_text = metadata.get("title", "") + " " + ", ".join(metadata.get("categories", []))

                        embedding = generate_multimodal_embedding(image_path, descriptive_text)

                        entry_id = f"{subcategory}_{base_name}"

                        entry = [
                            entry_id,
                            embedding,
                            metadata.get("title", ""),
                            metadata.get("url", ""),
                            metadata.get("categories", []),
                            "wikimedia",
                            subcategory,
                        ]

                        entries.append(entry)

                    except Exception as img_err:
                        logging.error(f"Error processing image '{image_path}': {img_err}")

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
    logging.info(f"Inserted {len(entries)} Wikimedia images into Milvus (multimodal embeddings).")
except Exception as insert_err:
    logging.error(f"Error inserting data into Milvus: {insert_err}")
    raise

# Load collection into memory (optional, recommended)
try:
    collection.load()
    logging.info(f"Collection '{COLLECTION_NAME}' loaded into memory successfully.")
except Exception as load_err:
    logging.error(f"Error loading collection '{COLLECTION_NAME}' into memory: {load_err}")

logging.info("Milvus insertion script for Wikimedia images completed.")