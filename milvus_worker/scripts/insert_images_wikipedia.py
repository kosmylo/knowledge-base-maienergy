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

# Load environment variables from .env
load_dotenv()

# Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION"))
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME")
DATA_DIR = "data/wikipedia"
COLLECTION_NAME = "wikipedia_images"

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

# Define schema if the collection does not exist
if not utility.has_collection(COLLECTION_NAME):
    fields = [
        FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=500),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIMENSION),
        FieldSchema("title", DataType.VARCHAR, max_length=500),
        FieldSchema("url", DataType.VARCHAR, max_length=1000),
        FieldSchema("caption", DataType.VARCHAR, max_length=2000),
        FieldSchema("dataset", DataType.VARCHAR, max_length=100),
        FieldSchema("resolution", DataType.VARCHAR, max_length=50),
    ]

    schema = CollectionSchema(fields, description="Wikipedia multimodal imagery")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create HNSW index
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
else:
    collection = Collection(COLLECTION_NAME)

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# Multimodal embedding (image + caption)
def generate_multimodal_embedding(image_path, text):
    image = Image.open(image_path).convert("RGB")

    # Tokenize text with truncation
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        image_emb = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_emb = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # Combine embeddings (multimodal embedding)
    embedding = (image_emb + text_emb) / 2.0
    embedding = embedding.cpu().numpy()[0]
    embedding /= np.linalg.norm(embedding)

    return embedding.tolist()

# Prepare data entries
entries = []

# Iterate over files
for file in tqdm(os.listdir(DATA_DIR), desc="Wikipedia Images"):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        base_name = os.path.splitext(file)[0]
        image_path = os.path.join(DATA_DIR, file)
        json_path = os.path.join(DATA_DIR, f"{base_name}.json")

        if not os.path.exists(json_path):
            print(f"Metadata not found for {image_path}, skipping.")
            continue

        with open(json_path, 'r') as f:
            metadata = json.load(f)

        descriptive_text = metadata.get("caption", "") or metadata.get("article_title", "")

        embedding = generate_multimodal_embedding(image_path, descriptive_text)

        entry_id = base_name

        entry = [
            entry_id,
            embedding,
            metadata.get("article_title", ""),
            metadata.get("image_url", ""),
            metadata.get("caption", ""),
            "wikipedia",
            metadata.get("additional_info", {}).get("resolution", ""),
        ]

        entries.append(entry)

# Insert data into Milvus in batches
BATCH_SIZE = 100
for i in tqdm(range(0, len(entries), BATCH_SIZE), desc="Inserting batches"):
    batch = entries[i:i+BATCH_SIZE]
    collection.insert(list(zip(*batch)))

collection.flush()

print(f"Inserted {len(entries)} Wikipedia images into Milvus.")

# Load collection into memory for immediate searchability
collection.load()