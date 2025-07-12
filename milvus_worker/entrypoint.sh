#!/bin/bash
set -e

echo "Waiting for Milvus at ${MILVUS_HOST}:${MILVUS_PORT}..."
until nc -z ${MILVUS_HOST} ${MILVUS_PORT}; do
  sleep 5
done
echo "Milvus is available!"

echo "Starting image embeddings insertion..."

python scripts/insert_copernicus_images.py
python scripts/insert_eprel_images.py
python scripts/insert_inria_images.py
python scripts/insert_irf_images.py
python scripts/insert_wikimedia_images.py
python scripts/insert_wikipedia_images.py

echo "Image embeddings insertion completed successfully!"