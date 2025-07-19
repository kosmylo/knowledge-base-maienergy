#!/bin/bash
set -e

echo "Waiting for Milvus at ${MILVUS_HOST}:${MILVUS_PORT}..."
until nc -z ${MILVUS_HOST} ${MILVUS_PORT}; do
  sleep 5
done
echo "Milvus is available!"

echo "Starting image embeddings insertion..."

python scripts/milvus/images/insert_images_copernicus.py
python scripts/milvus/images/insert_images_eprel.py
python scripts/milvus/images/insert_images_inria.py
python scripts/milvus/images/insert_images_irf.py
python scripts/milvus/images/insert_images_wikimedia.py
python scripts/milvus/images/insert_images_wikipedia.py

echo "Milvus image embeddings insertion completed successfully!"