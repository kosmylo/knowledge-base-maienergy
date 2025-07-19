#!/bin/bash
set -e

echo "Waiting for Neo4j at ${NEO4J_URL}..."
until nc -z neo4j 7687; do
  sleep 5
done
echo "Neo4j is available!"

# Cypher import scripts
echo "Starting Cypher data imports..."

cypher-shell -a ${NEO4J_URL} -u ${NEO4J_USER} -p ${NEO4J_PASSWORD} -f /scripts/neo4j/import/gridkit_import.cypher
cypher-shell -a ${NEO4J_URL} -u ${NEO4J_USER} -p ${NEO4J_PASSWORD} -f /scripts/neo4j/import/cordis_import.cypher
cypher-shell -a ${NEO4J_URL} -u ${NEO4J_USER} -p ${NEO4J_PASSWORD} -f /scripts/neo4j/import/powerplants_import.cypher
cypher-shell -a ${NEO4J_URL} -u ${NEO4J_USER} -p ${NEO4J_PASSWORD} -f /scripts/neo4j/import/tso_network_import.cypher
cypher-shell -a ${NEO4J_URL} -u ${NEO4J_USER} -p ${NEO4J_PASSWORD} -f /scripts/neo4j/import/osm_import.cypher

echo "Cypher data imports completed successfully!"

# Embedding generation scripts
echo "Starting embedding generation..."

python scripts/neo4j/embeddings/gridkit_embeddings.py
python scripts/neo4j/embeddings/cordis_embeddings.py
python scripts/neo4j/embeddings/powerplants_embeddings.py
python scripts/neo4j/embeddings/tso_network_embeddings.py
python scripts/neo4j/embeddings/osm_embeddings.py

echo "Embedding generation completed successfully!"