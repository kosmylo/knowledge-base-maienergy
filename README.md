# 🔋 mAiEnergy Knowledge Base

The `maienergy-knowledge-base` repository integrates three vector databases—**OpenSearch**, **Milvus**, and **Neo4j**—into one unified Docker-based environment. This setup efficiently handles diverse data types, including textual, numerical, image, and graph data, facilitating retrieval-augmented generative AI solutions for energy sector applications.

## 📂 Repository Structure

```plaintext
maienergy-knowledge-base
├── .env                                 # Environment variables for configurations
├── README.md                            # This documentation file
├── data                                 # Directory for all dataset files
├── docker-compose.yaml                  # Docker Compose orchestration file
├── unified_worker                       # Embedding and ingestion worker
│   ├── Dockerfile                       # Dockerfile for worker
│   ├── requirements.txt                 # Python dependencies
│   ├── entrypoints                      # Manual entrypoint scripts for ingestion
│   │   ├── entrypoint_opensearch.sh
│   │   ├── entrypoint_milvus.sh
│   │   └── entrypoint_neo4j.sh
│   └── scripts                          # Scripts for data embedding and ingestion
│       ├── opensearch
│       │   ├── articles                 # Article ingestion scripts
│       │   └── numerical                # Numerical ingestion scripts
│       ├── milvus
│       │   └── images                   # Image ingestion scripts
│       └── neo4j
│           ├── embeddings               # Neo4j graph embedding scripts
│           └── import                   # Cypher scripts for Neo4j imports
└── logs                                 # Directory for logging ingestion processes
```

## 🚀 Quick Start

### 1. Clone the Repository

Clone the repository explicitly into your VM:

```bash
git clone <your_repo_url>
cd maienergy-knowledge-base
```

### 2. Upload Datasets

Upload all relevant dataset files into the `data/` directory following this structure:

- Text/Numerical: `data/articles/` and `data/numerical/`

- Images: `data/copernicus/`, `data/eprel/`, `data/inria/`, `data/irf/`, `data/wikimedia/`, and `data/wikipedia/`

- Graph CSVs: `data/cordis/`, `data/gridkit/`, `data/osm/`, `data/powerplants/`, and `data/tso_network/`

Ensure all files exactly match the paths specified in the scripts.

### 3. Launch the Containers

Ensure Docker and Docker Compose are installed on your host system.

```bash
docker-compose build
docker-compose up -d
```

This command sets up your vector databases (OpenSearch, Milvus, Neo4j) and a unified worker container (idle by default).

### 4. Execute Data Ingestion

Data ingestion scripts are executed manually. Once your containers are running, trigger each ingestion process separately as follows:

- OpenSearch Ingestion (Textual and Numerical data):

```bash
docker compose exec embedding_worker/entrypoints/entrypoint_opensearch.sh
```

- Milvus Ingestion (Image data):

```bash
docker compose exec embedding_worker/entrypoints/entrypoint_milvus.sh
```

- Neo4j Ingestion (Graph/Geospatial data):

```bash
docker compose exec embedding_worker/entrypoints/entrypoint_neo4j.sh
```

Monitor the logs generated in the `logs/` directory for detailed progress and troubleshooting information.

### 🛠️ Technologies Used

- OpenSearch: For textual and numerical data embeddings and retrieval.
- Milvus: Efficient handling and retrieval of large-scale image embeddings.
- Neo4j: Graph-based data storage and semantic embedding queries.
- Docker & Docker Compose: Simplified deployment, scalability, and maintenance.

