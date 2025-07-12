# 🔋 mAiEnergy Knowledge Base

The `maienergy-knowledge-base` repository integrates three vector databases—**OpenSearch**, **Milvus**, and **Neo4j**—into one unified Docker-based environment. This setup efficiently handles diverse data types, including textual, numerical, image, and graph data, facilitating retrieval-augmented generative AI solutions for energy sector applications.

## 📂 Repository Structure

```plaintext
maienergy-knowledge-base
├── .env
├── README.md
├── data
├── docker-compose.yaml
├── milvus_worker
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── requirements.txt
│   └── scripts
│       ├── insert_images_copernicus.py
│       ├── insert_images_eprel.py
│       ├── insert_images_inria.py
│       ├── insert_images_irf.py
│       ├── insert_images_wikimedia.py
│       └── insert_images_wikipedia.py
├── neo4j_worker
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── requirements.txt
│   └── scripts
│       ├── embeddings
│       │   ├── cordis_embeddings.py
│       │   ├── gridkit_embeddings.py
│       │   ├── osm_embeddings.py
│       │   ├── powerplants_embeddings.py
│       │   └── tso_network_embeddings.py
│       └── import
│           ├── cordis_import.cypher
│           ├── gridkit_import.cypher
│           ├── osm_import.cypher
│           ├── powerplants_import.cypher
│           └── tso_network_import.cypher
└── opensearch_worker
    ├── Dockerfile
    ├── entrypoint.sh
    ├── requirements.txt
    └── scripts
        ├── articles
        │   ├── insert_arxiv_articles.py
        │   ├── insert_gov_articles.py
        │   ├── insert_news_articles.py
        │   └── insert_wiki_articles.py
        └── numerical
            ├── country_mapping.py
            ├── insert_annual_energy_balances.py
            ├── insert_building_stock.py
            ├── insert_electricity_prices.py
            ├── insert_energy_efficiency_indicators.py
            ├── insert_energy_import_dependency.py
            ├── insert_energy_intensity_of_economy.py
            ├── insert_energy_performance.py
            ├── insert_final_energy_consumption_households_per_capita.py
            ├── insert_financial_performance.py
            ├── insert_gas_prices.py
            ├── insert_gdp.py
            ├── insert_ghg_emissions_energy.py
            ├── insert_households_number.py
            ├── insert_inability_to_keep_home_warm.py
            ├── insert_population.py
            ├── insert_reference_buildings.py
            ├── insert_renewable_energy_share.py
            └── insert_social_performance.py
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

### 4. Execute Data Ingestion

Data ingestion begins automatically once the containers start. Monitor logs:

```bash
docker logs -f opensearch_ingestion_worker
docker logs -f milvus_embeddings_worker
docker logs -f neo4j_embeddings_worker
```

### 🛠️ Technologies Used

- OpenSearch: For textual and numerical data embeddings and retrieval.
- Milvus: Efficient handling and retrieval of large-scale image embeddings.
- Neo4j: Graph-based data storage and semantic embedding queries.
- Docker & Docker Compose: Simplified deployment, scalability, and maintenance.

