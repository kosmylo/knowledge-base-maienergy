import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from dotenv import load_dotenv
import logging

load_dotenv()

# Logging configuration
logging.basicConfig(
    filename="logs/neo4j_worker.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.info("Neo4j embeddings script for CORDIS started.")

# Initialize embeddings model
try:
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    embedding = SentenceTransformerEmbeddings(
        model_name=embedding_model_name,
        cache_folder="./models/embeddings"
    )
    logging.info(f"Embedding model '{embedding_model_name}' initialized successfully.")
except Exception as model_err:
    logging.error(f"Failed to initialize embedding model '{embedding_model_name}': {model_err}")
    raise

# Define node labels and respective properties explicitly matching Neo4j data
node_configs = {
    "Project": ["title", "acronym", "objective", "frameworkProgramme"],
    "Organization": ["name", "shortName", "country", "activityType"],
    "Topic": ["title"],
    "LegalBasis": ["title"]
}

# Neo4j connection details
neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

# Generate and store embeddings for each node label
for label, properties in node_configs.items():
    index_name = f"cordis_{label.lower()}_index"
    try:
        Neo4jVector.from_existing_graph(
            embedding=embedding,
            url=neo4j_url,
            username=neo4j_user,
            password=neo4j_password,
            index_name=index_name,
            node_label=label,
            text_node_properties=properties,
            embedding_node_property="embedding"
        )
        logging.info(f"Embeddings generated and indexed successfully for nodes with label: {label}")
    except Exception as e:
        logging.error(f"Failed to generate embeddings for label '{label}': {e}")

logging.info("Neo4j embeddings script for CORDIS completed.")