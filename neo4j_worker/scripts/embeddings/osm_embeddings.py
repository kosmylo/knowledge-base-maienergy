import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Initialize embeddings model explicitly
embedding = SentenceTransformerEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
    cache_folder="./models/embeddings"
)

# Define node labels and respective properties explicitly matching Neo4j OSM data
node_configs = {
    "ChargingStation": ["name", "operator", "country", "capacity", "socket_types"],
    "PowerPlant": ["name", "operator", "country", "source", "method", "capacity"],
    "SolarFarm": ["operator", "country", "source", "method", "capacity"],
    "WindTurbine": ["manufacturer", "model", "operator", "country", "capacity", "rotor_diameter"],
    "Substation": ["country"],
    "TransmissionLine": ["name", "operator", "country", "voltage"]
}

# Neo4j connection details explicitly loaded from environment
neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

# Generate and store embeddings explicitly for each OSM node label
for label, properties in node_configs.items():
    index_name = f"osm_{label.lower()}_index"

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

    logging.info(f"Embeddings generated and indexed for nodes with label: {label}")