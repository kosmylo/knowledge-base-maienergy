FROM python:3.11-slim

WORKDIR /

RUN apt-get update && apt-get install -y \
    build-essential curl unzip netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /logs

# Install Neo4j cypher-shell
RUN curl -O https://dist.neo4j.org/cypher-shell/cypher-shell_5.19.0_all.deb \
    && apt-get update && apt-get install -y ./cypher-shell_5.19.0_all.deb \
    && rm cypher-shell_5.19.0_all.deb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY scripts/ /scripts/
COPY entrypoints/ /entrypoints/
RUN chmod +x /entrypoints/*.sh

VOLUME ["/data"]

# Default idle entrypoint; ingestion scripts run manually
ENTRYPOINT ["tail", "-f", "/dev/null"]