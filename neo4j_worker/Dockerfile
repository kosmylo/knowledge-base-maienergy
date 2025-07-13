FROM python:3.11-slim

WORKDIR /

# Install necessary packages and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unzip \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install cypher-shell from Neo4j official distribution
RUN curl -O https://dist.neo4j.org/cypher-shell/cypher-shell_5.19.0_all.deb && \
    apt-get update && apt-get install -y ./cypher-shell_5.19.0_all.deb && \
    rm cypher-shell_5.19.0_all.deb && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy scripts and entrypoint
COPY scripts/ /scripts/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Mount data directory from host
VOLUME ["/data"]

# Entry point
ENTRYPOINT ["/entrypoint.sh"]