FROM python:3.11-slim

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /logs

# Copy requirements and install Python dependencies
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