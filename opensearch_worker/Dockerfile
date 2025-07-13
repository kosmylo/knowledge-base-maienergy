FROM python:3.11-slim

# Set working directory
WORKDIR /

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /logs

# Copy requirements file (ensure this file exists with the listed packages)
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy scripts directory
COPY scripts/ ./scripts/

# Copy entrypoint script
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Mount data directory from host
VOLUME ["/data"]

# Entry point
ENTRYPOINT ["./entrypoint.sh"]