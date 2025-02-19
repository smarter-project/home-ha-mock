# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and sample data
COPY hamock.py .
COPY state_replay.log .

# Make the script executable
RUN chmod +x hamock.py

# Set environment variables with defaults
ENV HAMOCK_HASS_URL=localhost:8123

# Dummy token so you know what you're looking for. Substitute with your own (you have to create it)
ENV HAMOCK_HASS_ACCESS_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmN2NjNTkyY2VkMjI0NTY2OWI4ZDY0OWQ4MmFkOGFlYiIsImlhdCI6MTcyODQwMzg4OSwiZXhwIjoyMDQzNzYzODg5fQ.OqEiy5SJFdjc70CFWR9IiP9eXgI7aAY8N7YeowfvEtM
ENV HAMOCK_OPENAI_URL=http://localhost:11434
ENV HAMOCK_OPENAI_MODEL=llama3.2-vision:latest
ENV HAMOCK_DISPLAY_STATS=false
ENV HAMOCK_INFER=true

# Use hamock.py as entrypoint
ENTRYPOINT ["./hamock.py", "replay", "--input", "state_replay.log"]
