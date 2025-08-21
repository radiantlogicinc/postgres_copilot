#!/bin/sh
#
# User-friendly entry point for PostgreSQL Co-Pilot
#

set -e

# Get the current user's UID and GID
export CURRENT_UID=$(id -u)
export CURRENT_GID=$(id -g)

echo "Starting PostgreSQL Co-Pilot..."
# Create the data directory and all necessary subdirectories for the container volume
echo "Creating local data directories..."
mkdir -p ./data/config
mkdir -p ./data/memory/feedback
mkdir -p ./data/memory/insights
mkdir -p ./data/memory/schema
mkdir -p ./data/memory/conversation_history
mkdir -p ./data/memory/lancedb_stores
mkdir -p ./data/memory/logs
mkdir -p ./data/approved_sql

echo "Pulling the latest image..."
docker-compose -f client-docker-compose.yml pull

echo "Running container..."
docker-compose -f client-docker-compose.yml up -d && docker-compose -f client-docker-compose.yml logs -f
