#!/bin/sh
#
# Container startup script for PostgreSQL Co-Pilot
#

set -e

# Define paths
DATA_DIR="/app/data"
CONFIG_DIR="$DATA_DIR/config"
MEMORY_DIR="$DATA_DIR/memory"
APPROVED_SQL_DIR="$DATA_DIR/approved_sql"
CONFIG_FILE="$CONFIG_DIR/config.json"

# Create directory structure if it doesn't exist
mkdir -p "$CONFIG_DIR"
mkdir -p "$MEMORY_DIR/feedback"
mkdir -p "$MEMORY_DIR/insights"
mkdir -p "$MEMORY_DIR/schema"
mkdir -p "$MEMORY_DIR/conversation_history"
mkdir -p "$MEMORY_DIR/lancedb_stores"
mkdir -p "$APPROVED_SQL_DIR"

# Set permissions
chown -R appuser:appuser "$DATA_DIR"

# Execute the main command
exec "$@"
