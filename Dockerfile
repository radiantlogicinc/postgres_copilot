# Use a single-stage build
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user and group
RUN useradd --create-home --shell /bin/bash appuser

# Create necessary directories and set permissions
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Set the entrypoint to run the application directly
ENTRYPOINT ["python", "postgres_copilot_chat.py"]
