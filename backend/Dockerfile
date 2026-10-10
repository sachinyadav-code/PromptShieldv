# syntax=docker/dockerfile:1
# PromptShield Enterprise LLM Security Gateway - Backend Dockerfile
# Python base: 3.14-slim (matches verified runtime environment and wheels)

FROM python:3.14-slim AS runtime

# Prevent Python from writing .pyc files and enable unbuffered output for container logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    PROMPTSHIELD_AUDIT_DB_PATH=/data/audit.db

# Install curl for healthchecks and clean up apt cache in a single layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user and group
RUN groupadd -r promptshield && useradd -r -g promptshield -d /app -s /sbin/nologin promptshield

# Prepare persistent data directory for SQLite audit storage
RUN mkdir -p /data && chown -R promptshield:promptshield /data

WORKDIR /app

# Copy dependency manifest and install production wheels
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend application source, model artifacts, and RAG dataset
COPY backend/app /app/backend/app
COPY datasets /app/datasets

# Set ownership of application files to non-root user
RUN chown -R promptshield:promptshield /app

# Switch to non-root runtime user
USER promptshield

# Expose backend ASGI gateway port
EXPOSE 8000

# Container healthcheck verified against the public /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI application using uvicorn ASGI server
CMD ["uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "8000"]