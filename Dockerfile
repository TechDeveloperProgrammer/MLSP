# Multi-stage build for MLSP-MACP Platform
FROM python:3.9-slim-bullseye AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VERSION=1.2.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock* ./
COPY . .

# Install project dependencies
RUN $POETRY_VENV/bin/poetry config virtualenvs.create false \
    && $POETRY_VENV/bin/poetry install --no-interaction --no-ansi

# Final stage
FROM python:3.9-slim-bullseye

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder /app /app
COPY --from=builder /opt/poetry-venv /opt/poetry-venv

# Set environment path
ENV PATH="/opt/poetry-venv/bin:$PATH"

# Expose necessary ports
EXPOSE 8000
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["python", "-m", "macp.core.unified_config"]
