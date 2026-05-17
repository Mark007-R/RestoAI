# RestoAI Phase-6 FastAPI service.
#
# Multi-stage build keeps the runtime image lean.
#   stage 1 (builder)  : install wheels into a virtualenv
#   stage 2 (runtime)  : copy the venv + source, run uvicorn
#
# Only the FastAPI service runs here (port 8000). The Flask app (app.py)
# stays out of the container so this image isn't coupled to MySQL /
# Flask-SQLAlchemy / NLTK downloads — they're not on the inference path.

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only the runtime subset of requirements.txt — everything the API
# service actually imports. Pinned to match the rest of the repo.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    UVICORN_WORKERS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Bring in only what api.py + src.* needs. The Flask app, datasets,
# notebooks, etc. are NOT copied — keeps the image small.
COPY api.py /app/api.py
COPY src /app/src
COPY models /app/models
COPY manager_system/rag_chat.py /app/manager_system/rag_chat.py
COPY manager_system/__init__.py /app/manager_system/__init__.py
COPY manager_system/vector_db /app/manager_system/vector_db

RUN mkdir -p /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}"]
