FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

# tau2-bench is a submodule: fetch the pinned commit instead of uploading its 880 MB of data.
ARG TAU2_COMMIT=cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2
RUN git init tau2-bench && cd tau2-bench \
 && git remote add origin https://github.com/asayenju/tau2-bench.git \
 && git fetch --depth 1 origin $TAU2_COMMIT && git checkout FETCH_HEAD && rm -rf .git

COPY backend/pyproject.toml backend/uv.lock backend/
RUN cd backend && uv sync --frozen --no-dev --extra jev --no-install-project
COPY backend backend
COPY frontend frontend

ENV PYTHONPATH=/app/backend/src PYTHONUNBUFFERED=1
WORKDIR /app/backend
CMD ["sh", "-c", "uv run --no-sync uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
