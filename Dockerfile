# One image for both services; docker-compose.yml picks the command (API or UI).
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.3 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    RP_CACHE_DIR=/data/cache \
    RP_TRACE_DIR=/data/traces

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra api --extra ui --no-install-project

COPY src ./src
COPY ui ./ui
COPY public ./public
RUN uv sync --frozen --no-dev --extra api --extra ui

RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501
CMD ["uvicorn", "research_pilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
