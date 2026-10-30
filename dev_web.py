"""Local stand-in for Vercel: the same API, with public/ served the way the CDN serves it.

    uv run uvicorn dev_web:app --port 8090
"""

from pathlib import Path

from fastapi.staticfiles import StaticFiles

from app import app

app.mount("/", StaticFiles(directory=Path(__file__).parent / "public", html=True), name="public")
