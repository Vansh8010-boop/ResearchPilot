"""Vercel entrypoint: the platform looks for a FastAPI instance named `app` in this file.

Static files in public/ are served by Vercel's CDN. Everything under /api reaches this app.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from research_pilot.web import create_web_app  # noqa: E402

app = create_web_app()
