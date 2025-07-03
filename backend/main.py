#!/usr/bin/env python3
"""
Entry point for CardioAI Pro Backend Deployment
This file ensures the correct FastAPI app is used for deployment
"""

import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent
app_dir = current_dir / "app"
sys.path.insert(0, str(app_dir))

os.environ.setdefault("STANDALONE_MODE", "true")
os.environ.setdefault("SECRET_KEY", "deployment-secret-key-cardioai-pro-2024")
os.environ.setdefault("ENVIRONMENT", "production")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./cardioai.db")

from app.main import app

application = app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
    )
