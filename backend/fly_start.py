#!/usr/bin/env python3
"""
Fly.io startup script for CardioAI Pro Backend
"""
import uvicorn
from app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
