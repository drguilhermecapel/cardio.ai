# CardioAI Pro Backend

Enterprise-grade ECG analysis backend with AI/ML capabilities.

## Features

- FastAPI with async SQLAlchemy
- ONNX Runtime for ML inference
- Comprehensive ECG analysis
- Regulatory compliance (ANVISA/FDA/LGPD/HIPAA)
- Background task processing with Celery
- Real-time notifications via WebSocket

## Setup

```bash
poetry install
poetry run uvicorn app.main:app --reload
```

## Testing

```bash
poetry run pytest --cov=app
```
