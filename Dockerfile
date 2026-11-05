# =====================================================================
# STAGE 1: Build the React + TypeScript + Vite Frontend
# =====================================================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# =====================================================================
# STAGE 2: Python 3.11 Slim Production Runtime
# =====================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    LLM_PROVIDER=local \
    DB_SCHEMA=school \
    DATABASE_URL="postgresql://neondb_owner:npg_Ssqzmd9cDo0l@ep-fancy-cell-bd9rqecw-pooler.c-8.us-east-2.aws.neon.tech/neondb?sslmode=require"

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python backend dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Copy backend code, SQL seed, and manuals
COPY backend/ ./backend/
COPY school_management_postgres92.sql ./
COPY ["School Management Database - User Manual.pdf", "./"]
COPY manual_extracted.txt ./

# Copy compiled frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Security: Create and run as non-root user (UID 1000 for Hugging Face Spaces)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/backend/data && \
    chown -R appuser:appuser /app

USER appuser

# Pre-train & warm up the in-memory Scikit-Learn ML intent model
RUN python -c "import sys; sys.path.insert(0, 'backend'); from app.agent.ml_intent import get_intent_classifier; get_intent_classifier()"

EXPOSE 7860

# Start unified FastAPI server serving both API and Frontend
CMD ["sh", "-c", "python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-7860}"]

