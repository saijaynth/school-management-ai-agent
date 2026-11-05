---
title: School Management AI Assistant
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# School Management AI Chatbot

An AI assistant that answers natural-language questions about a PostgreSQL school-management database. Every data answer is grounded in **live read-only SQL** executed against the `school` schema, and every workflow/constraint answer is grounded in the **School Management Database - User Manual**.

Built with an **embedded Scikit-Learn ML Agent core (zero API keys and zero cost)**, with optional plug-and-play support for external cloud LLMs (OpenAI, Anthropic, Gemini, Ollama).

Responses are labeled with their source (`Live Database Query`, `User Manual`, or `Database + User Manual`), show the exact SQL executed alongside returned rows in interactive data tables, and offer clickable follow-up suggestions.

---

## Architecture

```
frontend (React 18 + TS + Vite + Tailwind CSS)
   │  Served statically by FastAPI at / (in production) or via Vite dev proxy (/api/*)
   ▼
FastAPI backend (app/main.py)
   ├── agent/
   │    ├── ml_intent.py         Scikit-Learn TF-IDF + LogisticRegression intent classifier
   │    ├── entity_extractor.py  Entity/slot extraction & conversational pronoun resolution
   │    ├── sql_synthesizer.py   Schema-aware read-only SQL query synthesizer
   │    ├── response_generator.py Grounded markdown builder with tables & citations
   │    └── orchestrator.py      Multi-turn state tracking & hybrid provider router
   ├── sql_validator.py          3-layer AST/token read-only guardrail
   ├── database.py               psycopg3 pool: search_path=school, READ ONLY, 5s timeout
   └── manual_loader.py          Indexes the 24-page User Manual (TF-IDF retrieval)
   ▼
PostgreSQL 16 `school` schema (29 tables, 4 reporting views, full seed data)
```

---

## 100% Free Remote Cloud Deployment Guide ($0/month)

Deploy the entire full-stack application online completely free without credit cards:

### Step 1: Create Free Remote PostgreSQL Database (Neon.tech)
1. Go to [Neon.tech](https://neon.tech) and create a free account.
2. Create a project named `school_db`.
3. Copy your PostgreSQL connection string (ensure `?sslmode=require` is at the end):
   ```text
   postgresql://username:password@ep-xyz.neon.tech/school_db?sslmode=require
   ```
4. Load the `school` schema (29 tables, 4 views, 8 users) into your remote Neon database in one command:
   ```bash
   python backend/scripts/init_db.py --db-url "postgresql://username:password@ep-xyz.neon.tech/school_db?sslmode=require"
   ```

### Step 2: Option A — Deploy on Hugging Face Spaces (Recommended: 16 GB RAM Free)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Space name: `school-management-ai` | License: `mit` | SDK: **Docker** | Hardware: **Free (2 vCPU, 16 GB RAM)**.
3. In **Settings > Variables and secrets**, add a new **Secret**:
   - Name: `DATABASE_URL`
   - Value: `postgresql://username:password@ep-xyz.neon.tech/school_db?sslmode=require`
4. Push this repository to your Space:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/school-management-ai
   git push space main
   ```
5. Your Space builds automatically via the multi-stage `Dockerfile` and is live at `https://YOUR_USERNAME-school-management-ai.hf.space`!

### Step 2: Option B — Deploy on Render.com
1. Go to [Render.com](https://render.com) and create a new **Web Service**.
2. Connect your Git repository.
3. Select **Docker** environment.
4. Set Environment Variables:
   - `DATABASE_URL` = your Neon PostgreSQL connection string.
   - `PORT` = `7860`
   - `LLM_PROVIDER` = `local`
5. Click **Create Web Service**. Render builds the multi-stage Docker container and provides an HTTPS URL.

---

## Local Development Setup

### 1. PostgreSQL Database
Ensure PostgreSQL is running locally on port 5432:
```bash
# Verify or bootstrap local database
python backend/scripts/init_db.py
```

### 2. Backend Server
```bash
# Start FastAPI backend (serves API and local ML agent)
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

### 3. Frontend UI
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```
*Note: In production or when running via Docker, FastAPI serves the compiled frontend directly at `http://localhost:8000/`.*

---

## Automated Verification Suite

Run the end-to-end test suite testing database ground truth, validator security, and 10 multi-turn manager chat queries:
```bash
python backend/scripts/verify_agent.py
```
Expected output:
```text
SUMMARY: 50 passed, 0 failed, 0 blocked
```

---

## Safety & Security Model

- **First Layer (Database Session)**: `default_transaction_read_only = on` and `statement_timeout = 5000` ms on every connection.
- **Second Layer (SQL Validator)**: Only `SELECT`/`WITH...SELECT` statements are permitted. Mutating keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`) are strictly blocked.
- **Third Layer (Projection Masking)**: References to `password_hash` in the SELECT projection are replaced with `NULL AS password_hash` before execution, and the database driver replaces any sensitive credential strings with `***MASKED***`.
- **Policy Enforcement**: Mutation requests (e.g., "Delete rejected leave requests") are refused with reference guidelines and SQL templates from the User Manual.
