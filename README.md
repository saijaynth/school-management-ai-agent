# School Management AI Assistant

A full-stack conversational assistant designed for querying school management records and administrative workflows. The application answers natural-language questions grounded strictly in a PostgreSQL `school` database (29 relational tables, 4 reporting views) and the official system user manual.

---

## Features

- **Grounded Database Answers**: Queries attendance, timetables, course assignments, homework submissions, exam scores, and leave requests via parameterized, read-only SQL queries.
- **Conversational Memory**: Resolves multi-turn follow-up questions and pronouns (e.g., asking *"What is their Monday schedule?"* after querying a specific teacher or student).
- **Administrative Manual Guidance**: Explains system workflows, leave approval steps, mark entries, and role permissions based on the system user manual.
- **Data Protection Guardrails**: AST-level read-only query validator prevents data mutation (no `INSERT`, `UPDATE`, `DELETE`, or `DROP`), with automatic password masking.
- **Zero API Key Requirement**: Built-in natural language classification and SQL synthesis runs locally out of the box with zero external API costs. Optional integration with external LLMs (OpenAI, Anthropic, Gemini, Ollama) is supported via environment configuration.
- **Responsive Web Interface**: Mobile-first interface built with React and Tailwind CSS, featuring structured tables, attendance indicators, and suggested follow-ups.

---

## Project Structure

```text
├── backend/
│   ├── app/
│   │   ├── agent/             # Intent classification, entity extraction & query synthesis
│   │   ├── config.py          # Application settings & environment variables
│   │   ├── database.py        # PostgreSQL connection pool with read-only isolation
│   │   ├── main.py            # FastAPI endpoints & static frontend serving
│   │   ├── manual_loader.py   # Search and retrieval engine for the user manual
│   │   ├── schemas.py         # Request and response models
│   │   └── sql_validator.py   # Read-only SQL safety validator
│   ├── data/
│   │   ├── intent_model.joblib # Trained intent classifier
│   │   └── manual_extracted.txt
│   ├── scripts/
│   │   ├── create_zip_package.py # Distribution packager
│   │   ├── init_db.py           # Database schema & seed migrator
│   │   └── verify_agent.py      # Automated 50-test verification suite
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/               # API client
│   │   ├── components/        # React components (ChatWindow, Sidebar, MessageBubble, etc.)
│   │   └── types.ts           # TypeScript interfaces
│   ├── package.json
│   └── vite.config.ts
├── docker-compose.yml
├── Dockerfile
├── school_management_postgres92.sql # Database schema & seed records
└── School Management Database - User Manual.pdf
```

---

## Quick Start

### Option 1: Docker (Recommended)

Run the full-stack application (backend and built frontend) in a single container:

```bash
docker compose up -d --build
```

The web application will be accessible at `http://localhost:8000`.

### Option 2: Local Python Setup

#### 1. Backend

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Start the unified FastAPI server
python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

#### 2. Frontend Development (Optional)

To run the Vite development server with hot module replacement:

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs at `http://localhost:5173` and proxies `/api` calls to the backend on port 8000.

---

## Configuration

Set the following environment variables in a `.env` file or within your deployment environment:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | Cloud PostgreSQL URL | Connection string for PostgreSQL database |
| `DB_SCHEMA` | `school` | Target database schema |
| `LLM_PROVIDER` | `local` | `local` (built-in, zero cost), `openai`, `gemini`, `anthropic`, or `ollama` |
| `LLM_MODEL` | `scikit-learn-ml-agent` | Model identifier |
| `PORT` | `8000` | Port for the web service |

---

## Verification & Testing

The project includes an automated test suite verifying database ground truth, read-only safety guardrails, schema endpoints, and conversational query handling:

```bash
python backend/scripts/verify_agent.py
```

Expected result: `50 passed, 0 failed, 0 blocked`.
