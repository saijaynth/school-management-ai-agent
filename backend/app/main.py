"""FastAPI application for the School Management AI chatbot."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .config import llm_is_ready, resolve_llm, settings, PROJECT_ROOT
from .database import check_connectivity, db_connection, close_pool
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ModuleInfo,
    SchemaSummaryResponse,
    EntitySample,
)
from .agent.orchestrator import chat as agent_chat, clear_session

logger = logging.getLogger("school_ai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODULES: list[ModuleInfo] = [
    ModuleInfo(module="Security", tables=["roles", "users"]),
    ModuleInfo(module="Academic Structure", tables=["academic_years", "classes", "subjects", "classrooms"]),
    ModuleInfo(module="Profiles", tables=["admins", "faculty", "students"]),
    ModuleInfo(module="Course Allocation", tables=["course_assignments"]),
    ModuleInfo(module="Timetable", tables=["periods", "timetable"]),
    ModuleInfo(module="Attendance", tables=["student_attendance", "faculty_attendance"]),
    ModuleInfo(module="Leave Management", tables=["leave_types", "leave_requests"]),
    ModuleInfo(module="Homework", tables=["homework", "homework_submissions"]),
    ModuleInfo(module="Assignments", tables=["assignments", "assignment_submissions"]),
    ModuleInfo(module="Exams & Marks", tables=["exam_types", "exams", "exam_schedules", "grade_scale", "marks"]),
    ModuleInfo(module="Progress Reports", tables=["progress_reports", "progress_report_items"]),
    ModuleInfo(module="Notifications", tables=["notifications"]),
]

VIEWS = [
    {
        "name": "v_student_attendance_report",
        "description": "Student-level period counts, attended periods (Present or Late) and attendance percentage.",
    },
    {
        "name": "v_exam_report",
        "description": "Subject-wise student marks with Pass/Fail result and letter grade.",
    },
    {
        "name": "v_pending_leaves",
        "description": "Pending leave requests with applicant name, role, leave type, dates and reason.",
    },
    {
        "name": "v_timetable",
        "description": "Section timetable with weekday, period times, subject, faculty name and classroom.",
    },
]

app = FastAPI(
    title="School Management AI Chatbot",
    description="Natural-language question answering over the `school` PostgreSQL schema and the School Management User Manual.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/api/health", methods=["GET", "HEAD", "POST"], response_model=HealthResponse)
def health() -> HealthResponse:
    db = check_connectivity()
    cfg = resolve_llm()
    llm = {
        "ready": llm_is_ready(),
        "provider": cfg["provider"],
        "model": cfg["model"],
        "key_configured": bool(cfg["api_key"]),
    }
    counts = {
        "tables": int(db.get("tables") or 0),
        "views": int(db.get("views") or 0),
        "users": int(db.get("users") or 0),
    }
    return HealthResponse(
        status="ok" if db.get("ok") and llm["ready"] else ("degraded" if db.get("ok") else "unavailable"),
        database={"connected": bool(db.get("ok")), **{k: db.get(k) for k in ("schema", "tables", "views")}},
        llm=llm,
        counts=counts,
    )


@app.api_route("/health", methods=["GET", "HEAD", "POST"])
@app.api_route("/ping", methods=["GET", "HEAD", "POST"])
def uptime_ping() -> Dict[str, Any]:
    """Lightweight 200 OK probe for UptimeRobot, cron jobs, and keep-alive monitors."""
    return {"status": "ok", "service": "school-management-ai", "uptime": "active"}


@app.get("/api/schema-summary", response_model=SchemaSummaryResponse)
def schema_summary() -> SchemaSummaryResponse:
    """Structured overview for the sidebar Database Explorer.

    Live entity counts come from the database; the module list is static.
    """
    entities: list[EntitySample] = []
    db_ok, counts = True, {}
    try:
        with db_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM users) AS users,
                  (SELECT COUNT(*) FROM students) AS students,
                  (SELECT COUNT(*) FROM faculty) AS faculty,
                  (SELECT COUNT(*) FROM classes) AS classes,
                  (SELECT COUNT(*) FROM subjects) AS subjects,
                  (SELECT COUNT(*) FROM exams) AS exams,
                  (SELECT COUNT(*) FROM homework) AS homework,
                  (SELECT COUNT(*) FROM assignments) AS assignments
                """
            )
            counts = cur.fetchone()  # type: ignore[assignment]
    except Exception:
        db_ok = False

    labels = [
        ("users", "Users", "8 users: 1 Admin, 3 Faculty, 4 Students"),
        ("students", "Students", "Arjun Rao, Meera Iyer (Grade 9-A); Kiran Patel, Divya Nair (Grade 10-A)"),
        ("faculty", "Faculty", "Anita Sharma (Maths, EMP001), Vijay Reddy (Science, EMP002), Sara Thomas (English, EMP003)"),
        ("classes", "Classes", "Grade 9, Grade 10"),
        ("subjects", "Subjects", "Mathematics (MATH), Science (SCI), English (ENG)"),
        ("exams", "Exams", "Grade 10 Mid Term 2026 (Oct 12-16)"),
        ("homework", "Homework", "Algebra Worksheet 5, Essay: My Role Model"),
        ("assignments", "Assignments", "Science Project: Solar Energy, Geometry Assignment 1"),
    ]
    for key, label, detail in labels:
        n = counts.get(key) if db_ok else None
        suffix = f" ({n} rows)" if n is not None else ""
        entities.append(EntitySample(kind=key, label=f"{label}{suffix}", detail=detail if db_ok else "Database not connected"))

    sample_users = EntitySample(
        kind="users",
        label="Sample logins",
        detail="admin01, fac_anita, fac_vijay, fac_sara, stu_arjun, stu_meera, stu_kiran, stu_divya",
    )
    entities.insert(1, sample_users)
    entities.pop(2)

    return SchemaSummaryResponse(
        schema_name=settings.db_schema,
        modules=MODULES,
        views=VIEWS,
        entities=entities,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Empty message.")

    session_id = req.session_id.strip() or uuid.uuid4().hex[:12]
    try:
        payload: Dict[str, Any] = agent_chat(session_id, message)
    except Exception as exc:  # never leak stack traces to the client
        logger.exception("Agent failure")
        return JSONResponse(
            status_code=502,
            content={
                "session_id": session_id,
                "answer": "The assistant could not process that question. Please try again.",
                "source_type": "error",
                "detail": f"{type(exc).__name__}",
            },
        )
    return ChatResponse.model_validate(payload)


@app.delete("/api/chat/{session_id}")
def reset_session(session_id: str) -> Dict[str, Any]:
    removed = clear_session(session_id)
    return {"session_id": session_id, "cleared": removed}


# --- Static Files / Production Frontend Mounting ---
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.is_dir():
    if (frontend_dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
    async def serve_spa(full_path: str):
        target = frontend_dist / full_path
        if full_path and target.is_file():
            return FileResponse(target)
        return FileResponse(frontend_dist / "index.html")

