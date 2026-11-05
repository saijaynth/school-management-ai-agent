"""Tool-calling and Local ML agent orchestrator.

Supports two execution paths:
1. Local ML Agent Core (default): Zero-cost, in-memory Scikit-Learn intent classification,
   entity/slot resolution, schema-aware SQL generation, and grounded response synthesis.
2. External LLM Provider: Multi-turn tool-calling loop (OpenAI, Anthropic, Gemini, Ollama)
   when an external provider with valid credentials is explicitly configured.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from ..config import llm_is_ready, resolve_llm, settings
from .prompt import SYSTEM_PROMPT
from .tools import (
    TOOL_DEFINITIONS,
    execute_school_sql,
    inspect_table_schema,
    lookup_school_manual,
)
from .session_memory import clear_session as clear_session_state, get_session, update_session
from .entity_extractor import KNOWN_SECTIONS, extract_entities
from .ml_intent import predict_intent
from .sql_synthesizer import synthesize_and_execute
from .response_generator import build_response

logger = logging.getLogger("school_ai.orchestrator")

_MAX_TOOL_ROUNDS = settings.agent_max_tool_rounds
_sessions: Dict[str, List[Dict[str, Any]]] = {}
_sessions_lock = threading.Lock()
MAX_HISTORY_MESSAGES = 24


def _history(session_id: str) -> List[Dict[str, Any]]:
    with _sessions_lock:
        return list(_sessions.get(session_id, []))


def _append(session_id: str, message: Dict[str, Any]) -> None:
    with _sessions_lock:
        history = _sessions.setdefault(session_id, [])
        history.append(message)
        if len([m for m in history if m.get("role") == "user"]) > MAX_HISTORY_MESSAGES:
            _sessions[session_id] = history[-(MAX_HISTORY_MESSAGES * 3 * 2):]


def clear_session(session_id: str) -> bool:
    clear_session_state(session_id)
    with _sessions_lock:
        return _sessions.pop(session_id, None) is not None


def _build_client():
    from openai import OpenAI

    cfg = resolve_llm()
    api_key = cfg["api_key"] or "not-needed"
    base_url = cfg["base_url"] or None
    return cfg, OpenAI(api_key=api_key, base_url=base_url, timeout=90.0)


def _execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "execute_school_sql":
        return execute_school_sql(args.get("sql_query", ""), args.get("rationale", ""))
    if name == "inspect_table_schema":
        return inspect_table_schema(args.get("table_or_view_names", []))
    if name == "lookup_school_manual":
        return lookup_school_manual(args.get("topic_or_query", ""))
    return {"ok": False, "error": f"Unknown tool: {name}"}


def _local_ml_agent_turn(session_id: str, message: str) -> Dict[str, Any]:
    """Execute one turn using the in-memory Scikit-Learn ML Agent core."""
    session = get_session(session_id)
    entities = extract_entities(message, session)
    intent, confidence = predict_intent(message)

    # Multi-turn context resolution for homework/assignment follow-ups
    if intent == "homework_assignment_pending" or entities.refers_to_multi_submission:
        if not entities.sections:
            if session.last_section_id == 2:
                entities.sections.append(KNOWN_SECTIONS[1])
            else:
                entities.sections.append(KNOWN_SECTIONS[0])  # Grade 9-A default

    # Persist detected entities into conversational session state
    student_id = entities.students[0]["student_id"] if entities.students else None
    student_name = entities.students[0]["name"] if entities.students else None
    faculty_id = entities.faculty[0]["faculty_id"] if entities.faculty else None
    faculty_name = entities.faculty[0]["name"] if entities.faculty else None
    section_id = entities.sections[0]["section_id"] if entities.sections else None
    section_full = entities.sections[0]["full_name"] if entities.sections else None
    class_name = entities.classes[0] if entities.classes else None
    subject = entities.subjects[0]["name"] if entities.subjects else None
    subject_id = entities.subjects[0]["subject_id"] if entities.subjects else None

    update_session(
        session_id,
        student_id=student_id,
        student_name=student_name,
        faculty_id=faculty_id,
        faculty_name=faculty_name,
        section_id=section_id,
        section_full=section_full,
        class_name=class_name,
        subject=subject,
        subject_id=subject_id,
        intent=intent,
    )

    # Synthesize and execute verified read-only SQL
    sql_results, meta = synthesize_and_execute(intent, entities, message)

    # Build grounded natural language response
    payload = build_response(intent, message, entities, sql_results, meta)
    payload["session_id"] = session_id

    # If the response is a clarification or empty, consult the Full SQL Semantic Knowledge Base
    if payload.get("source_type") == "clarification" or (
        payload.get("source_type") == "database"
        and not sql_results
        and "0" not in payload.get("answer", "")
    ):
        from .semantic_engine import query_full_data_knowledge
        facts = query_full_data_knowledge(message, top_k=1)
        if facts and facts[0][1] >= 0.22:
            fact, score = facts[0]
            payload["answer"] = f"### {fact.title}\n\n{fact.text}"
            payload["source_type"] = "database"

    # Generate Chain-of-Thought reasoning for transparency & autonomous thinking
    from .thought_engine import AgentThoughtEngine
    sql_plans = [r.get("sql", "") for r in sql_results if isinstance(r, dict) and r.get("sql")]
    manual_sections = [m.get("section", "") for m in payload.get("manual_references", [])]
    thought_trace = AgentThoughtEngine.generate_thought_trace(
        query=message,
        intent=intent,
        confidence=confidence,
        slots={
            "student_name": student_name,
            "faculty_name": faculty_name,
            "section_name": section_full,
            "class_name": class_name,
            "subject_name": subject,
            "date": entities.dates[0] if entities.dates else None,
            "weekday": entities.weekdays[0] if entities.weekdays else None,
            "low_threshold": entities.thresholds[0] if entities.thresholds else None,
        },
        source_type=payload.get("source_type", "database"),
        sql_plans=sql_plans,
        manual_sections=manual_sections,
        session_id=session_id,
    )
    payload["thinking"] = thought_trace

    # Record message history in session
    update_session(
        session_id,
        user_message=message,
        assistant_message=payload["answer"],
    )

    return payload


def chat(session_id: str, message: str) -> Dict[str, Any]:
    """Run one agent turn end to end and return the API response payload."""
    session_id = session_id or uuid.uuid4().hex[:12]
    cfg = resolve_llm()

    # Route through Local ML Agent if provider is 'local' or no external key is configured
    if cfg["provider"] == "local" or not cfg.get("api_key"):
        return _local_ml_agent_turn(session_id, message)

    # External LLM provider execution path
    try:
        cfg, client = _build_client()
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(_history(session_id))
        messages.append({"role": "user", "content": message})

        sql_queries: List[Dict[str, Any]] = []
        manual_references: List[Dict[str, Any]] = []
        final_text = ""
        iterations = 0

        while iterations < _MAX_TOOL_ROUNDS:
            iterations += 1
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,  # type: ignore[arg-type]
                tools=TOOL_DEFINITIONS,
                temperature=settings.llm_temperature,
            )
            choice = response.choices[0]
            msg = choice.message

            assistant_entry: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            tool_calls = msg.tool_calls or []
            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_entry)

            if not tool_calls:
                final_text = msg.content or ""
                break

            for tc in tool_calls:
                t_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                tool_result = _execute_tool(t_name, args)
                if t_name == "execute_school_sql" and tool_result.get("ok"):
                    sql_queries.append(tool_result)
                if t_name == "lookup_school_manual" and tool_result.get("ok"):
                    for sec in tool_result.get("sections", []):
                        manual_references.append(sec)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": t_name,
                        "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                    }
                )

        _append(session_id, {"role": "user", "content": message})
        _append(session_id, {"role": "assistant", "content": final_text})

        source_type = "hybrid" if (sql_queries and manual_references) else ("database" if sql_queries else ("manual" if manual_references else "clarification"))

        return {
            "session_id": session_id,
            "answer": final_text,
            "source_type": source_type,
            "sql_queries": sql_queries,
            "manual_references": manual_references,
            "suggested_followups": [
                "Which leave requests are pending approval?",
                "Show all student attendance percentages",
                "What is Grade 10-A's Monday timetable?",
            ],
            "llm_ready": True,
        }
    except Exception as exc:
        logger.warning("External LLM failure, falling back to Local ML Agent: %s", exc)
        return _local_ml_agent_turn(session_id, message)
