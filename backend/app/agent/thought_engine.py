"""Agent Thought Engine: Chain-of-Thought Reasoning & Self-Reflection.

Empowers the AI agent to "think on its own" before executing queries:
1. Semantic intent perception & confidence calibration.
2. Contextual entity resolution (memory & pronoun grounding).
3. Schema constraints reasoning (views, foreign keys, attendance formulas).
4. Safety & mutation validation.
5. Structured thought summary generation for user transparency.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class AgentThoughtEngine:
    """Generates transparent, grounded chain-of-thought reasoning for each interaction."""

    @staticmethod
    def generate_thought_trace(
        query: str,
        intent: str,
        confidence: float,
        slots: Dict[str, Any],
        source_type: str,
        sql_plans: List[str],
        manual_sections: List[str],
        session_id: str,
    ) -> str:
        confidence_pct = round(confidence * 100, 1)
        
        # Step 1: Perception
        lines = [
            f"**1. Perception & Intent Analysis**",
            f"- Natural Language Query: \"{query}\"",
            f"- Classified Intent: `{intent}` (Confidence: {confidence_pct}%)",
            f"- Target Knowledge Source: `{source_type}`",
        ]

        # Step 2: Context & Entity Grounding
        active_entities = []
        if slots.get("student_name"):
            active_entities.append(f"Student: `{slots['student_name']}`")
        if slots.get("faculty_name"):
            active_entities.append(f"Faculty: `{slots['faculty_name']}`")
        if slots.get("section_name"):
            active_entities.append(f"Section: `{slots['section_name']}`")
        if slots.get("class_name"):
            active_entities.append(f"Class: `{slots['class_name']}`")
        if slots.get("subject_name"):
            active_entities.append(f"Subject: `{slots['subject_name']}`")
        if slots.get("date"):
            active_entities.append(f"Date: `{slots['date']}`")
        if slots.get("weekday"):
            active_entities.append(f"Weekday: `{slots['weekday']}`")
        if slots.get("low_threshold"):
            active_entities.append(f"Threshold: `< {slots['low_threshold']}%`")

        entity_summary = ", ".join(active_entities) if active_entities else "None (Global/Summary query)"
        lines.append(f"\n**2. Context & Entity Resolution**")
        lines.append(f"- Extracted Entities: {entity_summary}")
        lines.append(f"- Active Session: `{session_id}`")

        # Step 3: Schema Reasoning & Policy Rules
        lines.append(f"\n**3. Domain & Policy Reasoning**")
        if "attendance" in intent:
            lines.append("- Applied Rule: Student attendance counts both `'Present'` and `'Late'` as attended periods.")
            lines.append("- Target: Routing to precomputed reporting view `school.v_student_attendance_report`.")
        elif "leave" in intent:
            lines.append("- Applied Rule: Leave requests map applicant to `users.user_id` (not student_id/faculty_id).")
            lines.append("- Target: Utilizing `school.v_pending_leaves` for approval queue or `leave_requests`.")
        elif "timetable" in intent or "class_teacher" in intent:
            lines.append("- Applied Rule: Schedule references `school.v_timetable` and `school.course_assignments`.")
            lines.append("- Constraint: Section class teachers are configured on `sections.class_teacher_id`.")
        elif "exam" in intent or "progress" in intent:
            lines.append("- Applied Rule: Exam marks evaluated against `marks_obtained >= pass_marks`.")
            lines.append("- Data Grounding: Only Grade 10 Mid Term 2026 has scheduled exams in seed database.")
        elif "mutation" in intent:
            lines.append("- Safety Guardrail: Read-only enforcement triggered. No destructive operations allowed.")
        elif source_type == "manual":
            lines.append(f"- Grounding: Referencing User Manual documentation sections: {manual_sections}.")

        # Step 4: Execution Plan
        lines.append(f"\n**4. Execution & Synthesis Plan**")
        if sql_plans:
            lines.append(f"- Read-Only SQL Statements: {len(sql_plans)} query plan(s) formulated.")
            for idx, sql in enumerate(sql_plans, 1):
                clean_sql = " ".join(sql.split())
                lines.append(f"  {idx}. `{clean_sql[:120]}{'...' if len(clean_sql) > 120 else ''}`")
        elif source_type == "manual":
            lines.append("- Procedure Synthesis: Extracting policy instructions from manual corpus.")
        else:
            lines.append("- Response Mode: Direct clarification or refusal notice.")

        return "\n".join(lines)
