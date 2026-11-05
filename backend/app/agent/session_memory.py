"""Session memory and entity tracking across conversational turns."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    
    # Entity slots carried over for multi-turn resolution
    last_student_id: Optional[int] = None
    last_student_name: Optional[str] = None
    last_faculty_id: Optional[int] = None
    last_faculty_name: Optional[str] = None
    last_class_name: Optional[str] = None      # e.g. "Grade 9", "Grade 10"
    last_section_name: Optional[str] = None    # e.g. "A"
    last_section_full: Optional[str] = None    # e.g. "Grade 9-A", "Grade 10-A"
    last_section_id: Optional[int] = None      # 1 for 9-A, 2 for 10-A
    last_subject: Optional[str] = None         # "Mathematics", "Science", "English"
    last_subject_id: Optional[int] = None      # 1, 2, 3
    last_intent: Optional[str] = None
    last_topic: Optional[str] = None           # "homework", "assignment", "exam", "attendance", "leave"
    
    messages: List[Dict[str, Any]] = field(default_factory=list)


_sessions: Dict[str, SessionState] = {}
_lock = threading.Lock()
MAX_HISTORY_PER_SESSION = 30


def get_session(session_id: str) -> SessionState:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = SessionState(session_id=session_id)
        session = _sessions[session_id]
        session.last_active = time.time()
        return session


def update_session(
    session_id: str,
    *,
    student_id: Optional[int] = None,
    student_name: Optional[str] = None,
    faculty_id: Optional[int] = None,
    faculty_name: Optional[str] = None,
    class_name: Optional[str] = None,
    section_name: Optional[str] = None,
    section_full: Optional[str] = None,
    section_id: Optional[int] = None,
    subject: Optional[str] = None,
    subject_id: Optional[int] = None,
    intent: Optional[str] = None,
    topic: Optional[str] = None,
    user_message: Optional[str] = None,
    assistant_message: Optional[str] = None,
) -> SessionState:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = SessionState(session_id=session_id)
        s = _sessions[session_id]
        s.last_active = time.time()
        
        if student_id is not None:
            s.last_student_id = student_id
        if student_name is not None:
            s.last_student_name = student_name
        if faculty_id is not None:
            s.last_faculty_id = faculty_id
        if faculty_name is not None:
            s.last_faculty_name = faculty_name
        if class_name is not None:
            s.last_class_name = class_name
        if section_name is not None:
            s.last_section_name = section_name
        if section_full is not None:
            s.last_section_full = section_full
        if section_id is not None:
            s.last_section_id = section_id
        if subject is not None:
            s.last_subject = subject
        if subject_id is not None:
            s.last_subject_id = subject_id
        if intent is not None:
            s.last_intent = intent
        if topic is not None:
            s.last_topic = topic
            
        if user_message:
            s.messages.append({"role": "user", "content": user_message})
        if assistant_message:
            s.messages.append({"role": "assistant", "content": assistant_message})
            
        if len(s.messages) > MAX_HISTORY_PER_SESSION:
            s.messages = s.messages[-MAX_HISTORY_PER_SESSION:]
            
        return s


def clear_session(session_id: str) -> bool:
    with _lock:
        return _sessions.pop(session_id, None) is not None
