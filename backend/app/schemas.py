"""Pydantic API request/response models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(default="", min_length=0, max_length=64)
    message: str = Field(min_length=1, max_length=4000)


class SqlQueryRecord(BaseModel):
    sql: str
    row_count: int
    columns: List[str]
    rows: List[Dict[str, Any]]
    execution_time_ms: float


class ManualReference(BaseModel):
    section: str
    summary: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    source_type: str  # database | manual | hybrid | clarification
    sql_queries: List[SqlQueryRecord] = []
    manual_references: List[ManualReference] = []
    suggested_followups: List[str] = []
    llm_ready: bool = True
    elapsed_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    database: Dict[str, Any]
    llm: Dict[str, Any]
    counts: Dict[str, int]


class ModuleInfo(BaseModel):
    module: str
    tables: List[str]


class EntitySample(BaseModel):
    kind: str
    label: str
    detail: str


class SchemaSummaryResponse(BaseModel):
    schema_name: str
    modules: List[ModuleInfo]
    views: List[Dict[str, str]]
    entities: List[EntitySample]
