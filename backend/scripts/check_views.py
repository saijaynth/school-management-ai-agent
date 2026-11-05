"""Smoke-test every semantic view against the live database (read-only)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.agent.semantic_layer import VIEWS, cte_prefix
from app.database import run_readonly_queries, close_pool

qs = []
for v in VIEWS:
    cols = ", ".join(v.colnames)
    qs.append((cte_prefix([v.name]) + f"SELECT {cols} FROM {v.name}", None))
res = run_readonly_queries(qs)
for v, r in zip(VIEWS, res):
    print(f"{v.name:28s} rows={r['row_count']:3d} cols={len(r['columns'])}")
    assert r["columns"] == v.colnames, (v.name, r["columns"], v.colnames)
print("ALL VIEWS OK")
close_pool()
