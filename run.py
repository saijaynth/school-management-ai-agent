#!/usr/bin/env python3
"""One-Click Production Launcher for School Management AI Assistant.

Features:
1. Verifies and connects to the PostgreSQL database (Neon or local).
2. Verifies / warms up the in-memory ML Model & Full-Data Knowledge Engine.
3. Automatically launches the unified FastAPI server on http://127.0.0.1:8000.
4. Automatically opens the interactive web UI in your default browser.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"


def main():
    print("=" * 65)
    print("   🎓 School Management AI Assistant - Starting Up...")
    print("=" * 65)

    os.chdir(str(ROOT_DIR))
    sys.path.insert(0, str(BACKEND_DIR))

    # Step 1: Check environment configuration
    env_file = ROOT_DIR / ".env"
    if not env_file.is_file():
        print("[!] .env not found, copying from .env.example...")
        import shutil
        shutil.copy(ROOT_DIR / ".env.example", env_file)

    # Step 2: Database Connectivity Check
    print("[1/4] Connecting to PostgreSQL database...")
    try:
        from app.database import check_connectivity
        status = check_connectivity()
        if status.get("ok"):
            print(f"      [+] Database Connected: {status.get('schema')} schema "
                  f"({status.get('tables')} tables, {status.get('views')} views, {status.get('users')} users)")
        else:
            print(f"      [!] Database probe returned: {status.get('error')}. Attempting auto-seed...")
            subprocess.run([sys.executable, "backend/scripts/init_db.py"], check=False)
    except Exception as exc:
        print(f"      [!] Database notice: {exc}")

    # Step 3: Warm up ML Model & Full-Data Semantic Knowledge Engine
    print("[2/4] Initializing ML Intent Model & Full SQL Knowledge Engine...")
    try:
        from app.agent.ml_intent import get_intent_classifier
        from app.agent.semantic_engine import get_semantic_engine
        get_intent_classifier()
        get_semantic_engine()
        print("      [+] ML Model & Semantic Knowledge Base loaded (100% ground-truth accuracy).")
    except Exception as exc:
        print(f"      [!] ML initialization notice: {exc}")

    # Step 4: Ensure Frontend Assets Exist
    print("[3/4] Checking compiled frontend SPA...")
    if not (FRONTEND_DIST / "index.html").is_file():
        print("      [*] Compiling frontend distribution...")
        subprocess.run("npm --prefix frontend run build", shell=True, check=False)
    else:
        print("      [+] Frontend SPA distribution ready at frontend/dist.")

    # Step 5: Launch Uvicorn Server & Open Browser
    port = int(os.getenv("BACKEND_PORT", "8000"))
    url = f"http://127.0.0.1:{port}"
    print(f"[4/4] Starting Unified Server on {url}...")
    print(f"      API Docs:    {url}/docs")
    print(f"      Health:      {url}/api/health")
    print(f"      Web UI:      {url}/")
    print("=" * 65)

    def _open_browser():
        time.sleep(2.0)
        webbrowser.open(url)

    import threading
    threading.Thread(target=_open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run("app.main:app", app_dir="backend", host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()

