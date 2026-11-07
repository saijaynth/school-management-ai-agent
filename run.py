#!/usr/bin/env python3
"""Launcher script for the School Management application."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"


def main():
    os.chdir(str(ROOT_DIR))
    sys.path.insert(0, str(BACKEND_DIR))

    env_file = ROOT_DIR / ".env"
    if not env_file.is_file() and (ROOT_DIR / ".env.example").is_file():
        import shutil
        shutil.copy(ROOT_DIR / ".env.example", env_file)

    try:
        from app.database import check_connectivity
        status = check_connectivity()
        if not status.get("ok"):
            print("Database connection issue. Run 'python backend/scripts/init_db.py' if needed.")
    except Exception as exc:
        print(f"Database check: {exc}")

    if not (FRONTEND_DIST / "index.html").is_file():
        print("Compiling frontend assets...")
        subprocess.run("npm --prefix frontend run build", shell=True, check=False)

    port = int(os.getenv("PORT", "8000"))
    url = f"http://localhost:{port}"
    print(f"Starting server on {url}")

    def _open_browser():
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run("app.main:app", app_dir="backend", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
