"""Deploy School Management AI Agent to Hugging Face Spaces (Docker)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
def _get_token() -> str:
    token = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_ACCESS_TOKEN="):
                return line.split("=", 1)[1].strip()
    return ""

TOKEN = _get_token()

def deploy():
    api = HfApi(token=TOKEN)
    user = api.whoami()
    username = user["name"]
    repo_id = f"{username}/school-management-ai-agent"
    print(f"Authenticated as Hugging Face user: {username}")
    print(f"Target Space repository: {repo_id}")

    # 1. Create or verify the Space exists
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False,
        )
        print(f"Space repository confirmed: https://huggingface.co/spaces/{repo_id}")
    except Exception as exc:
        print(f"Notice on create_repo: {exc}")

    # 2. Upload repository files
    ignore_patterns = [
        ".git/*",
        ".git",
        ".env",
        ".venv/*",
        "venv/*",
        "node_modules/*",
        "frontend/node_modules/*",
        "*.pyc",
        "__pycache__/*",
        "*.safetensors",
        "*.bin",
        "*.pt",
        "_pgportable/*",
        "*.log",
    ]

    print("Uploading project files to Hugging Face Space...")
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=ignore_patterns,
        commit_message="Deploy School Management AI Agent (FastAPI + React Docker container)",
    )

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    direct_app_url = f"https://{username}-school-management-ai-agent.hf.space"

    print("\n" + "=" * 60)
    print("DEPLOYMENT SUCCESSFUL!")
    print(f"Hugging Face Space: {space_url}")
    print(f"Direct Live App URL: {direct_app_url}")
    print("=" * 60 + "\n")

    return space_url, direct_app_url

if __name__ == "__main__":
    deploy()
