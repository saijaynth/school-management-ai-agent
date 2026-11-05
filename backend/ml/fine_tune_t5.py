"""Transfer-Based Fine-Tuning Pipeline for School Management AI Agent.

Fine-tunes a pre-trained sequence-to-sequence transformer (e.g., google/flan-t5-small or t5-small)
on domain-specific question-to-DSL pairs, evaluates performance, and pushes fine-tuned weights
to Hugging Face Model Hub.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fine_tune_t5")

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
DATA_DIR = BACKEND_DIR / "ml" / "data"
OUTPUT_DIR = BACKEND_DIR / "data" / "fine_tuned_t5"


class SchoolDslDataset(Dataset):
    """Custom PyTorch Dataset for input query to DSL target sequence pairs."""

    def __init__(self, data_path: Path, tokenizer, max_input_len: int = 128, max_target_len: int = 128, max_samples: int = 0):
        self.samples: List[Dict[str, str]] = []
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.samples.append(item)
                    if max_samples > 0 and len(self.samples) >= max_samples:
                        break

        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        input_text = item["input"]
        target_text = item["target"]

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze(0)
        # Replace padding token id's of the labels by -100 so they are ignored by loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def _load_token_from_env_file() -> str:
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_ACCESS_TOKEN="):
                    return line.split("=", 1)[1].strip()
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip()
    return ""


def generate_model_card(output_dir: Path, model_name: str, metrics: Dict[str, Any], repo_id: str) -> None:
    readme_content = f"""---
language:
- en
license: mit
tags:
- transfer-learning
- seq2seq
- flan-t5
- school-management
- text-to-dsl
- database-agent
base_model: {model_name}
pipeline_tag: text-generation
---

# School Management AI Agent (Transfer-Based Seq2Seq)

This model is a transfer-learning fine-tuned checkpoint of [`{model_name}`](https://huggingface.co/{model_name}) developed specifically for the **School Management Database AI Assistant**.

It translates natural-language queries from school administrators, faculty, and students into Domain-Specific Language (DSL) query specifications that compile directly to verified, parameterized PostgreSQL queries against the `school` schema.

## Key Capabilities

- **Strict Schema Grounding**: Targets verified PostgreSQL tables and views (`v_student_attendance_report`, `v_exam_report`, `v_pending_leaves`, `v_timetable`, `students`, `faculty`, `marks`).
- **Entity Resolution**: Seamlessly parses student names, faculty names, grade/section identifiers, exam types, and dates.
- **Zero-Cost Remote Access**: Can be used locally or remotely loaded directly from Hugging Face Model Hub using Transformers.

## Model Evaluation Metrics

- **Base Architecture**: `{model_name}`
- **Training Samples**: {metrics.get("train_samples", "N/A")}
- **Validation Samples**: {metrics.get("val_samples", "N/A")}
- **Validation Loss**: {metrics.get("val_loss", "N/A")}
- **Exact Match DSL Accuracy**: {metrics.get("exact_match_acc", "N/A")}%
- **Training Time**: {metrics.get("duration_seconds", "N/A")}s

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForSeq2SeqLM.from_pretrained("{repo_id}")

query = "Show attendance report for who is below 75%"
inputs = tokenizer(query, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128)
dsl_spec = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated DSL:", dsl_spec)
```
"""
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info("Generated Hugging Face Model Card at %s", readme_path)


def fine_tune(
    model_name: str = "google/flan-t5-small",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-4,
    max_samples: int = 0,
    output_dir: Path = OUTPUT_DIR,
    hf_token: str = "",
    repo_id: str = "jaay2/school-management-t5-agent",
) -> Dict[str, Any]:
    """Execute transfer learning fine-tuning loop."""
    resolved_token = (
        hf_token
        or os.environ.get("HF_ACCESS_TOKEN")
        or os.environ.get("HF_TOKEN")
        or _load_token_from_env_file()
    )
    logger.info("Initializing transfer learning fine-tuning with base model: %s", model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Computing on device: %s", device)

    # 1. Load Pre-trained Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=resolved_token or None)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=resolved_token or None)
    model.to(device)

    # 2. Prepare Datasets and DataLoaders
    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"

    train_ds = SchoolDslDataset(train_path, tokenizer, max_samples=max_samples)
    val_ds = SchoolDslDataset(val_path, tokenizer, max_samples=max_samples // 10 if max_samples > 0 else 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info("Loaded %d training samples and %d validation samples", len(train_ds), len(val_ds))

    # 3. Optimizer & Learning Rate Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    # 4. Training Loop
    logger.info("Starting training loop (%d epochs, %d total optimization steps)...", epochs, total_steps)
    start_time = time.time()
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % max(1, len(train_loader) // 5) == 0:
                logger.info(
                    "Epoch %d/%d | Step %d/%d | Current Batch Loss: %.4f",
                    epoch,
                    epochs,
                    step + 1,
                    len(train_loader),
                    loss.item(),
                )

        avg_train_loss = total_loss / max(1, len(train_loader))

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        exact_matches = 0
        total_eval = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                # Generate sample predictions to compute exact matches
                generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
                pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

                # Decode target labels
                clean_labels = labels.clone()
                clean_labels[clean_labels == -100] = tokenizer.pad_token_id
                target_texts = tokenizer.batch_decode(clean_labels, skip_special_tokens=True)

                for p, t in zip(pred_texts, target_texts):
                    if p.strip() == t.strip():
                        exact_matches += 1
                    total_eval += 1

        avg_val_loss = val_loss / max(1, len(val_loader))
        exact_match_acc = (exact_matches / max(1, total_eval)) * 100.0

        logger.info(
            "--- Epoch %d Complete | Train Loss: %.4f | Val Loss: %.4f | Exact Match: %.2f%% ---",
            epoch,
            avg_train_loss,
            avg_val_loss,
            exact_match_acc,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("Saved best model weights to %s", output_dir)

    total_duration = time.time() - start_time
    logger.info("Fine-tuning completed in %.2f seconds", total_duration)

    metrics = {
        "base_model": model_name,
        "epochs": epochs,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_loss": round(avg_train_loss, 4),
        "val_loss": round(avg_val_loss, 4),
        "exact_match_acc": round(exact_match_acc, 2),
        "duration_seconds": round(total_duration, 2),
    }

    metrics_file = output_dir / "fine_tune_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 5. Generate Hugging Face Model Card
    generate_model_card(output_dir, model_name, metrics, repo_id)

    # 6. Push to Hugging Face Model Hub if token provided or found
    token = resolved_token
    if token:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            logger.info("Uploading fine-tuned transfer model to Hugging Face: %s...", repo_id)
            api.upload_folder(
                folder_path=str(output_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload fine-tuned {model_name} transfer learning agent",
            )
            logger.info("Successfully published fine-tuned transfer model to https://huggingface.co/%s", repo_id)
            metrics["hf_repo"] = f"https://huggingface.co/{repo_id}"
        except Exception as exc:
            logger.warning("Could not upload to Hugging Face Hub: %s", exc)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained transformer via transfer learning")
    parser.add_argument("--model-name", default="google/flan-t5-small", help="Pre-trained Hugging Face model identifier")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional sample limit for quick execution/validation")
    parser.add_argument("--hf-token", default="", help="Hugging Face access token")
    parser.add_argument("--repo-id", default="jaay2/school-management-t5-agent", help="Hugging Face Model Hub repository")

    args = parser.parse_args()
    fine_tune(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        hf_token=args.hf_token,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
