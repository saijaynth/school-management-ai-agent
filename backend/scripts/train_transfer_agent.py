#!/usr/bin/env python3
"""Transfer Learning & Fine-Tuning Pipeline for School Management AI Assistant.

Trains a sequence-to-sequence transformer (Flan-T5) on natural language queries
mapped to grounded school domain intents and entities, then publishes the fine-tuned
model to Hugging Face Model Hub.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_transfer_agent")

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "backend"))

HF_REPO_ID = "jaay2/school-management-t5-agent"
DEFAULT_BASE_MODEL = "google/flan-t5-small"


def build_training_dataset():
    """Build a comprehensive dataset of natural language queries to grounded domain targets."""
    from app.agent.ml_intent import TRAINING_DATA

    dataset = []
    for query, intent in TRAINING_DATA:
        # Formulate canonical target representation
        target = f"intent: {intent}"
        dataset.append({"input_text": f"school query: {query.strip()}", "target_text": target})

    # Add variations and profile patterns
    extra_pairs = [
        ("who is divya", "intent: person_profile | person: Divya Nair | role: student"),
        ("who is arjun", "intent: person_profile | person: Arjun Rao | role: student"),
        ("who is meera", "intent: person_profile | person: Meera Iyer | role: student"),
        ("who is kiran", "intent: person_profile | person: Kiran Patel | role: student"),
        ("who is anita", "intent: person_profile | person: Anita Sharma | role: faculty"),
        ("who is vijay", "intent: person_profile | person: Vijay Reddy | role: faculty"),
        ("who is sara", "intent: person_profile | person: Sara Thomas | role: faculty"),
        ("who is admin", "intent: person_profile | person: Ramesh Kumar | role: admin"),
        ("who is below 75% attendance?", "intent: student_attendance_low | threshold: 75"),
        ("what is grade 10-a timetable on monday?", "intent: timetable_section | section: Grade 10-A | day: Monday"),
        ("compare kiran and divya marks", "intent: exam_compare | students: Kiran Patel, Divya Nair"),
        ("which leave requests are pending?", "intent: pending_leaves | status: Pending"),
        ("how do i approve leave in the manual?", "intent: manual_workflow | topic: leave_approval"),
    ]

    for q, t in extra_pairs:
        dataset.append({"input_text": f"school query: {q}", "target_text": t})

    return dataset


def train_and_publish(
    base_model_name: str = DEFAULT_BASE_MODEL,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    push_to_hub: bool = True,
    output_dir: str = "backend/data/fine_tuned_t5",
):
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset

    token = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
    logger.info("Initializing transfer learning pipeline with base model: %s", base_model_name)
    logger.info("Hugging Face target repository: %s", HF_REPO_ID)

    raw_data = build_training_dataset()
    logger.info("Constructed %d training samples across domain intents", len(raw_data))

    # Split into train and validation
    split_idx = int(len(raw_data) * 0.85)
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, token=token)

    def preprocess(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")
        labels_matrix = [
            [(t if t != tokenizer.pad_token_id else -100) for t in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels_matrix
        return model_inputs

    tokenized_train = train_ds.map(preprocess, batched=True)
    tokenized_val = val_ds.map(preprocess, batched=True)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=epochs,
        logging_steps=10,
        load_best_model_at_end=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    start_time = time.time()
    logger.info("Starting fine-tuning...")
    trainer.train()
    duration = time.time() - start_time
    logger.info("Training completed in %.2f seconds", duration)

    # Save fine-tuned model and tokenizer
    logger.info("Saving model weights to %s", out_path)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    metrics = {
        "base_model": base_model_name,
        "epochs": epochs,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "duration_seconds": round(duration, 2),
    }
    (out_path / "fine_tune_metrics.json").write_text(json.dumps(metrics, indent=2))

    if push_to_hub and token:
        logger.info("Publishing fine-tuned model to Hugging Face Model Hub: %s", HF_REPO_ID)
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=str(out_path),
                repo_id=HF_REPO_ID,
                repo_type="model",
                commit_message=f"Fine-tune {base_model_name} on School Management domain",
                ignore_patterns=["checkpoints/**"],
            )
            logger.info("Successfully published model to https://huggingface.co/%s", HF_REPO_ID)
        except Exception as e:
            logger.error("Failed to upload model to Hugging Face Hub: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune transformer for School Management AI")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base HF model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--no-push", action="store_true", help="Do not push to Hugging Face Hub")
    args = parser.parse_args()

    train_and_publish(
        base_model_name=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        push_to_hub=not args.no_push,
    )


if __name__ == "__main__":
    main()
