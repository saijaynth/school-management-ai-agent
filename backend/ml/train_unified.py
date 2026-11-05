"""Unified MLOps Training & Publishing Pipeline for School Management AI Agent.

Combines:
1. Rich domain samples across 25 schema-grounded intents.
2. Synthetic variations from `backend/ml/data/train.jsonl` and `val.jsonl`.
3. Multi-representation FeatureUnion: Word TF-IDF (1-3 ngrams) + Character Subword TF-IDF (2-5 ngrams).
4. Soft-Voting Ensemble Classifier (Logistic Regression + SGD + Naive Bayes).
5. 5-Fold Stratified Cross Validation.
6. Serializes model to `backend/data/intent_model.joblib`.
7. Automatically publishes model artifacts to Hugging Face Hub (`jaay2/school-management-ml-agent`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
DATA_DIR = BACKEND_DIR / "data"
MODEL_PATH = DATA_DIR / "intent_model.joblib"
METRICS_PATH = DATA_DIR / "model_metrics.json"

sys.path.insert(0, str(BACKEND_DIR))
from scripts.train_ml_model import TRAINING_SAMPLES, build_ensemble_pipeline
from app.agent.training_data import get_training_corpus


def load_all_training_data() -> Tuple[List[str], List[str]]:
    """Aggregate training samples from hardcoded domain seeds and generated dataset."""
    texts: List[str] = []
    labels: List[str] = []

    # 1. Base Domain Samples (300+ samples)
    for text, label in TRAINING_SAMPLES:
        texts.append(text)
        labels.append(label)

    # 2. Schema Seeds from training_data.py
    topic_to_intent = {
        "student_info": "student_count",
        "faculty_info": "faculty_list",
        "class_teacher": "class_teacher",
        "directory": "users_list",
        "subjects": "academic_structure",
        "course_allocation": "timetable_teacher",
        "timetable": "timetable_section",
        "classrooms": "academic_structure",
        "periods": "academic_structure",
        "classes_sections": "academic_structure",
        "academic_year": "academic_structure",
        "student_attendance": "student_attendance_summary",
        "faculty_attendance": "faculty_attendance",
        "attendance_any": "student_attendance_summary",
        "leave_requests": "all_leaves",
        "leave_types": "all_leaves",
        "homework": "homework_list",
        "homework_submissions": "homework_submissions",
        "assignments": "assignments_list",
        "assignment_submissions": "assignments_submissions",
        "exams": "exam_report",
        "marks": "exam_report",
        "progress_reports": "progress_report",
        "grade_scale": "exam_report",
        "notifications": "notifications",
        "manual": "manual_workflow",
        "write_request": "mutation_refusal",
        "out_of_scope": "manual_workflow",
    }

    schema_corpus = get_training_corpus()
    for phrase, topic in schema_corpus:
        intent = topic_to_intent.get(topic)
        if intent:
            texts.append(phrase)
            labels.append(intent)

    # 3. Filter out rare classes (< 5 samples) to ensure robust cross-validation
    label_counts: Dict[str, int] = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1

    filtered_texts, filtered_labels = [], []
    for t, l in zip(texts, labels):
        if label_counts[l] >= 5:
            filtered_texts.append(t)
            filtered_labels.append(l)

    return filtered_texts, filtered_labels


def train_and_publish(hf_token: str = None) -> Dict:
    texts, labels = load_all_training_data()
    unique_classes = sorted(list(set(labels)))
    print(f"Total aggregated dataset: {len(texts)} samples across {len(unique_classes)} classes.")

    pipeline = build_ensemble_pipeline()

    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, texts, labels, cv=skf, scoring="f1_macro")
    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores))

    print(f"5-Fold Cross-Validation Macro F1: {mean_f1:.4f} (+/- {std_f1:.4f})")

    # Fit final ensemble model
    start = time.perf_counter()
    pipeline.fit(texts, labels)
    train_time_ms = round((time.perf_counter() - start) * 1000.0, 2)

    preds = pipeline.predict(texts)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    train_f1 = float(f1_score(labels, preds, average="macro"))
    print(f"Full Dataset Training Macro F1: {train_f1:.4f} (Fit time: {train_time_ms} ms)")

    # Serialize artifacts
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": len(texts),
        "total_classes": len(unique_classes),
        "classes": unique_classes,
        "cv_folds": 5,
        "cv_f1_macro_mean": round(mean_f1, 4),
        "cv_f1_macro_std": round(std_f1, 4),
        "train_f1_macro": round(train_f1, 4),
        "train_time_ms": train_time_ms,
        "per_class_metrics": {k: v for k, v in report.items() if isinstance(v, dict)},
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

    # Optional: Publish to Hugging Face Model Hub
    token = hf_token or os.environ.get("HF_ACCESS_TOKEN")
    if token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            repo_id = "jaay2/school-management-ml-agent"
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            api.upload_file(path_or_fileobj=str(MODEL_PATH), path_in_repo="intent_model.joblib", repo_id=repo_id, repo_type="model")
            api.upload_file(path_or_fileobj=str(METRICS_PATH), path_in_repo="model_metrics.json", repo_id=repo_id, repo_type="model")
            print(f"Successfully published model & metrics to Hugging Face Hub: {repo_id}")
        except Exception as exc:
            print(f"Could not publish to Hugging Face Hub: {exc}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", default=None, help="Hugging Face access token")
    args = parser.parse_args()
    res = train_and_publish(args.hf_token)
    if res["train_f1_macro"] < 0.90:
        sys.exit(1)
    sys.exit(0)
