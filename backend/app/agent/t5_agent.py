"""Inference module for fine-tuned Transfer-Learning Sequence-to-Sequence Agent.

Loads fine-tuned weights from local storage or Hugging Face Hub, predicts DSL queries from
natural language questions, and compiles them via the semantic DSL engine.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

from .dsl import DslError, compile_query, load_value_index, parse
from ..database import run_readonly_query

logger = logging.getLogger("school_ai.t5_agent")

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "fine_tuned_t5"
HF_REPO_ID = "jaay2/school-management-t5-agent"
FALLBACK_BASE_MODEL = "google/flan-t5-small"

_model = None
_tokenizer = None
_value_index = None
_lock = threading.Lock()


def _get_model_and_tokenizer():
    global _model, _tokenizer
    with _lock:
        if _model is not None:
            return _model, _tokenizer

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Priority 1: Local fine-tuned weights
            if MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists():
                logger.info("Loading fine-tuned transfer model from %s", MODEL_DIR)
                _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
                _model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
            else:
                # Priority 2: Hugging Face Model Hub repository
                token = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
                logger.info("Attempting to load transfer model from Hugging Face Hub: %s", HF_REPO_ID)
                try:
                    _tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, token=token)
                    _model = AutoModelForSeq2SeqLM.from_pretrained(HF_REPO_ID, token=token)
                except Exception as exc:
                    logger.info("HF repo not yet ready, using base transfer model %s: %s", FALLBACK_BASE_MODEL, exc)
                    _tokenizer = AutoTokenizer.from_pretrained(FALLBACK_BASE_MODEL)
                    _model = AutoModelForSeq2SeqLM.from_pretrained(FALLBACK_BASE_MODEL)

            _model.to(device)
            _model.eval()
            return _model, _tokenizer
        except Exception as exc:
            logger.warning("Could not initialize T5 transfer model: %s", exc)
            return None, None


def predict_dsl(question: str) -> Optional[str]:
    """Generate DSL query string for a natural language question using transfer model."""
    model, tokenizer = _get_model_and_tokenizer()
    if model is None or tokenizer is None:
        return None

    try:
        import torch

        device = next(model.parameters()).device
        prompt = f"q: {question.strip()}"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=2,
                early_stopping=True,
            )

        dsl_pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return dsl_pred if dsl_pred else None
    except Exception as exc:
        logger.warning("Transfer model prediction error: %s", exc)
        return None


def execute_transfer_query(question: str) -> Tuple[bool, Optional[list], Optional[str]]:
    """Predict DSL, compile to SQL, and execute against PostgreSQL."""
    dsl_text = predict_dsl(question)
    if not dsl_text:
        return False, None, None

    global _value_index
    with _lock:
        if _value_index is None:
            try:
                _value_index = load_value_index()
            except Exception as exc:
                logger.warning("Could not load ValueIndex: %s", exc)
                return False, None, None

    try:
        items = parse(dsl_text)
        if not items:
            return False, None, None
        first = items[0]
        if hasattr(first, "view"):
            compiled = compile_query(first, _value_index)
            res = run_readonly_query(compiled.sql, compiled.params)
            return True, res["rows"], compiled.sql
    except DslError:
        pass
    except Exception as exc:
        logger.debug("DSL compilation error for '%s': %s", dsl_text, exc)

    return False, None, None

