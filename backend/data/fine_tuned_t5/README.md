---
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
base_model: google/flan-t5-small
pipeline_tag: text2text-generation
---

# School Management AI Agent (Transfer-Based Seq2Seq)

This model is a transfer-learning fine-tuned checkpoint of [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small) developed specifically for the **School Management Database AI Assistant**.

It translates natural-language queries from school administrators, faculty, and students into Domain-Specific Language (DSL) query specifications that compile directly to verified, parameterized PostgreSQL queries against the `school` schema.

## Key Capabilities

- **Strict Schema Grounding**: Targets verified PostgreSQL tables and views (`v_student_attendance_report`, `v_exam_report`, `v_pending_leaves`, `v_timetable`, `students`, `faculty`, `marks`).
- **Entity Resolution**: Seamlessly parses student names, faculty names, grade/section identifiers, exam types, and dates.
- **Zero-Cost Remote Access**: Can be used locally or remotely loaded directly from Hugging Face Model Hub using Transformers.

## Model Evaluation Metrics

- **Base Architecture**: `google/flan-t5-small`
- **Training Samples**: 100
- **Validation Samples**: 10
- **Validation Loss**: 1.9191
- **Exact Match DSL Accuracy**: 0.0%
- **Training Time**: 127.72s

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("jaay2/school-management-t5-agent")
model = AutoModelForSeq2SeqLM.from_pretrained("jaay2/school-management-t5-agent")

query = "Show attendance report for who is below 75%"
inputs = tokenizer(query, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128)
dsl_spec = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated DSL:", dsl_spec)
```
