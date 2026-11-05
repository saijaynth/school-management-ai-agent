import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.agent.orchestrator import chat

qs = [
    "who teaches science?",
    "faculty attendance on 2026-11-02",
    "what is the room for Sara Thomas on Monday",
    "how many students are in Grade 10-A",
    "who is absent?",
    "show attendance",
    "what is Divya's email",
    "which subject has Kiran scored the lowest?",
    "tell me about the weather",
    "when is the next exam for grade 9",
    "how many periods per week for maths in grade 10",
]
for q in qs:
    r = chat("probe", q)
    print("Q:", q)
    print("A:", r["answer"][:600].replace("\n", "\n   "))
    print("-" * 60)
