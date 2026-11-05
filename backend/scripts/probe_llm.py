import json, time, urllib.request

prompt = """You write PostgreSQL. Schema (schema school):
students(student_id, user_id, admission_no, roll_no, section_id, gender)
users(user_id, first_name, last_name, email)
sections(section_id, class_id, section_name, class_teacher_id)
classes(class_id, class_name)
faculty(faculty_id, user_id, employee_no)
Question: Who is the class teacher of Grade 10-A?
Return only one SELECT statement."""
body = json.dumps({"model": "gemma:2b", "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_predict": 150}}).encode()
t = time.time()
req = urllib.request.Request("http://127.0.0.1:11434/api/generate", body, {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=300))
print("seconds:", round(time.time() - t, 1), "tokens/s:", round(r["eval_count"] / (r["eval_duration"] / 1e9), 1))
print(r["response"])
