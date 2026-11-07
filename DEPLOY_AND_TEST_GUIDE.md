# School Management AI Chatbot — Deployment & Testing Guide

This guide provides instructions for testing the AI agent remotely, deploying on AWS Free Tier, or running locally with Docker/Python.

---

## 1. Quick Testing (No Installation Required)

You can test the application immediately from any browser:

- **Live Remote App**: **[https://spicy-eggs-wink.loca.lt](https://spicy-eggs-wink.loca.lt)**
  *(If prompted on first visit, enter Tunnel IP: `103.120.51.218`)*
- **GitHub Repository**: **[https://github.com/saijaynth/school-management-ai-agent](https://github.com/saijaynth/school-management-ai-agent)**
- **Hugging Face Model Hub**: **[https://huggingface.co/jaay2/school-management-t5-agent](https://huggingface.co/jaay2/school-management-t5-agent)**

---

## 2. Cloud Database Architecture (AWS us-east-2)

The database and schema are already provisioned and hosted on **AWS us-east-2 (Ohio)** via Neon Serverless PostgreSQL:
- **Connection String**: `postgresql://neondb_owner:npg_Ssqzmd9cDo0l@ep-fancy-cell-bd9rqecw-pooler.c-8.us-east-2.aws.neon.tech/neondb?sslmode=require`
- **Schema**: `school` (29 relational tables, 4 analytical views)
- **Security**: Strictly read-only connection, client parameterization, and AST-level query validation.

---

## 3. Deploying on Amazon Free Tier (EC2 / Lightsail)

To run the application inside an AWS Free Tier instance (`t2.micro` / `t3.micro`):

### Option A: 1-Command Docker Run
```bash
# 1. SSH into your Amazon EC2 instance
ssh -i your-key.pem ec2-user@<YOUR-EC2-PUBLIC-IP>

# 2. Clone the repository
git clone https://github.com/saijaynth/school-management-ai-agent.git
cd school-management-ai-agent

# 3. Start the application container
docker compose up -d --build
```
The application will be live at `http://<YOUR-EC2-PUBLIC-IP>:8000`.

---

## 4. Running Locally in 1 Step

### Using Docker
```bash
docker compose up
# Open http://localhost:8000
```

### Using Python
```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Start the unified server (serves both API & Frontend)
python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

---

## 5. Automated Verification Test Suite

To verify all 50 ground-truth questions against the database:
```bash
python backend/scripts/verify_agent.py
# Result: 50 passed, 0 failed, 0 blocked
```

---

## 6. Recommended Test Questions

| Module | Test Question | Expected Ground-Truth Result |
| :--- | :--- | :--- |
| **Attendance** | *"Student attendance report - who is below 75%?"* | Identifies **Meera Iyer** (0.00% attendance due to sick leave on Nov 2). All other students are at 100%. |
| **Timetable** | *"What is Grade 10-A's Monday timetable?"* | Lists Period 1 English (Sara Thomas, Room 102) and Period 2 Science (Vijay Reddy, Room 101). |
| **Exam Marks** | *"Compare Kiran Patel and Divya Nair's marks"* | Displays Grade 10 Mid Term rankings: Divya Nair (Rank 1, 91.67%, Grade A+) and Kiran Patel (Rank 2, 82.00%, Grade A) with subject marks. |
| **Leaves** | *"Which leave requests are pending approval?"* | Lists Meera Iyer's Sick Leave request for Nov 2–4, 2026. |
| **Guidelines** | *"According to the manual, how do I approve a leave request?"* | Explains the Admin leave approval workflow from Section 6.2 of the User Manual. |
