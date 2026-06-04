# ⚡ Multi-Agent Recruitment Intelligence System

A production-grade AI recruitment platform powered by **LangGraph**, **LangChain**, **RAG**, and **Groq (Llama 3.3 70B)**. Five specialized agents collaborate through an automated pipeline to parse resumes, extract skills, rank candidates, generate interview plans, and deliver hiring recommendations — all in seconds.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agents](#agents)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Web App](#web-app-localhost)
  - [CLI](#cli)
- [API Reference](#api-reference)
- [RAG Pipeline](#rag-pipeline)
- [Sample Output](#sample-output)

---

## Overview

The system automatically:

| Step | What happens |
|------|-------------|
| 1 | Loads job descriptions into a **ChromaDB vector store** |
| 2 | **RAG retrieval** fetches the most relevant JD context for the role |
| 3 | **Agent 1** parses the raw resume into structured data |
| 4 | **Agent 2** extracts skills, detects gaps vs. job requirements |
| 5 | **Agent 3** scores the candidate across 6 weighted dimensions |
| 6 | **Agent 4** generates a tailored multi-round interview plan |
| 7 | **Agent 5** delivers a final hire/no-hire decision with rationale |

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │         LangGraph StateGraph         │
                        │                                      │
  Resume Text ──────►  │  [START]                             │
  Job Title   ──────►  │     │                                │
  Job Desc    ──────►  │     ▼                                │
                        │  rag_retrieval  ◄── ChromaDB RAG    │
                        │     │                                │
                        │     ▼                                │
                        │  parse_resume   ◄── Agent 1         │
                        │     │                                │
                        │     ▼                                │
                        │  extract_skills ◄── Agent 2         │
                        │     │                                │
                        │     ▼                                │
                        │  rank_candidate ◄── Agent 3         │
                        │     │                                │
                        │     ▼                                │
                        │  generate_questions ◄── Agent 4     │
                        │     │                                │
                        │     ▼                                │
                        │  generate_recommendation ◄─ Agent 5 │
                        │     │                                │
                        │  [END]                               │
                        └─────────────────────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │    RecruitmentState      │
                          │  (Pydantic v2 schema)    │
                          │  • ParsedResume           │
                          │  • SkillAnalysis          │
                          │  • CandidateScore         │
                          │  • InterviewPlan          │
                          │  • HiringRecommendation   │
                          └──────────────────────────┘
```

---

## Agents

### 1. Resume Parsing Agent
**File:** `agents/resume_parser.py`

Converts raw resume text (`.txt`, `.pdf`, `.docx`) into a fully structured `ParsedResume` object.

**Extracts:**
- Personal info — name, email, phone, location, LinkedIn, GitHub
- Work experience — company, title, duration, achievements, technologies
- Education — institution, degree, field, graduation year
- Skills, certifications, projects, languages
- Total years of experience + experience level (entry / mid / senior / lead / executive)

---

### 2. Skill Extraction Agent
**File:** `agents/skill_extractor.py`

Maps candidate skills against job requirements using a detailed taxonomy.

**Produces:**
- Categorised skill list (technical, language, framework, tool, database, soft, domain, certification, platform, methodology)
- Proficiency scores (0–100%) inferred from evidence
- Years of experience per skill
- Matched requirements
- Skill gap analysis with importance and severity scores
- Overall skill coverage score

---

### 3. Candidate Ranking Agent
**File:** `agents/candidate_ranker.py`

Scores the candidate across 6 weighted dimensions on a 0–100 scale.

| Dimension | Weight |
|-----------|--------|
| Technical Skills | 25% |
| Experience Relevance | 20% |
| Skill Coverage | 20% |
| Growth Trajectory | 15% |
| Cultural Indicators | 15% |
| Education Fit | 5% |

**Produces:** overall score, dimension breakdown, strengths, concerns, scoring rationale.

---

### 4. Interview Question Agent
**File:** `agents/interview_agent.py`

Generates a personalised, multi-round interview plan tailored to the candidate's profile and identified gaps.

**Interview rounds:**
- 📞 Screening
- 💻 Technical
- 🧠 Behavioral
- 🏗️ System Design
- 🤝 Culture Fit

Each question includes: purpose, expected answer points, follow-up questions, and red flags to probe.

---

### 5. Hiring Recommendation Agent
**File:** `agents/recommendation_agent.py`

Synthesises all upstream outputs into a final verdict.

**Decisions:** `STRONG_HIRE` · `HIRE` · `MAYBE` · `NO_HIRE` · `STRONG_NO_HIRE`

**Produces:**
- Decision + confidence score
- Executive summary
- Key reasons, risk factors, development plan
- Suggested compensation band
- Recommended start level
- Next steps

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq — `llama-3.3-70b-versatile` |
| Orchestration | LangGraph 1.2 |
| LLM Framework | LangChain 1.3 · langchain-groq |
| Vector Store | ChromaDB 1.5 + ONNX `all-MiniLM-L6-v2` embeddings |
| RAG | langchain-chroma |
| Schemas | Pydantic v2 |
| Web Server | FastAPI 0.136 + Uvicorn |
| Frontend | Tailwind CSS · Vanilla JS |
| CLI | Typer + Rich |
| Document Parsing | pypdf · python-docx · pdfminer.six |
| Config | python-dotenv |

---

## Project Structure

```
recruitment_system/
│
├── agents/                        # The 5 AI agents
│   ├── base.py                    # Shared BaseRecruitmentAgent (ChatGroq + JSON retry)
│   ├── resume_parser.py           # Agent 1 — structured resume extraction
│   ├── skill_extractor.py         # Agent 2 — skill taxonomy + gap detection
│   ├── candidate_ranker.py        # Agent 3 — multi-dimension scoring
│   ├── interview_agent.py         # Agent 4 — tailored interview plan
│   └── recommendation_agent.py   # Agent 5 — hire/no-hire decision
│
├── graph/
│   └── recruitment_graph.py       # LangGraph StateGraph — wires all 6 nodes
│
├── rag/
│   └── vector_store.py            # ChromaDB — job JD ingestion + semantic retrieval
│
├── models/
│   └── schemas.py                 # Pydantic v2 — all typed output models + RecruitmentState
│
├── utils/
│   ├── helpers.py                 # load_resume, extract_json_block, clamp, truncate
│   └── reporter.py                # Rich terminal report renderer
│
├── templates/
│   └── index.html                 # Full-featured web UI (Tailwind + vanilla JS)
│
├── data/
│   ├── sample_jobs.json           # 4 sample job descriptions for the vector store
│   └── sample_resume.txt          # Sample candidate resume for demo mode
│
├── server.py                      # FastAPI web server (localhost:8000)
├── main.py                        # Typer CLI — demo / analyse / load-jobs
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Environment variable template
└── .gitignore                     # Excludes .env, chroma_db/, __pycache__/
```

---

## Installation

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier available)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/dharshan9797/job.git
cd job/recruitment_system

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
# Required — get your key at https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Optional — where ChromaDB stores vector data (default: ./chroma_db)
CHROMA_PERSIST_DIR=./chroma_db

# Optional — Groq model to use
LLM_MODEL=llama-3.3-70b-versatile

# Optional — log verbosity: DEBUG | INFO | WARNING | ERROR
LOG_LEVEL=WARNING
```

> **Note:** `.env` is listed in `.gitignore` and will never be committed.

---

## Usage

### Web App (localhost)

```bash
cd recruitment_system
python server.py
# Open http://localhost:8000
```

**UI Features:**

| Feature | Description |
|---------|-------------|
| Paste Resume | Type or paste resume text directly |
| Upload File | Upload `.txt`, `.pdf`, or `.docx` resume |
| Analyse Resume | Run the full 5-agent pipeline on your input |
| Run Demo | One-click demo with bundled sample candidate |
| Live pipeline steps | Real-time progress indicators per agent |
| 👤 Profile tab | Candidate info, work history, education |
| 🔧 Skills tab | Skill bars, coverage %, gap table |
| 📊 Score tab | 6-dimension scoring + strengths/concerns |
| 🎯 Interview tab | Interview questions by round + red flags |
| 📋 Details tab | Decision, compensation, development plan |

---

### CLI

```bash
cd recruitment_system

# Run the built-in demo (sample resume + Senior Backend Engineer role)
python main.py demo

# Analyse a specific resume file
python main.py analyse \
  --resume path/to/resume.pdf \
  --job-title "Data Engineer" \
  --job-description "We need a data engineer with Spark and dbt experience..."

# Save results to JSON
python main.py analyse \
  --resume resume.txt \
  --job-title "ML Engineer" \
  --output results.json

# Pre-load job descriptions into the vector store
python main.py load-jobs --file data/sample_jobs.json
```

---

## API Reference

Base URL: `http://localhost:8000`

### `GET /`
Returns the web UI (HTML page).

---

### `POST /api/analyse`
Analyse a resume (pasted text) against a job description.

**Request body:**
```json
{
  "resume_text": "John Doe\njohn@example.com\n...",
  "job_title": "Senior Backend Engineer",
  "job_description": "We are looking for..."
}
```

**Response:** Full `RecruitmentState` as JSON — see [Response Schema](#response-schema).

---

### `POST /api/analyse-file`
Analyse an uploaded resume file.

**Form fields:**
| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Resume file (`.txt`, `.pdf`, `.docx`) |
| `job_title` | string | Target job title |
| `job_description` | string | Job description text |

---

### `POST /api/demo`
Run the pipeline with the bundled sample candidate against a Senior Backend Engineer role. No request body needed.

---

### `GET /api/health`
Health check.

```json
{ "status": "ok", "model": "llama-3.3-70b-versatile" }
```

---

### Response Schema

All `/api/analyse*` and `/api/demo` endpoints return:

```json
{
  "resume_text": "...",
  "job_title": "...",
  "job_description": "...",
  "retrieved_job_context": ["..."],

  "parsed_resume": {
    "candidate_name": "Alex Morgan",
    "email": "alex@email.com",
    "location": "San Francisco, CA",
    "total_experience_years": 7.0,
    "experience_level": "senior",
    "work_experience": [...],
    "education": [...],
    "raw_skills": [...],
    "certifications": [...]
  },

  "skill_analysis": {
    "skills": [
      { "name": "Python", "category": "language", "proficiency": 0.9, "years_experience": 7.0 }
    ],
    "skill_gaps": [
      { "skill": "Apache Spark", "importance": 0.6, "gap_severity": 1.0, "notes": "..." }
    ],
    "coverage_score": 0.8,
    "gap_summary": "..."
  },

  "candidate_score": {
    "overall_score": 82.0,
    "dimensions": [...],
    "strengths": [...],
    "concerns": [...]
  },

  "interview_plan": {
    "recommended_rounds": ["screening", "technical", "behavioral", "system_design", "culture_fit"],
    "questions": [
      {
        "round": "technical",
        "question": "...",
        "purpose": "...",
        "follow_ups": ["..."]
      }
    ],
    "time_estimate_hours": 4.5
  },

  "recommendation": {
    "decision": "hire",
    "confidence": 0.82,
    "summary": "...",
    "key_reasons": [...],
    "risk_factors": [...],
    "development_plan": [...],
    "suggested_compensation_band": "$145k–$170k / Senior IC",
    "suggested_start_level": "senior",
    "next_steps": [...]
  },

  "errors": [],
  "current_node": "generate_recommendation"
}
```

---

## RAG Pipeline

Job descriptions are embedded using ChromaDB's built-in `all-MiniLM-L6-v2` ONNX model (downloaded automatically on first run, ~79 MB, cached at `~/.cache/chroma/`).

**How it works:**
1. `load-jobs` or `demo` upserts job descriptions into the `job_descriptions` ChromaDB collection
2. Before each analysis, a semantic query `"{job_title}: {job_description}"` retrieves the top-3 most similar JDs
3. Retrieved context is injected into the Skill Extraction and Candidate Ranking agents to sharpen gap detection

**Adding your own jobs:**

Create a JSON file matching this structure and run `python main.py load-jobs --file your_jobs.json`:

```json
[
  {
    "id": "unique-job-id",
    "title": "Senior Data Engineer",
    "department": "Engineering",
    "level": "Senior",
    "description": "...",
    "requirements": ["5+ years Python", "Apache Spark", "dbt"],
    "nice_to_have": ["Kafka", "Airflow"],
    "tech_stack": ["Python", "Spark", "PostgreSQL", "Kubernetes"]
  }
]
```

---

## Sample Output

Running `python main.py demo` or clicking **Run Demo** in the web UI analyses the bundled candidate **Alex Morgan** against a Senior Backend Engineer role:

```
┌─────────────────────────── 👤 Candidate Profile ─────────────────────┐
│ Candidate    Alex Morgan                                              │
│ Email        alex.morgan@email.com                                    │
│ Experience   7.0 years (senior)                                       │
└───────────────────────────────────────────────────────────────────────┘

Skill Coverage ████████████████░░░░ 80%

Top Skills: Python 90% · FastAPI 80% · PostgreSQL 80% · Kubernetes 80%

Skill Gaps:  Apache Spark · dbt · GitOps · Security hardening

Overall Score: 82.0 / 100

┌────────────────────────── ⚖️  HIRING DECISION ──────────────────────┐
│  ✅ HIRE   Confidence: 82%   Compensation: $145k–$170k / Senior IC  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## License

MIT
