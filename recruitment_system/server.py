"""FastAPI web server for the Multi-Agent Recruitment Intelligence System."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graph.recruitment_graph import build_recruitment_graph
from models.schemas import RecruitmentState
from rag.vector_store import RecruitmentVectorStore
from utils.helpers import load_resume

import tempfile
import logging
logging.basicConfig(level=logging.WARNING)

app = FastAPI(title="Recruitment Intelligence System", version="1.0.0")

# Shared vector store
_vs: RecruitmentVectorStore | None = None

def get_vector_store() -> RecruitmentVectorStore:
    global _vs
    if _vs is None:
        _vs = RecruitmentVectorStore(
            persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        )
        jobs_file = Path("data/sample_jobs.json")
        if jobs_file.exists():
            _vs.load_jobs_from_file(jobs_file)
    return _vs


class AnalyseRequest(BaseModel):
    resume_text: str
    job_title: str
    job_description: str


def _run_pipeline(resume_text: str, job_title: str, job_description: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env")

    vs = get_vector_store()
    graph = build_recruitment_graph(vector_store=vs)
    initial = RecruitmentState(
        resume_text=resume_text,
        job_title=job_title,
        job_description=job_description,
    ).model_dump()

    result = graph.invoke(initial)
    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/analyse")
async def analyse(req: AnalyseRequest):
    try:
        result = _run_pipeline(req.resume_text, req.job_title, req.job_description)
        return JSONResponse(content=json.loads(json.dumps(result, default=str)))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyse-file")
async def analyse_file(
    file: UploadFile = File(...),
    job_title: str = Form(...),
    job_description: str = Form(...),
):
    suffix = Path(file.filename or "resume.txt").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        resume_text = load_resume(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    try:
        result = _run_pipeline(resume_text, job_title, job_description)
        return JSONResponse(content=json.loads(json.dumps(result, default=str)))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/demo")
async def demo():
    resume_file = Path("data/sample_resume.txt")
    if not resume_file.exists():
        raise HTTPException(status_code=404, detail="Sample resume not found")

    resume_text = resume_file.read_text(encoding="utf-8")
    job_title = "Senior Software Engineer — Backend"
    job_description = (
        "We are seeking a Senior Backend Engineer to design, build, and maintain scalable "
        "microservices powering our data platform. You will work closely with product and data "
        "teams to deliver high-quality, production-grade Python services. The ideal candidate "
        "has deep Python expertise, strong database skills, and experience with Kubernetes and "
        "cloud infrastructure."
    )
    try:
        result = _run_pipeline(resume_text, job_title, job_description)
        return JSONResponse(content=json.loads(json.dumps(result, default=str)))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
