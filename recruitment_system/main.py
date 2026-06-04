#!/usr/bin/env python3
"""
Multi-Agent Recruitment Intelligence System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Entry point — runs the full LangGraph pipeline and renders a terminal report.

Usage:
  # Analyse a resume against a job title / description
  python main.py analyse --resume data/sample_resume.txt \
                         --job-title "Senior Software Engineer — Backend" \
                         --job-description "..." \
                         [--output report.json]

  # Load job descriptions into the vector store
  python main.py load-jobs --file data/sample_jobs.json

  # Full demo with bundled sample data
  python main.py demo
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Force UTF-8 on Windows so emojis and box-drawing chars render correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()

# Add project root to sys.path so imports work cleanly
sys.path.insert(0, str(Path(__file__).parent))

from graph.recruitment_graph import build_recruitment_graph
from models.schemas import RecruitmentState
from rag.vector_store import RecruitmentVectorStore
from utils.helpers import load_resume
from utils.reporter import print_full_report

console = Console()
app = typer.Typer(
    name="recruitment-ai",
    help="Multi-Agent Recruitment Intelligence System",
    add_completion=False,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _get_vector_store() -> RecruitmentVectorStore:
    return RecruitmentVectorStore(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("load-jobs")
def load_jobs(
    file: Path = typer.Option(
        Path("data/sample_jobs.json"),
        "--file", "-f",
        help="JSON file containing job descriptions",
    )
) -> None:
    """Load job descriptions into the ChromaDB vector store."""
    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(1)

    vs = _get_vector_store()
    count = vs.load_jobs_from_file(file)
    console.print(f"[green]✓[/green] Loaded [bold]{count}[/bold] job descriptions into vector store.")


@app.command("analyse")
def analyse(
    resume: Path = typer.Option(..., "--resume", "-r", help="Path to resume file (.txt, .pdf, .docx)"),
    job_title: str = typer.Option(..., "--job-title", "-t", help="Target job title"),
    job_description: str = typer.Option("", "--job-description", "-d", help="Job description text"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Save JSON results to this file"),
    model: str | None = typer.Option(None, "--model", "-m", help="LLM model ID override"),
) -> None:
    """Analyse a single resume against a job description."""
    if not resume.exists():
        console.print(f"[red]Resume not found:[/red] {resume}")
        raise typer.Exit(1)

    console.print(f"\n[cyan]Loading resume:[/cyan] {resume}")
    resume_text = load_resume(resume)

    _run_pipeline(
        resume_text=resume_text,
        job_title=job_title,
        job_description=job_description or job_title,
        output=output,
        model=model,
    )


@app.command("demo")
def demo(
    output: Path | None = typer.Option(None, "--output", "-o", help="Save JSON results to this file"),
    model: str | None = typer.Option(None, "--model", "-m", help="LLM model ID override"),
) -> None:
    """
    Run the full pipeline with bundled sample data.
    Loads sample jobs + analyses sample_resume.txt against a Senior Backend Engineer role.
    """
    console.print("\n[bold cyan]⚡ Recruitment Intelligence — Demo Mode[/bold cyan]")

    # Auto-load sample jobs
    jobs_file = Path("data/sample_jobs.json")
    if jobs_file.exists():
        vs = _get_vector_store()
        count = vs.load_jobs_from_file(jobs_file)
        console.print(f"[green]✓[/green] Loaded {count} sample job descriptions into vector store")

    resume_file = Path("data/sample_resume.txt")
    if not resume_file.exists():
        console.print("[red]data/sample_resume.txt not found. Run from the recruitment_system/ directory.[/red]")
        raise typer.Exit(1)

    resume_text = load_resume(resume_file)

    _run_pipeline(
        resume_text=resume_text,
        job_title="Senior Software Engineer — Backend",
        job_description=(
            "We are seeking a Senior Backend Engineer to design, build, and maintain scalable "
            "microservices powering our data platform. You will work closely with product and data "
            "teams to deliver high-quality, production-grade Python services. The ideal candidate "
            "has deep Python expertise, strong database skills, and experience with Kubernetes and "
            "cloud infrastructure."
        ),
        output=output,
        model=model,
    )


# ---------------------------------------------------------------------------
# Pipeline runner (shared by all commands)
# ---------------------------------------------------------------------------

def _run_pipeline(
    resume_text: str,
    job_title: str,
    job_description: str,
    output: Path | None,
    model: str | None,
) -> None:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        console.print("[red bold]GROQ_API_KEY is not set.[/red bold]")
        console.print("Create a .env file based on .env.example and set your Groq API key.")
        raise typer.Exit(1)

    vs = _get_vector_store()
    graph = build_recruitment_graph(vector_store=vs, model=model)

    initial_state = RecruitmentState(
        resume_text=resume_text,
        job_title=job_title,
        job_description=job_description,
    ).model_dump()

    console.print(f"\n[bold]Job:[/bold] {job_title}")
    console.print("[dim]Running multi-agent pipeline...[/dim]\n")

    _STEPS = [
        ("rag_retrieval",           "RAG — Retrieving similar job context"),
        ("parse_resume",            "Agent 1 — Parsing resume"),
        ("extract_skills",          "Agent 2 — Extracting & analysing skills"),
        ("rank_candidate",          "Agent 3 — Scoring candidate"),
        ("generate_questions",      "Agent 4 — Generating interview plan"),
        ("generate_recommendation", "Agent 5 — Producing hiring recommendation"),
    ]

    final_state_dict: dict = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Starting pipeline...", total=len(_STEPS))

        for i, (node_name, label) in enumerate(_STEPS):
            progress.update(task, description=f"[{i+1}/{len(_STEPS)}] {label}", advance=0)

            # Stream one step at a time for live progress
            if i == 0:
                current = initial_state
            else:
                current = final_state_dict

            try:
                # Invoke the full graph (LangGraph handles checkpointing internally)
                if i == 0:
                    final_state_dict = graph.invoke(initial_state)
                    break  # graph.invoke runs all nodes end-to-end
            except Exception as exc:
                console.print(f"\n[red]Pipeline error at {node_name}: {exc}[/red]")
                raise typer.Exit(1)

            progress.advance(task)

    final_state = RecruitmentState(**final_state_dict)

    # Render rich terminal report
    print_full_report(final_state)

    # Optional JSON export
    if output:
        output.write_text(
            json.dumps(final_state_dict, indent=2, default=str),
            encoding="utf-8",
        )
        console.print(f"[green]✓[/green] Results saved to [bold]{output}[/bold]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
