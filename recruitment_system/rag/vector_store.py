"""
ChromaDB-backed vector store for job descriptions and skills knowledge base.
Supports RAG retrieval for the Candidate Ranking and Recommendation agents.
Uses a lightweight hash-based embedding so no model download is required
(works on serverless platforms like Vercel as well as locally).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings


_COLLECTION_JOBS = "job_descriptions"
_COLLECTION_SKILLS = "skills_knowledge_base"


class _HashEmbedding(EmbeddingFunction):
    """
    Zero-dependency TF-IDF-style hash embedding.
    No model download — works identically on local and serverless environments.
    Dimensionality: 512 floats (sufficient for small-corpus semantic retrieval).
    """
    _DIM = 512

    def __call__(self, input: Documents) -> Embeddings:
        results: Embeddings = []
        for doc in input:
            vec = [0.0] * self._DIM
            tokens = doc.lower().split()
            for token in tokens:
                h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
                vec[h % self._DIM] += 1.0
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            results.append(vec)
        return results


class RecruitmentVectorStore:
    """
    Manages two ChromaDB collections:
      - job_descriptions  : stores JD text for semantic retrieval
      - skills_knowledge_base : stores skill taxonomy for gap analysis
    Uses in-memory client on Vercel (VERCEL env var set) and persistent client locally.
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        self._ef = _HashEmbedding()

        # Use ephemeral (in-memory) client on serverless — no file-system writes needed
        if os.getenv("VERCEL") or os.getenv("VERCEL_ENV"):
            self._client = chromadb.EphemeralClient()
        else:
            self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)

        self._jobs_col = self._client.get_or_create_collection(
            name=_COLLECTION_JOBS,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._skills_col = self._client.get_or_create_collection(
            name=_COLLECTION_SKILLS,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        # On serverless every instance is cold — auto-load sample jobs if present
        if os.getenv("VERCEL") or os.getenv("VERCEL_ENV"):
            _jobs_file = Path(__file__).parent.parent / "data" / "sample_jobs.json"
            if _jobs_file.exists() and self._jobs_col.count() == 0:
                self.load_jobs_from_file(_jobs_file)

    # ------------------------------------------------------------------
    # Job descriptions
    # ------------------------------------------------------------------

    def add_job_description(self, job: dict[str, Any]) -> None:
        """Upsert a single job description into the vector store."""
        doc_text = self._format_job(job)
        self._jobs_col.upsert(
            ids=[job["id"]],
            documents=[doc_text],
            metadatas=[{
                "title": job.get("title", ""),
                "department": job.get("department", ""),
                "level": job.get("level", ""),
            }],
        )

    def load_jobs_from_file(self, path: str | Path) -> int:
        """Load all jobs from a JSON file. Returns count loaded."""
        with open(path, encoding="utf-8") as fh:
            jobs: list[dict[str, Any]] = json.load(fh)
        for job in jobs:
            self.add_job_description(job)
        return len(jobs)

    def retrieve_similar_jobs(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """Return the top-k job descriptions most similar to `query`."""
        count = self._jobs_col.count()
        if count == 0:
            return []
        results = self._jobs_col.query(
            query_texts=[query],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        return [
            {"text": d, "metadata": m, "distance": dist}
            for d, m, dist in zip(docs, metas, dists)
        ]

    def retrieve_job_requirements(self, job_title: str, job_description: str) -> list[str]:
        """
        Semantically retrieve the most relevant JD snippets for a given role,
        returning a list of requirement strings used by downstream agents.
        """
        query = f"{job_title}: {job_description}"
        hits = self.retrieve_similar_jobs(query, k=3)
        requirements: list[str] = []
        for hit in hits:
            requirements.append(hit["text"])
        return requirements

    # ------------------------------------------------------------------
    # Skills knowledge base
    # ------------------------------------------------------------------

    def add_skill(self, skill_name: str, skill_data: dict[str, Any]) -> None:
        doc = (
            f"Skill: {skill_name}\n"
            f"Category: {skill_data.get('category', '')}\n"
            f"Related: {', '.join(skill_data.get('related_skills', []))}\n"
            f"Description: {skill_data.get('description', '')}"
        )
        self._skills_col.upsert(
            ids=[skill_name.lower().replace(" ", "_")],
            documents=[doc],
            metadatas=[{"name": skill_name, "category": skill_data.get("category", "")}],
        )

    def find_related_skills(self, skill_name: str, k: int = 5) -> list[str]:
        if self._skills_col.count() == 0:
            return []
        results = self._skills_col.query(
            query_texts=[skill_name],
            n_results=min(k, self._skills_col.count()),
            include=["metadatas"],
        )
        metas = results.get("metadatas", [[]])[0]
        return [m["name"] for m in metas if m.get("name", "").lower() != skill_name.lower()]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_job(job: dict[str, Any]) -> str:
        lines = [
            f"Job Title: {job.get('title', '')}",
            f"Department: {job.get('department', '')}",
            f"Level: {job.get('level', '')}",
            f"Description: {job.get('description', '')}",
        ]
        if job.get("requirements"):
            lines.append("Requirements:")
            for r in job["requirements"]:
                lines.append(f"  - {r}")
        if job.get("nice_to_have"):
            lines.append("Nice to Have:")
            for n in job["nice_to_have"]:
                lines.append(f"  - {n}")
        if job.get("tech_stack"):
            lines.append(f"Tech Stack: {', '.join(job['tech_stack'])}")
        return "\n".join(lines)
