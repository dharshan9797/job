"""
Agent 1 — Resume Parsing Agent
Converts raw resume text into a structured ParsedResume object.
"""
from __future__ import annotations

from typing import Any

from models.schemas import (
    Education,
    ExperienceLevel,
    ParsedResume,
    WorkExperience,
)
from utils.helpers import truncate

from .base import BaseRecruitmentAgent

_SYSTEM = """You are an expert Resume Parser with 15+ years of HR experience.
Your task is to extract structured information from resume text with high accuracy.

Extract and return a single JSON object with these exact fields:
{
  "candidate_name": "string",
  "email": "string",
  "phone": "string",
  "location": "string (City, State/Country)",
  "linkedin": "string (URL or empty)",
  "github": "string (URL or empty)",
  "summary": "string (professional summary, max 300 chars)",
  "total_experience_years": number,
  "experience_level": "entry|mid|senior|lead|executive",
  "work_experience": [
    {
      "company": "string",
      "title": "string",
      "duration_months": number,
      "description": "string",
      "achievements": ["string"],
      "technologies": ["string"]
    }
  ],
  "education": [
    {
      "institution": "string",
      "degree": "string",
      "field": "string",
      "graduation_year": number or null,
      "gpa": number or null
    }
  ],
  "raw_skills": ["string"],
  "certifications": ["string"],
  "languages": ["string"],
  "projects": [
    {"name": "string", "description": "string", "technologies": ["string"]}
  ]
}

Rules:
- experience_level: entry (<2yr), mid (2-5yr), senior (5-10yr), lead (10-15yr), executive (15+yr)
- duration_months: estimate if dates are ambiguous
- raw_skills: all technical and soft skills mentioned anywhere in the resume
- Return ONLY the JSON, no extra text."""


class ResumeParsingAgent(BaseRecruitmentAgent):
    """Parses raw resume text into a structured ParsedResume."""

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    def run(self, resume_text: str) -> ParsedResume:
        self.logger.info("Parsing resume (%d chars)...", len(resume_text))
        safe_text = truncate(resume_text, max_chars=8000)

        human = f"Parse the following resume:\n\n{safe_text}"
        data: dict[str, Any] = self._chat_json(self.system_prompt, human)  # type: ignore[assignment]

        # Map raw dict → typed sub-models
        work_exp = [
            WorkExperience(**{k: v for k, v in exp.items() if k in WorkExperience.model_fields})
            for exp in data.get("work_experience", [])
        ]
        education = [
            Education(**{k: v for k, v in edu.items() if k in Education.model_fields})
            for edu in data.get("education", [])
        ]

        return ParsedResume(
            candidate_name=data.get("candidate_name", "Unknown"),
            email=data.get("email", ""),
            phone=data.get("phone", ""),
            location=data.get("location", ""),
            linkedin=data.get("linkedin", ""),
            github=data.get("github", ""),
            summary=data.get("summary", ""),
            total_experience_years=float(data.get("total_experience_years", 0)),
            experience_level=ExperienceLevel(data.get("experience_level", "entry")),
            work_experience=work_exp,
            education=education,
            raw_skills=data.get("raw_skills", []),
            certifications=data.get("certifications", []),
            languages=data.get("languages", []),
            projects=data.get("projects", []),
        )
