"""
Agent 2 — Skill Extraction Agent
Categorizes skills, detects gaps vs. job requirements, and scores coverage.
"""
from __future__ import annotations

from typing import Any

from models.schemas import (
    ParsedResume,
    Skill,
    SkillAnalysis,
    SkillCategory,
    SkillGap,
)
from utils.helpers import clamp

from .base import BaseRecruitmentAgent

_SYSTEM = """You are an expert Technical Skills Analyst with deep knowledge of:
- Software engineering skill taxonomies
- Job market requirements across all tech stacks
- Skill gap analysis and competency mapping

Given a candidate's resume data and job requirements, return a JSON object:
{
  "skills": [
    {
      "name": "string",
      "category": "technical|soft|domain|tool|language|certification",
      "proficiency": number (0.0-1.0, inferred from experience/context),
      "years_experience": number,
      "evidence": "string (brief quote or note from resume)"
    }
  ],
  "top_technical_skills": ["string (top 6 technical skills)"],
  "top_soft_skills": ["string (top 4 soft skills)"],
  "matched_requirements": ["string (job requirements the candidate clearly meets)"],
  "skill_gaps": [
    {
      "skill": "string",
      "importance": number (0.0-1.0, how critical this skill is for the role),
      "gap_severity": number (0.0-1.0, 1.0 = completely missing, 0.0 = fully covered),
      "notes": "string"
    }
  ],
  "coverage_score": number (0.0-1.0, overall % of job requirements met),
  "gap_summary": "string (2-3 sentence analysis of the most important gaps)"
}

Be precise with proficiency scores. Use evidence from the resume, not assumptions."""


class SkillExtractionAgent(BaseRecruitmentAgent):
    """Extracts, categorizes and gap-analyses candidate skills."""

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    def run(
        self,
        parsed_resume: ParsedResume,
        job_description: str,
        job_requirements: list[str],
    ) -> SkillAnalysis:
        self.logger.info(
            "Extracting skills for %s...", parsed_resume.candidate_name
        )

        jd_context = "\n".join(job_requirements) if job_requirements else job_description

        human = f"""CANDIDATE PROFILE:
Name: {parsed_resume.candidate_name}
Experience: {parsed_resume.total_experience_years} years ({parsed_resume.experience_level.value})
Summary: {parsed_resume.summary}
Raw Skills Listed: {', '.join(parsed_resume.raw_skills)}
Work History:
{self._format_experience(parsed_resume)}
Education:
{self._format_education(parsed_resume)}
Projects: {self._format_projects(parsed_resume)}
Certifications: {', '.join(parsed_resume.certifications)}

JOB REQUIREMENTS:
{jd_context}

Perform detailed skill extraction and gap analysis."""

        data: dict[str, Any] = self._chat_json(self.system_prompt, human)  # type: ignore[assignment]

        _valid_cats = {c.value for c in SkillCategory}

        def _safe_cat(raw: str) -> SkillCategory:
            val = (raw or "technical").lower().strip()
            return SkillCategory(val) if val in _valid_cats else SkillCategory.TECHNICAL

        skills = [
            Skill(
                name=s["name"],
                category=_safe_cat(s.get("category", "technical")),
                proficiency=clamp(float(s.get("proficiency", 0.5))),
                years_experience=float(s.get("years_experience", 0)),
                evidence=s.get("evidence", ""),
            )
            for s in data.get("skills", [])
        ]

        gaps = [
            SkillGap(
                skill=g["skill"],
                importance=clamp(float(g.get("importance", 0.5))),
                gap_severity=clamp(float(g.get("gap_severity", 0.5))),
                notes=g.get("notes", ""),
            )
            for g in data.get("skill_gaps", [])
        ]

        return SkillAnalysis(
            skills=skills,
            top_technical_skills=data.get("top_technical_skills", []),
            top_soft_skills=data.get("top_soft_skills", []),
            matched_requirements=data.get("matched_requirements", []),
            skill_gaps=gaps,
            coverage_score=clamp(float(data.get("coverage_score", 0.5))),
            gap_summary=data.get("gap_summary", ""),
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_experience(resume: ParsedResume) -> str:
        lines = []
        for exp in resume.work_experience:
            lines.append(
                f"  • {exp.title} @ {exp.company} ({exp.duration_months}mo): "
                f"{exp.description[:200]} | Tech: {', '.join(exp.technologies)}"
            )
        return "\n".join(lines) or "  (none listed)"

    @staticmethod
    def _format_education(resume: ParsedResume) -> str:
        lines = []
        for edu in resume.education:
            lines.append(
                f"  • {edu.degree} in {edu.field} — {edu.institution} "
                f"({edu.graduation_year or 'N/A'})"
            )
        return "\n".join(lines) or "  (none listed)"

    @staticmethod
    def _format_projects(resume: ParsedResume) -> str:
        if not resume.projects:
            return "(none)"
        return "; ".join(
            f"{p.get('name','?')} ({', '.join(p.get('technologies', []))})"
            for p in resume.projects[:5]
        )
