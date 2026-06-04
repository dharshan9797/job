"""
Agent 3 — Candidate Ranking Agent
Scores candidates across multiple dimensions and produces a ranked profile.
"""
from __future__ import annotations

from typing import Any

from models.schemas import (
    CandidateScore,
    ParsedResume,
    ScoreDimension,
    SkillAnalysis,
)
from utils.helpers import clamp

from .base import BaseRecruitmentAgent

_SYSTEM = """You are a Senior Talent Acquisition Specialist who objectively scores candidates.

Score the candidate across EXACTLY these 6 dimensions (0-10 scale each):
1. technical_skills       - Depth and breadth of relevant technical expertise
2. experience_relevance   - How closely their background matches the role
3. skill_coverage         - Percentage of required skills met (weighted by importance)
4. growth_trajectory      - Career progression speed and direction
5. education_fit          - Relevance of academic background
6. cultural_indicators    - Leadership, teamwork, communication signals from resume

Return a JSON object:
{
  "overall_score": number (0-100, weighted composite),
  "dimensions": [
    {
      "name": "dimension_name",
      "score": number (0-10),
      "weight": number (0-1, must sum to 1.0 across all 6),
      "rationale": "string (1-2 sentences)"
    }
  ],
  "strengths": ["string (top 4 candidate strengths)"],
  "concerns": ["string (top 3 hiring concerns)"],
  "scoring_rationale": "string (3-4 sentence overall narrative)"
}

Weights must sum to exactly 1.0. Be analytical and objective, not positive-biased."""


class CandidateRankingAgent(BaseRecruitmentAgent):
    """Scores and ranks a candidate against a specific job opening."""

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    def run(
        self,
        parsed_resume: ParsedResume,
        skill_analysis: SkillAnalysis,
        job_description: str,
        job_title: str,
    ) -> CandidateScore:
        self.logger.info(
            "Scoring candidate %s for %s...", parsed_resume.candidate_name, job_title
        )

        human = f"""JOB: {job_title}
DESCRIPTION: {job_description[:1500]}

CANDIDATE: {parsed_resume.candidate_name}
Experience: {parsed_resume.total_experience_years} yrs ({parsed_resume.experience_level.value} level)
Summary: {parsed_resume.summary}

TOP TECHNICAL SKILLS: {', '.join(skill_analysis.top_technical_skills)}
TOP SOFT SKILLS: {', '.join(skill_analysis.top_soft_skills)}
SKILL COVERAGE SCORE: {skill_analysis.coverage_score:.0%}

MATCHED REQUIREMENTS ({len(skill_analysis.matched_requirements)}):
{chr(10).join(f'  ✓ {r}' for r in skill_analysis.matched_requirements[:10])}

SKILL GAPS ({len(skill_analysis.skill_gaps)}):
{chr(10).join(f'  ✗ {g.skill} (severity: {g.gap_severity:.0%}, importance: {g.importance:.0%})' for g in skill_analysis.skill_gaps[:8])}

GAP SUMMARY: {skill_analysis.gap_summary}

Work History (most recent first):
{self._format_history(parsed_resume)}

Score this candidate objectively across all 6 dimensions."""

        data: dict[str, Any] = self._chat_json(self.system_prompt, human)  # type: ignore[assignment]

        dimensions = [
            ScoreDimension(
                name=d["name"],
                score=clamp(float(d.get("score", 5.0)), 0.0, 10.0),
                weight=clamp(float(d.get("weight", 1 / 6))),
                rationale=d.get("rationale", ""),
            )
            for d in data.get("dimensions", [])
        ]

        return CandidateScore(
            overall_score=clamp(float(data.get("overall_score", 50.0)), 0.0, 100.0),
            dimensions=dimensions,
            strengths=data.get("strengths", []),
            concerns=data.get("concerns", []),
            scoring_rationale=data.get("scoring_rationale", ""),
        )

    @staticmethod
    def _format_history(resume: ParsedResume) -> str:
        lines = []
        for exp in resume.work_experience[:4]:
            lines.append(
                f"  {exp.title} @ {exp.company} ({exp.duration_months}mo) — "
                f"{exp.description[:150]}"
            )
        return "\n".join(lines) or "(no work history)"
