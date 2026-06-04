"""
Agent 4 — Interview Question Agent
Generates a tailored interview plan targeting skill gaps and verifying strengths.
"""
from __future__ import annotations

from typing import Any

from models.schemas import (
    CandidateScore,
    InterviewPlan,
    InterviewQuestion,
    InterviewRound,
    ParsedResume,
    SkillAnalysis,
)
from .base import BaseRecruitmentAgent

_SYSTEM = """You are a Principal Technical Interviewer designing tailored interview plans.

Create a comprehensive, role-specific interview plan. Return JSON:
{
  "recommended_rounds": ["screening|technical|behavioral|system_design|culture_fit"],
  "questions": [
    {
      "round": "screening|technical|behavioral|system_design|culture_fit",
      "question": "string (the actual question)",
      "purpose": "string (why this question — what it reveals)",
      "expected_answer_points": ["string (key points a strong answer should cover)"],
      "follow_ups": ["string (1-2 follow-up probes)"]
    }
  ],
  "focus_areas": ["string (key areas to probe in depth)"],
  "red_flags_to_probe": ["string (concerns from resume that need validation)"],
  "time_estimate_hours": number
}

Guidelines:
- Include 3-5 questions per recommended round
- Mix of conceptual, practical, and situational questions
- Directly target skill gaps (to verify they're actually gaps)
- Include behavioral questions that surface soft skills evidence
- For senior/lead roles, include system design questions
- Questions must be specific to THIS candidate's background, not generic
- Cite concrete things from the resume in your questions"""


class InterviewQuestionAgent(BaseRecruitmentAgent):
    """Generates a candidate-specific interview plan with targeted questions."""

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    def run(
        self,
        parsed_resume: ParsedResume,
        skill_analysis: SkillAnalysis,
        candidate_score: CandidateScore,
        job_title: str,
        job_description: str,
    ) -> InterviewPlan:
        self.logger.info(
            "Generating interview plan for %s...", parsed_resume.candidate_name
        )

        human = f"""ROLE: {job_title}
JOB DESCRIPTION: {job_description[:1000]}

CANDIDATE: {parsed_resume.candidate_name}
Level: {parsed_resume.experience_level.value} | {parsed_resume.total_experience_years} years
Top Skills: {', '.join(skill_analysis.top_technical_skills)}

IDENTIFIED SKILL GAPS (probe these!):
{chr(10).join(f'  - {g.skill}: {g.notes}' for g in skill_analysis.skill_gaps[:6])}

MATCHED STRENGTHS (verify authenticity):
{chr(10).join(f'  - {r}' for r in skill_analysis.matched_requirements[:6])}

SCORING CONCERNS:
{chr(10).join(f'  - {c}' for c in candidate_score.concerns)}

CAREER HIGHLIGHTS:
{self._format_highlights(parsed_resume)}

Design a rigorous, personalised interview plan for this candidate."""

        data: dict[str, Any] = self._chat_json(self.system_prompt, human)  # type: ignore[assignment]

        rounds_raw = data.get("recommended_rounds", ["screening", "technical", "behavioral"])
        recommended_rounds = []
        for r in rounds_raw:
            try:
                recommended_rounds.append(InterviewRound(r))
            except ValueError:
                pass

        questions = []
        for q in data.get("questions", []):
            try:
                questions.append(
                    InterviewQuestion(
                        round=InterviewRound(q.get("round", "technical")),
                        question=q["question"],
                        purpose=q.get("purpose", ""),
                        expected_answer_points=q.get("expected_answer_points", []),
                        follow_ups=q.get("follow_ups", []),
                    )
                )
            except (KeyError, ValueError):
                continue

        return InterviewPlan(
            recommended_rounds=recommended_rounds,
            questions=questions,
            focus_areas=data.get("focus_areas", []),
            red_flags_to_probe=data.get("red_flags_to_probe", []),
            time_estimate_hours=float(data.get("time_estimate_hours", 2.0)),
        )

    @staticmethod
    def _format_highlights(resume: ParsedResume) -> str:
        lines = []
        for exp in resume.work_experience[:3]:
            if exp.achievements:
                lines.append(f"  @ {exp.company}: {'; '.join(exp.achievements[:2])}")
        for proj in resume.projects[:2]:
            lines.append(
                f"  Project '{proj.get('name','')}': "
                f"{proj.get('description','')[:100]}"
            )
        return "\n".join(lines) or "  (no specific highlights found)"
