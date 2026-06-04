"""
Agent 5 — Hiring Recommendation Agent
Synthesises all upstream agent outputs into a final hiring decision.
"""
from __future__ import annotations

from typing import Any

from models.schemas import (
    CandidateScore,
    ExperienceLevel,
    HiringDecision,
    HiringRecommendation,
    InterviewPlan,
    ParsedResume,
    SkillAnalysis,
)
from utils.helpers import clamp

from .base import BaseRecruitmentAgent

_SYSTEM = """You are a Chief People Officer making final, evidence-based hiring decisions.

Synthesise all evidence to produce a hiring recommendation. Return JSON:
{
  "decision": "strong_hire|hire|maybe|no_hire|strong_no_hire",
  "confidence": number (0.0-1.0),
  "summary": "string (executive summary, 3-4 sentences)",
  "key_reasons": ["string (top 5 specific reasons supporting the decision)"],
  "risk_factors": ["string (up to 3 risks if hired)"],
  "development_plan": ["string (3-4 onboarding/growth actions if hired)"],
  "suggested_compensation_band": "string (e.g. '$120k-$145k / Senior IC')",
  "suggested_start_level": "entry|mid|senior|lead|executive",
  "next_steps": ["string (immediate actions for the hiring team)"]
}

Decision thresholds:
- strong_hire  : Overall ≥85, coverage ≥80%, strong signals, minimal critical gaps
- hire         : Overall ≥70, coverage ≥60%, manageable gaps
- maybe        : Overall 55-69, mixed signals, needs deeper probe
- no_hire      : Overall <55 OR critical skill gaps in core requirements
- strong_no_hire: Fundamental mismatch, experience misrepresentation, or multiple red flags

Always ground your decision in specific evidence from the data provided."""


class HiringRecommendationAgent(BaseRecruitmentAgent):
    """Produces the final hiring recommendation by synthesising all agent outputs."""

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    def run(
        self,
        parsed_resume: ParsedResume,
        skill_analysis: SkillAnalysis,
        candidate_score: CandidateScore,
        interview_plan: InterviewPlan,
        job_title: str,
        job_description: str,
    ) -> HiringRecommendation:
        self.logger.info(
            "Generating hiring recommendation for %s...", parsed_resume.candidate_name
        )

        human = f"""HIRING DECISION REQUEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE: {job_title}
DESCRIPTION: {job_description[:800]}

CANDIDATE PROFILE
  Name: {parsed_resume.candidate_name}
  Experience: {parsed_resume.total_experience_years} yrs — {parsed_resume.experience_level.value}
  Location: {parsed_resume.location}
  Summary: {parsed_resume.summary}

SKILL ANALYSIS
  Coverage Score: {skill_analysis.coverage_score:.0%}
  Top Technical: {', '.join(skill_analysis.top_technical_skills)}
  Top Soft: {', '.join(skill_analysis.top_soft_skills)}
  Matched Requirements: {len(skill_analysis.matched_requirements)} of required skills
  Critical Gaps: {', '.join(g.skill for g in skill_analysis.skill_gaps if g.importance >= 0.7)[:5] or 'None'}
  Gap Summary: {skill_analysis.gap_summary}

COMPOSITE SCORE: {candidate_score.overall_score:.1f}/100
  Breakdown:
{self._format_dimensions(candidate_score)}
  Strengths: {'; '.join(candidate_score.strengths)}
  Concerns: {'; '.join(candidate_score.concerns)}
  Rationale: {candidate_score.scoring_rationale}

INTERVIEW PLAN
  Recommended Rounds: {', '.join(r.value for r in interview_plan.recommended_rounds)}
  Red Flags to Probe: {'; '.join(interview_plan.red_flags_to_probe) or 'None'}
  Focus Areas: {'; '.join(interview_plan.focus_areas[:3])}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Make the final hiring recommendation."""

        data: dict[str, Any] = self._chat_json(self.system_prompt, human)  # type: ignore[assignment]

        try:
            decision = HiringDecision(data.get("decision", "maybe"))
        except ValueError:
            decision = HiringDecision.MAYBE

        try:
            start_level = ExperienceLevel(data.get("suggested_start_level", "mid"))
        except ValueError:
            start_level = None

        return HiringRecommendation(
            decision=decision,
            confidence=clamp(float(data.get("confidence", 0.5))),
            summary=data.get("summary", ""),
            key_reasons=data.get("key_reasons", []),
            risk_factors=data.get("risk_factors", []),
            development_plan=data.get("development_plan", []),
            suggested_compensation_band=data.get("suggested_compensation_band", ""),
            suggested_start_level=start_level,
            next_steps=data.get("next_steps", []),
        )

    @staticmethod
    def _format_dimensions(score: CandidateScore) -> str:
        lines = []
        for dim in score.dimensions:
            bar = "█" * int(dim.score) + "░" * (10 - int(dim.score))
            lines.append(
                f"    {dim.name:<25} {bar} {dim.score:.1f}/10 (w={dim.weight:.2f})"
            )
        return "\n".join(lines)
