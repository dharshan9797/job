from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"


class HiringDecision(str, Enum):
    STRONG_HIRE = "strong_hire"
    HIRE = "hire"
    MAYBE = "maybe"
    NO_HIRE = "no_hire"
    STRONG_NO_HIRE = "strong_no_hire"


class SkillCategory(str, Enum):
    TECHNICAL = "technical"
    SOFT = "soft"
    DOMAIN = "domain"
    TOOL = "tool"
    LANGUAGE = "language"
    CERTIFICATION = "certification"
    FRAMEWORK = "framework"
    DATABASE = "database"
    PLATFORM = "platform"
    METHODOLOGY = "methodology"


class InterviewRound(str, Enum):
    SCREENING = "screening"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    SYSTEM_DESIGN = "system_design"
    CULTURE_FIT = "culture_fit"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class WorkExperience(BaseModel):
    company: str
    title: str
    duration_months: int = 0
    description: str = ""
    achievements: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str
    degree: str
    field: str = ""
    graduation_year: int | None = None
    gpa: float | None = None


class Skill(BaseModel):
    name: str
    category: SkillCategory
    proficiency: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    years_experience: float = 0.0
    evidence: str = ""


class SkillGap(BaseModel):
    skill: str
    importance: Annotated[float, Field(ge=0.0, le=1.0)]
    gap_severity: Annotated[float, Field(ge=0.0, le=1.0)]
    notes: str = ""


class InterviewQuestion(BaseModel):
    round: InterviewRound
    question: str
    purpose: str
    expected_answer_points: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)


class ScoreDimension(BaseModel):
    name: str
    score: Annotated[float, Field(ge=0.0, le=10.0)]
    weight: Annotated[float, Field(ge=0.0, le=1.0)]
    rationale: str


# ---------------------------------------------------------------------------
# Primary output models (one per agent)
# ---------------------------------------------------------------------------

class ParsedResume(BaseModel):
    """Output of the Resume Parsing Agent."""
    candidate_name: str = "Unknown"
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    summary: str = ""
    total_experience_years: float = 0.0
    experience_level: ExperienceLevel = ExperienceLevel.ENTRY
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    raw_skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    projects: list[dict[str, Any]] = Field(default_factory=list)


class SkillAnalysis(BaseModel):
    """Output of the Skill Extraction Agent."""
    skills: list[Skill] = Field(default_factory=list)
    top_technical_skills: list[str] = Field(default_factory=list)
    top_soft_skills: list[str] = Field(default_factory=list)
    matched_requirements: list[str] = Field(default_factory=list)
    skill_gaps: list[SkillGap] = Field(default_factory=list)
    coverage_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    gap_summary: str = ""


class CandidateScore(BaseModel):
    """Output of the Candidate Ranking Agent."""
    overall_score: Annotated[float, Field(ge=0.0, le=100.0)] = 0.0
    dimensions: list[ScoreDimension] = Field(default_factory=list)
    rank_percentile: float | None = None
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    scoring_rationale: str = ""


class InterviewPlan(BaseModel):
    """Output of the Interview Question Agent."""
    recommended_rounds: list[InterviewRound] = Field(default_factory=list)
    questions: list[InterviewQuestion] = Field(default_factory=list)
    focus_areas: list[str] = Field(default_factory=list)
    red_flags_to_probe: list[str] = Field(default_factory=list)
    time_estimate_hours: float = 1.0


class HiringRecommendation(BaseModel):
    """Output of the Hiring Recommendation Agent — the final verdict."""
    decision: HiringDecision
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    summary: str
    key_reasons: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    development_plan: list[str] = Field(default_factory=list)
    suggested_compensation_band: str = ""
    suggested_start_level: ExperienceLevel | None = None
    next_steps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------

class RecruitmentState(BaseModel):
    """Full pipeline state threaded through every LangGraph node."""
    # --- inputs ---
    resume_text: str = ""
    job_description: str = ""
    job_title: str = ""
    retrieved_job_context: list[str] = Field(default_factory=list)

    # --- agent outputs ---
    parsed_resume: ParsedResume | None = None
    skill_analysis: SkillAnalysis | None = None
    candidate_score: CandidateScore | None = None
    interview_plan: InterviewPlan | None = None
    recommendation: HiringRecommendation | None = None

    # --- control ---
    errors: list[str] = Field(default_factory=list)
    current_node: str = "start"
