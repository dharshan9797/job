"""
LangGraph Recruitment Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Node execution order:

  [START]
     │
     ▼
  rag_retrieval          ← RAG: fetch similar JD context from ChromaDB
     │
     ▼
  parse_resume           ← Agent 1: structured resume extraction
     │
     ▼
  extract_skills         ← Agent 2: skill taxonomy + gap detection
     │
     ▼
  rank_candidate         ← Agent 3: multi-dimension scoring
     │
     ▼
  generate_questions     ← Agent 4: tailored interview plan
     │
     ▼
  generate_recommendation← Agent 5: final hire/no-hire decision
     │
     ▼
  [END]

All nodes operate on RecruitmentState and propagate errors gracefully.
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents import (
    CandidateRankingAgent,
    HiringRecommendationAgent,
    InterviewQuestionAgent,
    ResumeParsingAgent,
    SkillExtractionAgent,
)
from models.schemas import RecruitmentState
from rag.vector_store import RecruitmentVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed state dict for LangGraph (uses plain dict under the hood)
# ---------------------------------------------------------------------------

def _state_to_dict(state: RecruitmentState) -> dict[str, Any]:
    return state.model_dump()


def _dict_to_state(d: dict[str, Any]) -> RecruitmentState:
    return RecruitmentState(**d)


# ---------------------------------------------------------------------------
# Node functions — each receives and returns a dict (LangGraph requirement)
# ---------------------------------------------------------------------------

def make_nodes(
    vector_store: RecruitmentVectorStore,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Factory that closes over shared resources (vector store, agent instances)
    and returns a mapping of node_name → node_function.
    """
    resume_agent = ResumeParsingAgent(model=model)
    skill_agent = SkillExtractionAgent(model=model)
    rank_agent = CandidateRankingAgent(model=model)
    interview_agent = InterviewQuestionAgent(model=model)
    rec_agent = HiringRecommendationAgent(model=model)

    # ------------------------------------------------------------------
    # Node 1 — RAG Retrieval
    # ------------------------------------------------------------------
    def rag_retrieval(state: dict[str, Any]) -> dict[str, Any]:
        logger.info("[Node] rag_retrieval")
        s = _dict_to_state(state)
        try:
            contexts = vector_store.retrieve_job_requirements(
                job_title=s.job_title,
                job_description=s.job_description,
            )
            s.retrieved_job_context = contexts
            s.current_node = "rag_retrieval"
        except Exception as exc:
            logger.error("RAG retrieval failed: %s", exc)
            s.errors.append(f"RAG retrieval: {exc}")
            # Non-fatal — agents will fall back to raw job_description
        return _state_to_dict(s)

    # ------------------------------------------------------------------
    # Node 2 — Resume Parsing
    # ------------------------------------------------------------------
    def parse_resume(state: dict[str, Any]) -> dict[str, Any]:
        logger.info("[Node] parse_resume")
        s = _dict_to_state(state)
        try:
            s.parsed_resume = resume_agent.run(resume_text=s.resume_text)
            s.current_node = "parse_resume"
        except Exception as exc:
            logger.error("Resume parsing failed: %s", exc)
            s.errors.append(f"parse_resume: {exc}")
        return _state_to_dict(s)

    # ------------------------------------------------------------------
    # Node 3 — Skill Extraction
    # ------------------------------------------------------------------
    def extract_skills(state: dict[str, Any]) -> dict[str, Any]:
        logger.info("[Node] extract_skills")
        s = _dict_to_state(state)
        if s.parsed_resume is None:
            s.errors.append("extract_skills: no parsed_resume available")
            return _state_to_dict(s)
        try:
            jd_context = s.retrieved_job_context or [s.job_description]
            s.skill_analysis = skill_agent.run(
                parsed_resume=s.parsed_resume,
                job_description=s.job_description,
                job_requirements=jd_context,
            )
            s.current_node = "extract_skills"
        except Exception as exc:
            logger.error("Skill extraction failed: %s", exc)
            s.errors.append(f"extract_skills: {exc}")
        return _state_to_dict(s)

    # ------------------------------------------------------------------
    # Node 4 — Candidate Ranking
    # ------------------------------------------------------------------
    def rank_candidate(state: dict[str, Any]) -> dict[str, Any]:
        logger.info("[Node] rank_candidate")
        s = _dict_to_state(state)
        if s.parsed_resume is None or s.skill_analysis is None:
            s.errors.append("rank_candidate: missing upstream outputs")
            return _state_to_dict(s)
        try:
            s.candidate_score = rank_agent.run(
                parsed_resume=s.parsed_resume,
                skill_analysis=s.skill_analysis,
                job_description=s.job_description,
                job_title=s.job_title,
            )
            s.current_node = "rank_candidate"
        except Exception as exc:
            logger.error("Candidate ranking failed: %s", exc)
            s.errors.append(f"rank_candidate: {exc}")
        return _state_to_dict(s)

    # ------------------------------------------------------------------
    # Node 5 — Interview Question Generation
    # ------------------------------------------------------------------
    def generate_questions(state: dict[str, Any]) -> dict[str, Any]:
        logger.info("[Node] generate_questions")
        s = _dict_to_state(state)
        if s.parsed_resume is None or s.skill_analysis is None or s.candidate_score is None:
            s.errors.append("generate_questions: missing upstream outputs")
            return _state_to_dict(s)
        try:
            s.interview_plan = interview_agent.run(
                parsed_resume=s.parsed_resume,
                skill_analysis=s.skill_analysis,
                candidate_score=s.candidate_score,
                job_title=s.job_title,
                job_description=s.job_description,
            )
            s.current_node = "generate_questions"
        except Exception as exc:
            logger.error("Interview generation failed: %s", exc)
            s.errors.append(f"generate_questions: {exc}")
        return _state_to_dict(s)

    # ------------------------------------------------------------------
    # Node 6 — Hiring Recommendation
    # ------------------------------------------------------------------
    def generate_recommendation(state: dict[str, Any]) -> dict[str, Any]:
        logger.info("[Node] generate_recommendation")
        s = _dict_to_state(state)
        required = [s.parsed_resume, s.skill_analysis, s.candidate_score, s.interview_plan]
        if any(x is None for x in required):
            s.errors.append("generate_recommendation: missing upstream outputs")
            return _state_to_dict(s)
        try:
            s.recommendation = rec_agent.run(
                parsed_resume=s.parsed_resume,
                skill_analysis=s.skill_analysis,
                candidate_score=s.candidate_score,
                interview_plan=s.interview_plan,
                job_title=s.job_title,
                job_description=s.job_description,
            )
            s.current_node = "generate_recommendation"
        except Exception as exc:
            logger.error("Recommendation generation failed: %s", exc)
            s.errors.append(f"generate_recommendation: {exc}")
        return _state_to_dict(s)

    return {
        "rag_retrieval": rag_retrieval,
        "parse_resume": parse_resume,
        "extract_skills": extract_skills,
        "rank_candidate": rank_candidate,
        "generate_questions": generate_questions,
        "generate_recommendation": generate_recommendation,
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_recruitment_graph(
    vector_store: RecruitmentVectorStore,
    model: str | None = None,
) -> Any:
    """
    Construct and compile the LangGraph StateGraph for the full recruitment pipeline.
    Returns a compiled graph (Runnable) that accepts a dict and returns a dict.
    """
    nodes = make_nodes(vector_store=vector_store, model=model)

    graph = StateGraph(dict)

    # Register nodes
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    # Linear pipeline edges
    graph.add_edge(START, "rag_retrieval")
    graph.add_edge("rag_retrieval", "parse_resume")
    graph.add_edge("parse_resume", "extract_skills")
    graph.add_edge("extract_skills", "rank_candidate")
    graph.add_edge("rank_candidate", "generate_questions")
    graph.add_edge("generate_questions", "generate_recommendation")
    graph.add_edge("generate_recommendation", END)

    return graph.compile()
