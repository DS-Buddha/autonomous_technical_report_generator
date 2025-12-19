"""
Pydantic schemas for agent output validation.

These schemas ensure agents return properly structured data,
preventing silent failures from typos or missing fields.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional


class PlannerOutput(BaseModel):
    """Validated output schema for Planner agent."""

    plan: Dict = Field(..., description="Hierarchical task structure")
    subtasks: List[Dict] = Field(..., min_length=1, description="Actionable subtasks")
    search_queries: List[str] = Field(..., min_length=1, max_length=20)
    code_specifications: List[Dict] = Field(default_factory=list)
    dependencies: Dict = Field(default_factory=dict)

    @field_validator('search_queries')
    @classmethod
    def validate_queries(cls, v):
        """Validate search queries are meaningful."""
        if not v:
            raise ValueError("Must have at least one search query")
        if any(len(q) < 3 for q in v):
            raise ValueError("Search queries must be at least 3 characters")
        return v

    @field_validator('subtasks')
    @classmethod
    def validate_subtasks(cls, v):
        """Validate subtasks structure."""
        if len(v) < 1:
            raise ValueError("Must have at least one subtask")
        if len(v) > 20:
            raise ValueError("Too many subtasks (max 20)")
        return v


class ResearcherOutput(BaseModel):
    """Validated output schema for Researcher agent."""

    research_papers: List[Dict] = Field(..., min_length=1, description="Papers found")
    key_findings: List[Dict] = Field(..., description="Extracted insights")
    literature_summary: str = Field(..., min_length=100, description="Summary")

    @field_validator('research_papers')
    @classmethod
    def validate_papers(cls, v):
        """Validate sufficient research."""
        if len(v) < 3:
            raise ValueError(f"Insufficient research: only {len(v)} papers (minimum 3)")
        return v

    @field_validator('literature_summary')
    @classmethod
    def validate_summary(cls, v):
        """Validate summary quality."""
        if len(v) < 100:
            raise ValueError("Literature summary too short (minimum 100 characters)")
        if "no research" in v.lower():
            raise ValueError("Literature summary indicates failure")
        return v


class CoderOutput(BaseModel):
    """Validated output schema for Coder agent."""

    generated_code: Dict[str, str] = Field(..., min_length=1, description="Code blocks")
    code_dependencies: List[str] = Field(default_factory=list, description="Required packages")

    @field_validator('generated_code')
    @classmethod
    def validate_code(cls, v):
        """Validate code blocks."""
        if not v:
            raise ValueError("Must generate at least one code block")
        if any(len(code) < 10 for code in v.values()):
            raise ValueError("Code blocks too short (minimum 10 characters)")
        return v


class TesterOutput(BaseModel):
    """Validated output schema for Tester agent."""

    test_results: List[Dict] = Field(default_factory=list)
    validation_errors: List[Dict] = Field(default_factory=list)
    executable_code: Dict[str, str] = Field(default_factory=dict)
    test_coverage: float = Field(..., ge=0.0, le=100.0)

    @field_validator('test_coverage')
    @classmethod
    def validate_coverage(cls, v):
        """Validate coverage is in valid range."""
        if v < 0 or v > 100:
            raise ValueError(f"Invalid coverage: {v} (must be 0-100)")
        return v


class CriticOutput(BaseModel):
    """Validated output schema for Critic agent."""

    quality_scores: Dict[str, float] = Field(...)
    overall_score: float = Field(..., ge=0.0, le=10.0)
    feedback: Dict[str, str] = Field(...)
    needs_revision: bool = Field(...)
    priority_issues: List[str] = Field(default_factory=list)

    @field_validator('quality_scores')
    @classmethod
    def validate_scores(cls, v):
        """Validate all required dimensions present."""
        required_dimensions = ['accuracy', 'completeness', 'code_quality', 'clarity', 'executability']
        if not all(dim in v for dim in required_dimensions):
            raise ValueError(f"Missing required dimensions. Need: {required_dimensions}")
        if not all(0 <= score <= 10 for score in v.values()):
            raise ValueError("Scores must be between 0 and 10")
        return v

    @field_validator('overall_score')
    @classmethod
    def validate_overall_score(cls, v):
        """Validate overall score is in range."""
        if v < 0 or v > 10:
            raise ValueError(f"Invalid overall score: {v} (must be 0-10)")
        return v


class SynthesizerOutput(BaseModel):
    """Validated output schema for Synthesizer agent."""

    final_report: str = Field(..., min_length=500, description="Complete markdown")
    report_metadata: Dict = Field(..., description="Statistics")

    @field_validator('final_report')
    @classmethod
    def validate_report(cls, v):
        """Validate report quality."""
        if len(v) < 500:
            raise ValueError("Report too short (minimum 500 characters)")
        if '# ' not in v:
            raise ValueError("Report missing markdown headers")
        return v

    @field_validator('report_metadata')
    @classmethod
    def validate_metadata(cls, v):
        """Validate metadata structure."""
        if 'word_count' not in v:
            raise ValueError("Metadata missing word_count")
        return v
