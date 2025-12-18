"""
Configuration settings for the Hybrid Agentic System.
Uses Pydantic for validation and environment variable loading.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Google AI Configuration
    google_api_key: str = Field(..., description="Google AI API key")
    google_ai_model: str = Field(
        default="gemini-2.0-flash-exp",
        description="Google AI model to use"
    )

    # External APIs
    semantic_scholar_api_key: Optional[str] = Field(
        default=None,
        description="Semantic Scholar API key (optional)"
    )
    arxiv_max_results: int = Field(
        default=20,
        description="Maximum results to fetch from arXiv"
    )

    # LangGraph Configuration
    langgraph_checkpoint_db: str = Field(
        default="sqlite:///data/checkpoints.db",
        description="Database connection string for LangGraph checkpoints"
    )
    langgraph_max_iterations: int = Field(
        default=3,
        description="Maximum reflection loop iterations"
    )

    # Vector Store Configuration
    faiss_index_path: str = Field(
        default="data/vector_store",
        description="Path to store FAISS index"
    )
    embedding_dimension: int = Field(
        default=768,
        description="Dimension of embedding vectors"
    )
    embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Google embedding model name"
    )

    # Code Execution Configuration
    code_execution_timeout: int = Field(
        default=30,
        description="Timeout for code execution in seconds"
    )
    max_code_retries: int = Field(
        default=3,
        description="Maximum retries for code execution failures"
    )

    # Output Configuration
    reports_output_dir: str = Field(
        default="outputs/reports",
        description="Directory for generated reports"
    )
    enable_logging: bool = Field(
        default=True,
        description="Enable structured logging"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # Quality Thresholds
    min_quality_score: float = Field(
        default=7.0,
        description="Minimum quality score for report approval (0-10)"
    )
    min_research_papers: int = Field(
        default=5,
        description="Minimum number of research papers required"
    )
    min_key_findings: int = Field(
        default=10,
        description="Minimum number of key findings required"
    )
    min_test_coverage: float = Field(
        default=80.0,
        description="Minimum test coverage percentage for code"
    )

    # Performance Settings
    max_concurrent_searches: int = Field(
        default=5,
        description="Maximum concurrent API searches"
    )
    request_timeout: int = Field(
        default=60,
        description="HTTP request timeout in seconds"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
