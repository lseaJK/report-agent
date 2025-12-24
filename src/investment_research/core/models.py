"""Core data models for the investment research system."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    String, Text, DateTime, Float, Integer, Boolean, 
    JSON, ForeignKey, Enum as SQLEnum
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from .database import Base


class TaskStatus(str, Enum):
    """Research task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceType(str, Enum):
    """Data source type enumeration."""
    MCP = "mcp"
    RAG = "rag"
    API = "api"
    DATABASE = "database"
    FILE = "file"
    WEB = "web"
    MANUAL = "manual"


class AgentType(str, Enum):
    """Agent type enumeration."""
    INDUSTRY = "industry"
    FINANCIAL = "financial"
    MARKET = "market"
    RISK = "risk"


class OutputFormat(str, Enum):
    """Report output format enumeration."""
    PDF = "pdf"
    WORD = "word"
    HTML = "html"
    MARKDOWN = "markdown"


class ResearchTask(Base):
    """Research task model."""
    
    __tablename__ = "research_tasks"
    
    task_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    template_id: Mapped[str] = mapped_column(String(100), nullable=False)
    parameters: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[TaskStatus] = mapped_column(
        SQLEnum(TaskStatus), 
        default=TaskStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )
    deadline: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    assigned_agents: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Relationships
    analysis_results: Mapped[List["AnalysisResult"]] = relationship(
        "AnalysisResult", 
        back_populates="task",
        cascade="all, delete-orphan"
    )


class Source(Base):
    """Data source model."""
    
    __tablename__ = "sources"
    
    source_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    source_type: Mapped[SourceType] = mapped_column(SQLEnum(SourceType))
    url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    author: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    publication_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime, 
        nullable=True
    )
    access_date: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow
    )
    reliability_score: Mapped[float] = mapped_column(Float, default=0.5)
    source_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class AnalysisResult(Base):
    """Agent analysis result model."""
    
    __tablename__ = "analysis_results"
    
    result_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    task_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("research_tasks.task_id")
    )
    agent_id: Mapped[str] = mapped_column(String(100), nullable=False)
    agent_type: Mapped[AgentType] = mapped_column(SQLEnum(AgentType))
    analysis_type: Mapped[str] = mapped_column(String(100), nullable=False)
    content: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow
    )
    data_references: Mapped[List[UUID]] = mapped_column(JSON, default=list)
    
    # Relationships
    task: Mapped["ResearchTask"] = relationship(
        "ResearchTask", 
        back_populates="analysis_results"
    )


class ReportTemplate(Base):
    """Report template model."""
    
    __tablename__ = "report_templates"
    
    template_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    sections: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    required_agents: Mapped[List[str]] = mapped_column(JSON, default=list)
    output_format: Mapped[OutputFormat] = mapped_column(
        SQLEnum(OutputFormat), 
        default=OutputFormat.PDF
    )
    style_guide: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )


class KBDocument(Base):
    """Knowledge base document model."""
    
    __tablename__ = "kb_documents"
    
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(String(100), nullable=False)
    domain: Mapped[str] = mapped_column(String(100), nullable=False)
    tags: Mapped[List[str]] = mapped_column(JSON, default=list)
    embedding: Mapped[Optional[List[float]]] = mapped_column(JSON, nullable=True)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    document_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class Tool(Base):
    """Tool configuration model."""
    
    __tablename__ = "tools"
    
    tool_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    tool_type: Mapped[str] = mapped_column(String(50), nullable=False)
    agent_types: Mapped[List[str]] = mapped_column(JSON, default=list)
    api_endpoint: Mapped[Optional[str]] = mapped_column(
        String(500), 
        nullable=True
    )
    authentication: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    rate_limits: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    parameters_schema: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_health_check: Mapped[Optional[datetime]] = mapped_column(
        DateTime, 
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )


class DataSource(Base):
    """Data source configuration model."""
    
    __tablename__ = "data_sources"
    
    source_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    source_name: Mapped[str] = mapped_column(String(200), nullable=False)
    source_type: Mapped[SourceType] = mapped_column(SQLEnum(SourceType))
    agent_types: Mapped[List[str]] = mapped_column(JSON, default=list)
    access_config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    update_frequency: Mapped[str] = mapped_column(String(50), default="daily")
    reliability_score: Mapped[float] = mapped_column(Float, default=0.5)
    data_schema: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_sync: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )