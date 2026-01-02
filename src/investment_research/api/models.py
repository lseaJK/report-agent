"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


# Enums
class ResearchType(str, Enum):
    """Research type enumeration."""
    COMPREHENSIVE = "comprehensive"
    FINANCIAL = "financial"
    MARKET = "market"
    INDUSTRY = "industry"
    RISK = "risk"


class TaskPriority(str, Enum):
    """Task priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReportFormat(str, Enum):
    """Report format enumeration."""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    JSON = "json"


# Authentication models
class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str


# Task models
class CreateTaskRequest(BaseModel):
    """Create research task request model."""
    company_symbol: str = Field(..., min_length=1, max_length=10, description="Company stock symbol")
    research_type: ResearchType = Field(..., description="Type of research to perform")
    priority: TaskPriority = Field(TaskPriority.NORMAL, description="Task priority")
    deadline: Optional[datetime] = Field(None, description="Task deadline")
    requirements: Optional[Dict[str, Any]] = Field(None, description="Additional requirements")
    
    @validator('company_symbol')
    def validate_symbol(cls, v):
        """Validate company symbol format."""
        if not v.isalnum():
            raise ValueError('Company symbol must be alphanumeric')
        return v.upper()
    
    @validator('deadline')
    def validate_deadline(cls, v):
        """Validate deadline is in the future."""
        if v and v <= datetime.utcnow():
            raise ValueError('Deadline must be in the future')
        return v


class TaskResponse(BaseModel):
    """Task response model."""
    task_id: str
    status: str
    created_at: Optional[datetime] = None
    message: str


class TaskSummary(BaseModel):
    """Task summary model for list responses."""
    task_id: str
    company_symbol: str
    research_type: str
    status: str
    priority: str
    created_at: datetime
    deadline: Optional[datetime] = None
    progress: float = Field(0.0, ge=0.0, le=1.0)


class TaskDetail(BaseModel):
    """Detailed task information model."""
    task_id: str
    company_symbol: str
    research_type: str
    status: str
    priority: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    progress: float = Field(0.0, ge=0.0, le=1.0)
    requirements: Optional[Dict[str, Any]] = None
    assigned_agents: List[str] = []
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# Report models
class ReportSummary(BaseModel):
    """Report summary model for list responses."""
    report_id: str
    task_id: str
    company_symbol: str
    report_type: str
    status: str
    created_at: datetime
    file_size: Optional[int] = None
    format: str


class ReportDetail(BaseModel):
    """Detailed report information model."""
    report_id: str
    task_id: str
    company_symbol: str
    report_type: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    file_size: Optional[int] = None
    format: str
    title: Optional[str] = None
    summary: Optional[str] = None
    sections: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None


class GenerateReportRequest(BaseModel):
    """Generate report request model."""
    task_id: str
    format: ReportFormat = ReportFormat.PDF
    template: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


# System monitoring models
class SystemStatus(BaseModel):
    """System status model."""
    health_score: float = Field(..., ge=0.0, le=100.0)
    health_status: str
    active_alerts: int = Field(..., ge=0)
    current_metrics: Dict[str, Any]
    timestamp: datetime


class MetricData(BaseModel):
    """Metric data model."""
    metric_type: str
    value: float
    timestamp: datetime
    tags: Optional[Dict[str, str]] = None


class AlertInfo(BaseModel):
    """Alert information model."""
    alert_id: str
    rule_id: str
    severity: str
    message: str
    status: str
    triggered_at: datetime
    current_value: float
    threshold_value: float
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


# Configuration models
class UpdateRuleRequest(BaseModel):
    """Update rule request model."""
    name: str
    description: str
    trigger_type: str
    schedule_expression: Optional[str] = None
    monitoring_metrics: Optional[List[str]] = None
    threshold_conditions: Optional[Dict[str, Any]] = None
    target_reports: Optional[List[str]] = None
    is_active: bool = True


class AlertRuleRequest(BaseModel):
    """Alert rule request model."""
    name: str
    metric_type: str
    condition: str = Field(..., description="Condition like '> 80' or '< 0.95'")
    severity: str
    description: str
    cooldown_minutes: int = Field(15, ge=1, le=1440)
    is_active: bool = True
    
    @validator('condition')
    def validate_condition(cls, v):
        """Validate condition format."""
        valid_operators = ['>', '<', '>=', '<=', '==', '!=']
        if not any(op in v for op in valid_operators):
            raise ValueError('Condition must contain a valid operator (>, <, >=, <=, ==, !=)')
        return v.strip()


# User management models
class UserInfo(BaseModel):
    """User information model."""
    user_id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_admin: bool = False
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None


class CreateUserRequest(BaseModel):
    """Create user request model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: Optional[str] = Field(None, max_length=100)
    is_admin: bool = False


class UpdateUserRequest(BaseModel):
    """Update user request model."""
    email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: Optional[str] = Field(None, max_length=100)
    is_admin: Optional[bool] = None
    is_active: Optional[bool] = None


class ChangePasswordRequest(BaseModel):
    """Change password request model."""
    current_password: str
    new_password: str = Field(..., min_length=6)


# Analytics models
class AnalyticsRequest(BaseModel):
    """Analytics request model."""
    start_date: datetime
    end_date: datetime
    metrics: Optional[List[str]] = None
    group_by: Optional[str] = None
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Validate date range."""
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('End date must be after start date')
        return v


class AnalyticsResponse(BaseModel):
    """Analytics response model."""
    period: Dict[str, datetime]
    metrics: Dict[str, Any]
    summary: Dict[str, float]
    generated_at: datetime


# Batch operation models
class BatchTaskRequest(BaseModel):
    """Batch task creation request model."""
    tasks: List[CreateTaskRequest] = Field(..., min_items=1, max_items=100)


class BatchTaskResponse(BaseModel):
    """Batch task creation response model."""
    created_tasks: List[TaskResponse]
    failed_tasks: List[Dict[str, str]]
    total_requested: int
    total_created: int
    total_failed: int


# Export models
class ExportRequest(BaseModel):
    """Export request model."""
    export_type: str = Field(..., description="Type of data to export")
    format: str = Field("json", description="Export format")
    filters: Optional[Dict[str, Any]] = None
    date_range: Optional[Dict[str, datetime]] = None


class ExportResponse(BaseModel):
    """Export response model."""
    export_id: str
    status: str
    created_at: datetime
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None


# Webhook models
class WebhookRequest(BaseModel):
    """Webhook configuration request model."""
    url: str = Field(..., regex=r'^https?://.+')
    events: List[str] = Field(..., min_items=1)
    secret: Optional[str] = None
    is_active: bool = True


class WebhookResponse(BaseModel):
    """Webhook configuration response model."""
    webhook_id: str
    url: str
    events: List[str]
    is_active: bool
    created_at: datetime
    last_triggered: Optional[datetime] = None


# Error models
class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, Union[int, str]]


class ValidationErrorResponse(BaseModel):
    """Validation error response model."""
    error: Dict[str, Union[int, str]]
    details: List[Dict[str, Any]]


# Pagination models
class PaginationParams(BaseModel):
    """Pagination parameters model."""
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)
    sort_by: Optional[str] = None
    sort_order: str = Field("desc", regex=r'^(asc|desc)$')


class PaginatedResponse(BaseModel):
    """Paginated response model."""
    items: List[Any]
    total: int
    limit: int
    offset: int
    has_next: bool
    has_prev: bool


# Health check models
class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    uptime: Optional[float] = None
    dependencies: Optional[Dict[str, str]] = None