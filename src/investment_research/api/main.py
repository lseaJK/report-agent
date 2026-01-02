"""Main FastAPI application for investment research system."""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import io

from .models import *
from .auth import AuthManager, get_current_user
from .dependencies import get_task_coordinator, get_workflow_manager, get_system_monitor
from ..core.task_coordinator import TaskCoordinator
from ..core.workflow_manager import WorkflowManager
from ..services.system_monitor import SystemMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Investment Research System API",
    description="API for automated investment research report generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
auth_manager = AuthManager()


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# Authentication endpoints
@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(credentials: LoginRequest):
    """Authenticate user and return access token."""
    try:
        token_data = await auth_manager.authenticate_user(
            credentials.username, 
            credentials.password
        )
        
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        return TokenResponse(**token_data)
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@app.post("/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(refresh_request: RefreshTokenRequest):
    """Refresh access token."""
    try:
        token_data = await auth_manager.refresh_token(refresh_request.refresh_token)
        
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        return TokenResponse(**token_data)
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


# Research task endpoints
@app.post("/tasks", response_model=TaskResponse, tags=["Research Tasks"])
async def create_research_task(
    task_request: CreateTaskRequest,
    current_user: dict = Depends(get_current_user),
    task_coordinator: TaskCoordinator = Depends(get_task_coordinator)
):
    """Create a new research task."""
    try:
        task_id = await task_coordinator.create_research_task(
            company_symbol=task_request.company_symbol,
            research_type=task_request.research_type,
            priority=task_request.priority,
            deadline=task_request.deadline,
            requirements=task_request.requirements,
            created_by=current_user["user_id"]
        )
        
        task_status = await task_coordinator.get_task_status(task_id)
        
        return TaskResponse(
            task_id=task_id,
            status=task_status.status.value,
            created_at=task_status.created_at,
            message="Research task created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating research task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create research task: {str(e)}"
        )


@app.get("/tasks", response_model=List[TaskSummary], tags=["Research Tasks"])
async def list_research_tasks(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    task_coordinator: TaskCoordinator = Depends(get_task_coordinator)
):
    """List research tasks."""
    try:
        tasks = await task_coordinator.list_tasks(
            status_filter=status,
            limit=limit,
            offset=offset,
            user_id=current_user["user_id"] if not current_user.get("is_admin") else None
        )
        
        return [
            TaskSummary(
                task_id=task.task_id,
                company_symbol=task.company_symbol,
                research_type=task.research_type,
                status=task.status.value,
                priority=task.priority.value,
                created_at=task.created_at,
                deadline=task.deadline,
                progress=task.progress
            )
            for task in tasks
        ]
        
    except Exception as e:
        logger.error(f"Error listing research tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve research tasks"
        )


@app.get("/tasks/{task_id}", response_model=TaskDetail, tags=["Research Tasks"])
async def get_research_task(
    task_id: str,
    current_user: dict = Depends(get_current_user),
    task_coordinator: TaskCoordinator = Depends(get_task_coordinator)
):
    """Get detailed information about a research task."""
    try:
        task_status = await task_coordinator.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Research task not found"
            )
        
        # Check permissions
        if not current_user.get("is_admin") and task_status.created_by != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return TaskDetail(
            task_id=task_status.task_id,
            company_symbol=task_status.company_symbol,
            research_type=task_status.research_type,
            status=task_status.status.value,
            priority=task_status.priority.value,
            created_at=task_status.created_at,
            started_at=task_status.started_at,
            completed_at=task_status.completed_at,
            deadline=task_status.deadline,
            progress=task_status.progress,
            requirements=task_status.requirements,
            assigned_agents=task_status.assigned_agents,
            results=task_status.results,
            error_message=task_status.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting research task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve research task"
        )


@app.post("/tasks/{task_id}/cancel", response_model=TaskResponse, tags=["Research Tasks"])
async def cancel_research_task(
    task_id: str,
    current_user: dict = Depends(get_current_user),
    task_coordinator: TaskCoordinator = Depends(get_task_coordinator)
):
    """Cancel a research task."""
    try:
        success = await task_coordinator.cancel_task(task_id, current_user["user_id"])
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Research task not found or cannot be cancelled"
            )
        
        return TaskResponse(
            task_id=task_id,
            status="cancelled",
            message="Research task cancelled successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling research task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel research task"
        )


# Report endpoints
@app.get("/reports", response_model=List[ReportSummary], tags=["Reports"])
async def list_reports(
    task_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """List generated reports."""
    try:
        reports = await workflow_manager.list_reports(
            task_id=task_id,
            status_filter=status,
            limit=limit,
            offset=offset,
            user_id=current_user["user_id"] if not current_user.get("is_admin") else None
        )
        
        return [
            ReportSummary(
                report_id=report.report_id,
                task_id=report.task_id,
                company_symbol=report.company_symbol,
                report_type=report.report_type,
                status=report.status,
                created_at=report.created_at,
                file_size=report.file_size,
                format=report.format
            )
            for report in reports
        ]
        
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve reports"
        )


@app.get("/reports/{report_id}", response_model=ReportDetail, tags=["Reports"])
async def get_report_details(
    report_id: str,
    current_user: dict = Depends(get_current_user),
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """Get detailed information about a report."""
    try:
        report = await workflow_manager.get_report_details(report_id)
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report not found"
            )
        
        # Check permissions
        if not current_user.get("is_admin") and report.created_by != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return ReportDetail(**report.__dict__)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report {report_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve report"
        )


@app.get("/reports/{report_id}/download", tags=["Reports"])
async def download_report(
    report_id: str,
    format: Optional[str] = "pdf",
    current_user: dict = Depends(get_current_user),
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """Download a report file."""
    try:
        report = await workflow_manager.get_report_details(report_id)
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report not found"
            )
        
        # Check permissions
        if not current_user.get("is_admin") and report.created_by != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Get report content
        content = await workflow_manager.get_report_content(report_id, format)
        
        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report content not found"
            )
        
        # Determine content type and filename
        content_types = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "html": "text/html",
            "json": "application/json"
        }
        
        content_type = content_types.get(format, "application/octet-stream")
        filename = f"report_{report_id}.{format}"
        
        # Create streaming response
        def generate():
            yield content
        
        return StreamingResponse(
            generate(),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report {report_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download report"
        )


# System monitoring endpoints
@app.get("/system/status", response_model=SystemStatus, tags=["System"])
async def get_system_status(
    current_user: dict = Depends(get_current_user),
    system_monitor: SystemMonitor = Depends(get_system_monitor)
):
    """Get current system status."""
    try:
        if not current_user.get("is_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        health_score = system_monitor.get_system_health_score()
        current_metrics = system_monitor.get_current_metrics()
        active_alerts = system_monitor.get_active_alerts()
        
        return SystemStatus(
            health_score=health_score["overall_score"],
            health_status=health_score["health_status"],
            active_alerts=len(active_alerts),
            current_metrics=current_metrics,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system status"
        )


@app.get("/system/metrics", tags=["System"])
async def get_system_metrics(
    metric_type: Optional[str] = None,
    hours: int = 24,
    current_user: dict = Depends(get_current_user),
    system_monitor: SystemMonitor = Depends(get_system_monitor)
):
    """Get system metrics history."""
    try:
        if not current_user.get("is_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        if metric_type:
            from ..services.system_monitor import MetricType
            try:
                metric_enum = MetricType(metric_type)
                history = system_monitor.get_metric_history(metric_enum, hours)
                return {"metric_type": metric_type, "history": history}
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid metric type: {metric_type}"
                )
        else:
            # Return all current metrics
            return system_monitor.get_current_metrics()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system metrics"
        )


@app.get("/system/alerts", tags=["System"])
async def get_system_alerts(
    status: Optional[str] = None,
    hours: int = 24,
    current_user: dict = Depends(get_current_user),
    system_monitor: SystemMonitor = Depends(get_system_monitor)
):
    """Get system alerts."""
    try:
        if not current_user.get("is_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        if status == "active":
            alerts = system_monitor.get_active_alerts()
        else:
            alerts = system_monitor.get_alert_history(hours)
        
        return {
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "rule_id": alert.rule_id,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "status": alert.status.value,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value
                }
                for alert in alerts
            ],
            "total": len(alerts)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system alerts"
        )


@app.post("/system/alerts/{alert_id}/acknowledge", tags=["System"])
async def acknowledge_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user),
    system_monitor: SystemMonitor = Depends(get_system_monitor)
):
    """Acknowledge a system alert."""
    try:
        if not current_user.get("is_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        success = await system_monitor.acknowledge_alert(alert_id, current_user["username"])
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        return {"message": "Alert acknowledged successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acknowledge alert"
        )


# Background task for system maintenance
@app.post("/system/maintenance", tags=["System"])
async def trigger_maintenance(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Trigger system maintenance tasks."""
    try:
        if not current_user.get("is_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        background_tasks.add_task(run_maintenance_tasks)
        
        return {"message": "Maintenance tasks scheduled"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering maintenance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to trigger maintenance"
        )


async def run_maintenance_tasks():
    """Run system maintenance tasks in background."""
    try:
        logger.info("Starting system maintenance tasks")
        
        # Clear old data, optimize databases, etc.
        # This would be implemented based on specific requirements
        
        logger.info("System maintenance tasks completed")
        
    except Exception as e:
        logger.error(f"Error in maintenance tasks: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return {
        "error": {
            "code": exc.status_code,
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": {
            "code": 500,
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )