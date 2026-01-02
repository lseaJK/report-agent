"""FastAPI dependencies for dependency injection."""

from typing import Optional
from fastapi import Depends, HTTPException, status
import logging

from ..core.task_coordinator import TaskCoordinator
from ..core.workflow_manager import WorkflowManager
from ..services.system_monitor import SystemMonitor
from ..services.configuration_manager import ConfigurationManager
from ..services.tool_manager import ToolManager
from ..services.rag_service import RAGService
from ..services.update_scheduler import UpdateScheduler
from ..services.quality_controller import QualityController
from ..services.error_handler import ErrorHandler
from ..services.citation_manager import CitationManager
from ..services.content_aggregator import ContentAggregator

logger = logging.getLogger(__name__)

# Global service instances (in production, these would be properly initialized)
_task_coordinator: Optional[TaskCoordinator] = None
_workflow_manager: Optional[WorkflowManager] = None
_system_monitor: Optional[SystemMonitor] = None
_configuration_manager: Optional[ConfigurationManager] = None
_tool_manager: Optional[ToolManager] = None
_rag_service: Optional[RAGService] = None
_update_scheduler: Optional[UpdateScheduler] = None
_quality_controller: Optional[QualityController] = None
_error_handler: Optional[ErrorHandler] = None
_citation_manager: Optional[CitationManager] = None
_content_aggregator: Optional[ContentAggregator] = None


async def get_task_coordinator() -> TaskCoordinator:
    """Get task coordinator instance."""
    global _task_coordinator
    
    if _task_coordinator is None:
        try:
            _task_coordinator = TaskCoordinator()
            logger.info("Initialized TaskCoordinator")
        except Exception as e:
            logger.error(f"Failed to initialize TaskCoordinator: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Task coordination service unavailable"
            )
    
    return _task_coordinator


async def get_workflow_manager() -> WorkflowManager:
    """Get workflow manager instance."""
    global _workflow_manager
    
    if _workflow_manager is None:
        try:
            _workflow_manager = WorkflowManager()
            logger.info("Initialized WorkflowManager")
        except Exception as e:
            logger.error(f"Failed to initialize WorkflowManager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Workflow management service unavailable"
            )
    
    return _workflow_manager


async def get_system_monitor() -> SystemMonitor:
    """Get system monitor instance."""
    global _system_monitor
    
    if _system_monitor is None:
        try:
            _system_monitor = SystemMonitor()
            await _system_monitor.start_monitoring()
            logger.info("Initialized SystemMonitor")
        except Exception as e:
            logger.error(f"Failed to initialize SystemMonitor: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System monitoring service unavailable"
            )
    
    return _system_monitor


async def get_configuration_manager() -> ConfigurationManager:
    """Get configuration manager instance."""
    global _configuration_manager
    
    if _configuration_manager is None:
        try:
            _configuration_manager = ConfigurationManager()
            logger.info("Initialized ConfigurationManager")
        except Exception as e:
            logger.error(f"Failed to initialize ConfigurationManager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Configuration management service unavailable"
            )
    
    return _configuration_manager


async def get_tool_manager() -> ToolManager:
    """Get tool manager instance."""
    global _tool_manager
    
    if _tool_manager is None:
        try:
            config_manager = await get_configuration_manager()
            _tool_manager = ToolManager(config_manager)
            logger.info("Initialized ToolManager")
        except Exception as e:
            logger.error(f"Failed to initialize ToolManager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tool management service unavailable"
            )
    
    return _tool_manager


async def get_rag_service() -> RAGService:
    """Get RAG service instance."""
    global _rag_service
    
    if _rag_service is None:
        try:
            _rag_service = RAGService()
            logger.info("Initialized RAGService")
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service unavailable"
            )
    
    return _rag_service


async def get_update_scheduler() -> UpdateScheduler:
    """Get update scheduler instance."""
    global _update_scheduler
    
    if _update_scheduler is None:
        try:
            _update_scheduler = UpdateScheduler()
            await _update_scheduler.start_scheduler()
            logger.info("Initialized UpdateScheduler")
        except Exception as e:
            logger.error(f"Failed to initialize UpdateScheduler: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Update scheduling service unavailable"
            )
    
    return _update_scheduler


async def get_quality_controller() -> QualityController:
    """Get quality controller instance."""
    global _quality_controller
    
    if _quality_controller is None:
        try:
            _quality_controller = QualityController()
            logger.info("Initialized QualityController")
        except Exception as e:
            logger.error(f"Failed to initialize QualityController: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Quality control service unavailable"
            )
    
    return _quality_controller


async def get_error_handler() -> ErrorHandler:
    """Get error handler instance."""
    global _error_handler
    
    if _error_handler is None:
        try:
            _error_handler = ErrorHandler()
            logger.info("Initialized ErrorHandler")
        except Exception as e:
            logger.error(f"Failed to initialize ErrorHandler: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Error handling service unavailable"
            )
    
    return _error_handler


async def get_citation_manager() -> CitationManager:
    """Get citation manager instance."""
    global _citation_manager
    
    if _citation_manager is None:
        try:
            _citation_manager = CitationManager()
            logger.info("Initialized CitationManager")
        except Exception as e:
            logger.error(f"Failed to initialize CitationManager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Citation management service unavailable"
            )
    
    return _citation_manager


async def get_content_aggregator() -> ContentAggregator:
    """Get content aggregator instance."""
    global _content_aggregator
    
    if _content_aggregator is None:
        try:
            _content_aggregator = ContentAggregator()
            logger.info("Initialized ContentAggregator")
        except Exception as e:
            logger.error(f"Failed to initialize ContentAggregator: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Content aggregation service unavailable"
            )
    
    return _content_aggregator


# Service health check dependencies
async def check_service_health():
    """Check health of all services."""
    services_status = {}
    
    try:
        # Check task coordinator
        task_coordinator = await get_task_coordinator()
        services_status["task_coordinator"] = "healthy"
    except Exception:
        services_status["task_coordinator"] = "unhealthy"
    
    try:
        # Check workflow manager
        workflow_manager = await get_workflow_manager()
        services_status["workflow_manager"] = "healthy"
    except Exception:
        services_status["workflow_manager"] = "unhealthy"
    
    try:
        # Check system monitor
        system_monitor = await get_system_monitor()
        services_status["system_monitor"] = "healthy"
    except Exception:
        services_status["system_monitor"] = "unhealthy"
    
    return services_status


# Cleanup functions for graceful shutdown
async def cleanup_services():
    """Cleanup all services on shutdown."""
    global _system_monitor, _update_scheduler
    
    try:
        if _system_monitor:
            await _system_monitor.stop_monitoring()
            logger.info("Stopped SystemMonitor")
        
        if _update_scheduler:
            await _update_scheduler.stop_scheduler()
            logger.info("Stopped UpdateScheduler")
        
        logger.info("All services cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during service cleanup: {e}")


# Database connection dependencies (placeholder)
async def get_database_connection():
    """Get database connection (placeholder)."""
    # In a real implementation, this would return a database connection
    # For now, we'll return None since we're using in-memory storage
    return None


# Cache dependencies (placeholder)
async def get_cache_client():
    """Get cache client (placeholder)."""
    # In a real implementation, this would return a Redis or Memcached client
    return None


# Message queue dependencies (placeholder)
async def get_message_queue():
    """Get message queue client (placeholder)."""
    # In a real implementation, this would return a RabbitMQ or Kafka client
    return None


# Configuration dependencies
async def get_app_config():
    """Get application configuration."""
    # In a real implementation, this would load configuration from files/environment
    return {
        "debug": True,
        "max_concurrent_tasks": 10,
        "default_timeout": 300,
        "api_rate_limit": 1000,
        "report_storage_path": "/tmp/reports",
        "log_level": "INFO"
    }


# Rate limiting dependency
class RateLimiter:
    """Simple rate limiter (placeholder)."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        # Simple implementation - in production, use Redis or similar
        import time
        
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(current_time)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit(client_id: str = "default"):
    """Check rate limit for client."""
    if not await rate_limiter.check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    return True