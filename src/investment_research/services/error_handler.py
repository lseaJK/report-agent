"""Error handling and recovery mechanism service."""

import logging
import traceback
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import time
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error category types."""
    NETWORK = "network"
    API = "api"
    DATA = "data"
    PROCESSING = "processing"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ESCALATE = "escalate"
    MANUAL = "manual"


@dataclass
class ErrorRecord:
    """Records an error occurrence."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    error_category: ErrorCategory
    severity: ErrorSeverity
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    retry_count: int = 0
    resolved: bool = False
    
    def __post_init__(self):
        """Initialize error ID if not provided."""
        if not self.error_id:
            import hashlib
            content = f"{self.error_type}_{self.error_message}_{self.timestamp}"
            self.error_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_delay
        
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


@dataclass
class FallbackConfig:
    """Configuration for fallback mechanisms."""
    fallback_function: Optional[Callable] = None
    fallback_data: Optional[Any] = None
    use_cached_result: bool = True
    cache_max_age: timedelta = timedelta(hours=1)


class ErrorHandler:
    """Service for handling errors and implementing recovery mechanisms."""
    
    def __init__(self):
        """Initialize error handler."""
        self._error_records: List[ErrorRecord] = []
        self._error_patterns: Dict[str, Dict[str, Any]] = {}
        self._retry_configs: Dict[str, RetryConfig] = {}
        self._fallback_configs: Dict[str, FallbackConfig] = {}
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._error_cache: Dict[str, Any] = {}
        
        # Default configurations
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default error handling configurations."""
        # Default retry configurations
        self._retry_configs.update({
            "network": RetryConfig(max_attempts=3, base_delay=1.0, exponential_backoff=True),
            "api": RetryConfig(max_attempts=5, base_delay=2.0, exponential_backoff=True),
            "rate_limit": RetryConfig(max_attempts=3, base_delay=60.0, exponential_backoff=False),
            "timeout": RetryConfig(max_attempts=2, base_delay=5.0, exponential_backoff=True),
            "default": RetryConfig(max_attempts=3, base_delay=1.0, exponential_backoff=True)
        })
        
        # Error pattern recognition
        self._error_patterns.update({
            "connection_error": {
                "keywords": ["connection", "network", "unreachable", "timeout"],
                "category": ErrorCategory.NETWORK,
                "severity": ErrorSeverity.MEDIUM,
                "strategy": RecoveryStrategy.RETRY
            },
            "rate_limit_error": {
                "keywords": ["rate limit", "too many requests", "429"],
                "category": ErrorCategory.RATE_LIMIT,
                "severity": ErrorSeverity.MEDIUM,
                "strategy": RecoveryStrategy.RETRY
            },
            "authentication_error": {
                "keywords": ["unauthorized", "authentication", "401", "403"],
                "category": ErrorCategory.AUTHENTICATION,
                "severity": ErrorSeverity.HIGH,
                "strategy": RecoveryStrategy.ESCALATE
            },
            "data_validation_error": {
                "keywords": ["validation", "invalid", "malformed", "schema"],
                "category": ErrorCategory.VALIDATION,
                "severity": ErrorSeverity.MEDIUM,
                "strategy": RecoveryStrategy.FALLBACK
            },
            "system_error": {
                "keywords": ["system", "internal", "500", "server error"],
                "category": ErrorCategory.SYSTEM,
                "severity": ErrorSeverity.HIGH,
                "strategy": RecoveryStrategy.RETRY
            }
        })
    
    async def handle_error(self, error: Exception, context: Dict[str, Any], 
                          operation_name: str = "unknown") -> Dict[str, Any]:
        """Handle an error with appropriate recovery strategy."""
        try:
            # Classify the error
            error_category, severity, strategy = self._classify_error(error)
            
            # Create error record
            error_record = ErrorRecord(
                error_id="",  # Will be auto-generated
                timestamp=datetime.utcnow(),
                error_type=type(error).__name__,
                error_message=str(error),
                error_category=error_category,
                severity=severity,
                context=context,
                stack_trace=traceback.format_exc(),
                recovery_strategy=strategy
            )
            
            # Store error record
            self._error_records.append(error_record)
            
            # Log the error
            logger.error(f"Error in {operation_name}: {error_record.error_message}", 
                        extra={"error_id": error_record.error_id})
            
            # Apply recovery strategy
            recovery_result = await self._apply_recovery_strategy(error_record, context, operation_name)
            
            # Update error record with recovery results
            error_record.recovery_attempted = True
            error_record.recovery_successful = recovery_result.get("success", False)
            error_record.resolved = recovery_result.get("success", False)
            
            return {
                "error_id": error_record.error_id,
                "handled": True,
                "recovery_attempted": True,
                "recovery_successful": recovery_result.get("success", False),
                "recovery_result": recovery_result,
                "error_category": error_category.value,
                "severity": severity.value
            }
            
        except Exception as e:
            logger.critical(f"Error in error handler: {e}")
            return {
                "error_id": "handler_error",
                "handled": False,
                "recovery_attempted": False,
                "recovery_successful": False,
                "error_message": f"Error handler failed: {str(e)}"
            }
    
    def _classify_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity, RecoveryStrategy]:
        """Classify an error based on patterns."""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Check against known patterns
        for pattern_name, pattern_config in self._error_patterns.items():
            keywords = pattern_config["keywords"]
            if any(keyword in error_message or keyword in error_type for keyword in keywords):
                return (
                    pattern_config["category"],
                    pattern_config["severity"],
                    pattern_config["strategy"]
                )
        
        # Default classification
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY
    
    async def _apply_recovery_strategy(self, error_record: ErrorRecord, 
                                     context: Dict[str, Any], operation_name: str) -> Dict[str, Any]:
        """Apply the appropriate recovery strategy."""
        strategy = error_record.recovery_strategy
        
        if strategy == RecoveryStrategy.RETRY:
            return await self._apply_retry_strategy(error_record, context, operation_name)
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._apply_fallback_strategy(error_record, context, operation_name)
        elif strategy == RecoveryStrategy.SKIP:
            return await self._apply_skip_strategy(error_record, context, operation_name)
        elif strategy == RecoveryStrategy.ESCALATE:
            return await self._apply_escalation_strategy(error_record, context, operation_name)
        else:
            return {"success": False, "message": "No recovery strategy available"}
    
    async def _apply_retry_strategy(self, error_record: ErrorRecord, 
                                   context: Dict[str, Any], operation_name: str) -> Dict[str, Any]:
        """Apply retry recovery strategy."""
        try:
            # Get retry configuration
            retry_config = self._retry_configs.get(
                error_record.error_category.value,
                self._retry_configs["default"]
            )
            
            # Check circuit breaker
            if self._is_circuit_breaker_open(operation_name):
                return {
                    "success": False,
                    "message": "Circuit breaker is open",
                    "strategy": "retry_blocked"
                }
            
            # Get the original function to retry
            original_function = context.get("function")
            if not original_function:
                return {
                    "success": False,
                    "message": "No function to retry",
                    "strategy": "retry_failed"
                }
            
            # Attempt retries
            for attempt in range(1, retry_config.max_attempts + 1):
                try:
                    # Wait before retry (except first attempt)
                    if attempt > 1:
                        delay = retry_config.get_delay(attempt - 1)
                        logger.info(f"Retrying {operation_name} in {delay:.2f} seconds (attempt {attempt})")
                        await asyncio.sleep(delay)
                    
                    # Attempt the operation
                    args = context.get("args", ())
                    kwargs = context.get("kwargs", {})
                    
                    if asyncio.iscoroutinefunction(original_function):
                        result = await original_function(*args, **kwargs)
                    else:
                        result = original_function(*args, **kwargs)
                    
                    # Success - update circuit breaker
                    self._record_success(operation_name)
                    error_record.retry_count = attempt
                    
                    return {
                        "success": True,
                        "result": result,
                        "strategy": "retry_successful",
                        "attempts": attempt
                    }
                    
                except Exception as retry_error:
                    error_record.retry_count = attempt
                    logger.warning(f"Retry attempt {attempt} failed for {operation_name}: {retry_error}")
                    
                    # Record failure for circuit breaker
                    self._record_failure(operation_name)
                    
                    # If this was the last attempt, break
                    if attempt == retry_config.max_attempts:
                        break
            
            return {
                "success": False,
                "message": f"All {retry_config.max_attempts} retry attempts failed",
                "strategy": "retry_exhausted",
                "attempts": retry_config.max_attempts
            }
            
        except Exception as e:
            logger.error(f"Error in retry strategy: {e}")
            return {
                "success": False,
                "message": f"Retry strategy failed: {str(e)}",
                "strategy": "retry_error"
            }
    
    async def _apply_fallback_strategy(self, error_record: ErrorRecord, 
                                      context: Dict[str, Any], operation_name: str) -> Dict[str, Any]:
        """Apply fallback recovery strategy."""
        try:
            # Get fallback configuration
            fallback_config = self._fallback_configs.get(operation_name)
            
            if not fallback_config:
                # Try to use cached result
                cached_result = self._get_cached_result(operation_name, context)
                if cached_result:
                    return {
                        "success": True,
                        "result": cached_result,
                        "strategy": "fallback_cached"
                    }
                
                # Use default fallback data
                return {
                    "success": True,
                    "result": {"status": "fallback", "message": "Using fallback data"},
                    "strategy": "fallback_default"
                }
            
            # Use configured fallback
            if fallback_config.fallback_function:
                args = context.get("args", ())
                kwargs = context.get("kwargs", {})
                
                if asyncio.iscoroutinefunction(fallback_config.fallback_function):
                    result = await fallback_config.fallback_function(*args, **kwargs)
                else:
                    result = fallback_config.fallback_function(*args, **kwargs)
                
                return {
                    "success": True,
                    "result": result,
                    "strategy": "fallback_function"
                }
            
            elif fallback_config.fallback_data is not None:
                return {
                    "success": True,
                    "result": fallback_config.fallback_data,
                    "strategy": "fallback_data"
                }
            
            elif fallback_config.use_cached_result:
                cached_result = self._get_cached_result(operation_name, context)
                if cached_result:
                    return {
                        "success": True,
                        "result": cached_result,
                        "strategy": "fallback_cached"
                    }
            
            return {
                "success": False,
                "message": "No fallback option available",
                "strategy": "fallback_unavailable"
            }
            
        except Exception as e:
            logger.error(f"Error in fallback strategy: {e}")
            return {
                "success": False,
                "message": f"Fallback strategy failed: {str(e)}",
                "strategy": "fallback_error"
            }
    
    async def _apply_skip_strategy(self, error_record: ErrorRecord, 
                                  context: Dict[str, Any], operation_name: str) -> Dict[str, Any]:
        """Apply skip recovery strategy."""
        logger.info(f"Skipping operation {operation_name} due to error: {error_record.error_message}")
        
        return {
            "success": True,
            "result": None,
            "strategy": "skip",
            "message": "Operation skipped due to error"
        }
    
    async def _apply_escalation_strategy(self, error_record: ErrorRecord, 
                                        context: Dict[str, Any], operation_name: str) -> Dict[str, Any]:
        """Apply escalation recovery strategy."""
        # Log critical error for manual intervention
        logger.critical(f"ESCALATION REQUIRED - {operation_name}: {error_record.error_message}",
                       extra={
                           "error_id": error_record.error_id,
                           "context": context,
                           "requires_manual_intervention": True
                       })
        
        # Could integrate with alerting systems here
        # For now, we'll just mark it for manual review
        
        return {
            "success": False,
            "message": "Error escalated for manual intervention",
            "strategy": "escalated",
            "requires_manual_intervention": True
        }
    
    def _is_circuit_breaker_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open for an operation."""
        if operation_name not in self._circuit_breakers:
            return False
        
        breaker = self._circuit_breakers[operation_name]
        
        # Check if breaker is in open state
        if breaker["state"] == "open":
            # Check if enough time has passed to try again
            if datetime.utcnow() > breaker["next_attempt"]:
                breaker["state"] = "half_open"
                return False
            return True
        
        return False
    
    def _record_success(self, operation_name: str):
        """Record a successful operation for circuit breaker."""
        if operation_name not in self._circuit_breakers:
            self._circuit_breakers[operation_name] = {
                "failure_count": 0,
                "success_count": 0,
                "state": "closed",
                "next_attempt": None
            }
        
        breaker = self._circuit_breakers[operation_name]
        breaker["success_count"] += 1
        breaker["failure_count"] = 0  # Reset failure count on success
        
        # Close circuit breaker if it was open
        if breaker["state"] in ["open", "half_open"]:
            breaker["state"] = "closed"
            breaker["next_attempt"] = None
    
    def _record_failure(self, operation_name: str):
        """Record a failed operation for circuit breaker."""
        if operation_name not in self._circuit_breakers:
            self._circuit_breakers[operation_name] = {
                "failure_count": 0,
                "success_count": 0,
                "state": "closed",
                "next_attempt": None
            }
        
        breaker = self._circuit_breakers[operation_name]
        breaker["failure_count"] += 1
        
        # Open circuit breaker if failure threshold is reached
        failure_threshold = 5  # Configurable
        if breaker["failure_count"] >= failure_threshold:
            breaker["state"] = "open"
            breaker["next_attempt"] = datetime.utcnow() + timedelta(minutes=5)  # Configurable
            logger.warning(f"Circuit breaker opened for {operation_name}")
    
    def _get_cached_result(self, operation_name: str, context: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for fallback."""
        cache_key = f"{operation_name}_{hash(str(context))}"
        
        if cache_key in self._error_cache:
            cached_entry = self._error_cache[cache_key]
            
            # Check if cache is still valid
            max_age = timedelta(hours=1)  # Configurable
            if datetime.utcnow() - cached_entry["timestamp"] < max_age:
                return cached_entry["result"]
            else:
                # Remove expired cache entry
                del self._error_cache[cache_key]
        
        return None
    
    def cache_result(self, operation_name: str, context: Dict[str, Any], result: Any):
        """Cache a successful result for potential fallback use."""
        cache_key = f"{operation_name}_{hash(str(context))}"
        
        self._error_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.utcnow()
        }
        
        # Limit cache size
        max_cache_size = 1000  # Configurable
        if len(self._error_cache) > max_cache_size:
            # Remove oldest entries
            sorted_entries = sorted(
                self._error_cache.items(),
                key=lambda x: x[1]["timestamp"]
            )
            
            entries_to_remove = len(sorted_entries) - max_cache_size + 100
            for i in range(entries_to_remove):
                del self._error_cache[sorted_entries[i][0]]
    
    def register_retry_config(self, operation_name: str, config: RetryConfig):
        """Register retry configuration for an operation."""
        self._retry_configs[operation_name] = config
        logger.info(f"Registered retry config for {operation_name}")
    
    def register_fallback_config(self, operation_name: str, config: FallbackConfig):
        """Register fallback configuration for an operation."""
        self._fallback_configs[operation_name] = config
        logger.info(f"Registered fallback config for {operation_name}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and patterns."""
        if not self._error_records:
            return {"message": "No error records available"}
        
        # Count by category
        errors_by_category = {}
        for record in self._error_records:
            category = record.error_category.value
            errors_by_category[category] = errors_by_category.get(category, 0) + 1
        
        # Count by severity
        errors_by_severity = {}
        for record in self._error_records:
            severity = record.severity.value
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1
        
        # Recovery success rate
        recovery_attempted = len([r for r in self._error_records if r.recovery_attempted])
        recovery_successful = len([r for r in self._error_records if r.recovery_successful])
        recovery_rate = recovery_successful / recovery_attempted if recovery_attempted > 0 else 0
        
        # Most common errors
        error_types = {}
        for record in self._error_records:
            error_type = record.error_type
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        most_common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Recent errors (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_errors = len([r for r in self._error_records if r.timestamp > recent_cutoff])
        
        return {
            "total_errors": len(self._error_records),
            "recent_errors_24h": recent_errors,
            "errors_by_category": errors_by_category,
            "errors_by_severity": errors_by_severity,
            "recovery_attempts": recovery_attempted,
            "recovery_successes": recovery_successful,
            "recovery_success_rate": recovery_rate,
            "most_common_errors": most_common_errors,
            "circuit_breakers": {
                name: {
                    "state": breaker["state"],
                    "failure_count": breaker["failure_count"],
                    "success_count": breaker["success_count"]
                }
                for name, breaker in self._circuit_breakers.items()
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def get_unresolved_errors(self) -> List[ErrorRecord]:
        """Get list of unresolved errors."""
        return [record for record in self._error_records if not record.resolved]
    
    def mark_error_resolved(self, error_id: str) -> bool:
        """Mark an error as resolved."""
        for record in self._error_records:
            if record.error_id == error_id:
                record.resolved = True
                logger.info(f"Marked error {error_id} as resolved")
                return True
        return False
    
    def clear_error_history(self, older_than_days: int = 30):
        """Clear old error records."""
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        
        initial_count = len(self._error_records)
        self._error_records = [
            record for record in self._error_records
            if record.timestamp > cutoff_date
        ]
        
        cleared_count = initial_count - len(self._error_records)
        logger.info(f"Cleared {cleared_count} error records older than {older_than_days} days")
        
        return cleared_count


def with_error_handling(operation_name: str = None, 
                       retry_config: RetryConfig = None,
                       fallback_config: FallbackConfig = None):
    """Decorator for automatic error handling."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = ErrorHandler()  # In practice, this would be injected
            op_name = operation_name or func.__name__
            
            # Register configurations if provided
            if retry_config:
                error_handler.register_retry_config(op_name, retry_config)
            if fallback_config:
                error_handler.register_fallback_config(op_name, fallback_config)
            
            try:
                result = await func(*args, **kwargs)
                # Cache successful result
                error_handler.cache_result(op_name, {"args": args, "kwargs": kwargs}, result)
                return result
                
            except Exception as e:
                context = {
                    "function": func,
                    "args": args,
                    "kwargs": kwargs
                }
                
                recovery_result = await error_handler.handle_error(e, context, op_name)
                
                if recovery_result.get("recovery_successful"):
                    return recovery_result["recovery_result"]["result"]
                else:
                    # Re-raise the original exception if recovery failed
                    raise e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, we can't use async error handling
            # This would need a synchronous version of the error handler
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator