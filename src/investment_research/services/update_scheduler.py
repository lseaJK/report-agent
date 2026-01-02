"""Dynamic update and monitoring system for automated report updates."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from croniter import croniter

logger = logging.getLogger(__name__)


class UpdateTrigger(str, Enum):
    """Update trigger types."""
    SCHEDULED = "scheduled"
    MARKET_CHANGE = "market_change"
    DATA_CHANGE = "data_change"
    MANUAL = "manual"
    EVENT_DRIVEN = "event_driven"


class UpdateStatus(str, Enum):
    """Update status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MonitoringMetric(str, Enum):
    """Monitoring metric types."""
    PRICE_CHANGE = "price_change"
    VOLUME_CHANGE = "volume_change"
    NEWS_SENTIMENT = "news_sentiment"
    EARNINGS_UPDATE = "earnings_update"
    ANALYST_RATING = "analyst_rating"
    MARKET_INDEX = "market_index"
    ECONOMIC_INDICATOR = "economic_indicator"


@dataclass
class UpdateRule:
    """Defines when and how updates should be triggered."""
    rule_id: str
    name: str
    description: str
    trigger_type: UpdateTrigger
    schedule_expression: Optional[str] = None  # Cron expression for scheduled updates
    monitoring_metrics: List[MonitoringMetric] = None
    threshold_conditions: Dict[str, Any] = None
    target_reports: List[str] = None  # Report IDs to update
    is_active: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamps and defaults."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.monitoring_metrics is None:
            self.monitoring_metrics = []
        if self.threshold_conditions is None:
            self.threshold_conditions = {}
        if self.target_reports is None:
            self.target_reports = []
        
        if not self.rule_id:
            # Generate rule ID
            content = f"{self.name}_{self.trigger_type}_{self.created_at}"
            self.rule_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class UpdateTask:
    """Represents an update task."""
    task_id: str
    rule_id: str
    report_id: str
    trigger_type: UpdateTrigger
    trigger_data: Dict[str, Any]
    status: UpdateStatus
    scheduled_time: datetime
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        """Initialize task ID if not provided."""
        if not self.task_id:
            content = f"{self.rule_id}_{self.report_id}_{self.scheduled_time}"
            self.task_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class MarketDataPoint:
    """Represents a market data point for monitoring."""
    symbol: str
    metric: MonitoringMetric
    value: float
    timestamp: datetime
    source: str
    previous_value: Optional[float] = None
    change_percentage: Optional[float] = None
    
    def __post_init__(self):
        """Calculate change percentage if previous value is available."""
        if self.previous_value is not None and self.previous_value != 0:
            self.change_percentage = ((self.value - self.previous_value) / self.previous_value) * 100


@dataclass
class UpdateNotification:
    """Represents an update notification."""
    notification_id: str
    update_task_id: str
    report_id: str
    notification_type: str
    message: str
    recipients: List[str]
    sent: bool = False
    sent_time: Optional[datetime] = None
    created_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamps and notification ID."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        if not self.notification_id:
            content = f"{self.update_task_id}_{self.notification_type}_{self.created_at}"
            self.notification_id = hashlib.md5(content.encode()).hexdigest()[:8]


class UpdateScheduler:
    """Service for managing dynamic updates and monitoring."""
    
    def __init__(self):
        """Initialize update scheduler."""
        self._update_rules: Dict[str, UpdateRule] = {}
        self._update_tasks: Dict[str, UpdateTask] = {}
        self._market_data: Dict[str, List[MarketDataPoint]] = {}
        self._notifications: List[UpdateNotification] = []
        self._running_tasks: Set[str] = set()
        self._scheduler_running = False
        self._monitoring_running = False
        
        # Callbacks for external integrations
        self._update_callbacks: Dict[str, Callable] = {}
        self._notification_callbacks: List[Callable] = []
    
    async def start_scheduler(self):
        """Start the update scheduler."""
        if self._scheduler_running:
            logger.warning("Scheduler is already running")
            return
        
        self._scheduler_running = True
        logger.info("Starting update scheduler")
        
        # Start scheduler loop
        asyncio.create_task(self._scheduler_loop())
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_scheduler(self):
        """Stop the update scheduler."""
        self._scheduler_running = False
        self._monitoring_running = False
        logger.info("Stopping update scheduler")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._scheduler_running:
            try:
                await self._process_scheduled_updates()
                await self._process_pending_tasks()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)
    
    async def _monitoring_loop(self):
        """Market monitoring loop."""
        self._monitoring_running = True
        
        while self._monitoring_running:
            try:
                await self._monitor_market_changes()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)
    
    async def register_update_rule(self, rule: UpdateRule) -> str:
        """Register a new update rule."""
        try:
            # Validate rule
            if rule.trigger_type == UpdateTrigger.SCHEDULED and not rule.schedule_expression:
                raise ValueError("Scheduled updates require a schedule expression")
            
            if rule.trigger_type == UpdateTrigger.MARKET_CHANGE and not rule.monitoring_metrics:
                raise ValueError("Market change updates require monitoring metrics")
            
            # Validate cron expression if provided
            if rule.schedule_expression:
                try:
                    croniter(rule.schedule_expression)
                except ValueError as e:
                    raise ValueError(f"Invalid cron expression: {e}")
            
            # Store rule
            self._update_rules[rule.rule_id] = rule
            
            logger.info(f"Registered update rule: {rule.rule_id}")
            return rule.rule_id
            
        except Exception as e:
            logger.error(f"Error registering update rule: {e}")
            raise
    
    async def _process_scheduled_updates(self):
        """Process scheduled update rules."""
        current_time = datetime.utcnow()
        
        for rule in self._update_rules.values():
            if not rule.is_active or rule.trigger_type != UpdateTrigger.SCHEDULED:
                continue
            
            if not rule.schedule_expression:
                continue
            
            try:
                # Check if it's time to run this rule
                cron = croniter(rule.schedule_expression, current_time)
                next_run = cron.get_prev(datetime)
                
                # Check if we should have run this in the last minute
                if current_time - next_run <= timedelta(minutes=1):
                    # Check if we already have a recent task for this rule
                    recent_tasks = [
                        task for task in self._update_tasks.values()
                        if (task.rule_id == rule.rule_id and 
                            current_time - task.scheduled_time <= timedelta(minutes=2))
                    ]
                    
                    if not recent_tasks:
                        # Create update tasks for each target report
                        for report_id in rule.target_reports:
                            await self._create_update_task(
                                rule_id=rule.rule_id,
                                report_id=report_id,
                                trigger_type=UpdateTrigger.SCHEDULED,
                                trigger_data={"scheduled_time": current_time.isoformat()},
                                scheduled_time=current_time
                            )
                
            except Exception as e:
                logger.error(f"Error processing scheduled rule {rule.rule_id}: {e}")
    
    async def _monitor_market_changes(self):
        """Monitor market changes and trigger updates."""
        for rule in self._update_rules.values():
            if not rule.is_active or rule.trigger_type != UpdateTrigger.MARKET_CHANGE:
                continue
            
            try:
                # Check each monitoring metric
                for metric in rule.monitoring_metrics:
                    await self._check_metric_thresholds(rule, metric)
                    
            except Exception as e:
                logger.error(f"Error monitoring rule {rule.rule_id}: {e}")
    
    async def _check_metric_thresholds(self, rule: UpdateRule, metric: MonitoringMetric):
        """Check if metric thresholds are exceeded."""
        # This would integrate with actual market data sources
        # For now, we'll simulate some market data
        
        # Get threshold conditions for this metric
        threshold_key = f"{metric.value}_threshold"
        if threshold_key not in rule.threshold_conditions:
            return
        
        threshold = rule.threshold_conditions[threshold_key]
        
        # Simulate getting current market data
        current_data = await self._get_market_data(metric)
        
        if current_data and self._threshold_exceeded(current_data, threshold):
            # Create update tasks
            for report_id in rule.target_reports:
                await self._create_update_task(
                    rule_id=rule.rule_id,
                    report_id=report_id,
                    trigger_type=UpdateTrigger.MARKET_CHANGE,
                    trigger_data={
                        "metric": metric.value,
                        "current_value": current_data.value,
                        "change_percentage": current_data.change_percentage,
                        "threshold": threshold
                    },
                    scheduled_time=datetime.utcnow()
                )
    
    async def _get_market_data(self, metric: MonitoringMetric) -> Optional[MarketDataPoint]:
        """Get current market data for a metric."""
        # This would integrate with real market data APIs
        # For now, return simulated data
        import random
        
        current_time = datetime.utcnow()
        
        # Simulate market data
        base_value = 100.0
        if metric == MonitoringMetric.PRICE_CHANGE:
            value = base_value + random.uniform(-10, 10)
        elif metric == MonitoringMetric.VOLUME_CHANGE:
            value = base_value * random.uniform(0.5, 2.0)
        else:
            value = random.uniform(0, 100)
        
        # Get previous value for comparison
        previous_value = base_value
        
        return MarketDataPoint(
            symbol="EXAMPLE",
            metric=metric,
            value=value,
            timestamp=current_time,
            source="simulated",
            previous_value=previous_value
        )
    
    def _threshold_exceeded(self, data_point: MarketDataPoint, threshold: Dict[str, Any]) -> bool:
        """Check if a threshold is exceeded."""
        threshold_type = threshold.get("type", "percentage_change")
        threshold_value = threshold.get("value", 5.0)
        
        if threshold_type == "percentage_change":
            return abs(data_point.change_percentage or 0) >= threshold_value
        elif threshold_type == "absolute_value":
            return data_point.value >= threshold_value
        elif threshold_type == "absolute_change":
            if data_point.previous_value is not None:
                return abs(data_point.value - data_point.previous_value) >= threshold_value
        
        return False
    
    async def _create_update_task(self, rule_id: str, report_id: str, 
                                 trigger_type: UpdateTrigger, trigger_data: Dict[str, Any],
                                 scheduled_time: datetime) -> str:
        """Create a new update task."""
        task = UpdateTask(
            task_id="",  # Will be auto-generated
            rule_id=rule_id,
            report_id=report_id,
            trigger_type=trigger_type,
            trigger_data=trigger_data,
            status=UpdateStatus.PENDING,
            scheduled_time=scheduled_time
        )
        
        self._update_tasks[task.task_id] = task
        
        logger.info(f"Created update task: {task.task_id} for report {report_id}")
        return task.task_id
    
    async def _process_pending_tasks(self):
        """Process pending update tasks."""
        pending_tasks = [
            task for task in self._update_tasks.values()
            if task.status == UpdateStatus.PENDING and task.task_id not in self._running_tasks
        ]
        
        # Sort by scheduled time
        pending_tasks.sort(key=lambda t: t.scheduled_time)
        
        # Process tasks (limit concurrent tasks)
        max_concurrent = 5
        current_running = len(self._running_tasks)
        
        for task in pending_tasks[:max_concurrent - current_running]:
            if task.scheduled_time <= datetime.utcnow():
                asyncio.create_task(self._execute_update_task(task))
    
    async def _execute_update_task(self, task: UpdateTask):
        """Execute an update task."""
        try:
            self._running_tasks.add(task.task_id)
            task.status = UpdateStatus.RUNNING
            task.started_time = datetime.utcnow()
            
            logger.info(f"Executing update task: {task.task_id}")
            
            # Get the update callback for this report type
            callback = self._update_callbacks.get("default")
            if not callback:
                raise ValueError("No update callback registered")
            
            # Execute the update
            result = await callback(task.report_id, task.trigger_data)
            
            # Mark task as completed
            task.status = UpdateStatus.COMPLETED
            task.completed_time = datetime.utcnow()
            task.result_data = result
            
            # Send notification
            await self._send_update_notification(task, "success")
            
            logger.info(f"Update task completed: {task.task_id}")
            
        except Exception as e:
            logger.error(f"Error executing update task {task.task_id}: {e}")
            
            task.status = UpdateStatus.FAILED
            task.error_message = str(e)
            task.completed_time = datetime.utcnow()
            
            # Retry if possible
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = UpdateStatus.PENDING
                task.scheduled_time = datetime.utcnow() + timedelta(minutes=5)  # Retry in 5 minutes
                logger.info(f"Retrying update task: {task.task_id} (attempt {task.retry_count})")
            else:
                # Send failure notification
                await self._send_update_notification(task, "failure")
        
        finally:
            self._running_tasks.discard(task.task_id)
    
    async def _send_update_notification(self, task: UpdateTask, notification_type: str):
        """Send update notification."""
        try:
            # Create notification message
            if notification_type == "success":
                message = f"Report {task.report_id} has been successfully updated"
            else:
                message = f"Failed to update report {task.report_id}: {task.error_message}"
            
            notification = UpdateNotification(
                notification_id="",  # Will be auto-generated
                update_task_id=task.task_id,
                report_id=task.report_id,
                notification_type=notification_type,
                message=message,
                recipients=["admin@example.com"]  # This would be configurable
            )
            
            self._notifications.append(notification)
            
            # Send through registered callbacks
            for callback in self._notification_callbacks:
                try:
                    await callback(notification)
                    notification.sent = True
                    notification.sent_time = datetime.utcnow()
                except Exception as e:
                    logger.error(f"Error sending notification: {e}")
            
        except Exception as e:
            logger.error(f"Error creating notification: {e}")
    
    def register_update_callback(self, report_type: str, callback: Callable):
        """Register callback for report updates."""
        self._update_callbacks[report_type] = callback
        logger.info(f"Registered update callback for {report_type}")
    
    def register_notification_callback(self, callback: Callable):
        """Register callback for notifications."""
        self._notification_callbacks.append(callback)
        logger.info("Registered notification callback")
    
    async def trigger_manual_update(self, report_id: str, trigger_data: Optional[Dict[str, Any]] = None) -> str:
        """Trigger a manual update for a report."""
        task_id = await self._create_update_task(
            rule_id="manual",
            report_id=report_id,
            trigger_type=UpdateTrigger.MANUAL,
            trigger_data=trigger_data or {},
            scheduled_time=datetime.utcnow()
        )
        
        logger.info(f"Triggered manual update for report {report_id}")
        return task_id
    
    def get_update_status(self, task_id: str) -> Optional[UpdateTask]:
        """Get status of an update task."""
        return self._update_tasks.get(task_id)
    
    def get_report_update_history(self, report_id: str, limit: int = 10) -> List[UpdateTask]:
        """Get update history for a report."""
        tasks = [
            task for task in self._update_tasks.values()
            if task.report_id == report_id
        ]
        
        # Sort by scheduled time (most recent first)
        tasks.sort(key=lambda t: t.scheduled_time, reverse=True)
        
        return tasks[:limit]
    
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        total_tasks = len(self._update_tasks)
        
        # Count by status
        status_counts = {}
        for status in UpdateStatus:
            status_counts[status.value] = len([
                task for task in self._update_tasks.values()
                if task.status == status
            ])
        
        # Count by trigger type
        trigger_counts = {}
        for trigger in UpdateTrigger:
            trigger_counts[trigger.value] = len([
                task for task in self._update_tasks.values()
                if task.trigger_type == trigger
            ])
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_tasks = len([
            task for task in self._update_tasks.values()
            if task.scheduled_time > recent_cutoff
        ])
        
        # Success rate
        completed_tasks = [task for task in self._update_tasks.values() if task.status == UpdateStatus.COMPLETED]
        failed_tasks = [task for task in self._update_tasks.values() if task.status == UpdateStatus.FAILED]
        total_finished = len(completed_tasks) + len(failed_tasks)
        success_rate = len(completed_tasks) / total_finished if total_finished > 0 else 0
        
        return {
            "scheduler_running": self._scheduler_running,
            "monitoring_running": self._monitoring_running,
            "total_rules": len(self._update_rules),
            "active_rules": len([r for r in self._update_rules.values() if r.is_active]),
            "total_tasks": total_tasks,
            "running_tasks": len(self._running_tasks),
            "recent_tasks_24h": recent_tasks,
            "status_counts": status_counts,
            "trigger_counts": trigger_counts,
            "success_rate": success_rate,
            "total_notifications": len(self._notifications),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing rule."""
        if rule_id not in self._update_rules:
            return False
        
        rule = self._update_rules[rule_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        rule.updated_at = datetime.utcnow()
        
        logger.info(f"Updated rule: {rule_id}")
        return True
    
    async def delete_rule(self, rule_id: str) -> bool:
        """Delete an update rule."""
        if rule_id not in self._update_rules:
            return False
        
        del self._update_rules[rule_id]
        logger.info(f"Deleted rule: {rule_id}")
        return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending update task."""
        if task_id not in self._update_tasks:
            return False
        
        task = self._update_tasks[task_id]
        
        if task.status == UpdateStatus.PENDING:
            task.status = UpdateStatus.CANCELLED
            task.completed_time = datetime.utcnow()
            logger.info(f"Cancelled task: {task_id}")
            return True
        
        return False
    
    def clear_old_tasks(self, older_than_days: int = 30):
        """Clear old update tasks."""
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        
        initial_count = len(self._update_tasks)
        self._update_tasks = {
            task_id: task for task_id, task in self._update_tasks.items()
            if task.scheduled_time > cutoff_date
        }
        
        cleared_count = initial_count - len(self._update_tasks)
        logger.info(f"Cleared {cleared_count} old update tasks")
        
        return cleared_count