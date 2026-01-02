"""System monitoring and performance management service."""

import asyncio
import logging
import psutil
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """System metric types."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    API_RESPONSE_TIME = "api_response_time"
    API_REQUEST_COUNT = "api_request_count"
    API_ERROR_RATE = "api_error_rate"
    TASK_COMPLETION_TIME = "task_completion_time"
    QUEUE_SIZE = "queue_size"
    ACTIVE_CONNECTIONS = "active_connections"
    CACHE_HIT_RATE = "cache_hit_rate"
    DATABASE_CONNECTIONS = "database_connections"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class MetricData:
    """Represents a system metric data point."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize tags if not provided."""
        if self.tags is None:
            self.tags = {}


@dataclass
class AlertRule:
    """Defines conditions for triggering alerts."""
    rule_id: str
    name: str
    metric_type: MetricType
    condition: str  # e.g., "> 80", "< 0.95"
    severity: AlertSeverity
    description: str
    cooldown_minutes: int = 15
    is_active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamp."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class Alert:
    """Represents a system alert."""
    alert_id: str
    rule_id: str
    metric_type: MetricType
    severity: AlertSeverity
    message: str
    current_value: float
    threshold_value: float
    status: AlertStatus
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    def __post_init__(self):
        """Initialize alert ID if not provided."""
        if not self.alert_id:
            import hashlib
            content = f"{self.rule_id}_{self.triggered_at}_{self.current_value}"
            self.alert_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class PerformanceReport:
    """Performance analysis report."""
    report_id: str
    start_time: datetime
    end_time: datetime
    metrics_summary: Dict[str, Dict[str, float]]
    alerts_summary: Dict[str, int]
    recommendations: List[str]
    generated_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamp and report ID."""
        if self.generated_at is None:
            self.generated_at = datetime.utcnow()
        
        if not self.report_id:
            import hashlib
            content = f"{self.start_time}_{self.end_time}_{self.generated_at}"
            self.report_id = hashlib.md5(content.encode()).hexdigest()[:12]


class SystemMonitor:
    """Service for system monitoring and performance management."""
    
    def __init__(self, max_data_points: int = 10000):
        """Initialize system monitor."""
        self.max_data_points = max_data_points
        self._metrics_data: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=max_data_points))
        self._alert_rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._api_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "request_count": 0,
            "error_count": 0,
            "total_response_time": 0.0,
            "last_reset": datetime.utcnow()
        })
        self._monitoring_running = False
        self._alert_callbacks: List[Callable] = []
        
        # Performance thresholds
        self._default_thresholds = {
            MetricType.CPU_USAGE: 80.0,
            MetricType.MEMORY_USAGE: 85.0,
            MetricType.DISK_USAGE: 90.0,
            MetricType.API_RESPONSE_TIME: 5000.0,  # 5 seconds
            MetricType.API_ERROR_RATE: 5.0,  # 5%
            MetricType.CACHE_HIT_RATE: 0.8  # 80%
        }
        
        # Setup default alert rules
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="cpu_high",
                name="High CPU Usage",
                metric_type=MetricType.CPU_USAGE,
                condition="> 80",
                severity=AlertSeverity.WARNING,
                description="CPU usage is above 80%"
            ),
            AlertRule(
                rule_id="memory_high",
                name="High Memory Usage",
                metric_type=MetricType.MEMORY_USAGE,
                condition="> 85",
                severity=AlertSeverity.WARNING,
                description="Memory usage is above 85%"
            ),
            AlertRule(
                rule_id="disk_critical",
                name="Critical Disk Usage",
                metric_type=MetricType.DISK_USAGE,
                condition="> 95",
                severity=AlertSeverity.CRITICAL,
                description="Disk usage is above 95%"
            ),
            AlertRule(
                rule_id="api_slow",
                name="Slow API Response",
                metric_type=MetricType.API_RESPONSE_TIME,
                condition="> 5000",
                severity=AlertSeverity.WARNING,
                description="API response time is above 5 seconds"
            ),
            AlertRule(
                rule_id="api_errors",
                name="High API Error Rate",
                metric_type=MetricType.API_ERROR_RATE,
                condition="> 10",
                severity=AlertSeverity.ERROR,
                description="API error rate is above 10%"
            )
        ]
        
        for rule in default_rules:
            self._alert_rules[rule.rule_id] = rule
    
    async def start_monitoring(self):
        """Start system monitoring."""
        if self._monitoring_running:
            logger.warning("Monitoring is already running")
            return
        
        self._monitoring_running = True
        logger.info("Starting system monitoring")
        
        # Start monitoring loops
        asyncio.create_task(self._system_metrics_loop())
        asyncio.create_task(self._alert_processing_loop())
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring_running = False
        logger.info("Stopping system monitoring")
    
    async def _system_metrics_loop(self):
        """Main system metrics collection loop."""
        while self._monitoring_running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in system metrics loop: {e}")
                await asyncio.sleep(30)
    
    async def _alert_processing_loop(self):
        """Alert processing loop."""
        while self._monitoring_running:
            try:
                await self._process_alerts()
                await asyncio.sleep(60)  # Check alerts every minute
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        current_time = datetime.utcnow()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric(MetricType.CPU_USAGE, cpu_percent, current_time)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._add_metric(MetricType.MEMORY_USAGE, memory_percent, current_time)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._add_metric(MetricType.DISK_USAGE, disk_percent, current_time)
            
            # Network I/O
            network = psutil.net_io_counters()
            if network:
                # Calculate bytes per second (simplified)
                self._add_metric(MetricType.NETWORK_IO, network.bytes_sent + network.bytes_recv, current_time)
            
            # Database connections (simulated)
            # In a real implementation, this would query the actual database
            db_connections = len(self._active_alerts)  # Placeholder
            self._add_metric(MetricType.DATABASE_CONNECTIONS, db_connections, current_time)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _add_metric(self, metric_type: MetricType, value: float, timestamp: datetime, tags: Dict[str, str] = None):
        """Add a metric data point."""
        metric_data = MetricData(
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            tags=tags or {}
        )
        
        self._metrics_data[metric_type].append(metric_data)
    
    async def record_api_call(self, endpoint: str, response_time_ms: float, success: bool):
        """Record API call statistics."""
        stats = self._api_stats[endpoint]
        stats["request_count"] += 1
        stats["total_response_time"] += response_time_ms
        
        if not success:
            stats["error_count"] += 1
        
        # Add metrics
        current_time = datetime.utcnow()
        self._add_metric(MetricType.API_RESPONSE_TIME, response_time_ms, current_time, {"endpoint": endpoint})
        self._add_metric(MetricType.API_REQUEST_COUNT, 1, current_time, {"endpoint": endpoint})
        
        # Calculate error rate
        error_rate = (stats["error_count"] / stats["request_count"]) * 100
        self._add_metric(MetricType.API_ERROR_RATE, error_rate, current_time, {"endpoint": endpoint})
    
    async def record_task_completion(self, task_type: str, completion_time_ms: float):
        """Record task completion time."""
        current_time = datetime.utcnow()
        self._add_metric(MetricType.TASK_COMPLETION_TIME, completion_time_ms, current_time, {"task_type": task_type})
    
    async def record_queue_size(self, queue_name: str, size: int):
        """Record queue size."""
        current_time = datetime.utcnow()
        self._add_metric(MetricType.QUEUE_SIZE, size, current_time, {"queue": queue_name})
    
    async def record_cache_hit_rate(self, cache_name: str, hit_rate: float):
        """Record cache hit rate."""
        current_time = datetime.utcnow()
        self._add_metric(MetricType.CACHE_HIT_RATE, hit_rate, current_time, {"cache": cache_name})
    
    async def _process_alerts(self):
        """Process alert rules and trigger alerts."""
        for rule in self._alert_rules.values():
            if not rule.is_active:
                continue
            
            try:
                await self._check_alert_rule(rule)
            except Exception as e:
                logger.error(f"Error checking alert rule {rule.rule_id}: {e}")
    
    async def _check_alert_rule(self, rule: AlertRule):
        """Check if an alert rule should trigger."""
        # Get recent metrics for this rule
        if rule.metric_type not in self._metrics_data:
            return
        
        metrics = self._metrics_data[rule.metric_type]
        if not metrics:
            return
        
        # Get the most recent metric
        latest_metric = metrics[-1]
        
        # Check if condition is met
        if self._evaluate_condition(latest_metric.value, rule.condition):
            # Check if we're in cooldown period
            if self._is_in_cooldown(rule):
                return
            
            # Create alert
            await self._create_alert(rule, latest_metric)
    
    def _evaluate_condition(self, value: float, condition: str) -> bool:
        """Evaluate alert condition."""
        try:
            # Parse condition (e.g., "> 80", "< 0.95")
            condition = condition.strip()
            
            if condition.startswith(">"):
                threshold = float(condition[1:].strip())
                return value > threshold
            elif condition.startswith("<"):
                threshold = float(condition[1:].strip())
                return value < threshold
            elif condition.startswith(">="):
                threshold = float(condition[2:].strip())
                return value >= threshold
            elif condition.startswith("<="):
                threshold = float(condition[2:].strip())
                return value <= threshold
            elif condition.startswith("=="):
                threshold = float(condition[2:].strip())
                return value == threshold
            elif condition.startswith("!="):
                threshold = float(condition[2:].strip())
                return value != threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _is_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if alert rule is in cooldown period."""
        # Find the most recent alert for this rule
        recent_alerts = [
            alert for alert in self._alert_history
            if alert.rule_id == rule.rule_id
        ]
        
        if not recent_alerts:
            return False
        
        # Sort by triggered time
        recent_alerts.sort(key=lambda a: a.triggered_at, reverse=True)
        latest_alert = recent_alerts[0]
        
        # Check if cooldown period has passed
        cooldown_period = timedelta(minutes=rule.cooldown_minutes)
        return datetime.utcnow() - latest_alert.triggered_at < cooldown_period
    
    async def _create_alert(self, rule: AlertRule, metric: MetricData):
        """Create a new alert."""
        try:
            # Extract threshold from condition
            threshold_value = float(rule.condition.split()[-1]) if rule.condition.split() else 0.0
            
            alert = Alert(
                alert_id="",  # Will be auto-generated
                rule_id=rule.rule_id,
                metric_type=rule.metric_type,
                severity=rule.severity,
                message=f"{rule.description}. Current value: {metric.value:.2f}",
                current_value=metric.value,
                threshold_value=threshold_value,
                status=AlertStatus.ACTIVE,
                triggered_at=datetime.utcnow()
            )
            
            # Store alert
            self._active_alerts[alert.alert_id] = alert
            self._alert_history.append(alert)
            
            # Send alert notifications
            await self._send_alert_notification(alert)
            
            logger.warning(f"Alert triggered: {alert.alert_id} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification through registered callbacks."""
        for callback in self._alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error sending alert notification: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alert notifications."""
        self._alert_callbacks.append(callback)
        logger.info("Registered alert callback")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self._alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.rule_id}")
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self._alert_rules:
            del self._alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
        return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            
            # Remove from active alerts
            del self._active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        current_metrics = {}
        
        for metric_type, data_points in self._metrics_data.items():
            if data_points:
                latest = data_points[-1]
                current_metrics[metric_type.value] = {
                    "value": latest.value,
                    "timestamp": latest.timestamp.isoformat(),
                    "tags": latest.tags
                }
        
        return current_metrics
    
    def get_metric_history(self, metric_type: MetricType, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metric history for a specified time period."""
        if metric_type not in self._metrics_data:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        history = []
        for data_point in self._metrics_data[metric_type]:
            if data_point.timestamp >= cutoff_time:
                history.append({
                    "value": data_point.value,
                    "timestamp": data_point.timestamp.isoformat(),
                    "tags": data_point.tags
                })
        
        return history
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self._active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for a specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            alert for alert in self._alert_history
            if alert.triggered_at >= cutoff_time
        ]
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        stats = {}
        
        for endpoint, data in self._api_stats.items():
            if data["request_count"] > 0:
                avg_response_time = data["total_response_time"] / data["request_count"]
                error_rate = (data["error_count"] / data["request_count"]) * 100
                
                stats[endpoint] = {
                    "request_count": data["request_count"],
                    "error_count": data["error_count"],
                    "error_rate": error_rate,
                    "avg_response_time": avg_response_time,
                    "last_reset": data["last_reset"].isoformat()
                }
        
        return stats
    
    async def generate_performance_report(self, hours: int = 24) -> PerformanceReport:
        """Generate a performance analysis report."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Calculate metrics summary
        metrics_summary = {}
        
        for metric_type, data_points in self._metrics_data.items():
            # Filter data points within time range
            relevant_points = [
                dp for dp in data_points
                if start_time <= dp.timestamp <= end_time
            ]
            
            if relevant_points:
                values = [dp.value for dp in relevant_points]
                metrics_summary[metric_type.value] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": statistics.mean(values),
                    "median": statistics.median(values),
                    "count": len(values)
                }
        
        # Calculate alerts summary
        recent_alerts = self.get_alert_history(hours)
        alerts_summary = {}
        
        for severity in AlertSeverity:
            alerts_summary[severity.value] = len([
                alert for alert in recent_alerts
                if alert.severity == severity
            ])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics_summary, recent_alerts)
        
        report = PerformanceReport(
            report_id="",  # Will be auto-generated
            start_time=start_time,
            end_time=end_time,
            metrics_summary=metrics_summary,
            alerts_summary=alerts_summary,
            recommendations=recommendations
        )
        
        return report
    
    def _generate_recommendations(self, metrics_summary: Dict[str, Dict[str, float]], 
                                 recent_alerts: List[Alert]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Check CPU usage
        if MetricType.CPU_USAGE.value in metrics_summary:
            cpu_stats = metrics_summary[MetricType.CPU_USAGE.value]
            if cpu_stats["avg"] > 70:
                recommendations.append("Consider scaling up CPU resources or optimizing CPU-intensive operations")
        
        # Check memory usage
        if MetricType.MEMORY_USAGE.value in metrics_summary:
            memory_stats = metrics_summary[MetricType.MEMORY_USAGE.value]
            if memory_stats["avg"] > 80:
                recommendations.append("Monitor memory usage closely and consider increasing available memory")
        
        # Check API performance
        if MetricType.API_RESPONSE_TIME.value in metrics_summary:
            api_stats = metrics_summary[MetricType.API_RESPONSE_TIME.value]
            if api_stats["avg"] > 2000:  # 2 seconds
                recommendations.append("API response times are high. Consider optimizing database queries or adding caching")
        
        # Check error rates
        critical_alerts = [alert for alert in recent_alerts if alert.severity == AlertSeverity.CRITICAL]
        if len(critical_alerts) > 5:
            recommendations.append("High number of critical alerts. Review system stability and error handling")
        
        # Check cache performance
        if MetricType.CACHE_HIT_RATE.value in metrics_summary:
            cache_stats = metrics_summary[MetricType.CACHE_HIT_RATE.value]
            if cache_stats["avg"] < 0.8:
                recommendations.append("Cache hit rate is low. Review caching strategy and cache expiration policies")
        
        if not recommendations:
            recommendations.append("System performance appears to be within normal parameters")
        
        return recommendations
    
    def reset_api_statistics(self, endpoint: Optional[str] = None):
        """Reset API statistics."""
        if endpoint:
            if endpoint in self._api_stats:
                self._api_stats[endpoint] = {
                    "request_count": 0,
                    "error_count": 0,
                    "total_response_time": 0.0,
                    "last_reset": datetime.utcnow()
                }
                logger.info(f"Reset API statistics for {endpoint}")
        else:
            for endpoint in self._api_stats:
                self._api_stats[endpoint] = {
                    "request_count": 0,
                    "error_count": 0,
                    "total_response_time": 0.0,
                    "last_reset": datetime.utcnow()
                }
            logger.info("Reset all API statistics")
    
    def get_system_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        scores = {}
        weights = {
            MetricType.CPU_USAGE: 0.2,
            MetricType.MEMORY_USAGE: 0.2,
            MetricType.DISK_USAGE: 0.15,
            MetricType.API_RESPONSE_TIME: 0.2,
            MetricType.API_ERROR_RATE: 0.15,
            MetricType.CACHE_HIT_RATE: 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric_type, weight in weights.items():
            if metric_type in self._metrics_data and self._metrics_data[metric_type]:
                latest_value = self._metrics_data[metric_type][-1].value
                threshold = self._default_thresholds.get(metric_type, 100.0)
                
                # Calculate score (0-100)
                if metric_type in [MetricType.CPU_USAGE, MetricType.MEMORY_USAGE, MetricType.DISK_USAGE, MetricType.API_ERROR_RATE]:
                    # Lower is better
                    score = max(0, 100 - (latest_value / threshold) * 100)
                elif metric_type == MetricType.API_RESPONSE_TIME:
                    # Lower is better (response time in ms)
                    score = max(0, 100 - (latest_value / threshold) * 100)
                elif metric_type == MetricType.CACHE_HIT_RATE:
                    # Higher is better
                    score = (latest_value / threshold) * 100
                else:
                    score = 100  # Default good score
                
                scores[metric_type.value] = min(100, max(0, score))
                total_score += score * weight
                total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 100
        
        # Determine health status
        if overall_score >= 90:
            health_status = "excellent"
        elif overall_score >= 80:
            health_status = "good"
        elif overall_score >= 70:
            health_status = "fair"
        elif overall_score >= 60:
            health_status = "poor"
        else:
            health_status = "critical"
        
        return {
            "overall_score": round(overall_score, 2),
            "health_status": health_status,
            "metric_scores": scores,
            "active_alerts": len(self._active_alerts),
            "generated_at": datetime.utcnow().isoformat()
        }