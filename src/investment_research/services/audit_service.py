"""Audit and compliance service for tracking operations and ensuring regulatory compliance."""

import logging
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Audit event types."""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_CANCELLED = "task_cancelled"
    REPORT_GENERATED = "report_generated"
    REPORT_DOWNLOADED = "report_downloaded"
    DATA_ACCESSED = "data_accessed"
    CONFIGURATION_CHANGED = "configuration_changed"
    SYSTEM_ERROR = "system_error"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"
    DATA_EXPORT = "data_export"
    API_CALL = "api_call"


class ComplianceStatus(str, Enum):
    """Compliance status enumeration."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    REQUIRES_ACTION = "requires_action"


class DataClassification(str, Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class AuditEvent:
    """Represents an audit event."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    resource_id: Optional[str]
    resource_type: Optional[str]
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize event ID if not provided."""
        if not self.event_id:
            content = f"{self.event_type}_{self.timestamp}_{self.user_id}_{self.action}"
            self.event_id = hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class DataLineage:
    """Tracks data lineage and processing chain."""
    lineage_id: str
    source_id: str
    source_type: str
    processing_steps: List[Dict[str, Any]]
    final_output_id: str
    created_at: datetime
    data_classification: DataClassification
    retention_period_days: Optional[int] = None
    
    def __post_init__(self):
        """Initialize lineage ID if not provided."""
        if not self.lineage_id:
            content = f"{self.source_id}_{self.final_output_id}_{self.created_at}"
            self.lineage_id = hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class ComplianceRule:
    """Defines a compliance rule."""
    rule_id: str
    name: str
    description: str
    regulation: str  # e.g., "GDPR", "SOX", "MiFID II"
    rule_type: str
    conditions: Dict[str, Any]
    severity: str
    is_active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamp."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ComplianceViolation:
    """Represents a compliance violation."""
    violation_id: str
    rule_id: str
    event_id: Optional[str]
    violation_type: str
    description: str
    severity: str
    status: ComplianceStatus
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    assigned_to: Optional[str] = None
    
    def __post_init__(self):
        """Initialize violation ID if not provided."""
        if not self.violation_id:
            content = f"{self.rule_id}_{self.detected_at}_{self.violation_type}"
            self.violation_id = hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class AuditReport:
    """Audit report structure."""
    report_id: str
    report_type: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    generated_by: str
    summary: Dict[str, Any]
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    compliance_status: ComplianceStatus
    
    def __post_init__(self):
        """Initialize report ID if not provided."""
        if not self.report_id:
            content = f"{self.report_type}_{self.period_start}_{self.period_end}"
            self.report_id = hashlib.md5(content.encode()).hexdigest()[:16]


class AuditService:
    """Service for audit logging and compliance management."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize audit service."""
        self.storage_path = Path(storage_path) if storage_path else Path("./audit_logs")
        self.storage_path.mkdir(exist_ok=True)
        
        self._audit_events: List[AuditEvent] = []
        self._data_lineages: Dict[str, DataLineage] = {}
        self._compliance_rules: Dict[str, ComplianceRule] = {}
        self._compliance_violations: List[ComplianceViolation] = []
        self._audit_reports: Dict[str, AuditReport] = {}
        
        # Setup default compliance rules
        self._setup_default_compliance_rules()
    
    def _setup_default_compliance_rules(self):
        """Setup default compliance rules."""
        default_rules = [
            ComplianceRule(
                rule_id="data_retention",
                name="Data Retention Policy",
                description="Ensure data is not retained beyond specified periods",
                regulation="GDPR",
                rule_type="data_retention",
                conditions={"max_retention_days": 365},
                severity="high"
            ),
            ComplianceRule(
                rule_id="access_logging",
                name="Access Logging Requirement",
                description="All data access must be logged",
                regulation="SOX",
                rule_type="access_control",
                conditions={"require_logging": True},
                severity="critical"
            ),
            ComplianceRule(
                rule_id="data_classification",
                name="Data Classification Requirement",
                description="All data must be properly classified",
                regulation="Internal Policy",
                rule_type="data_governance",
                conditions={"require_classification": True},
                severity="medium"
            ),
            ComplianceRule(
                rule_id="user_authentication",
                name="User Authentication Requirement",
                description="All actions must be performed by authenticated users",
                regulation="Security Policy",
                rule_type="authentication",
                conditions={"require_authentication": True},
                severity="critical"
            ),
            ComplianceRule(
                rule_id="sensitive_data_access",
                name="Sensitive Data Access Control",
                description="Access to sensitive data must be authorized and logged",
                regulation="Privacy Policy",
                rule_type="access_control",
                conditions={"require_authorization": True, "log_access": True},
                severity="high"
            )
        ]
        
        for rule in default_rules:
            self._compliance_rules[rule.rule_id] = rule
    
    async def log_event(self, event_type: AuditEventType, action: str, 
                       user_id: Optional[str] = None, session_id: Optional[str] = None,
                       resource_id: Optional[str] = None, resource_type: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None, 
                       ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                       success: bool = True, error_message: Optional[str] = None) -> str:
        """Log an audit event."""
        try:
            event = AuditEvent(
                event_id="",  # Will be auto-generated
                event_type=event_type,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                session_id=session_id,
                resource_id=resource_id,
                resource_type=resource_type,
                action=action,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent,
                success=success,
                error_message=error_message
            )
            
            # Store event
            self._audit_events.append(event)
            
            # Persist to file
            await self._persist_event(event)
            
            # Check compliance
            await self._check_compliance(event)
            
            logger.info(f"Audit event logged: {event.event_id}")
            return event.event_id
            
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            raise
    
    async def _persist_event(self, event: AuditEvent):
        """Persist audit event to file."""
        try:
            # Create daily log file
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = self.storage_path / f"audit_{date_str}.jsonl"
            
            # Append event to file
            with open(log_file, "a") as f:
                event_data = asdict(event)
                # Convert datetime to ISO string
                event_data["timestamp"] = event.timestamp.isoformat()
                f.write(json.dumps(event_data) + "\n")
                
        except Exception as e:
            logger.error(f"Error persisting audit event: {e}")
    
    async def create_data_lineage(self, source_id: str, source_type: str,
                                 processing_steps: List[Dict[str, Any]],
                                 final_output_id: str, 
                                 data_classification: DataClassification,
                                 retention_period_days: Optional[int] = None) -> str:
        """Create data lineage record."""
        try:
            lineage = DataLineage(
                lineage_id="",  # Will be auto-generated
                source_id=source_id,
                source_type=source_type,
                processing_steps=processing_steps,
                final_output_id=final_output_id,
                created_at=datetime.utcnow(),
                data_classification=data_classification,
                retention_period_days=retention_period_days
            )
            
            # Store lineage
            self._data_lineages[lineage.lineage_id] = lineage
            
            # Log lineage creation
            await self.log_event(
                event_type=AuditEventType.DATA_ACCESSED,
                action="create_data_lineage",
                resource_id=lineage.lineage_id,
                resource_type="data_lineage",
                details={
                    "source_id": source_id,
                    "source_type": source_type,
                    "final_output_id": final_output_id,
                    "data_classification": data_classification.value,
                    "processing_steps_count": len(processing_steps)
                }
            )
            
            logger.info(f"Data lineage created: {lineage.lineage_id}")
            return lineage.lineage_id
            
        except Exception as e:
            logger.error(f"Error creating data lineage: {e}")
            raise
    
    async def _check_compliance(self, event: AuditEvent):
        """Check event against compliance rules."""
        try:
            for rule in self._compliance_rules.values():
                if not rule.is_active:
                    continue
                
                violation = await self._evaluate_compliance_rule(rule, event)
                if violation:
                    self._compliance_violations.append(violation)
                    
                    # Log compliance violation
                    await self.log_event(
                        event_type=AuditEventType.COMPLIANCE_CHECK,
                        action="compliance_violation_detected",
                        resource_id=violation.violation_id,
                        resource_type="compliance_violation",
                        details={
                            "rule_id": rule.rule_id,
                            "violation_type": violation.violation_type,
                            "severity": violation.severity
                        }
                    )
                    
                    logger.warning(f"Compliance violation detected: {violation.violation_id}")
            
        except Exception as e:
            logger.error(f"Error checking compliance: {e}")
    
    async def _evaluate_compliance_rule(self, rule: ComplianceRule, event: AuditEvent) -> Optional[ComplianceViolation]:
        """Evaluate a compliance rule against an event."""
        try:
            if rule.rule_type == "access_control":
                return await self._check_access_control_rule(rule, event)
            elif rule.rule_type == "data_retention":
                return await self._check_data_retention_rule(rule, event)
            elif rule.rule_type == "authentication":
                return await self._check_authentication_rule(rule, event)
            elif rule.rule_type == "data_governance":
                return await self._check_data_governance_rule(rule, event)
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating compliance rule {rule.rule_id}: {e}")
            return None
    
    async def _check_access_control_rule(self, rule: ComplianceRule, event: AuditEvent) -> Optional[ComplianceViolation]:
        """Check access control compliance rule."""
        conditions = rule.conditions
        
        # Check if logging is required
        if conditions.get("require_logging") and event.event_type != AuditEventType.DATA_ACCESSED:
            return None  # This rule only applies to data access events
        
        if event.event_type == AuditEventType.DATA_ACCESSED:
            # Check if access is properly logged
            if not event.details.get("data_classification"):
                return ComplianceViolation(
                    violation_id="",
                    rule_id=rule.rule_id,
                    event_id=event.event_id,
                    violation_type="missing_data_classification",
                    description="Data access without proper classification",
                    severity=rule.severity,
                    status=ComplianceStatus.REQUIRES_ACTION,
                    detected_at=datetime.utcnow()
                )
        
        return None
    
    async def _check_data_retention_rule(self, rule: ComplianceRule, event: AuditEvent) -> Optional[ComplianceViolation]:
        """Check data retention compliance rule."""
        conditions = rule.conditions
        max_retention_days = conditions.get("max_retention_days", 365)
        
        # Check if we have old data that should be purged
        cutoff_date = datetime.utcnow() - timedelta(days=max_retention_days)
        
        # Check audit events
        old_events = [e for e in self._audit_events if e.timestamp < cutoff_date]
        if len(old_events) > 1000:  # Threshold for violation
            return ComplianceViolation(
                violation_id="",
                rule_id=rule.rule_id,
                event_id=None,
                violation_type="data_retention_exceeded",
                description=f"Found {len(old_events)} events older than {max_retention_days} days",
                severity=rule.severity,
                status=ComplianceStatus.REQUIRES_ACTION,
                detected_at=datetime.utcnow()
            )
        
        return None
    
    async def _check_authentication_rule(self, rule: ComplianceRule, event: AuditEvent) -> Optional[ComplianceViolation]:
        """Check authentication compliance rule."""
        conditions = rule.conditions
        
        if conditions.get("require_authentication"):
            # Check if user is authenticated for sensitive operations
            sensitive_events = [
                AuditEventType.TASK_CREATED,
                AuditEventType.REPORT_GENERATED,
                AuditEventType.DATA_ACCESSED,
                AuditEventType.CONFIGURATION_CHANGED
            ]
            
            if event.event_type in sensitive_events and not event.user_id:
                return ComplianceViolation(
                    violation_id="",
                    rule_id=rule.rule_id,
                    event_id=event.event_id,
                    violation_type="unauthenticated_access",
                    description=f"Sensitive operation {event.action} performed without authentication",
                    severity=rule.severity,
                    status=ComplianceStatus.REQUIRES_ACTION,
                    detected_at=datetime.utcnow()
                )
        
        return None
    
    async def _check_data_governance_rule(self, rule: ComplianceRule, event: AuditEvent) -> Optional[ComplianceViolation]:
        """Check data governance compliance rule."""
        conditions = rule.conditions
        
        if conditions.get("require_classification"):
            # Check if data is properly classified
            if event.event_type == AuditEventType.DATA_ACCESSED:
                if not event.details.get("data_classification"):
                    return ComplianceViolation(
                        violation_id="",
                        rule_id=rule.rule_id,
                        event_id=event.event_id,
                        violation_type="missing_data_classification",
                        description="Data accessed without proper classification",
                        severity=rule.severity,
                        status=ComplianceStatus.REQUIRES_ACTION,
                        detected_at=datetime.utcnow()
                    )
        
        return None
    
    async def generate_audit_report(self, report_type: str, period_start: datetime,
                                   period_end: datetime, generated_by: str) -> str:
        """Generate audit report for specified period."""
        try:
            # Filter events for the period
            period_events = [
                event for event in self._audit_events
                if period_start <= event.timestamp <= period_end
            ]
            
            # Generate summary
            summary = self._generate_audit_summary(period_events)
            
            # Generate findings
            findings = self._generate_audit_findings(period_events)
            
            # Generate recommendations
            recommendations = self._generate_audit_recommendations(period_events)
            
            # Determine overall compliance status
            compliance_status = self._determine_compliance_status(period_events)
            
            # Create report
            report = AuditReport(
                report_id="",  # Will be auto-generated
                report_type=report_type,
                period_start=period_start,
                period_end=period_end,
                generated_at=datetime.utcnow(),
                generated_by=generated_by,
                summary=summary,
                findings=findings,
                recommendations=recommendations,
                compliance_status=compliance_status
            )
            
            # Store report
            self._audit_reports[report.report_id] = report
            
            # Log report generation
            await self.log_event(
                event_type=AuditEventType.REPORT_GENERATED,
                action="generate_audit_report",
                user_id=generated_by,
                resource_id=report.report_id,
                resource_type="audit_report",
                details={
                    "report_type": report_type,
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                    "events_analyzed": len(period_events)
                }
            )
            
            logger.info(f"Audit report generated: {report.report_id}")
            return report.report_id
            
        except Exception as e:
            logger.error(f"Error generating audit report: {e}")
            raise
    
    def _generate_audit_summary(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate audit summary from events."""
        summary = {
            "total_events": len(events),
            "events_by_type": {},
            "events_by_user": {},
            "success_rate": 0.0,
            "period_violations": len([
                v for v in self._compliance_violations
                if any(e.event_id == v.event_id for e in events)
            ])
        }
        
        # Count events by type
        for event in events:
            event_type = event.event_type.value
            summary["events_by_type"][event_type] = summary["events_by_type"].get(event_type, 0) + 1
        
        # Count events by user
        for event in events:
            user_id = event.user_id or "anonymous"
            summary["events_by_user"][user_id] = summary["events_by_user"].get(user_id, 0) + 1
        
        # Calculate success rate
        successful_events = len([e for e in events if e.success])
        summary["success_rate"] = successful_events / len(events) if events else 1.0
        
        return summary
    
    def _generate_audit_findings(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Generate audit findings from events."""
        findings = []
        
        # Check for security events
        security_events = [e for e in events if e.event_type == AuditEventType.SECURITY_EVENT]
        if security_events:
            findings.append({
                "type": "security_concern",
                "severity": "high",
                "description": f"Found {len(security_events)} security events",
                "details": {"event_count": len(security_events)}
            })
        
        # Check for failed operations
        failed_events = [e for e in events if not e.success]
        if len(failed_events) > len(events) * 0.1:  # More than 10% failure rate
            findings.append({
                "type": "high_failure_rate",
                "severity": "medium",
                "description": f"High failure rate: {len(failed_events)}/{len(events)} operations failed",
                "details": {"failure_rate": len(failed_events) / len(events)}
            })
        
        # Check for compliance violations
        period_violations = [
            v for v in self._compliance_violations
            if any(e.event_id == v.event_id for e in events)
        ]
        if period_violations:
            findings.append({
                "type": "compliance_violations",
                "severity": "high",
                "description": f"Found {len(period_violations)} compliance violations",
                "details": {
                    "violation_count": len(period_violations),
                    "violations_by_type": {}
                }
            })
        
        return findings
    
    def _generate_audit_recommendations(self, events: List[AuditEvent]) -> List[str]:
        """Generate audit recommendations."""
        recommendations = []
        
        # Check for authentication issues
        unauthenticated_events = [e for e in events if not e.user_id and e.event_type in [
            AuditEventType.TASK_CREATED, AuditEventType.DATA_ACCESSED
        ]]
        if unauthenticated_events:
            recommendations.append("Implement stronger authentication requirements for sensitive operations")
        
        # Check for error patterns
        failed_events = [e for e in events if not e.success]
        if len(failed_events) > 10:
            recommendations.append("Investigate and address recurring system errors")
        
        # Check for data access patterns
        data_access_events = [e for e in events if e.event_type == AuditEventType.DATA_ACCESSED]
        if len(data_access_events) > len(events) * 0.5:
            recommendations.append("Review data access patterns and implement data minimization practices")
        
        if not recommendations:
            recommendations.append("Continue current security and compliance practices")
        
        return recommendations
    
    def _determine_compliance_status(self, events: List[AuditEvent]) -> ComplianceStatus:
        """Determine overall compliance status."""
        period_violations = [
            v for v in self._compliance_violations
            if any(e.event_id == v.event_id for e in events)
        ]
        
        critical_violations = [v for v in period_violations if v.severity == "critical"]
        high_violations = [v for v in period_violations if v.severity == "high"]
        
        if critical_violations:
            return ComplianceStatus.NON_COMPLIANT
        elif high_violations:
            return ComplianceStatus.REQUIRES_ACTION
        elif period_violations:
            return ComplianceStatus.PENDING_REVIEW
        else:
            return ComplianceStatus.COMPLIANT
    
    def get_audit_events(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        event_type: Optional[AuditEventType] = None,
                        user_id: Optional[str] = None,
                        limit: int = 1000) -> List[AuditEvent]:
        """Get audit events with filtering."""
        events = self._audit_events
        
        # Apply filters
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        # Sort by timestamp (most recent first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return events[:limit]
    
    def get_data_lineage(self, lineage_id: str) -> Optional[DataLineage]:
        """Get data lineage by ID."""
        return self._data_lineages.get(lineage_id)
    
    def get_compliance_violations(self, status: Optional[ComplianceStatus] = None,
                                 severity: Optional[str] = None) -> List[ComplianceViolation]:
        """Get compliance violations with filtering."""
        violations = self._compliance_violations
        
        if status:
            violations = [v for v in violations if v.status == status]
        
        if severity:
            violations = [v for v in violations if v.severity == severity]
        
        # Sort by detection time (most recent first)
        violations.sort(key=lambda v: v.detected_at, reverse=True)
        
        return violations
    
    async def resolve_compliance_violation(self, violation_id: str, resolution_notes: str,
                                          resolved_by: str) -> bool:
        """Resolve a compliance violation."""
        try:
            violation = None
            for v in self._compliance_violations:
                if v.violation_id == violation_id:
                    violation = v
                    break
            
            if not violation:
                return False
            
            violation.status = ComplianceStatus.COMPLIANT
            violation.resolved_at = datetime.utcnow()
            violation.resolution_notes = resolution_notes
            violation.assigned_to = resolved_by
            
            # Log resolution
            await self.log_event(
                event_type=AuditEventType.COMPLIANCE_CHECK,
                action="resolve_compliance_violation",
                user_id=resolved_by,
                resource_id=violation_id,
                resource_type="compliance_violation",
                details={
                    "violation_type": violation.violation_type,
                    "resolution_notes": resolution_notes
                }
            )
            
            logger.info(f"Compliance violation resolved: {violation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving compliance violation {violation_id}: {e}")
            return False
    
    def get_audit_report(self, report_id: str) -> Optional[AuditReport]:
        """Get audit report by ID."""
        return self._audit_reports.get(report_id)
    
    def list_audit_reports(self) -> List[AuditReport]:
        """List all audit reports."""
        reports = list(self._audit_reports.values())
        reports.sort(key=lambda r: r.generated_at, reverse=True)
        return reports
    
    async def cleanup_old_data(self, retention_days: int = 365) -> Dict[str, int]:
        """Clean up old audit data based on retention policy."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            # Clean up old events
            initial_events = len(self._audit_events)
            self._audit_events = [e for e in self._audit_events if e.timestamp > cutoff_date]
            cleaned_events = initial_events - len(self._audit_events)
            
            # Clean up old violations
            initial_violations = len(self._compliance_violations)
            self._compliance_violations = [
                v for v in self._compliance_violations 
                if v.detected_at > cutoff_date
            ]
            cleaned_violations = initial_violations - len(self._compliance_violations)
            
            # Clean up old lineages
            initial_lineages = len(self._data_lineages)
            self._data_lineages = {
                k: v for k, v in self._data_lineages.items()
                if v.created_at > cutoff_date
            }
            cleaned_lineages = initial_lineages - len(self._data_lineages)
            
            # Log cleanup
            await self.log_event(
                event_type=AuditEventType.SYSTEM_ERROR,  # Using as maintenance event
                action="cleanup_old_audit_data",
                details={
                    "retention_days": retention_days,
                    "cleaned_events": cleaned_events,
                    "cleaned_violations": cleaned_violations,
                    "cleaned_lineages": cleaned_lineages
                }
            )
            
            logger.info(f"Cleaned up old audit data: {cleaned_events} events, {cleaned_violations} violations, {cleaned_lineages} lineages")
            
            return {
                "cleaned_events": cleaned_events,
                "cleaned_violations": cleaned_violations,
                "cleaned_lineages": cleaned_lineages
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up old audit data: {e}")
            raise
    
    def export_audit_data(self, start_date: datetime, end_date: datetime,
                         format_type: str = "json") -> str:
        """Export audit data for specified period."""
        try:
            # Get events for period
            events = self.get_audit_events(start_date, end_date)
            
            # Get violations for period
            violations = [
                v for v in self._compliance_violations
                if start_date <= v.detected_at <= end_date
            ]
            
            # Prepare export data
            export_data = {
                "export_metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "period_start": start_date.isoformat(),
                    "period_end": end_date.isoformat(),
                    "format": format_type
                },
                "audit_events": [asdict(event) for event in events],
                "compliance_violations": [asdict(violation) for violation in violations],
                "summary": {
                    "total_events": len(events),
                    "total_violations": len(violations)
                }
            }
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            # Serialize to JSON
            if format_type == "json":
                return json.dumps(export_data, default=convert_datetime, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
        except Exception as e:
            logger.error(f"Error exporting audit data: {e}")
            raise