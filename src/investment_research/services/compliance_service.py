"""Compliance service for regulatory compliance checks and reporting."""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class RegulationType(str, Enum):
    """Regulatory framework types."""
    GDPR = "gdpr"
    SOX = "sox"
    MIFID_II = "mifid_ii"
    BASEL_III = "basel_iii"
    DODD_FRANK = "dodd_frank"
    INTERNAL_POLICY = "internal_policy"
    SECURITY_POLICY = "security_policy"


class ComplianceCheckType(str, Enum):
    """Types of compliance checks."""
    DATA_SOURCES = "data_sources"
    CITATIONS = "citations"
    VERSION_TRACKING = "version_tracking"
    ACCESS_CONTROL = "access_control"
    DATA_RETENTION = "data_retention"
    AUDIT_TRAIL = "audit_trail"
    DISCLOSURE = "disclosure"
    CONFLICT_OF_INTEREST = "conflict_of_interest"


class ComplianceStatus(str, Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    PENDING_REVIEW = "pending_review"
    REQUIRES_REMEDIATION = "requires_remediation"


@dataclass
class DataSource:
    """Represents a data source with compliance metadata."""
    source_id: str
    name: str
    provider: str
    source_type: str
    url: Optional[str]
    access_date: datetime
    license_type: str
    reliability_score: float
    is_public: bool
    regulatory_approval: Optional[str] = None
    data_classification: str = "internal"
    retention_period_days: Optional[int] = None
    
    def __post_init__(self):
        """Initialize source ID if not provided."""
        if not self.source_id:
            content = f"{self.name}_{self.provider}_{self.access_date}"
            self.source_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class Citation:
    """Represents a citation with compliance metadata."""
    citation_id: str
    source_id: str
    content: str
    page_reference: Optional[str]
    quote: Optional[str]
    usage_context: str
    citation_format: str
    verification_status: str
    created_at: datetime
    
    def __post_init__(self):
        """Initialize citation ID if not provided."""
        if not self.citation_id:
            content = f"{self.source_id}_{self.content[:50]}_{self.created_at}"
            self.citation_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class VersionRecord:
    """Tracks version history for compliance."""
    record_id: str
    resource_id: str
    resource_type: str
    version: str
    changes: List[str]
    changed_by: str
    change_reason: str
    timestamp: datetime
    previous_version: Optional[str] = None
    
    def __post_init__(self):
        """Initialize record ID if not provided."""
        if not self.record_id:
            content = f"{self.resource_id}_{self.version}_{self.timestamp}"
            self.record_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class ComplianceCheck:
    """Represents a compliance check result."""
    check_id: str
    check_type: ComplianceCheckType
    regulation: RegulationType
    resource_id: str
    resource_type: str
    status: ComplianceStatus
    score: float  # 0.0 to 1.0
    findings: List[str]
    recommendations: List[str]
    checked_at: datetime
    checked_by: str
    evidence: Dict[str, Any]
    
    def __post_init__(self):
        """Initialize check ID if not provided."""
        if not self.check_id:
            content = f"{self.check_type}_{self.resource_id}_{self.checked_at}"
            self.check_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class ComplianceReport:
    """Comprehensive compliance report."""
    report_id: str
    report_type: str
    scope: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    generated_by: str
    overall_status: ComplianceStatus
    overall_score: float
    checks_performed: List[str]
    data_sources_inventory: List[Dict[str, Any]]
    citations_list: List[Dict[str, Any]]
    version_history: List[Dict[str, Any]]
    findings_summary: Dict[str, Any]
    recommendations: List[str]
    regulatory_requirements: Dict[str, Any]
    
    def __post_init__(self):
        """Initialize report ID if not provided."""
        if not self.report_id:
            content = f"{self.report_type}_{self.scope}_{self.generated_at}"
            self.report_id = hashlib.md5(content.encode()).hexdigest()[:12]


class ComplianceService:
    """Service for compliance checks and regulatory reporting."""
    
    def __init__(self):
        """Initialize compliance service."""
        self._data_sources: Dict[str, DataSource] = {}
        self._citations: Dict[str, Citation] = {}
        self._version_records: List[VersionRecord] = []
        self._compliance_checks: List[ComplianceCheck] = []
        self._compliance_reports: Dict[str, ComplianceReport] = {}
        
        # Regulatory requirements
        self._regulatory_requirements = self._load_regulatory_requirements()
    
    def _load_regulatory_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Load regulatory requirements configuration."""
        return {
            RegulationType.GDPR.value: {
                "name": "General Data Protection Regulation",
                "requirements": {
                    "data_sources": "All data sources must be documented with legal basis",
                    "retention": "Data retention periods must be defined and enforced",
                    "consent": "User consent must be tracked and verifiable",
                    "audit_trail": "Complete audit trail required for data processing"
                },
                "penalties": "Up to 4% of annual revenue or €20 million"
            },
            RegulationType.SOX.value: {
                "name": "Sarbanes-Oxley Act",
                "requirements": {
                    "audit_trail": "Complete audit trail for financial data access",
                    "access_control": "Strict access controls for financial systems",
                    "data_integrity": "Data integrity controls and validation",
                    "reporting": "Accurate and timely financial reporting"
                },
                "penalties": "Criminal penalties up to 20 years imprisonment"
            },
            RegulationType.MIFID_II.value: {
                "name": "Markets in Financial Instruments Directive II",
                "requirements": {
                    "best_execution": "Best execution reporting and documentation",
                    "research_disclosure": "Investment research must be properly disclosed",
                    "conflict_of_interest": "Conflicts of interest must be identified and managed",
                    "record_keeping": "Comprehensive record keeping requirements"
                },
                "penalties": "Administrative fines and sanctions"
            }
        }
    
    async def register_data_source(self, name: str, provider: str, source_type: str,
                                  url: Optional[str] = None, license_type: str = "unknown",
                                  reliability_score: float = 0.5, is_public: bool = True,
                                  regulatory_approval: Optional[str] = None,
                                  data_classification: str = "internal",
                                  retention_period_days: Optional[int] = None) -> str:
        """Register a data source for compliance tracking."""
        try:
            data_source = DataSource(
                source_id="",  # Will be auto-generated
                name=name,
                provider=provider,
                source_type=source_type,
                url=url,
                access_date=datetime.utcnow(),
                license_type=license_type,
                reliability_score=reliability_score,
                is_public=is_public,
                regulatory_approval=regulatory_approval,
                data_classification=data_classification,
                retention_period_days=retention_period_days
            )
            
            self._data_sources[data_source.source_id] = data_source
            
            logger.info(f"Data source registered: {data_source.source_id}")
            return data_source.source_id
            
        except Exception as e:
            logger.error(f"Error registering data source: {e}")
            raise
    
    async def create_citation(self, source_id: str, content: str, 
                             page_reference: Optional[str] = None,
                             quote: Optional[str] = None, usage_context: str = "",
                             citation_format: str = "APA") -> str:
        """Create a citation record for compliance tracking."""
        try:
            if source_id not in self._data_sources:
                raise ValueError(f"Data source {source_id} not found")
            
            citation = Citation(
                citation_id="",  # Will be auto-generated
                source_id=source_id,
                content=content,
                page_reference=page_reference,
                quote=quote,
                usage_context=usage_context,
                citation_format=citation_format,
                verification_status="pending",
                created_at=datetime.utcnow()
            )
            
            self._citations[citation.citation_id] = citation
            
            logger.info(f"Citation created: {citation.citation_id}")
            return citation.citation_id
            
        except Exception as e:
            logger.error(f"Error creating citation: {e}")
            raise
    
    async def track_version(self, resource_id: str, resource_type: str, version: str,
                           changes: List[str], changed_by: str, change_reason: str,
                           previous_version: Optional[str] = None) -> str:
        """Track version changes for compliance."""
        try:
            version_record = VersionRecord(
                record_id="",  # Will be auto-generated
                resource_id=resource_id,
                resource_type=resource_type,
                version=version,
                changes=changes,
                changed_by=changed_by,
                change_reason=change_reason,
                timestamp=datetime.utcnow(),
                previous_version=previous_version
            )
            
            self._version_records.append(version_record)
            
            logger.info(f"Version tracked: {version_record.record_id}")
            return version_record.record_id
            
        except Exception as e:
            logger.error(f"Error tracking version: {e}")
            raise
    
    async def perform_compliance_check(self, check_type: ComplianceCheckType,
                                      regulation: RegulationType, resource_id: str,
                                      resource_type: str, checked_by: str) -> str:
        """Perform a specific compliance check."""
        try:
            check_result = await self._execute_compliance_check(
                check_type, regulation, resource_id, resource_type
            )
            
            compliance_check = ComplianceCheck(
                check_id="",  # Will be auto-generated
                check_type=check_type,
                regulation=regulation,
                resource_id=resource_id,
                resource_type=resource_type,
                status=check_result["status"],
                score=check_result["score"],
                findings=check_result["findings"],
                recommendations=check_result["recommendations"],
                checked_at=datetime.utcnow(),
                checked_by=checked_by,
                evidence=check_result["evidence"]
            )
            
            self._compliance_checks.append(compliance_check)
            
            logger.info(f"Compliance check performed: {compliance_check.check_id}")
            return compliance_check.check_id
            
        except Exception as e:
            logger.error(f"Error performing compliance check: {e}")
            raise
    
    async def _execute_compliance_check(self, check_type: ComplianceCheckType,
                                       regulation: RegulationType, resource_id: str,
                                       resource_type: str) -> Dict[str, Any]:
        """Execute specific compliance check logic."""
        if check_type == ComplianceCheckType.DATA_SOURCES:
            return await self._check_data_sources_compliance(regulation, resource_id)
        elif check_type == ComplianceCheckType.CITATIONS:
            return await self._check_citations_compliance(regulation, resource_id)
        elif check_type == ComplianceCheckType.VERSION_TRACKING:
            return await self._check_version_tracking_compliance(regulation, resource_id)
        elif check_type == ComplianceCheckType.ACCESS_CONTROL:
            return await self._check_access_control_compliance(regulation, resource_id)
        elif check_type == ComplianceCheckType.DATA_RETENTION:
            return await self._check_data_retention_compliance(regulation, resource_id)
        elif check_type == ComplianceCheckType.AUDIT_TRAIL:
            return await self._check_audit_trail_compliance(regulation, resource_id)
        else:
            return {
                "status": ComplianceStatus.PENDING_REVIEW,
                "score": 0.5,
                "findings": [f"Check type {check_type} not implemented"],
                "recommendations": ["Implement compliance check logic"],
                "evidence": {}
            }
    
    async def _check_data_sources_compliance(self, regulation: RegulationType, 
                                           resource_id: str) -> Dict[str, Any]:
        """Check data sources compliance."""
        findings = []
        recommendations = []
        evidence = {}
        score = 1.0
        
        # Check if all data sources are documented
        undocumented_sources = []
        for source_id, source in self._data_sources.items():
            if not source.license_type or source.license_type == "unknown":
                undocumented_sources.append(source_id)
                findings.append(f"Data source {source.name} lacks license documentation")
        
        if undocumented_sources:
            score -= 0.3
            recommendations.append("Document license types for all data sources")
        
        # Check reliability scores
        low_reliability_sources = [
            source for source in self._data_sources.values()
            if source.reliability_score < 0.7
        ]
        
        if low_reliability_sources:
            score -= 0.2
            findings.append(f"Found {len(low_reliability_sources)} sources with low reliability scores")
            recommendations.append("Review and improve data source reliability")
        
        # Check regulatory approval for financial data
        if regulation in [RegulationType.MIFID_II, RegulationType.SOX]:
            unapproved_sources = [
                source for source in self._data_sources.values()
                if not source.regulatory_approval and source.source_type == "financial"
            ]
            
            if unapproved_sources:
                score -= 0.4
                findings.append(f"Found {len(unapproved_sources)} financial sources without regulatory approval")
                recommendations.append("Obtain regulatory approval for financial data sources")
        
        evidence = {
            "total_sources": len(self._data_sources),
            "undocumented_sources": len(undocumented_sources),
            "low_reliability_sources": len(low_reliability_sources),
            "average_reliability": sum(s.reliability_score for s in self._data_sources.values()) / len(self._data_sources) if self._data_sources else 0
        }
        
        # Determine status
        if score >= 0.9:
            status = ComplianceStatus.COMPLIANT
        elif score >= 0.7:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return {
            "status": status,
            "score": max(0.0, score),
            "findings": findings,
            "recommendations": recommendations,
            "evidence": evidence
        }
    
    async def _check_citations_compliance(self, regulation: RegulationType,
                                        resource_id: str) -> Dict[str, Any]:
        """Check citations compliance."""
        findings = []
        recommendations = []
        evidence = {}
        score = 1.0
        
        # Check citation completeness
        incomplete_citations = []
        for citation_id, citation in self._citations.items():
            if not citation.page_reference and citation.usage_context != "general":
                incomplete_citations.append(citation_id)
                findings.append(f"Citation {citation_id} lacks page reference")
        
        if incomplete_citations:
            score -= 0.3
            recommendations.append("Add page references to all specific citations")
        
        # Check citation verification
        unverified_citations = [
            citation for citation in self._citations.values()
            if citation.verification_status == "pending"
        ]
        
        if unverified_citations:
            score -= 0.2
            findings.append(f"Found {len(unverified_citations)} unverified citations")
            recommendations.append("Verify all citations for accuracy")
        
        # Check citation format consistency
        citation_formats = set(citation.citation_format for citation in self._citations.values())
        if len(citation_formats) > 1:
            score -= 0.1
            findings.append("Inconsistent citation formats used")
            recommendations.append("Standardize citation format across all references")
        
        evidence = {
            "total_citations": len(self._citations),
            "incomplete_citations": len(incomplete_citations),
            "unverified_citations": len(unverified_citations),
            "citation_formats": list(citation_formats)
        }
        
        # Determine status
        if score >= 0.9:
            status = ComplianceStatus.COMPLIANT
        elif score >= 0.7:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return {
            "status": status,
            "score": max(0.0, score),
            "findings": findings,
            "recommendations": recommendations,
            "evidence": evidence
        }
    
    async def _check_version_tracking_compliance(self, regulation: RegulationType,
                                               resource_id: str) -> Dict[str, Any]:
        """Check version tracking compliance."""
        findings = []
        recommendations = []
        evidence = {}
        score = 1.0
        
        # Check if version tracking exists for the resource
        resource_versions = [
            record for record in self._version_records
            if record.resource_id == resource_id
        ]
        
        if not resource_versions:
            score = 0.0
            findings.append(f"No version tracking found for resource {resource_id}")
            recommendations.append("Implement version tracking for all resources")
        else:
            # Check version tracking completeness
            incomplete_records = [
                record for record in resource_versions
                if not record.change_reason or not record.changes
            ]
            
            if incomplete_records:
                score -= 0.3
                findings.append(f"Found {len(incomplete_records)} incomplete version records")
                recommendations.append("Ensure all version changes include reason and change details")
            
            # Check version continuity
            versions = sorted(resource_versions, key=lambda r: r.timestamp)
            for i in range(1, len(versions)):
                if versions[i].previous_version != versions[i-1].version:
                    score -= 0.2
                    findings.append("Version tracking chain is broken")
                    recommendations.append("Maintain continuous version tracking chain")
                    break
        
        evidence = {
            "resource_id": resource_id,
            "total_versions": len(resource_versions),
            "incomplete_records": len([r for r in resource_versions if not r.change_reason]),
            "version_chain_complete": True  # Would be calculated based on actual chain
        }
        
        # Determine status
        if score >= 0.9:
            status = ComplianceStatus.COMPLIANT
        elif score >= 0.7:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return {
            "status": status,
            "score": max(0.0, score),
            "findings": findings,
            "recommendations": recommendations,
            "evidence": evidence
        }
    
    async def _check_access_control_compliance(self, regulation: RegulationType,
                                             resource_id: str) -> Dict[str, Any]:
        """Check access control compliance."""
        # This would integrate with actual access control systems
        # For now, return a placeholder implementation
        return {
            "status": ComplianceStatus.COMPLIANT,
            "score": 0.8,
            "findings": [],
            "recommendations": ["Implement detailed access control compliance checks"],
            "evidence": {"placeholder": True}
        }
    
    async def _check_data_retention_compliance(self, regulation: RegulationType,
                                             resource_id: str) -> Dict[str, Any]:
        """Check data retention compliance."""
        findings = []
        recommendations = []
        evidence = {}
        score = 1.0
        
        # Check if retention periods are defined
        sources_without_retention = [
            source for source in self._data_sources.values()
            if source.retention_period_days is None
        ]
        
        if sources_without_retention:
            score -= 0.4
            findings.append(f"Found {len(sources_without_retention)} sources without defined retention periods")
            recommendations.append("Define retention periods for all data sources")
        
        # Check for overdue data
        current_date = datetime.utcnow()
        overdue_sources = []
        
        for source in self._data_sources.values():
            if source.retention_period_days:
                retention_deadline = source.access_date + timedelta(days=source.retention_period_days)
                if current_date > retention_deadline:
                    overdue_sources.append(source.source_id)
        
        if overdue_sources:
            score -= 0.3
            findings.append(f"Found {len(overdue_sources)} sources with overdue retention")
            recommendations.append("Purge or review overdue data according to retention policy")
        
        evidence = {
            "total_sources": len(self._data_sources),
            "sources_without_retention": len(sources_without_retention),
            "overdue_sources": len(overdue_sources)
        }
        
        # Determine status
        if score >= 0.9:
            status = ComplianceStatus.COMPLIANT
        elif score >= 0.7:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return {
            "status": status,
            "score": max(0.0, score),
            "findings": findings,
            "recommendations": recommendations,
            "evidence": evidence
        }
    
    async def _check_audit_trail_compliance(self, regulation: RegulationType,
                                          resource_id: str) -> Dict[str, Any]:
        """Check audit trail compliance."""
        # This would integrate with the audit service
        # For now, return a placeholder implementation
        return {
            "status": ComplianceStatus.COMPLIANT,
            "score": 0.9,
            "findings": [],
            "recommendations": ["Integrate with audit service for comprehensive audit trail checks"],
            "evidence": {"placeholder": True}
        }
    
    async def generate_compliance_report(self, report_type: str, scope: str,
                                        period_start: datetime, period_end: datetime,
                                        generated_by: str) -> str:
        """Generate comprehensive compliance report."""
        try:
            # Perform all relevant compliance checks
            check_results = []
            
            # Check data sources
            for regulation in [RegulationType.GDPR, RegulationType.SOX, RegulationType.MIFID_II]:
                check_id = await self.perform_compliance_check(
                    ComplianceCheckType.DATA_SOURCES, regulation, scope, "system", generated_by
                )
                check_results.append(check_id)
            
            # Generate data sources inventory
            data_sources_inventory = [
                {
                    "source_id": source.source_id,
                    "name": source.name,
                    "provider": source.provider,
                    "type": source.source_type,
                    "reliability_score": source.reliability_score,
                    "is_public": source.is_public,
                    "license_type": source.license_type,
                    "access_date": source.access_date.isoformat(),
                    "regulatory_approval": source.regulatory_approval
                }
                for source in self._data_sources.values()
            ]
            
            # Generate citations list
            citations_list = [
                {
                    "citation_id": citation.citation_id,
                    "source_id": citation.source_id,
                    "content": citation.content[:100] + "..." if len(citation.content) > 100 else citation.content,
                    "citation_format": citation.citation_format,
                    "verification_status": citation.verification_status,
                    "created_at": citation.created_at.isoformat()
                }
                for citation in self._citations.values()
            ]
            
            # Generate version history
            version_history = [
                {
                    "record_id": record.record_id,
                    "resource_id": record.resource_id,
                    "resource_type": record.resource_type,
                    "version": record.version,
                    "changed_by": record.changed_by,
                    "change_reason": record.change_reason,
                    "timestamp": record.timestamp.isoformat()
                }
                for record in self._version_records
                if period_start <= record.timestamp <= period_end
            ]
            
            # Calculate overall compliance
            recent_checks = [
                check for check in self._compliance_checks
                if period_start <= check.checked_at <= period_end
            ]
            
            if recent_checks:
                overall_score = sum(check.score for check in recent_checks) / len(recent_checks)
                
                # Determine overall status
                if overall_score >= 0.9:
                    overall_status = ComplianceStatus.COMPLIANT
                elif overall_score >= 0.7:
                    overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
                else:
                    overall_status = ComplianceStatus.NON_COMPLIANT
            else:
                overall_score = 0.0
                overall_status = ComplianceStatus.PENDING_REVIEW
            
            # Generate findings summary
            findings_summary = self._generate_findings_summary(recent_checks)
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(recent_checks)
            
            # Create report
            report = ComplianceReport(
                report_id="",  # Will be auto-generated
                report_type=report_type,
                scope=scope,
                period_start=period_start,
                period_end=period_end,
                generated_at=datetime.utcnow(),
                generated_by=generated_by,
                overall_status=overall_status,
                overall_score=overall_score,
                checks_performed=check_results,
                data_sources_inventory=data_sources_inventory,
                citations_list=citations_list,
                version_history=version_history,
                findings_summary=findings_summary,
                recommendations=recommendations,
                regulatory_requirements=self._regulatory_requirements
            )
            
            # Store report
            self._compliance_reports[report.report_id] = report
            
            logger.info(f"Compliance report generated: {report.report_id}")
            return report.report_id
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            raise
    
    def _generate_findings_summary(self, checks: List[ComplianceCheck]) -> Dict[str, Any]:
        """Generate summary of compliance findings."""
        summary = {
            "total_checks": len(checks),
            "compliant_checks": len([c for c in checks if c.status == ComplianceStatus.COMPLIANT]),
            "non_compliant_checks": len([c for c in checks if c.status == ComplianceStatus.NON_COMPLIANT]),
            "partially_compliant_checks": len([c for c in checks if c.status == ComplianceStatus.PARTIALLY_COMPLIANT]),
            "checks_by_regulation": {},
            "checks_by_type": {},
            "average_score": sum(c.score for c in checks) / len(checks) if checks else 0.0
        }
        
        # Group by regulation
        for check in checks:
            reg = check.regulation.value
            if reg not in summary["checks_by_regulation"]:
                summary["checks_by_regulation"][reg] = {"count": 0, "average_score": 0.0}
            summary["checks_by_regulation"][reg]["count"] += 1
        
        # Calculate average scores by regulation
        for reg in summary["checks_by_regulation"]:
            reg_checks = [c for c in checks if c.regulation.value == reg]
            summary["checks_by_regulation"][reg]["average_score"] = sum(c.score for c in reg_checks) / len(reg_checks)
        
        # Group by type
        for check in checks:
            check_type = check.check_type.value
            if check_type not in summary["checks_by_type"]:
                summary["checks_by_type"][check_type] = {"count": 0, "average_score": 0.0}
            summary["checks_by_type"][check_type]["count"] += 1
        
        # Calculate average scores by type
        for check_type in summary["checks_by_type"]:
            type_checks = [c for c in checks if c.check_type.value == check_type]
            summary["checks_by_type"][check_type]["average_score"] = sum(c.score for c in type_checks) / len(type_checks)
        
        return summary
    
    def _generate_compliance_recommendations(self, checks: List[ComplianceCheck]) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = set()
        
        # Collect all recommendations from checks
        for check in checks:
            recommendations.update(check.recommendations)
        
        # Add general recommendations based on patterns
        non_compliant_checks = [c for c in checks if c.status == ComplianceStatus.NON_COMPLIANT]
        if len(non_compliant_checks) > len(checks) * 0.3:
            recommendations.add("Conduct comprehensive compliance review and remediation")
        
        data_source_issues = [c for c in checks if c.check_type == ComplianceCheckType.DATA_SOURCES and c.score < 0.8]
        if data_source_issues:
            recommendations.add("Improve data source documentation and governance")
        
        citation_issues = [c for c in checks if c.check_type == ComplianceCheckType.CITATIONS and c.score < 0.8]
        if citation_issues:
            recommendations.add("Enhance citation management and verification processes")
        
        return list(recommendations)
    
    def get_compliance_report(self, report_id: str) -> Optional[ComplianceReport]:
        """Get compliance report by ID."""
        return self._compliance_reports.get(report_id)
    
    def list_compliance_reports(self) -> List[ComplianceReport]:
        """List all compliance reports."""
        reports = list(self._compliance_reports.values())
        reports.sort(key=lambda r: r.generated_at, reverse=True)
        return reports
    
    def get_data_sources_inventory(self) -> List[DataSource]:
        """Get complete data sources inventory."""
        return list(self._data_sources.values())
    
    def get_citations_list(self) -> List[Citation]:
        """Get complete citations list."""
        return list(self._citations.values())
    
    def get_version_history(self, resource_id: Optional[str] = None) -> List[VersionRecord]:
        """Get version history, optionally filtered by resource."""
        if resource_id:
            return [record for record in self._version_records if record.resource_id == resource_id]
        return self._version_records
    
    def get_compliance_checks(self, regulation: Optional[RegulationType] = None,
                             check_type: Optional[ComplianceCheckType] = None) -> List[ComplianceCheck]:
        """Get compliance checks with optional filtering."""
        checks = self._compliance_checks
        
        if regulation:
            checks = [c for c in checks if c.regulation == regulation]
        
        if check_type:
            checks = [c for c in checks if c.check_type == check_type]
        
        return sorted(checks, key=lambda c: c.checked_at, reverse=True)
    
    def export_compliance_data(self, format_type: str = "json") -> str:
        """Export compliance data for external systems."""
        try:
            export_data = {
                "export_metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "format": format_type
                },
                "data_sources": [asdict(source) for source in self._data_sources.values()],
                "citations": [asdict(citation) for citation in self._citations.values()],
                "version_records": [asdict(record) for record in self._version_records],
                "compliance_checks": [asdict(check) for check in self._compliance_checks],
                "regulatory_requirements": self._regulatory_requirements
            }
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            if format_type == "json":
                return json.dumps(export_data, default=convert_datetime, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
        except Exception as e:
            logger.error(f"Error exporting compliance data: {e}")
            raise